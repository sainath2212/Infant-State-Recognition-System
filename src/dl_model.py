"""
CryNet — Hybrid CNN-BiLSTM-Attention Architecture for Infant Cry Recognition.

This module implements the complete CryNet architecture from scratch using
only PyTorch primitives. Every component is hand-built with mathematical
justification — no pre-trained weights or library architectures are used.

Architecture Overview:
=====================

Input: Mel-Spectrogram (1 × 128 × 188) — single-channel spectro-temporal image

Stage 1: Convolutional Feature Extractor
  - 3 convolutional blocks with increasing channel depth (32 → 64 → 128)
  - Each block: Conv2D → BatchNorm → GELU → MaxPool
  - Squeeze-and-Excitation (SE) block after 2nd conv for channel attention
  - Residual skip connection from Block 2 to Block 3

Stage 2: Temporal Encoder
  - Reshape CNN output: (B, C, H', W') → (B, W', C×H')
  - 2-layer Bidirectional LSTM captures forward and backward temporal context
  - Layer Normalization for stable training

Stage 3: Multi-Head Self-Attention
  - 4-head scaled dot-product attention over temporal sequence
  - Learns which time steps are most discriminative
  - Residual connection + LayerNorm

Stage 4: Classification Head
  - Attention-weighted temporal pooling
  - FC(256→128) → GELU → Dropout → FC(128→8)

Theoretical Foundations:
=======================

1. **Convolutions exploit translation equivariance**: spectral patterns
   (harmonics, formants) can appear at any frequency/time position.
   Weight sharing across spatial positions gives  O(k²·C) params instead
   of O(H·W·C).

2. **Batch Normalization** (Ioffe & Szegedy, 2015): normalizes activations
   to N(0,1) per channel, reducing internal covariate shift. This allows
   higher learning rates and makes training less sensitive to initialization.

3. **GELU activation** (Hendrycks & Gimpel, 2016): x·Φ(x) where Φ is the
   standard Gaussian CDF. Smoother than ReLU, avoids dead neurons. Used in
   BERT, GPT, and modern audio models.

4. **Squeeze-and-Excitation** (Hu et al., 2018): learns channel-wise
   attention via global average pooling → FC → ReLU → FC → Sigmoid.
   This allows the network to emphasize informative frequency bands
   (e.g., fundamental frequency vs. harmonics) adaptively per sample.

5. **Bidirectional LSTM**: captures both past→future and future→past
   temporal dependencies. Critical because cry patterns have meaningful
   onset, sustain, and release phases that inform classification.

6. **Multi-Head Self-Attention** (Vaswani et al., 2017):
   Attention(Q,K,V) = softmax(QK^T / √d_k) · V
   Multiple heads allow attending to different aspects simultaneously
   (e.g., one head for onset, another for pitch contour).

7. **Residual connections** (He et al., 2016): enable gradient flow through
   deep networks by providing an identity shortcut. Solves the degradation
   problem where deeper networks paradoxically perform worse.

8. **Weight Initialization**:
   - Conv layers: Kaiming He initialization (accounts for ReLU/GELU nonlinearity)
   - LSTM: Orthogonal initialization (preserves gradient magnitude across time)
   - Linear: Xavier/Glorot (balances variance for symmetric activations)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Building Block 1: Squeeze-and-Excitation (SE) Block
# ============================================================

class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel-wise attention.

    The SE block adaptively recalibrates channel-wise feature responses
    by explicitly modelling interdependencies between channels.

    Architecture:
        Input: (B, C, H, W)
        → Global Average Pool: (B, C, 1, 1) — "Squeeze"
        → FC(C → C//r) → ReLU — bottleneck compression
        → FC(C//r → C) → Sigmoid — "Excitation"
        → Scale input channels by excitation weights

    The reduction ratio r controls the bottleneck. r=16 is standard,
    but we use r=4 for our small channel counts to retain capacity.

    Mathematical formulation:
        z_c = (1/H·W) · Σ_i Σ_j u_c(i,j)          [squeeze]
        s = σ(W₂ · δ(W₁ · z))                       [excite]
        x̃_c = s_c · u_c                              [scale]

    Where δ = ReLU, σ = Sigmoid, and W₁ ∈ R^(C/r × C), W₂ ∈ R^(C × C/r).
    """

    def __init__(self, channels, reduction=4):
        """
        Parameters
        ----------
        channels : int
            Number of input/output channels.
        reduction : int
            Bottleneck reduction ratio.
        """
        super().__init__()
        mid_channels = max(channels // reduction, 4)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

        # Initialize excitation weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the FC layers."""
        for m in self.excitation:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        out : torch.Tensor
            Channel-recalibrated tensor, same shape as input.
        """
        B, C, H, W = x.shape

        # Squeeze: Global Average Pooling
        z = self.squeeze(x).view(B, C)

        # Excitation: bottleneck FC → channel weights
        s = self.excitation(z).view(B, C, 1, 1)

        # Scale: element-wise multiplication
        return x * s


# ============================================================
# Building Block 2: Convolutional Block with Optional Residual
# ============================================================

class ConvBlock(nn.Module):
    """
    A single convolutional block: Conv2D → BatchNorm → GELU → MaxPool.

    Optionally supports residual connections when input and output
    channels match (or via a 1×1 projection convolution).

    Why this specific design:
    - Conv2D extracts local spectro-temporal patterns
    - BatchNorm stabilizes training and acts as regularizer
    - GELU provides smooth, non-zero gradients everywhere
    - MaxPool(2,2) reduces spatial dimensions by 2x, increasing
      the receptive field of subsequent layers
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, pool_size=2, use_residual=False):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels (filters).
        kernel_size : int
            Spatial extent of convolutional filters.
        stride : int
            Convolution stride.
        padding : int
            Zero-padding applied to input.
        pool_size : int
            MaxPool kernel size. Set to 0 to skip pooling.
        use_residual : bool
            Whether to add a residual/skip connection.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.pool = nn.MaxPool2d(pool_size) if pool_size > 0 else nn.Identity()
        self.use_residual = use_residual

        # 1×1 projection for residual when channels change
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif use_residual:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = None

        # Kaiming He initialization for Conv layers with GELU
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming He initialization for convolutional layers.

        For layers followed by ReLU/GELU, Kaiming initialization sets:
            Var(w) = 2 / fan_in
        This preserves the variance of activations across layers,
        preventing vanishing/exploding activations in deep networks.
        """
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out',
                                nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x):
        """Forward pass with optional residual connection."""
        identity = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)

        if self.use_residual and self.residual_proj is not None:
            # Residual must match spatial dims after pooling
            identity = self.residual_proj(identity)
            if isinstance(self.pool, nn.MaxPool2d):
                identity = F.adaptive_avg_pool2d(
                    identity, output_size=out.shape[2:]
                )
            out = out + identity

        out = self.pool(out)
        return out


# ============================================================
# Building Block 3: Multi-Head Self-Attention
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism (Vaswani et al., 2017).

    Implemented from scratch — no nn.MultiheadAttention used.

    The attention mechanism computes:
        Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

    Where Q (queries), K (keys), V (values) are linear projections
    of the input. Multiple heads allow the model to jointly attend
    to information from different representation subspaces:

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
        where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

    For infant cry analysis, different heads might learn to attend to:
    - Head 1: cry onset and attack phase
    - Head 2: sustained cry fundamental frequency
    - Head 3: breathing pauses between cry bursts
    - Head 4: cry offset and decay

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension.
    num_heads : int
        Number of attention heads. embed_dim must be divisible by num_heads.
    dropout : float
        Dropout rate for attention weights.
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Store attention weights for visualization
        self.attention_weights = None

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for attention projections."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (B, T, D) where
            B = batch, T = sequence length, D = embed_dim.

        Returns
        -------
        out : torch.Tensor
            Attention output of shape (B, T, D).
        """
        B, T, D = x.shape
        residual = x

        # Compute Q, K, V projections
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)  # (B, T, D)
        V = self.W_v(x)  # (B, T, D)

        # Reshape for multi-head: (B, T, D) → (B, h, T, d_k)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # scores: (B, h, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Softmax normalization
        attn_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attn_weights.detach()  # Store for visualization
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        # context: (B, h, T, d_k)
        context = torch.matmul(attn_weights, V)

        # Reshape back: (B, h, T, d_k) → (B, T, D)
        context = context.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        out = self.W_o(context)

        # Residual connection + Layer Normalization
        out = self.layer_norm(out + residual)

        return out


# ============================================================
# Building Block 4: Attention Pooling
# ============================================================

class AttentionPooling(nn.Module):
    """
    Learned attention-based temporal pooling.

    Instead of simple mean/max pooling over the time axis, we learn
    a weighted combination where the weights indicate the importance
    of each time step for classification.

    This is particularly useful for infant cry because:
    - Not all portions of the cry are equally informative
    - The onset (first 0.5s) often contains the most discriminative features
    - Breathing pauses should receive low attention
    - The sustained cry carries class-specific pitch information

    Formulation:
        e_t = tanh(W · h_t + b)     [energy function]
        α_t = softmax(e_t)           [attention weights]
        c = Σ α_t · h_t              [context vector]
    """

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1, bias=False)
        )
        self.pool_weights = None  # Store for visualization

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, T, D).

        Returns
        -------
        context : torch.Tensor
            Pooled output of shape (B, D).
        """
        # Compute attention energies: (B, T, 1)
        energies = self.attention(x)

        # Normalize to attention weights: (B, T, 1)
        weights = F.softmax(energies, dim=1)
        self.pool_weights = weights.detach()

        # Weighted sum: (B, D)
        context = (weights * x).sum(dim=1)

        return context


# ============================================================
# CryNet: Complete Architecture
# ============================================================

class CryNet(nn.Module):
    """
    CryNet — Hybrid CNN-BiLSTM-Attention for Infant Cry Classification.

    A novel architecture that combines:
    1. CNN for local spectro-temporal feature extraction
    2. SE blocks for adaptive channel attention
    3. BiLSTM for temporal sequence modeling
    4. Multi-Head Self-Attention for global context
    5. Attention pooling for discriminative temporal weighting

    This design reflects the multi-scale nature of infant cry signals:
    - Micro-scale (CNN): formant frequencies, harmonic structure
    - Meso-scale (BiLSTM): cry phrase dynamics (onset → sustain → release)
    - Macro-scale (Attention): cross-phrase patterns and global structure

    Parameters
    ----------
    n_classes : int
        Number of output classes (8 for infant cry).
    n_mels : int
        Number of mel frequency bins.
    dropout : float
        Dropout probability for regularization.
    """

    def __init__(self, n_classes=8, n_mels=128, dropout=0.3):
        super().__init__()

        self.n_classes = n_classes

        # ---- Stage 1: Convolutional Feature Extractor ----

        # Block 1: 1 → 32 channels
        # Input: (B, 1, 128, 188) → Output: (B, 32, 64, 94)
        self.conv1 = ConvBlock(1, 32, kernel_size=3, padding=1, pool_size=2)

        # Block 2: 32 → 64 channels
        # Input: (B, 32, 64, 94) → Output: (B, 64, 32, 47)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, padding=1, pool_size=2)

        # Squeeze-and-Excitation after Block 2
        self.se_block = SqueezeExcitationBlock(64, reduction=4)

        # Block 3: 64 → 128 channels with residual connection
        # Input: (B, 64, 32, 47) → Output: (B, 128, 16, 23)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, padding=1, pool_size=2,
                               use_residual=True)

        # Spatial dropout after CNN
        self.cnn_dropout = nn.Dropout2d(dropout)

        # ---- Stage 2: Temporal Encoder (BiLSTM) ----

        # After CNN: (B, 128, 16, 23) → reshape to (B, 23, 128*16) = (B, 23, 2048)
        # Project down to manageable size first
        self.temporal_proj = nn.Linear(128 * 16, 256)
        self.temporal_ln = nn.LayerNorm(256)

        # BiLSTM: processes temporal sequence
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=128,          # 128 per direction → 256 total
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Layer normalization after LSTM
        self.lstm_ln = nn.LayerNorm(256)  # 128 * 2 (bidirectional)

        # ---- Stage 3: Multi-Head Self-Attention ----
        self.self_attention = MultiHeadSelfAttention(
            embed_dim=256, num_heads=4, dropout=0.1
        )

        # ---- Stage 4: Attention Pooling ----
        self.attention_pool = AttentionPooling(256)

        # ---- Stage 5: Classification Head ----
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

        # Apply custom weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Apply architecture-appropriate weight initialization.

        Strategy:
        - Conv layers: Kaiming He (handled in ConvBlock)
        - LSTM: Orthogonal initialization preserves gradient norm
          across time steps, crucial for long sequences
        - Linear (classifier): Xavier/Glorot for symmetric activations
        """
        # LSTM: Orthogonal initialization
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden: Xavier
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-to-hidden: Orthogonal (preserves gradient norm)
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Forget gate bias = 1 (remember by default)
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

        # Classifier: Xavier initialization
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Temporal projection
        nn.init.xavier_uniform_(self.temporal_proj.weight)
        nn.init.zeros_(self.temporal_proj.bias)

    def forward(self, x):
        """
        Forward pass through the complete CryNet architecture.

        Parameters
        ----------
        x : torch.Tensor
            Input mel-spectrogram of shape (B, 1, n_mels, T).

        Returns
        -------
        logits : torch.Tensor
            Raw class scores of shape (B, n_classes).
        """
        # ---- Stage 1: CNN Feature Extraction ----
        h = self.conv1(x)          # (B, 32, 64, 94)
        h = self.conv2(h)          # (B, 64, 32, 47)
        h = self.se_block(h)       # (B, 64, 32, 47) — channel attention
        h = self.conv3(h)          # (B, 128, 16, 23)
        h = self.cnn_dropout(h)    # Spatial dropout

        # ---- Reshape for temporal processing ----
        B, C, H, W = h.shape
        # Treat width (time) as sequence, height×channels as features
        h = h.permute(0, 3, 1, 2)           # (B, W, C, H)
        h = h.contiguous().view(B, W, C * H) # (B, 23, 2048)

        # Project to LSTM input size
        h = self.temporal_proj(h)   # (B, 23, 256)
        h = self.temporal_ln(h)     # Layer norm

        # ---- Stage 2: BiLSTM ----
        h, _ = self.bilstm(h)       # (B, 23, 256)
        h = self.lstm_ln(h)         # Layer norm

        # ---- Stage 3: Self-Attention ----
        h = self.self_attention(h)   # (B, 23, 256)

        # ---- Stage 4: Attention Pooling ----
        h = self.attention_pool(h)   # (B, 256)

        # ---- Stage 5: Classification ----
        logits = self.classifier(h)  # (B, 8)

        return logits

    def get_embeddings(self, x):
        """
        Extract penultimate layer embeddings for visualization (t-SNE/UMAP).

        Parameters
        ----------
        x : torch.Tensor
            Input mel-spectrogram of shape (B, 1, n_mels, T).

        Returns
        -------
        embeddings : torch.Tensor
            Feature embeddings of shape (B, 256).
        """
        # Run through all stages except final classifier
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.se_block(h)
        h = self.conv3(h)
        h = self.cnn_dropout(h)

        B, C, H, W = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        h = self.temporal_proj(h)
        h = self.temporal_ln(h)

        h, _ = self.bilstm(h)
        h = self.lstm_ln(h)
        h = self.self_attention(h)
        h = self.attention_pool(h)

        return h

    def get_attention_maps(self):
        """
        Retrieve stored attention weights for visualization.

        Returns
        -------
        self_attn : torch.Tensor or None
            Multi-head self-attention weights (B, heads, T, T).
        pool_attn : torch.Tensor or None
            Temporal pooling attention weights (B, T, 1).
        """
        self_attn = self.self_attention.attention_weights
        pool_attn = self.attention_pool.pool_weights
        return self_attn, pool_attn

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================
# Ablation Variants — For Studying Component Contributions
# ============================================================

class CryNet_NoAttention(CryNet):
    """CryNet without multi-head self-attention (ablation study)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace attention with identity
        self.self_attention = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.se_block(h)
        h = self.conv3(h)
        h = self.cnn_dropout(h)

        B, C, H, W = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        h = self.temporal_proj(h)
        h = self.temporal_ln(h)

        h, _ = self.bilstm(h)
        h = self.lstm_ln(h)
        # No self-attention here
        h = self.attention_pool(h)
        logits = self.classifier(h)
        return logits


class CryNet_NoSE(CryNet):
    """CryNet without Squeeze-and-Excitation block (ablation study)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.se_block = nn.Identity()


class CryNet_CNNOnly(nn.Module):
    """CNN-only variant without BiLSTM or Attention (ablation study)."""

    def __init__(self, n_classes=8, n_mels=128, dropout=0.3):
        super().__init__()
        self.conv1 = ConvBlock(1, 32, kernel_size=3, padding=1, pool_size=2)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, padding=1, pool_size=2)
        self.se_block = SqueezeExcitationBlock(64, reduction=4)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, padding=1, pool_size=2)
        self.cnn_dropout = nn.Dropout2d(dropout)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.se_block(h)
        h = self.conv3(h)
        h = self.cnn_dropout(h)
        h = self.global_pool(h).view(h.size(0), -1)
        logits = self.classifier(h)
        return logits


class CryNet_NoSkip(CryNet):
    """CryNet without residual skip connections (ablation study)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace residual conv3 with non-residual
        self.conv3 = ConvBlock(64, 128, kernel_size=3, padding=1,
                               pool_size=2, use_residual=False)
