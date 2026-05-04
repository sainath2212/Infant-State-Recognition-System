"""
CryNetV2 — Dual-Stream DST × Mel Cross-Attention Hybrid Architecture.

Architecture Overview
=====================
                    ┌──────────────────────────────────┐
                    │         Input Audio (3s)          │
                    └──────────────┬───────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                                          ▼
    ┌─────────────────┐                     ┌──────────────────────┐
    │  Mel-Spectrogram │                    │   DST Spectrogram    │
    │  (1 × 128 × 188) │                   │   (1 ×  64 × 188)    │
    └────────┬─────────┘                   └──────────┬───────────┘
             │                                        │
    ┌────────▼──────────┐               ┌─────────────▼───────────┐
    │  CNN-SE Encoder   │               │   CNN-SE Encoder         │
    │  (3 conv blocks)  │               │   (3 conv blocks)        │
    │  32→64→128 ch     │               │   32→64→128 ch           │
    └────────┬──────────┘               └─────────────┬───────────┘
             │                                        │
    ┌────────▼──────────┐               ┌─────────────▼───────────┐
    │   BiLSTM Encoder  │               │   BiLSTM Encoder         │
    │   (2L, 256-dim)   │               │   (2L, 256-dim)          │
    └────────┬──────────┘               └─────────────┬───────────┘
             │  (B, 23, 256)                          │  (B, 23, 256)
             └──────────────────┬─────────────────────┘
                                │
                   ┌────────────▼──────────────┐
                   │  Cross-Attention Fusion    │
                   │  Mel→DST  +  DST→Mel       │
                   │  → Concat → Linear(512→256)│
                   └────────────┬──────────────┘
                                │  (B, 23, 256)
                   ┌────────────▼──────────────┐
                   │  Transformer Encoder (2L)  │
                   │  8 heads, d=256            │
                   └────────────┬──────────────┘
                                │
                   ┌────────────▼──────────────┐
                   │   Attention Pooling        │
                   └────────────┬──────────────┘
                                │  (B, 256)
                   ┌────────────▼──────────────┐
                   │   Classification Head      │
                   │  256→128→64→8 classes      │
                   └───────────────────────────┘

Key Innovations
===============
1. Dual-Stream Processing: Mel captures perceptual spectral envelope;
   DST captures adaptive time-frequency energy (frequency-dependent resolution)

2. Cross-Attention Fusion: Each stream attends to the other's features,
   enabling inter-modal reasoning about spectro-temporal patterns

3. Transformer Encoder: Global context modeling after cross-modal fusion

4. EnsembleCryNet: Meta-ensemble combining CryNet + CryNetV2 + ML tier

References
==========
- Jayasree & Blessy (2025): DST for infant cry classification
- Vaswani et al. (2017): Attention Is All You Need (Transformer)
- He et al. (2016): Deep Residual Learning (skip connections)
- Hu et al. (2018): Squeeze-and-Excitation Networks
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse building blocks from original CryNet
from src.dl_model import SqueezeExcitationBlock, ConvBlock, AttentionPooling


# ============================================================
# Cross-Attention Block
# ============================================================

class CrossAttentionBlock(nn.Module):
    """
    Cross-Modal Attention: Query one stream with the other's Keys and Values.

    Given two sequence representations Q_src and KV_src:
        CrossAttn(Q, K, V) = softmax(QKᵀ / √d_k) · V

    This allows the Mel-stream to attend to DST features and vice-versa,
    learning which DST time-frequency patterns correspond to which Mel features.

    Parameters
    ----------
    embed_dim : int
        Query/Key/Value embedding dimension.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout on attention weights.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        # Xavier init
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        query_src: torch.Tensor,
        kv_src: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query_src : (B, T, D)  — generates Queries
        kv_src    : (B, T, D)  — generates Keys and Values

        Returns
        -------
        out : (B, T, D)
        """
        B, T, D = query_src.shape
        residual = query_src

        Q = self.W_q(query_src).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(kv_src).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(kv_src).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)

        context = torch.matmul(attn_w, V)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(context)

        return self.norm(out + residual)


# ============================================================
# Single-Stream Encoder (shared architecture for Mel and DST)
# ============================================================

class StreamEncoder(nn.Module):
    """
    CNN-SE-BiLSTM encoder for a single spectrogram stream.

    Processes either Mel (128 mel-bins) or DST (64 freq-bins) as input,
    producing a temporal sequence of 256-dimensional embeddings.

    Parameters
    ----------
    in_height : int
        Number of frequency bins (128 for Mel, 64 for DST).
    dropout : float
        Spatial dropout after CNN; recurrent dropout in BiLSTM.
    """

    def __init__(self, in_height: int = 128, dropout: float = 0.3):
        super().__init__()

        # --- Convolutional Stage ---
        # Block 1: 1 → 32 channels, halves spatial dims
        self.conv1 = ConvBlock(1, 32, kernel_size=3, padding=1, pool_size=2)
        # Block 2: 32 → 64 channels + SE channel attention
        self.conv2 = ConvBlock(32, 64, kernel_size=3, padding=1, pool_size=2)
        self.se = SqueezeExcitationBlock(64, reduction=4)
        # Block 3: 64 → 128 channels with residual skip
        self.conv3 = ConvBlock(64, 128, kernel_size=3, padding=1, pool_size=2,
                               use_residual=True)
        self.cnn_dropout = nn.Dropout2d(dropout)

        # Compute temporal projection dimension after 3× pooling
        # Mel: 128 → 64 → 32 → 16  ⟹  128 × 16 = 2048 per time step
        # DST:  64 → 32 → 16 →  8  ⟹  128 ×  8 = 1024 per time step
        proj_in = 128 * (in_height // 8)

        # --- Temporal Projection ---
        self.temporal_proj = nn.Linear(proj_in, 256)
        self.temporal_ln = nn.LayerNorm(256)

        # --- BiLSTM Stage ---
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=128,     # × 2 directions = 256
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstm_ln = nn.LayerNorm(256)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.temporal_proj.weight)
        nn.init.zeros_(self.temporal_proj.bias)
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)  # forget gate bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W)  — single-channel spectrogram

        Returns
        -------
        h : (B, W', 256)  — temporal feature sequence
        """
        # CNN
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.se(h)
        h = self.conv3(h)
        h = self.cnn_dropout(h)

        # Reshape: (B, C, H', W') → (B, W', C*H')
        B, C, Hp, Wp = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(B, Wp, C * Hp)

        # Project + normalize
        h = self.temporal_proj(h)
        h = self.temporal_ln(h)

        # BiLSTM
        h, _ = self.bilstm(h)
        h = self.lstm_ln(h)

        return h  # (B, Wp, 256)


# ============================================================
# Cross-Modal Fusion Module
# ============================================================

class CrossModalFusion(nn.Module):
    """
    Bidirectional cross-attention fusion of Mel and DST streams.

    Step 1: Mel attends to DST  → mel_enriched    (B, T, 256)
    Step 2: DST attends to Mel  → dst_enriched    (B, T, 256)
    Step 3: Concatenate both    → (B, T, 512)
    Step 4: Project back        → (B, T, 256)

    This symmetric design lets each modality incorporate
    complementary information from the other.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mel_to_dst = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.dst_to_mel = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        nn.init.xavier_uniform_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)

    def forward(
        self,
        mel_features: torch.Tensor,  # (B, T, 256)
        dst_features: torch.Tensor,  # (B, T, 256)
    ) -> torch.Tensor:
        """
        Returns
        -------
        fused : (B, T, 256)
        """
        mel_enriched = self.mel_to_dst(query_src=mel_features, kv_src=dst_features)
        dst_enriched = self.dst_to_mel(query_src=dst_features, kv_src=mel_features)
        combined = torch.cat([mel_enriched, dst_enriched], dim=-1)
        fused = self.fusion_proj(combined)
        return self.fusion_norm(fused)


# ============================================================
# CryNetV2 — Main Hybrid Model
# ============================================================

class CryNetV2(nn.Module):
    """
    CryNetV2: Dual-Stream DST × Mel Cross-Attention Hybrid Architecture.

    Combines complementary spectro-temporal representations:
    - Mel-Spectrogram: perceptually-weighted, fixed frequency resolution
    - DST-Spectrogram: physically-principled, adaptive frequency resolution

    Parameters
    ----------
    n_classes : int
        Number of output classes (8 for infant cry).
    n_mels : int
        Number of Mel filterbank bins (height of Mel input).
    n_dst_freqs : int
        Number of DST frequency bins (height of DST input).
    dropout : float
        Dropout probability.
    num_transformer_layers : int
        Number of Transformer encoder layers after cross-attention fusion.
    """

    def __init__(
        self,
        n_classes: int = 8,
        n_mels: int = 128,
        n_dst_freqs: int = 64,
        dropout: float = 0.3,
        num_transformer_layers: int = 2,
    ):
        super().__init__()

        self.n_classes = n_classes

        # --- Stream Encoders ---
        self.mel_encoder = StreamEncoder(in_height=n_mels, dropout=dropout)
        self.dst_encoder = StreamEncoder(in_height=n_dst_freqs, dropout=dropout)

        # --- Cross-Modal Fusion ---
        self.cross_fusion = CrossModalFusion(embed_dim=256, num_heads=8, dropout=0.1)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # --- Attention Pooling ---
        self.attention_pool = AttentionPooling(input_dim=256)

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        mel: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mel : (B, 1, n_mels, T)   — Mel-spectrogram
        dst : (B, 1, n_dst, T)    — DST-spectrogram

        Returns
        -------
        logits : (B, n_classes)
        """
        # Encode each stream
        mel_feat = self.mel_encoder(mel)   # (B, T', 256)
        dst_feat = self.dst_encoder(dst)   # (B, T', 256)

        # Align temporal dimension (both should be same after 3× pooling)
        T = min(mel_feat.size(1), dst_feat.size(1))
        mel_feat = mel_feat[:, :T, :]
        dst_feat = dst_feat[:, :T, :]

        # Cross-modal fusion
        fused = self.cross_fusion(mel_feat, dst_feat)   # (B, T, 256)

        # Global Transformer context
        fused = self.transformer(fused)                  # (B, T, 256)

        # Temporal aggregation
        pooled = self.attention_pool(fused)              # (B, 256)

        # Classification
        logits = self.classifier(pooled)                 # (B, n_classes)
        return logits

    def get_embeddings(self, mel: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Extract penultimate-layer embeddings for visualization."""
        mel_feat = self.mel_encoder(mel)
        dst_feat = self.dst_encoder(dst)
        T = min(mel_feat.size(1), dst_feat.size(1))
        fused = self.cross_fusion(mel_feat[:, :T], dst_feat[:, :T])
        fused = self.transformer(fused)
        return self.attention_pool(fused)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================
# Meta-Ensemble: ML + DL Fusion
# ============================================================

class HybridEnsemble(nn.Module):
    """
    Full Hybrid Ensemble combining ML and DL predictions.

    Prediction Flow:
    ----------------
    1. ML Ensemble  →  ml_proba  (N, 8)
    2. CryNet       →  dl1_proba (N, 8)
    3. CryNetV2     →  dl2_proba (N, 8)
    4. Weighted avg →  final_proba

    Weights are empirically set (tunable):
        ML:       0.20
        CryNet:   0.30
        CryNetV2: 0.50

    Parameters
    ----------
    crynet : nn.Module
        Trained original CryNet.
    crynetv2 : nn.Module
        Trained CryNetV2.
    ml_ensemble : MLEnsemble
        Fitted ML ensemble.
    """

    def __init__(
        self,
        crynet: nn.Module,
        crynetv2: nn.Module,
        ml_ensemble,
        weights: tuple = (0.20, 0.30, 0.50),
        device: torch.device = None,
    ):
        super().__init__()
        self.crynet = crynet
        self.crynetv2 = crynetv2
        self.ml_ensemble = ml_ensemble
        self.w_ml, self.w_dl1, self.w_dl2 = weights
        self.device = device or torch.device('cpu')

    @torch.no_grad()
    def predict_proba_dl(
        self,
        mel: torch.Tensor,
        dst: torch.Tensor,
    ) -> tuple:
        """
        Get soft probability outputs from both DL models.

        Returns
        -------
        (crynet_proba, crynetv2_proba) : tuple of (N, 8) tensors
        """
        self.crynet.eval()
        self.crynetv2.eval()

        # CryNet takes only Mel
        logits1 = self.crynet(mel)
        proba1 = F.softmax(logits1, dim=1)

        # CryNetV2 takes both Mel + DST
        logits2 = self.crynetv2(mel, dst)
        proba2 = F.softmax(logits2, dim=1)

        return proba1, proba2

    def predict_proba_full(
        self,
        mel: torch.Tensor,
        dst: torch.Tensor,
        ml_features: np.ndarray,
    ) -> np.ndarray:
        """
        Full ensemble soft probability prediction.

        Parameters
        ----------
        mel : (B, 1, 128, 188)
        dst : (B, 1, 64, 188)
        ml_features : (B, n_ml_features)

        Returns
        -------
        proba : np.ndarray of shape (B, 8)
        """
        p1, p2 = self.predict_proba_dl(mel, dst)
        p_ml = self.ml_ensemble.predict_proba(ml_features)

        p1_np = p1.cpu().numpy()
        p2_np = p2.cpu().numpy()

        proba = (
            self.w_ml  * p_ml  +
            self.w_dl1 * p1_np +
            self.w_dl2 * p2_np
        )
        return proba

    def predict(
        self,
        mel: torch.Tensor,
        dst: torch.Tensor,
        ml_features: np.ndarray,
    ) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba_full(mel, dst, ml_features), axis=1)
