"""
Deep Learning Data Pipeline for Infant Cry Recognition — Phase 2.

This module implements a production-grade PyTorch data pipeline for
mel-spectrogram-based infant cry classification. It addresses several
critical challenges:

1. **Variable-Length Audio → Fixed-Size Spectrograms**
   Raw audio files vary in duration (0.5s–10s+). We standardize by
   computing mel-spectrograms and either padding or truncating to a
   fixed number of time frames, ensuring uniform tensor dimensions.

2. **Severe Class Imbalance (36:1 ratio)**
   We implement a WeightedRandomSampler that oversamples minority
   classes proportionally, preventing the model from learning
   a "majority-class guessing" strategy.

3. **Advanced Augmentation**
   - SpecAugment (Park et al., 2019): Masks random frequency bands
     and time steps, forcing the model to learn redundant representations.
   - MixUp (Zhang et al., 2018): Linearly interpolates training pairs,
     regularizing the model by encouraging linear behavior between samples.

4. **Optimized I/O**
   Uses num_workers and pin_memory for efficient CPU→GPU data transfer.

Mathematical Foundation:
========================
Mel-spectrogram computation:
  1. STFT: X(t,f) = Σ x(n) · w(n-t·H) · e^(-j2πfn/N)
  2. Mel filterbank: M(t,m) = Σ H_m(f) · |X(t,f)|²
  3. Log compression: S(t,m) = log(M(t,m) + ε)

Where H_m(f) are triangular filters spaced linearly on the mel scale:
  mel(f) = 2595 · log₁₀(1 + f/700)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa

from src.utils import (
    TARGET_SR, CLEANED_DIR, NOISY_DIR, AUDIO_EXTENSIONS,
    CLASS_LABELS, ensure_dir
)


# ============================================================
# Mel-Spectrogram Configuration
# ============================================================

# These parameters are carefully chosen for infant cry analysis:
MEL_CONFIG = {
    'n_mels': 128,        # 128 mel bands — high resolution for cry harmonics
    'n_fft': 1024,        # ~64ms window at 16kHz — captures low fundamentals (250Hz+)
    'hop_length': 256,    # ~16ms hop — fine temporal resolution for cry dynamics
    'fmax': 8000,         # Max frequency = Nyquist at 16kHz
    'power': 2.0,         # Power spectrogram (magnitude squared)
}

# Fixed spectrogram width: 3 seconds of audio at 16kHz with hop=256
# = (3 * 16000) / 256 = 187.5 ≈ 188 frames
MAX_TIME_FRAMES = 188
AUDIO_DURATION_SEC = 3.0


# ============================================================
# Audio → Mel-Spectrogram Conversion
# ============================================================

def audio_to_mel_spectrogram(y, sr=TARGET_SR, config=None):
    """
    Convert raw audio waveform to log-mel-spectrogram.

    The mel-spectrogram is the primary input representation for our
    CNN-BiLSTM architecture. It provides a 2D time-frequency image
    that CNNs can process with spatial convolutions.

    Why log-mel over raw STFT?
    - Mel scale compresses frequency axis to match human perception
    - Log compression reduces dynamic range (60dB+ → ~6 units)
    - Together, they create a perceptually-meaningful representation
      where Euclidean distance correlates with perceived similarity

    Parameters
    ----------
    y : np.ndarray
        Audio waveform (1D).
    sr : int
        Sample rate.
    config : dict, optional
        Mel-spectrogram parameters.

    Returns
    -------
    log_mel : np.ndarray
        Log-mel-spectrogram of shape (n_mels, T).
    """
    if config is None:
        config = MEL_CONFIG

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=config['n_mels'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        fmax=config['fmax'],
        power=config['power']
    )

    # Log compression with small epsilon to avoid log(0)
    # ε = 1e-10 ensures numerical stability without affecting scale
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel


def pad_or_truncate_spectrogram(spec, max_frames=MAX_TIME_FRAMES):
    """
    Ensure spectrogram has exactly max_frames time steps.

    Short signals are zero-padded on the right (silence).
    Long signals are center-cropped to preserve the most
    acoustically active portion.

    Parameters
    ----------
    spec : np.ndarray
        Spectrogram of shape (n_mels, T).
    max_frames : int
        Target number of time frames.

    Returns
    -------
    spec_fixed : np.ndarray
        Fixed-size spectrogram of shape (n_mels, max_frames).
    """
    n_mels, T = spec.shape

    if T < max_frames:
        # Pad with minimum value (silence in dB scale)
        pad_width = max_frames - T
        spec = np.pad(spec, ((0, 0), (0, pad_width)),
                      mode='constant', constant_values=spec.min())
    elif T > max_frames:
        # Center crop to keep the most active portion
        start = (T - max_frames) // 2
        spec = spec[:, start:start + max_frames]

    return spec


# ============================================================
# SpecAugment — Frequency and Time Masking
# ============================================================

def spec_augment(spec, freq_mask_param=15, time_mask_param=25,
                 n_freq_masks=2, n_time_masks=2):
    """
    Apply SpecAugment (Park et al., 2019) to a mel-spectrogram.

    SpecAugment applies two types of masks:
    1. Frequency masking: zeroes out F consecutive mel bins
       → Forces the model to not rely on any single frequency band
    2. Time masking: zeroes out T consecutive time frames
       → Forces the model to classify from partial temporal context

    This acts as a strong regularizer, analogous to Dropout but
    operating in the input space rather than hidden representations.

    Original paper: "SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition" — Park et al., Interspeech 2019.

    Parameters
    ----------
    spec : np.ndarray
        Mel-spectrogram of shape (n_mels, T).
    freq_mask_param : int
        Maximum width of frequency mask (F parameter).
    time_mask_param : int
        Maximum width of time mask (T parameter).
    n_freq_masks : int
        Number of frequency masks to apply.
    n_time_masks : int
        Number of time masks to apply.

    Returns
    -------
    spec_aug : np.ndarray
        Augmented spectrogram.
    """
    spec_aug = spec.copy()
    n_mels, n_frames = spec_aug.shape

    # Frequency masking
    for _ in range(n_freq_masks):
        f = np.random.randint(0, min(freq_mask_param, n_mels))
        f0 = np.random.randint(0, max(1, n_mels - f))
        spec_aug[f0:f0 + f, :] = spec_aug.min()

    # Time masking
    for _ in range(n_time_masks):
        t = np.random.randint(0, min(time_mask_param, n_frames))
        t0 = np.random.randint(0, max(1, n_frames - t))
        spec_aug[:, t0:t0 + t] = spec_aug.min()

    return spec_aug


# ============================================================
# MixUp Augmentation
# ============================================================

def mixup_data(x, y, alpha=0.4):
    """
    Apply MixUp augmentation (Zhang et al., 2018).

    MixUp creates virtual training examples by linearly interpolating
    between pairs of training samples and their labels:

        x̃ = λ · x_i + (1-λ) · x_j
        ỹ = λ · y_i + (1-λ) · y_j

    where λ ~ Beta(α, α).

    This encourages the model to behave linearly between training
    examples, providing a strong regularization effect. It also
    smooths the label space, reducing overconfident predictions.

    Parameters
    ----------
    x : torch.Tensor
        Batch of inputs (B, C, H, W).
    y : torch.Tensor
        Batch of labels (B,) — integer class indices.
    alpha : float
        MixUp interpolation strength. α=0 → no mixing, α=1 → uniform.

    Returns
    -------
    mixed_x : torch.Tensor
        Mixed inputs.
    y_a, y_b : torch.Tensor
        Original and shuffled labels.
    lam : float
        Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute MixUp loss: λ · L(pred, y_a) + (1-λ) · L(pred, y_b).
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# PyTorch Dataset
# ============================================================

class InfantCryDataset(Dataset):
    """
    PyTorch Dataset for infant cry mel-spectrograms.

    Loads audio files on-the-fly, converts to mel-spectrograms,
    and optionally applies SpecAugment augmentation.

    On-the-fly computation (vs. pre-computed) trades CPU time for:
    - Reduced disk usage (no duplicate spectrograms)
    - Different augmentations each epoch (stochastic SpecAugment)
    - Flexibility to change spectrogram parameters without regenerating

    Attributes
    ----------
    file_paths : list of str
        Paths to audio files.
    labels : list of int
        Encoded class labels.
    class_names : list of str
        Human-readable class names.
    augment : bool
        Whether to apply SpecAugment.
    """

    def __init__(self, file_paths, labels, class_names=None,
                 augment=False, sr=TARGET_SR, max_frames=MAX_TIME_FRAMES):
        """
        Parameters
        ----------
        file_paths : list of str
            Paths to audio files.
        labels : list of int
            Integer-encoded labels.
        class_names : list of str, optional
            Class name list.
        augment : bool
            Apply SpecAugment during loading.
        sr : int
            Target sample rate.
        max_frames : int
            Fixed spectrogram width.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.class_names = class_names or CLASS_LABELS
        self.augment = augment
        self.sr = sr
        self.max_frames = max_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load a single audio file and return (spectrogram_tensor, label).

        Returns
        -------
        spec_tensor : torch.Tensor
            Shape (1, n_mels, max_frames) — single-channel "image".
        label : int
            Integer class label.
        """
        filepath = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio, truncate/pad to fixed duration
        y, _ = librosa.load(filepath, sr=self.sr, mono=True,
                            duration=AUDIO_DURATION_SEC)

        # Pad short audio to full duration
        target_len = int(self.sr * AUDIO_DURATION_SEC)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')

        # Convert to mel-spectrogram
        mel_spec = audio_to_mel_spectrogram(y, sr=self.sr)
        mel_spec = pad_or_truncate_spectrogram(mel_spec, self.max_frames)

        # Apply SpecAugment if in training mode
        if self.augment:
            mel_spec = spec_augment(mel_spec)

        # Normalize to zero mean, unit variance (per-sample)
        mean = mel_spec.mean()
        std = mel_spec.std()
        if std > 0:
            mel_spec = (mel_spec - mean) / std

        # Add channel dimension: (n_mels, T) → (1, n_mels, T)
        spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)

        return spec_tensor, label

    def get_class_weights(self):
        """
        Compute inverse-frequency class weights for loss function.

        For class i with n_i samples out of N total:
            w_i = N / (n_classes × n_i)

        This ensures minority classes contribute equally to the loss
        gradient, preventing the optimizer from ignoring them.

        Returns
        -------
        weights : torch.Tensor
            Class weights of shape (n_classes,).
        """
        labels_array = np.array(self.labels)
        n_classes = len(self.class_names)
        class_counts = np.bincount(labels_array, minlength=n_classes)
        # Avoid division by zero for empty classes
        class_counts = np.maximum(class_counts, 1)
        weights = len(labels_array) / (n_classes * class_counts)
        return torch.FloatTensor(weights)

    def get_sample_weights(self):
        """
        Compute per-sample weights for WeightedRandomSampler.

        Each sample gets the weight of its class, so minority-class
        samples are drawn more frequently during training.

        Returns
        -------
        sample_weights : list of float
        """
        class_weights = self.get_class_weights().numpy()
        return [class_weights[label] for label in self.labels]


# ============================================================
# Data Discovery & Splitting
# ============================================================

def discover_audio_for_dl(data_dirs, label_list=None):
    """
    Discover all audio files across directories for DL training.

    Parameters
    ----------
    data_dirs : list of str
        Directories containing class-labeled subfolders.
    label_list : list of str, optional
        Ordered class labels. Defaults to CLASS_LABELS.

    Returns
    -------
    file_paths : list of str
    labels : list of int
    label_names : list of str
    """
    if label_list is None:
        label_list = CLASS_LABELS

    label_to_idx = {name: idx for idx, name in enumerate(label_list)}
    file_paths = []
    labels = []

    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} does not exist, skipping.")
            continue

        for folder_name in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path) or folder_name.startswith('.'):
                continue

            if folder_name not in label_to_idx:
                continue

            label_idx = label_to_idx[folder_name]

            for fname in sorted(os.listdir(folder_path)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in AUDIO_EXTENSIONS and not fname.startswith('.'):
                    file_paths.append(os.path.join(folder_path, fname))
                    labels.append(label_idx)

    return file_paths, labels, label_list


def create_data_loaders(train_paths, train_labels, val_paths, val_labels,
                        class_names=None, batch_size=32, num_workers=0):
    """
    Create PyTorch DataLoaders with class-balanced sampling.

    Parameters
    ----------
    train_paths, val_paths : list of str
    train_labels, val_labels : list of int
    class_names : list of str
    batch_size : int
    num_workers : int
        Number of parallel data loading workers.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    train_dataset : InfantCryDataset
    val_dataset : InfantCryDataset
    """
    train_dataset = InfantCryDataset(
        train_paths, train_labels, class_names,
        augment=True  # SpecAugment during training
    )
    val_dataset = InfantCryDataset(
        val_paths, val_labels, class_names,
        augment=False  # No augmentation during validation
    )

    # Weighted sampler for class-balanced batches
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,  # CPU-only on macOS
        drop_last=True     # Ensures consistent batch sizes for BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, train_dataset, val_dataset
