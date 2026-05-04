"""
Dual-Stream Data Pipeline for CryNetV2 (Mel + DST).

Extends the existing InfantCryDataset to return triplets:
    (mel_spectrogram, dst_spectrogram, label)

Both representations are computed on-the-fly during training,
giving different augmented views each epoch.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa

from src.utils import TARGET_SR, CLEANED_DIR, NOISY_DIR, AUDIO_EXTENSIONS, CLASS_LABELS
from src.dl_data import (
    audio_to_mel_spectrogram, pad_or_truncate_spectrogram,
    spec_augment, mixup_data, mixup_criterion,
    MEL_CONFIG, MAX_TIME_FRAMES, AUDIO_DURATION_SEC,
)
from src.dst_features import stockwell_transform, DST_CONFIG


# ============================================================
# Dual-Stream SpecAugment
# ============================================================

def dual_spec_augment(
    mel: np.ndarray,
    dst: np.ndarray,
    freq_mask_param: int = 15,
    time_mask_param: int = 25,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
) -> tuple:
    """
    Apply SpecAugment consistently to both Mel and DST spectrograms.

    The same time masks are applied to both to maintain temporal alignment.
    Frequency masks are applied independently (different frequency axes).

    Parameters
    ----------
    mel : np.ndarray of shape (n_mels, T)
    dst : np.ndarray of shape (n_dst_freqs, T)

    Returns
    -------
    (mel_aug, dst_aug) : augmented spectrograms
    """
    mel_aug = mel.copy()
    dst_aug = dst.copy()

    T = mel.shape[1]

    # Time masking — same for both (preserves temporal alignment)
    for _ in range(n_time_masks):
        t = np.random.randint(0, min(time_mask_param, T))
        t0 = np.random.randint(0, max(1, T - t))
        mel_aug[:, t0:t0 + t] = mel_aug.min()
        dst_aug[:, t0:t0 + t] = dst_aug.min()

    # Frequency masking — independent for each stream
    n_mel = mel.shape[0]
    for _ in range(n_freq_masks):
        f = np.random.randint(0, min(freq_mask_param, n_mel))
        f0 = np.random.randint(0, max(1, n_mel - f))
        mel_aug[f0:f0 + f, :] = mel_aug.min()

    n_dst = dst.shape[0]
    for _ in range(n_freq_masks):
        f = np.random.randint(0, min(freq_mask_param // 2, n_dst))
        f0 = np.random.randint(0, max(1, n_dst - f))
        dst_aug[f0:f0 + f, :] = dst_aug.min()

    return mel_aug, dst_aug


# ============================================================
# Dual-Stream Dataset
# ============================================================

class HybridInfantCryDataset(Dataset):
    """
    PyTorch Dataset returning (mel_tensor, dst_tensor, label) triplets.

    On each __getitem__ call:
    1. Load audio file
    2. Compute Mel-spectrogram (128 × 188)
    3. Compute DST spectrogram (64 × 188)
    4. Optionally apply dual SpecAugment
    5. Normalize each spectrogram independently

    Parameters
    ----------
    file_paths : list of str
    labels : list of int
    class_names : list of str
    augment : bool
        Apply SpecAugment (training only).
    sr : int
    max_frames : int
    """

    def __init__(
        self,
        file_paths,
        labels,
        class_names=None,
        augment=False,
        sr=TARGET_SR,
        max_frames=MAX_TIME_FRAMES,
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.class_names = class_names or CLASS_LABELS
        self.augment = augment
        self.sr = sr
        self.max_frames = max_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio
        y, _ = librosa.load(filepath, sr=self.sr, mono=True,
                            duration=AUDIO_DURATION_SEC)
        target_len = int(self.sr * AUDIO_DURATION_SEC)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')

        # Mel-spectrogram
        mel = audio_to_mel_spectrogram(y, sr=self.sr, config=MEL_CONFIG)
        mel = pad_or_truncate_spectrogram(mel, self.max_frames)

        # DST spectrogram
        dst = stockwell_transform(
            y, sr=self.sr,
            n_freqs=DST_CONFIG['n_freqs'],
            fmin=DST_CONFIG['fmin'],
            fmax=DST_CONFIG['fmax'],
            hop_length=DST_CONFIG['hop_length'],
            max_frames=self.max_frames,
        )

        # Augmentation
        if self.augment:
            mel, dst = dual_spec_augment(mel, dst)

        # Per-sample normalization
        for spec in [mel, dst]:
            # (already normalized in dst computation; mel needs it)
            pass

        # Normalize Mel
        m, s = mel.mean(), mel.std()
        if s > 0:
            mel = (mel - m) / s

        # Convert to tensors: (1, H, W)
        mel_t = torch.FloatTensor(mel).unsqueeze(0)   # (1, 128, 188)
        dst_t = torch.FloatTensor(dst).unsqueeze(0)   # (1, 64, 188)

        return mel_t, dst_t, label

    def get_class_weights(self):
        labels_array = np.array(self.labels)
        n_classes = len(self.class_names)
        counts = np.bincount(labels_array, minlength=n_classes)
        counts = np.maximum(counts, 1)
        weights = len(labels_array) / (n_classes * counts)
        return torch.FloatTensor(weights)

    def get_sample_weights(self):
        cw = self.get_class_weights().numpy()
        return [cw[lbl] for lbl in self.labels]


# ============================================================
# DataLoader Factory
# ============================================================

def create_hybrid_loaders(
    train_paths, train_labels,
    val_paths, val_labels,
    class_names=None,
    batch_size=32,
    num_workers=0,
):
    """
    Create train and validation DataLoaders for CryNetV2.

    Parameters
    ----------
    train_paths, val_paths : list of str
    train_labels, val_labels : list of int
    class_names : list of str
    batch_size : int
    num_workers : int

    Returns
    -------
    train_loader, val_loader, train_dataset, val_dataset
    """
    train_ds = HybridInfantCryDataset(
        train_paths, train_labels, class_names, augment=True
    )
    val_ds = HybridInfantCryDataset(
        val_paths, val_labels, class_names, augment=False
    )

    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, train_ds, val_ds


# ============================================================
# Hybrid MixUp (both streams simultaneously)
# ============================================================

def hybrid_mixup_data(mel, dst, y, alpha=0.4):
    """
    Apply MixUp augmentation to both Mel and DST simultaneously.

    Same λ and permutation are used for both streams to maintain
    the cross-modal correspondence.

    Parameters
    ----------
    mel : (B, 1, 128, T)
    dst : (B, 1, 64, T)
    y   : (B,) — integer labels
    alpha : float

    Returns
    -------
    mixed_mel, mixed_dst, y_a, y_b, lam
    """
    import numpy as np
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B = mel.size(0)
    idx = torch.randperm(B, device=mel.device)

    mixed_mel = lam * mel + (1 - lam) * mel[idx]
    mixed_dst = lam * dst + (1 - lam) * dst[idx]
    y_a, y_b = y, y[idx]

    return mixed_mel, mixed_dst, y_a, y_b, lam
