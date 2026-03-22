"""
Noise Augmentation Module for Infant Cry Recognition System.

Real-world infant cry recordings are captured in diverse acoustic environments:
hospitals, homes, cars, outdoors. A model trained only on clean studio-quality
recordings will fail when deployed in these conditions.

This module implements three complementary augmentation strategies:

1. **Gaussian Noise** — Simulates sensor/electronic noise. Controlled via SNR.
2. **Background Noise** — Simulates environmental sounds using colored noise
   (pink noise for natural environments, brown noise for HVAC/traffic).
3. **Time Shifting** — Simulates temporal misalignment from different recording
   start points, improving the model's temporal invariance.

These augmentations serve as implicit regularization: they expand the training
distribution and force the model to learn noise-invariant representations,
directly improving generalization (analogous to dropout in neural networks).


"""

import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from src.utils import (
    TARGET_SR, CLEANED_DIR, NOISY_DIR, AUDIO_EXTENSIONS,
    CLASS_LABELS, ensure_dir
)


# ============================================================
# 1. Gaussian (White) Noise
# ============================================================

def add_gaussian_noise(y, snr_db=20):
    """
    Add Gaussian (white) noise at a specified Signal-to-Noise Ratio.

    White noise has equal power across all frequencies and simulates
    electronic/sensor noise from microphones and ADCs.

    Parameters
    ----------
    y : np.ndarray
        Clean audio signal.
    snr_db : float
        Desired signal-to-noise ratio in dB. Lower values = more noise.
        - 30 dB: barely perceptible noise
        - 20 dB: moderate noise (typical indoor recording)
        - 10 dB: significant noise (noisy environment)

    Returns
    -------
    y_noisy : np.ndarray
        Noise-augmented signal.

    Notes
    -----
    SNR(dB) = 10 * log10(P_signal / P_noise)
    We compute the required noise power from the signal power and desired SNR.
    """
    signal_power = np.mean(np.square(y))
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    y_noisy = y + noise
    # Clip to prevent clipping artifacts
    return np.clip(y_noisy, -1.0, 1.0)


# ============================================================
# 2. Colored Background Noise (Pink Noise)
# ============================================================

def _generate_pink_noise(n_samples):
    """
    Generate pink noise (1/f noise) using the Voss-McCartney algorithm.

    Pink noise has equal energy per octave, making it perceptually
    similar to natural environmental sounds (wind, crowd murmur,
    distant traffic). It is a more realistic noise model than white noise
    for simulating real-world recording conditions.
    """
    # Simple spectral method: generate white noise, shape spectrum
    white = np.random.randn(n_samples)
    # FFT-based pink noise generation
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    # Avoid division by zero at DC
    freqs[0] = 1
    # Scale by 1/sqrt(f) to get pink noise spectrum
    fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n_samples)
    # Normalize
    pink = pink / (np.max(np.abs(pink)) + 1e-8)
    return pink


def add_background_noise(y, snr_db=15):
    """
    Add pink (1/f) background noise at a specified SNR.

    Pink noise simulates natural environmental sounds that are
    commonly present during infant cry recordings (e.g., hospital
    ambient noise, household sounds, ventilation systems).

    Parameters
    ----------
    y : np.ndarray
        Clean audio signal.
    snr_db : float
        Target SNR in dB.

    Returns
    -------
    y_noisy : np.ndarray
        Signal with added background noise.
    """
    pink_noise = _generate_pink_noise(len(y))
    signal_power = np.mean(np.square(y))
    noise_power = signal_power / (10 ** (snr_db / 10))
    current_noise_power = np.mean(np.square(pink_noise))
    if current_noise_power > 0:
        pink_noise = pink_noise * np.sqrt(noise_power / current_noise_power)
    y_noisy = y + pink_noise
    return np.clip(y_noisy, -1.0, 1.0)


# ============================================================
# 3. Time Shifting
# ============================================================

def time_shift(y, shift_fraction=0.1):
    """
    Shift audio signal in time by a random amount.

    Time shifting simulates recordings that start at different points
    relative to the onset of crying. This teaches the model to be
    invariant to the exact temporal position of acoustic events,
    which is critical since real-world recordings have unpredictable
    onset times.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    shift_fraction : float
        Maximum shift as a fraction of total length. 0.1 = up to 10%.

    Returns
    -------
    y_shifted : np.ndarray
        Time-shifted signal (shifted portion filled with zeros).
    """
    shift_max = int(len(y) * shift_fraction)
    shift = np.random.randint(-shift_max, shift_max)
    y_shifted = np.roll(y, shift)
    # Zero-fill the wrapped portion to avoid artificial periodicity
    if shift > 0:
        y_shifted[:shift] = 0
    elif shift < 0:
        y_shifted[shift:] = 0
    return y_shifted


# ============================================================
# 4. Combined Augmentation Pipeline
# ============================================================

def augment_audio(y, augmentations=None):
    """
    Apply a sequence of augmentations to an audio signal.

    Parameters
    ----------
    y : np.ndarray
        Input audio signal.
    augmentations : list of str, optional
        List of augmentations to apply. Options: 'gaussian', 'background', 'shift'.
        If None, applies all three.

    Returns
    -------
    y_aug : np.ndarray
        Augmented audio signal.
    """
    if augmentations is None:
        augmentations = ['gaussian', 'background', 'shift']

    y_aug = y.copy()

    for aug in augmentations:
        if aug == 'gaussian':
            y_aug = add_gaussian_noise(y_aug, snr_db=20)
        elif aug == 'background':
            y_aug = add_background_noise(y_aug, snr_db=15)
        elif aug == 'shift':
            y_aug = time_shift(y_aug, shift_fraction=0.1)
        else:
            raise ValueError(f"Unknown augmentation: {aug}")

    return y_aug


# ============================================================
# 5. Batch Dataset Augmentation
# ============================================================

def augment_dataset(input_dir=CLEANED_DIR, output_dir=NOISY_DIR, sr=TARGET_SR):
    """
    Generate noisy versions of all cleaned audio files.

    For each cleaned audio file, we generate ONE augmented version
    applying all three noise types in sequence.

    Parameters
    ----------
    input_dir : str
        Path to cleaned data directory.
    output_dir : str
        Path to noisy data output directory.
    sr : int
        Sample rate for saving.

    Returns
    -------
    results : list of dict
        Augmentation results.
    """
    results: list[dict[str, str | float | None]] = []
    ensure_dir(output_dir)

    for label in sorted(os.listdir(input_dir)):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir) or label.startswith('.'):
            continue

        out_class_dir = ensure_dir(os.path.join(output_dir, label))

        audio_files = [f for f in sorted(os.listdir(label_dir))
                       if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
                       and not f.startswith('.')]

        for fname in tqdm(audio_files, desc=f'Augmenting {label}', leave=False):
            filepath = os.path.join(label_dir, fname)
            out_name = f'noisy_{fname}'
            out_path = os.path.join(out_class_dir, out_name)

            try:
                y, _ = librosa.load(filepath, sr=sr, mono=True)
                y_aug = augment_audio(y)
                sf.write(out_path, y_aug, sr, subtype='PCM_16')
                results.append({
                    'original_file': filepath,
                    'noisy_file': out_path,
                    'label': label,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'original_file': filepath,
                    'noisy_file': None,
                    'label': label,
                    'status': f'error: {str(e)}'
                })

    return results
