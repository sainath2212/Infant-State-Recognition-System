"""
Discrete Stockwell Transform (DST) Feature Extraction for Infant Cry Recognition.

The S-Transform (Stockwell Transform) is a time-frequency analysis method that
combines the best properties of STFT and wavelet transforms. It provides:

  - Frequency-dependent Gaussian windows (unlike STFT's fixed resolution)
  - Absolute phase information
  - Multi-resolution time-frequency representation

Mathematical Foundation:
========================
The continuous S-Transform of a signal x(t):
    S(τ, f) = ∫ x(t) · (|f|/√2π) · exp(-(τ-t)²f²/2) · exp(-j2πft) dt

Efficiently computed in the spectral domain:
    S(τ, f) = IFFT_α { X(α + f) · G(α, f) }

where:
    X(f)         = DFT of x(t)
    G(α, f)      = exp(-2π²α²/f²)  [Gaussian filter, scales with 1/f]
    α            = integration frequency variable

Why DST for Infant Cry?
=======================
Infant cry signals are non-stationary — the fundamental frequency (250–700 Hz)
and harmonics evolve over time. The DST's adaptive frequency resolution gives:
  - HIGH frequency resolution at low frequencies (cry fundamentals)
  - HIGH time resolution at high frequencies (transients/harmonics)

This is superior to Mel-spectrograms which have fixed time-frequency resolution,
and it complements the Mel representation in our dual-stream hybrid architecture.

Reference: Jayasree & Blessy, "Infant cry classification via deep learning based
Infant cry networks using Discrete Stockwell Transform", Engineering Applications
of Artificial Intelligence, Vol. 160, Aug 2025. DOI: 10.1016/j.engappai.2025.112008
"""

import numpy as np
import librosa
from typing import Optional

# ============================================================
# DST Configuration
# ============================================================

DST_CONFIG = {
    'n_freqs': 64,        # Number of frequency bins (logarithmically spaced)
    'fmin': 50.0,         # Minimum frequency (Hz) — captures low cry fundamentals
    'fmax': 8000.0,       # Maximum frequency (Hz) — Nyquist at 16kHz
    'hop_length': 256,    # Temporal downsampling (same as Mel pipeline)
    'max_frames': 188,    # Fixed width: 3s at 16kHz with hop=256
}

TARGET_SR = 16000         # Standard sample rate


# ============================================================
# Core S-Transform (Spectral Domain Implementation)
# ============================================================

def _gaussian_filter(N: int, freq_index: int) -> np.ndarray:
    """
    Compute Gaussian window in the frequency domain for a given frequency index.

    G(α, f) = exp(-2π²α²/f²)

    where f is the frequency index (in samples), and α ranges over the DFT spectrum.

    For f=0 (DC), returns a flat window (equivalent to temporal mean).

    Parameters
    ----------
    N : int
        Signal length (DFT size).
    freq_index : int
        Frequency index k such that physical frequency f = k * sr / N.

    Returns
    -------
    gaussian : np.ndarray of shape (N,)
        Gaussian window in frequency domain.
    """
    if freq_index == 0:
        return np.ones(N)  # DC: no frequency localization

    # Frequency axis (shifted so DC is at center for Gaussian computation)
    alpha = np.fft.fftfreq(N) * N  # integer frequency indices: [0, 1, ..., N/2, -N/2+1, ..., -1]

    # Gaussian filter: width inversely proportional to frequency
    gaussian = np.exp(-2.0 * np.pi**2 * alpha**2 / (freq_index**2))
    return gaussian


def stockwell_transform(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_freqs: int = DST_CONFIG['n_freqs'],
    fmin: float = DST_CONFIG['fmin'],
    fmax: float = DST_CONFIG['fmax'],
    hop_length: int = DST_CONFIG['hop_length'],
    max_frames: int = DST_CONFIG['max_frames'],
) -> np.ndarray:
    """
    Compute the log-magnitude Discrete Stockwell Transform spectrogram.

    Algorithm:
    ----------
    1. Compute DFT of signal: X[k] = DFT{x[n]}
    2. For each analysis frequency fᵢ (log-spaced, fmin → fmax):
       a. Convert fᵢ to frequency index k_i = round(fᵢ · N / sr)
       b. Compute Gaussian window G_i[α] = exp(-2π²α²/k_i²)
       c. Cyclic-shift DFT: X_shifted[α] = X[α + k_i]  (modulo N)
       d. IDFT of product: s_i[τ] = IDFT{X_shifted · G_i}
       e. Store |s_i[τ]| as row i of S-matrix
    3. Temporally downsample by averaging over hop_length windows
    4. Apply log compression: log(|S| + ε)
    5. Normalize to zero mean, unit variance (per-sample)
    6. Pad or crop to max_frames columns

    Parameters
    ----------
    y : np.ndarray
        Audio waveform (1D), mono.
    sr : int
        Sample rate (Hz).
    n_freqs : int
        Number of output frequency bins (rows of DST matrix).
    fmin : float
        Minimum analysis frequency (Hz).
    fmax : float
        Maximum analysis frequency (Hz).
    hop_length : int
        Temporal stride for downsampling (samples).
    max_frames : int
        Target number of time frames (columns of DST matrix).

    Returns
    -------
    dst_spec : np.ndarray of shape (n_freqs, max_frames)
        Log-magnitude DST spectrogram, normalized to ~N(0,1).
    """
    N = len(y)

    # Step 1: DFT of signal
    X = np.fft.fft(y)

    # Step 2: Log-spaced analysis frequencies
    freqs_hz = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    freq_indices = np.clip(
        np.round(freqs_hz * N / sr).astype(int), 1, N // 2
    )

    # Step 3: S-Transform matrix (n_freqs × N)
    S_raw = np.zeros((n_freqs, N), dtype=np.float32)

    for i, (f_idx) in enumerate(freq_indices):
        # Gaussian window for this frequency
        G = _gaussian_filter(N, f_idx)

        # Cyclic shift of DFT
        X_shifted = np.roll(X, -f_idx)

        # IDFT of filtered, shifted spectrum → S-transform row
        s_row = np.fft.ifft(X_shifted * G)
        S_raw[i, :] = np.abs(s_row).astype(np.float32)

    # Step 4: Temporal downsampling — average over hop_length windows
    n_frames = N // hop_length
    S_down = np.zeros((n_freqs, n_frames), dtype=np.float32)
    for t in range(n_frames):
        start = t * hop_length
        end = min(start + hop_length, N)
        S_down[:, t] = S_raw[:, start:end].mean(axis=1)

    # Step 5: Log compression
    S_log = np.log(S_down + 1e-10).astype(np.float32)

    # Step 6: Normalize (per-sample statistics)
    mean = S_log.mean()
    std = S_log.std()
    if std > 1e-8:
        S_log = (S_log - mean) / std

    # Step 7: Pad or center-crop to fixed size
    n_actual = S_log.shape[1]
    if n_actual < max_frames:
        pad = max_frames - n_actual
        S_log = np.pad(S_log, ((0, 0), (0, pad)),
                       mode='constant', constant_values=0.0)
    elif n_actual > max_frames:
        start = (n_actual - max_frames) // 2
        S_log = S_log[:, start:start + max_frames]

    return S_log


# ============================================================
# DST Statistical Features (for ML models)
# ============================================================

def dst_statistical_features(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_freqs: int = DST_CONFIG['n_freqs'],
    fmin: float = DST_CONFIG['fmin'],
    fmax: float = DST_CONFIG['fmax'],
) -> np.ndarray:
    """
    Extract compact statistical features from the S-Transform matrix.

    These 5 statistics per frequency bin provide rich descriptors for ML models:
      - Mean energy in each frequency band over time
      - Standard deviation (temporal variability)
      - Maximum activation (peak energy)
      - Entropy of time-frequency energy distribution
      - Centroid (temporal center of energy)

    Parameters
    ----------
    y : np.ndarray
        Audio waveform.
    sr : int
        Sample rate.
    n_freqs : int
        Number of frequency bins.
    fmin, fmax : float
        Frequency range.

    Returns
    -------
    features : np.ndarray of shape (n_freqs * 5,)
        Flattened statistical feature vector.
    """
    N = len(y)
    X = np.fft.fft(y)

    freqs_hz = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    freq_indices = np.clip(np.round(freqs_hz * N / sr).astype(int), 1, N // 2)

    means, stds, maxs, entropies, centroids = [], [], [], [], []

    for f_idx in freq_indices:
        G = _gaussian_filter(N, f_idx)
        X_shifted = np.roll(X, -f_idx)
        s_row = np.abs(np.fft.ifft(X_shifted * G))

        energy = s_row + 1e-10
        p = energy / energy.sum()

        means.append(np.mean(s_row))
        stds.append(np.std(s_row))
        maxs.append(np.max(s_row))
        entropies.append(-np.sum(p * np.log(p + 1e-10)))
        t = np.arange(N)
        centroids.append(np.sum(t * energy) / energy.sum())

    features = np.concatenate([
        np.array(means),
        np.array(stds),
        np.array(maxs),
        np.array(entropies),
        np.array(centroids),
    ]).astype(np.float32)

    return features


# ============================================================
# High-level Helpers
# ============================================================

def audio_to_dst_spectrogram(
    filepath: str,
    sr: int = TARGET_SR,
    duration: float = 3.0,
    config: Optional[dict] = None,
) -> np.ndarray:
    """
    Load audio file and compute DST spectrogram.

    Parameters
    ----------
    filepath : str
        Path to audio file.
    sr : int
        Target sample rate.
    duration : float
        Audio duration to load (seconds).
    config : dict, optional
        DST configuration overrides.

    Returns
    -------
    dst_spec : np.ndarray of shape (n_freqs, max_frames)
    """
    if config is None:
        config = DST_CONFIG

    y, _ = librosa.load(filepath, sr=sr, mono=True, duration=duration)

    # Pad to target duration
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')

    return stockwell_transform(
        y, sr=sr,
        n_freqs=config['n_freqs'],
        fmin=config['fmin'],
        fmax=config['fmax'],
        hop_length=config['hop_length'],
        max_frames=config['max_frames'],
    )


def audio_to_dst_features(
    filepath: str,
    sr: int = TARGET_SR,
    duration: float = 3.0,
    n_freqs: int = DST_CONFIG['n_freqs'],
) -> np.ndarray:
    """
    Load audio and extract DST statistical features for ML models.

    Returns
    -------
    features : np.ndarray of shape (n_freqs * 5,)
    """
    y, _ = librosa.load(filepath, sr=sr, mono=True, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')

    return dst_statistical_features(y, sr=sr, n_freqs=n_freqs)
