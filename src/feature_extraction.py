"""
Feature Extraction Module for Infant Cry Recognition System.

This module extracts handcrafted acoustic features grounded in signal processing
theory. Unlike deep learning approaches that learn representations implicitly,
these features provide interpretable, physically meaningful descriptors of the
audio signal.

Features extracted:
1. **MFCC** (Mel-Frequency Cepstral Coefficients) — Primary feature
   - Mimics human auditory perception via mel filterbank
   - Captures spectral envelope (vocal tract characteristics)
   - Standard in speech/audio classification since Davis & Mermelstein (1980)

2. **Spectral Centroid** — "Center of mass" of the spectrum
   - Correlates with perceived brightness/sharpness of sound
   - Higher values = more high-frequency energy (e.g., pain cry vs. tired cry)

3. **Zero Crossing Rate (ZCR)** — Rate of sign changes in the signal
   - Indicator of noisiness vs. harmonicity
   - High ZCR suggests noisy/breathy cry; low ZCR suggests tonal/harmonic cry

4. **Chroma Features** — Pitch class energy distribution
   - Captures melodic/tonal patterns
   - Different cry states may have distinct pitch contour patterns

Theoretical Foundation:
- We extract features from short frames (25ms, hop 10ms) assuming
  quasi-stationarity: within short windows, the vocal tract configuration
  is approximately constant, making spectral analysis meaningful.
- Feature statistics (mean, std) over the full signal provide a fixed-length
  representation regardless of audio duration.
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from src.utils import (
    TARGET_SR, AUDIO_EXTENSIONS, FEATURES_DIR, ensure_dir
)


# ============================================================
# Frame Parameters
# ============================================================

# Frame length of 25ms at 16kHz = 400 samples
# This is standard in speech processing: short enough for stationarity,
# long enough for reliable spectral estimation (at least 2 pitch periods)
FRAME_LENGTH = 400    # 25ms at 16kHz
HOP_LENGTH = 160      # 10ms hop → 60% overlap for smooth feature trajectories
N_MFCC = 13           # Standard: 13 coefficients capture spectral envelope


# ============================================================
# Individual Feature Extractors
# ============================================================

def extract_mfcc(y, sr=TARGET_SR, n_mfcc=N_MFCC):
    """
    Extract MFCC features with delta and delta-delta coefficients.

    MFCCs capture the spectral envelope of the signal, which encodes
    vocal tract shape information. The computation pipeline:
    1. Pre-emphasis → emphasize high frequencies (compensate -6dB/octave rolloff)
    2. Windowing → apply Hamming window to reduce spectral leakage
    3. FFT → convert to frequency domain
    4. Mel filterbank → warp to perceptual frequency scale
    5. Log → compress dynamic range (mimics loudness perception)
    6. DCT → decorrelate filter outputs → MFCCs

    Delta (Δ) and delta-delta (ΔΔ) coefficients capture temporal dynamics:
    - Δ-MFCC: velocity of spectral change (formant transitions)
    - ΔΔ-MFCC: acceleration of spectral change

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    n_mfcc : int
        Number of MFCC coefficients.

    Returns
    -------
    features : dict
        Keys: 'mfcc_mean', 'mfcc_std', 'delta_mfcc_mean', 'delta_mfcc_std',
              'delta2_mfcc_mean', 'delta2_mfcc_std'
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    features = {}
    for i in range(n_mfcc):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfcc[i])
        features[f'delta_mfcc_{i+1}_mean'] = np.mean(delta_mfcc[i])
        features[f'delta_mfcc_{i+1}_std'] = np.std(delta_mfcc[i])
        features[f'delta2_mfcc_{i+1}_mean'] = np.mean(delta2_mfcc[i])
        features[f'delta2_mfcc_{i+1}_std'] = np.std(delta2_mfcc[i])

    return features


def extract_spectral_centroid(y, sr=TARGET_SR):
    """
    Extract spectral centroid statistics.

    The spectral centroid is the weighted mean of frequencies present in the
    signal, where magnitudes are the weights:

        SC = Σ(f * |X(f)|) / Σ|X(f)|

    It correlates with the perceived "brightness" of a sound. Pain cries
    tend to have higher spectral centroids (more high-frequency energy)
    compared to hunger or tiredness cries.

    Returns
    -------
    features : dict
        Keys: 'spectral_centroid_mean', 'spectral_centroid_std'
    """
    sc = librosa.feature.spectral_centroid(y=y, sr=sr,
                                            n_fft=FRAME_LENGTH,
                                            hop_length=HOP_LENGTH)[0]
    return {
        'spectral_centroid_mean': np.mean(sc),
        'spectral_centroid_std': np.std(sc)
    }


def extract_zcr(y):
    """
    Extract zero crossing rate statistics.

    ZCR measures how often the signal crosses the zero axis per frame.
    It is a simple time-domain feature that distinguishes:
    - High ZCR: noisy, unvoiced, or breathy sounds
    - Low ZCR: harmonic, voiced sounds (sustained crying)

    ZCR is computationally cheap and provides complementary information
    to spectral features. It is particularly useful for detecting the
    difference between voiced cry segments and unvoiced breath sounds.

    Returns
    -------
    features : dict
        Keys: 'zcr_mean', 'zcr_std'
    """
    zcr = librosa.feature.zero_crossing_rate(y,
                                              frame_length=FRAME_LENGTH,
                                              hop_length=HOP_LENGTH)[0]
    return {
        'zcr_mean': np.mean(zcr),
        'zcr_std': np.std(zcr)
    }


def extract_chroma(y, sr=TARGET_SR):
    """
    Extract chroma feature statistics.

    Chroma features project the spectrum onto 12 pitch classes (C, C#, ..., B).
    They capture the harmonic/melodic content of the cry independent of octave.

    While originally designed for music, chroma features are useful for cry
    analysis because:
    - Different emotional states produce different pitch contour patterns
    - Chroma captures tonal patterns complementary to MFCC's timbral features
    - They are relatively robust to noise compared to raw pitch tracking

    Returns
    -------
    features : dict
        Keys: 'chroma_{i}_mean', 'chroma_{i}_std' for i in 1..12
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_fft=FRAME_LENGTH,
                                          hop_length=HOP_LENGTH)
    features = {}
    for i in range(12):
        features[f'chroma_{i+1}_mean'] = np.mean(chroma[i])
        features[f'chroma_{i+1}_std'] = np.std(chroma[i])

    return features


# ============================================================
# Combined Feature Extraction
# ============================================================

def extract_all_features(y, sr=TARGET_SR):
    """
    Extract the complete feature vector for a single audio signal.

    Combines:
    - 13 MFCCs × (mean + std) + deltas + delta-deltas = 78 features
    - Spectral centroid (mean + std) = 2 features
    - ZCR (mean + std) = 2 features
    - 12 chroma (mean + std) = 24 features
    - Total: 106 features per audio file

    Parameters
    ----------
    y : np.ndarray
        Audio signal (preprocessed).
    sr : int
        Sample rate.

    Returns
    -------
    features : dict
        Combined feature dictionary.
    """
    features = {}
    features.update(extract_mfcc(y, sr))
    features.update(extract_spectral_centroid(y, sr))
    features.update(extract_zcr(y))
    features.update(extract_chroma(y, sr))
    return features


# ============================================================
# Batch Feature Extraction
# ============================================================

def process_feature_dataset(data_dirs, output_path=None, sr=TARGET_SR):
    """
    Extract features from all audio files across multiple directories.

    Processes both cleaned and noisy data directories, tagging each
    sample with a 'source' column ('cleaned' or 'noisy').

    Parameters
    ----------
    data_dirs : dict
        Mapping of source_name → directory_path.
        Example: {'cleaned': 'data/cleaned/', 'noisy': 'data/noisy/'}
    output_path : str, optional
        Path to save feature CSV. Defaults to data/features/features.csv.
    sr : int
        Sample rate.

    Returns
    -------
    df : pd.DataFrame
        Feature matrix with columns for features, label, source, and filename.
    """
    if output_path is None:
        output_path = os.path.join(FEATURES_DIR, 'features.csv')
    ensure_dir(os.path.dirname(output_path))

    all_records = []

    for source_name, data_dir in data_dirs.items():
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} does not exist, skipping.")
            continue

        for label in sorted(os.listdir(data_dir)):
            label_dir = os.path.join(data_dir, label)
            if not os.path.isdir(label_dir) or label.startswith('.'):
                continue

            audio_files = [f for f in sorted(os.listdir(label_dir))
                           if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
                           and not f.startswith('.')]

            for fname in tqdm(audio_files,
                              desc=f'Features [{source_name}/{label}]',
                              leave=False):
                filepath = os.path.join(label_dir, fname)
                try:
                    y, _ = librosa.load(filepath, sr=sr, mono=True)
                    features = extract_all_features(y, sr)
                    features['label'] = label
                    features['source'] = source_name
                    features['filename'] = fname
                    all_records.append(features)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    df = pd.DataFrame(all_records)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} feature vectors to {output_path}")
    return df
