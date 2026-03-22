"""
Audio Preprocessing Module for Infant Cry Recognition System.

This module implements a principled signal preprocessing pipeline:
  1. Loading — universal format support via librosa
  2. Resampling — standardize to 16 kHz (Nyquist ≥ 8 kHz covers cry fundamentals)
  3. Normalization — peak normalization to [-1, 1] for consistent amplitude
  4. Silence Trimming — remove non-informative leading/trailing silence

Each step is justified by signal processing theory and its impact on
downstream feature extraction and classification.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from src.utils import (
    TARGET_SR, RAW_DIR, CLEANED_DIR, AUDIO_EXTENSIONS,
    FOLDER_TO_LABEL, ensure_dir
)


# ============================================================
# 1. Audio Loading
# ============================================================

def load_audio(filepath, sr=TARGET_SR):
    """
    Load an audio file and resample to target sample rate.

    Parameters
    ----------
    filepath : str
        Path to audio file (wav, mp3, ogg, flac, m4a).
    sr : int
        Target sample rate. Default 16000 Hz.

    Returns
    -------
    y : np.ndarray
        Audio time series.
    sr : int
        Sample rate.

    Notes
    -----
    We use 16 kHz because:
    - Infant cry fundamental frequency is typically 300–600 Hz
    - Harmonics extend up to ~6–8 kHz
    - 16 kHz (Nyquist = 8 kHz) captures all relevant spectral information
    - Standard in speech processing literature (matches telephony/ASR conventions)
    """
    y, sr_out = librosa.load(filepath, sr=sr, mono=True)
    return y, sr_out


# ============================================================
# 2. Normalization
# ============================================================

def normalize_audio(y):
    """
    Peak normalize audio signal to [-1, 1].

    This ensures consistent amplitude across recordings made with
    different microphones, distances, and gain settings. Without
    normalization, amplitude-sensitive features like energy and
    MFCC magnitudes would be confounded by recording conditions.

    Parameters
    ----------
    y : np.ndarray
        Audio time series.

    Returns
    -------
    y_norm : np.ndarray
        Normalized audio signal.
    """
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y


# ============================================================
# 3. Silence Trimming
# ============================================================

def trim_silence(y, sr=TARGET_SR, top_db=25):
    """
    Remove leading and trailing silence from audio.

    Uses librosa's energy-based trimming with a threshold of 25 dB
    below the peak energy. This removes non-informative silence that
    would dilute feature statistics (e.g., lowering mean MFCC values)
    and waste computation.

    Parameters
    ----------
    y : np.ndarray
        Audio time series.
    sr : int
        Sample rate.
    top_db : int
        Threshold in dB below peak to consider as silence.

    Returns
    -------
    y_trimmed : np.ndarray
        Trimmed audio signal.
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


# ============================================================
# 4. Full Preprocessing Pipeline
# ============================================================

def preprocess_audio(filepath, sr=TARGET_SR, top_db=25):
    """
    Complete preprocessing pipeline for a single audio file.

    Pipeline: Load → Normalize → Trim Silence

    Parameters
    ----------
    filepath : str
        Path to input audio file.
    sr : int
        Target sample rate.
    top_db : int
        Silence trimming threshold.

    Returns
    -------
    y_processed : np.ndarray
        Preprocessed audio signal.
    sr : int
        Sample rate.
    """
    # Step 1: Load and resample
    y, sr = load_audio(filepath, sr=sr)

    # Step 2: Normalize amplitude
    y = normalize_audio(y)

    # Step 3: Trim silence
    y = trim_silence(y, sr=sr, top_db=top_db)

    return y, sr


# ============================================================
# 5. Batch Dataset Processing
# ============================================================

def process_dataset(input_dir=RAW_DIR, output_dir=CLEANED_DIR, sr=TARGET_SR):
    """
    Batch preprocess all audio files from input_dir and save to output_dir.

    Preserves the class-folder structure:
        input_dir/class_name/file.wav → output_dir/class_name/file.wav

    All output files are saved as 16-bit WAV at the target sample rate.

    Parameters
    ----------
    input_dir : str
        Path to raw data directory.
    output_dir : str
        Path to cleaned data directory.
    sr : int
        Target sample rate.

    Returns
    -------
    results : list of dict
        Processing results with file paths, durations, and status.
    """
    results = []
    ensure_dir(output_dir)

    for folder_name in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder_name)
        if not os.path.isdir(folder_path) or folder_name.startswith('.'):
            continue

        label = FOLDER_TO_LABEL.get(folder_name, folder_name)
        out_class_dir = ensure_dir(os.path.join(output_dir, label))

        audio_files = [f for f in sorted(os.listdir(folder_path))
                       if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
                       and not f.startswith('.')]

        for fname in tqdm(audio_files, desc=f'Processing {label}', leave=False):
            filepath = os.path.join(folder_path, fname)
            out_name = os.path.splitext(fname)[0] + '.wav'
            out_path = os.path.join(out_class_dir, out_name)

            try:
                y, sr_out = preprocess_audio(filepath, sr=sr)
                sf.write(out_path, y, sr_out, subtype='PCM_16')
                results.append({
                    'original_file': filepath,
                    'cleaned_file': out_path,
                    'label': label,
                    'duration_sec': len(y) / sr_out,
                    'samples': len(y),
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'original_file': filepath,
                    'cleaned_file': None,
                    'label': label,
                    'duration_sec': 0,
                    'samples': 0,
                    'status': f'error: {str(e)}'
                })

    return results
