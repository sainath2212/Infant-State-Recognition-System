"""
Utility functions for the Infant Cry Recognition System.
Provides common helpers for path management, class labeling, and visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')
NOISY_DIR = os.path.join(DATA_DIR, 'noisy')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')

# ============================================================
# Class Labels
# ============================================================

CLASS_LABELS = [
    'belly_pain', 'burping', 'cold_hot', 'discomfort',
    'hungry', 'lonely', 'scared', 'tired'
]

# Mapping from folder names (as they appear on disk) to normalized labels
FOLDER_TO_LABEL = {
    'belly pain': 'belly_pain',
    'belly_pain': 'belly_pain',
    'burping': 'burping',
    'cold_hot': 'cold_hot',
    'discomfort': 'discomfort',
    'hungry': 'hungry',
    'lonely': 'lonely',
    'scared': 'scared',
    'tired': 'tired',
}

# Target sample rate for all audio processing (16 kHz is standard for speech/cry)
TARGET_SR = 16000

# ============================================================
# Audio Dataset Discovery
# ============================================================

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}


def discover_audio_files(data_dir):
    """
    Walk a directory and return list of (filepath, label) tuples.
    Skips hidden files like .DS_Store.
    """
    files = []
    for folder_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path) or folder_name.startswith('.'):
            continue
        label = FOLDER_TO_LABEL.get(folder_name, folder_name)
        for fname in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in AUDIO_EXTENSIONS and not fname.startswith('.'):
                files.append((os.path.join(folder_path, fname), label))
    return files


def get_class_distribution(file_list):
    """Return a dict of label -> count from a list of (filepath, label) tuples."""
    from collections import Counter
    return dict(Counter(label for _, label in file_list))


# ============================================================
# Visualization Helpers
# ============================================================

def set_plot_style():
    """Set a clean, academic plot style."""
    sns.set_theme(style='whitegrid', palette='deep', font_scale=1.1)
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 100,
    })


def plot_class_distribution(distribution, title='Class Distribution', ax=None):
    """Plot a bar chart of class distribution with value labels."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    labels = sorted(distribution.keys())
    counts = [distribution[l] for l in labels]
    colors = sns.color_palette('viridis', len(labels))
    bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_xlabel('Cry Category')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return ax


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path
