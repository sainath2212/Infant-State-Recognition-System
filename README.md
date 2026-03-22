# Infant Cry Recognition System — Phase 1: Interpretable Acoustic Baseline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]()

An academic-grade infant cry classification pipeline that prioritizes **interpretability, signal processing rigor, and scientific reasoning** over black-box deep learning. This repository presents the foundational Phase 1 of the system, establishing a robust classical machine learning baseline.

---

## 1. The Clinical & Acoustic Problem

Infant crying is the primary pre-linguistic communication mechanism—a highly complex, non-stationary acoustic signal encoding critical information about an infant's physiological and emotional state. Automated classification of cry types (hunger, pain, discomfort, etc.) has profound clinical applications in neonatal intensive care units (NICUs), early developmental screening, and assistive caregiver support technologies.

**The Algorithmic Challenge:**
Existing deep learning approaches (such as ECAPA-TDNN or Blueprint Separable CNNs combined with TF-RNNs) achieve high empirical accuracy. However, they frequently sacrifice interpretability and computational efficiency—two mandatory requirements for verifiable and lightweight clinical deployment. Furthermore, these signals suffer from **pathological class imbalance** and **extreme acoustic overlap**, making naive classification highly susceptible to "majority-guessing" traps.

## 2. Dataset Overview & Pathological Imbalance

The data consists of annotated acoustic arrays captured from infants, categorized into 8 distinct physiological and emotional states:

| State | Samples | State | Samples |
| :--- | :--- | :--- | :--- |
| **belly_pain** | 127 | **hungry** | 382 |
| **burping** | 118 | **lonely** | 11 |
| **cold_hot** | 115 | **scared** | 27 |
| **discomfort** | 138 | **tired** | 136 |

*Total Audio Files: ~1,054*

**Note on Imbalance:** The dataset exhibits a severe 36:1 class imbalance ratio (Hungry vs. Lonely). This necessitates specialized validation metrics (Macro F1-Score) and algorithm penalization techniques (`class_weight='balanced'`) to prevent the model from ignoring underrepresented target states.

---

## 3. Comprehensive Pipeline Architecture

We engineered a structured, interpretable pipeline utilizing **first-principles signal processing theory** and **classical Statistical Machine Learning**. Our architecture is completely transparent, allowing us to scientifically audit the transformation of raw sound-waves into high-order geometric decision boundaries.

![Pipeline Architecture](architecture.png)

### Phase I: Acoustic Preprocessing
Raw audio is highly unstandardized. We strictly enforce uniform signal geometry using our `preprocessing.py` module:
- **Spatial Resampling (16 kHz):** Normalizes the Nyquist frequency limit, capturing the primary vocal tract formants without processing unnecessary ultrasonic noise.
- **Micro-Silence Trimming:** Eradicates dead atmospheric space at the edges of the waveform (top 15db threshold), heavily localizing the true acoustic activation.
- **Global Peak Normalization:** Uniformly scales amplitude vectors between `[-1.0, 1.0]`, completely removing hardware/microphone gain bias from the classification calculus.

### Phase II: Environmental Noise Augmentation
Clinical and home environments are never perfectly silent. To force the algorithm to learn the *structure* of the cry rather than the silence of the recording room, we inject robust synthetic artifacts via `noise_augmentation.py`:
- **Gaussian White Noise (SNR = 15dB):** Simulates standard thermal sensor noise.
- **Pink Background Noise:** Simulates organic atmospheric ambiance (1/f power spectral density) where lower-frequency environmental hums dominate.
- **Time Shifting (10% limit):** Randomly rolls the acoustic vector on the temporal axis, ensuring our models achieve translation invariance.

### Phase III: Multi-Dimensional Feature Engineering
Direct classification on 48,000-dimensional raw float arrays is computationally disastrous. We project the data into a dense, mathematically continuous 106-dimensional manifold (`feature_extraction.py`):
- **Mel-Frequency Cepstral Coefficients (MFCCs) [20 dimensions]:** Captures the human-auditory envelope (log-mel scaled spectrum).
- **Δ & ΔΔ MFCCs [40 dimensions]:** Tracks the exact velocity and acceleration (first and second order derivatives) of the infant's vocal tract over time.
- **Spectral Centroids & ZCR [4 dimensions]:** Measures the "center of mass" of frequencies and temporal volatility.
- **Chroma Dynamics [24 dimensions]:** Tracks harmonic progression and tonal centers.
*Every feature extracted inherently possesses a direct physical and acoustic interpretation.*

### Phase IV: Algorithmic Classification Baseline
With a processed 106-dimensional space, we deploy our solver (`model.py`):
- **Support Vector Machine (SVM) mapped with a Radial Basis Function (RBF) Kernel.** 
- **Non-Linear Manifolding:** The RBF projects non-linear data planes into infinite-dimensional Hilbert spaces, discovering hyper-separations that standard linear algebra cannot solve.
- **Validation:** Grid Search Optimization utilizing a strict 5-Fold Stratified Cross-Validation protocol to guarantee absolute statistical confidence and eliminate data leakage.

---

## 4. Visualizing the Acoustic Manifold (Jupyter Suite)

Our research strongly heavily relies on human-readable visualization to validate mathematical hypotheses. Inside the `/notebooks` directory, you will find exhaustive visual interpretations of the data:
- **`01_eda.ipynb`**: Showcases Probability Density Functions (PDFs) of cry durations, alongside deep-dive Peak Acoustic Profiles and Spectrogram analyses.
- **`02_preprocessing.ipynb`**: Demonstrates the empirical effect of peak normalization and silence stripping on waveform integrity.
- **`03_noise_augmentation.ipynb`**: Validates SNR levels over Frequency Profiles to ensure no critical acoustic topologies are overwritten by injected noise.
- **`04_feature_engineering.ipynb`**: Maps Multivariate Kernel Density Estimates (Pairplots) and Hierarchical Clustermaps to audit collinearity and feature separation dynamics.
- **`05_baseline_ml.ipynb`**: Features multi-class ROC curves, Precision-Recall Polar Radar Coordinates, Confusion Matrices (Error mapping), and the final 2D PCA Algorithmic Decision Boundary.

---

## 5. Project Structure

```text
Infant-State-Recognition-System/
├── data/
│   ├── raw/                 # Unaltered foundational dataset
│   ├── cleaned/             # Normalized, trimmed, 16kHz WAV projections
│   ├── noisy/               # Noise-augmented stress-test variants
│   └── features/            # Processed 106-dim extracted CSV arrays
├── notebooks/
│   ├── 00_literature_review.ipynb    # Theoretical foundation & research gap analysis
│   ├── 01_eda.ipynb                  # Deep-dive Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb        # Signal conditioning visual walk-through
│   ├── 03_noise_augmentation.ipynb   # Spectrograms & Synthetic stress testing
│   ├── 04_feature_engineering.ipynb  # Extraction, KDEs, and Collinearity Audits
│   └── 05_baseline_ml.ipynb          # SVM convergence, ROCs, & PCA Boundaries
├── src/
│   ├── utils.py             # Global constants & robust plotting configurators
│   ├── preprocessing.py     # Functional signal conditioning logic
│   ├── noise_augmentation.py# Artifact injection generators (Pink, Gaussian, Shift)
│   ├── feature_extraction.py# Fast Fourier Transform (FFT) & extraction pipelines
│   └── model.py             # SVM estimators, cross-validation & evaluation tooling
├── README.md
└── requirements.txt
```

---

## 6. Setup & Execution

**1. Clone & Install Dependencies:**
```bash
git clone https://github.com/sainath2212/Infant-State-Recognition-System.git
cd Infant-State-Recognition-System
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Execute the Iterative Pipeline:**
Navigate sequentially through the `notebooks/` directory. The architectural flow is completely sequential.
* `02_preprocessing.ipynb` populates the `/cleaned` directory.
* `03_noise_augmentation.ipynb` populates the `/noisy` directory.
* `04_feature_engineering.ipynb` compiles the global `audio_features.csv` dataset.
* `05_baseline_ml.ipynb` reads the dataset, calculates High-Level Accuracy Metrics, and renders the Algorithmic Solution Space.

---

## 7. Key Academic References

1. *Davis & Mermelstein (1980)* — Comparison of parametric representations for monosyllabic word recognition (MFCC foundation).
2. *Cortes & Vapnik (1995)* — Support-Vector Networks (SVM mathematical theory).
3. Current Deep Literature: *ECAPA-TDNN for Infant Cry Emotion Recognition* and *Blueprint Separable CNN + TF-RNN*.

---

## 8. Strategic Roadmap

- **Phase 1 (Current):** Establish a scientifically rigorous, fully interpretable signal processing and classical SVM algorithmic baseline.
- **Phase 2 (Upcoming):** Design non-linear Deep Learning architectures (CNNs for spectrogram topography, LSTMs for temporal dynamics) integrated with our hybrid features.
- **Phase 3 (Deployment):** Distill the resulting optimal topologies into lightweight, edge-deployable matrices for real-time mobile and clinical hardware application.
