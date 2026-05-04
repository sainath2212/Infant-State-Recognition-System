"""
Machine Learning Tier for Infant Cry Recognition.

This module implements the classical ML component of the hybrid ensemble:

Models
------
1. SVM (RBF Kernel)        — tuned on MFCC + spectral features
2. Random Forest           — 200 trees, feature importance analysis
3. XGBoost                 — gradient boosting with DST + MFCC features
4. Logistic Regression     — interpretable baseline

Feature Set
-----------
- MFCC (40 coefficients × mean/std)        = 80 features
- Delta MFCC (40 × mean/std)               = 80 features
- Spectral features (centroid/rolloff/flux) = 12 features
- ZCR, RMS energy                          = 4 features
- DST statistical features (64 × 5)        = 320 features
Total                                       = 496 features

Training Strategy
-----------------
- SMOTE oversampling (handles 36:1 class imbalance)
- StandardScaler normalization
- GridSearchCV / RandomizedSearchCV for hyperparameter tuning
- Class-weighted loss for remaining imbalance
"""

import os
import numpy as np
import joblib
import librosa
from typing import List, Tuple, Dict, Optional

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = False
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from src.dst_features import dst_statistical_features, DST_CONFIG
from src.utils import TARGET_SR, CLASS_LABELS


# ============================================================
# Feature Extraction
# ============================================================

def extract_handcrafted_features(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract comprehensive handcrafted features from a waveform.

    Feature Groups:
    ---------------
    1. MFCC (40 coefficients): captures timbral/spectral envelope
    2. Delta MFCC: first-order temporal derivatives (cry dynamics)
    3. Spectral features: centroid, rolloff, bandwidth, flux, contrast
    4. Prosodic: zero-crossing rate, RMS energy, fundamental frequency

    Parameters
    ----------
    y : np.ndarray
        Audio waveform (1D).
    sr : int
        Sample rate.

    Returns
    -------
    features : np.ndarray of shape (n_features,)
    """
    features = []

    # 1. MFCC (40 coefficients) — mean and std
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=1024, hop_length=256)
    features.extend(mfcc.mean(axis=1))   # 40
    features.extend(mfcc.std(axis=1))    # 40

    # 2. Delta MFCC — mean and std
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(delta_mfcc.mean(axis=1))  # 40
    features.extend(delta_mfcc.std(axis=1))   # 40

    # 3. Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=256)
    features.append(centroid.mean())
    features.append(centroid.std())

    # 4. Spectral rolloff (85% and 95%)
    rolloff_85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, n_fft=1024, hop_length=256)
    rolloff_95 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95, n_fft=1024, hop_length=256)
    features.append(rolloff_85.mean())
    features.append(rolloff_85.std())
    features.append(rolloff_95.mean())
    features.append(rolloff_95.std())

    # 5. Spectral bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=1024, hop_length=256)
    features.append(bw.mean())
    features.append(bw.std())

    # 6. Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=1024, hop_length=256)
    features.extend(contrast.mean(axis=1))  # 7

    # 7. Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=256)
    features.append(zcr.mean())
    features.append(zcr.std())

    # 8. RMS energy
    rms = librosa.feature.rms(y=y, hop_length=256)
    features.append(rms.mean())
    features.append(rms.std())

    return np.array(features, dtype=np.float32)


def extract_full_feature_vector(
    y: np.ndarray,
    sr: int = TARGET_SR,
    include_dst: bool = True,
) -> np.ndarray:
    """
    Extract the full feature vector combining handcrafted + DST features.

    Parameters
    ----------
    y : np.ndarray
        Audio waveform.
    sr : int
        Sample rate.
    include_dst : bool
        Whether to include DST statistical features (adds 320 dims).

    Returns
    -------
    features : np.ndarray
        Full feature vector.
    """
    hc = extract_handcrafted_features(y, sr)
    if include_dst:
        dst = dst_statistical_features(y, sr, n_freqs=DST_CONFIG['n_freqs'])
        return np.concatenate([hc, dst])
    return hc


def build_feature_matrix(
    file_paths: List[str],
    labels: List[int],
    sr: int = TARGET_SR,
    duration: float = 3.0,
    include_dst: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix from a list of audio files.

    Parameters
    ----------
    file_paths : list of str
    labels : list of int
    sr : int
    duration : float
    include_dst : bool
    verbose : bool

    Returns
    -------
    X : np.ndarray of shape (N, n_features)
    y : np.ndarray of shape (N,)
    """
    X, y = [], []
    target_len = int(sr * duration)

    for i, (fp, lbl) in enumerate(zip(file_paths, labels)):
        if verbose and i % 50 == 0:
            print(f"  Extracting features: {i}/{len(file_paths)}")
        try:
            audio, _ = librosa.load(fp, sr=sr, mono=True, duration=duration)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            feat = extract_full_feature_vector(audio, sr, include_dst)
            X.append(feat)
            y.append(lbl)
        except Exception as e:
            if verbose:
                print(f"  Warning: skipping {fp}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ============================================================
# Model Definitions
# ============================================================

def build_svm_pipeline(class_weight: str = 'balanced') -> Pipeline:
    """
    SVM with RBF kernel wrapped in a sklearn Pipeline with scaling.

    C=10, gamma='scale' — tuned for MFCC + DST feature space.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            class_weight=class_weight,
            probability=True,   # enable predict_proba
            random_state=42,
        ))
    ])


def build_random_forest(n_estimators: int = 300) -> Pipeline:
    """
    Random Forest with class-balanced sampling.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=1,
            random_state=42,
        ))
    ])


def build_xgboost(n_estimators: int = 400) -> Pipeline:
    """
    XGBoost classifier — handles feature interactions well.
    Uses scale_pos_weight for imbalance.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    return Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=1,
            random_state=42,
            verbosity=0,
        ))
    ])


def build_logistic_regression() -> Pipeline:
    """
    Logistic Regression — fast, interpretable baseline.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=1.0,
            class_weight='balanced',
            solver='lbfgs',
            max_iter=2000,
            multi_class='multinomial',
            random_state=42,
        ))
    ])


# ============================================================
# ML Ensemble
# ============================================================

class MLEnsemble:
    """
    Ensemble of classical ML models for infant cry classification.

    Produces soft probability predictions (averaged across models)
    for use in the meta-ensemble with DL models.

    Parameters
    ----------
    use_smote : bool
        Apply SMOTE oversampling during training.
    include_dst : bool
        Include DST statistical features in the feature vector.
    """

    def __init__(
        self,
        use_smote: bool = True,
        include_dst: bool = True,
        class_names: Optional[List[str]] = None,
    ):
        self.use_smote = use_smote and SMOTE_AVAILABLE
        self.include_dst = include_dst
        self.class_names = class_names or CLASS_LABELS

        # Individual models
        self.svm = build_svm_pipeline()
        self.rf = build_random_forest()
        self.lr = build_logistic_regression()

        self.models = {'svm': self.svm, 'rf': self.rf, 'lr': self.lr}

        if XGBOOST_AVAILABLE:
            self.xgb = build_xgboost()
            self.models['xgb'] = self.xgb

        # Weights for soft voting (sum = 1)
        self._weights = {'svm': 0.40, 'rf': 0.35, 'lr': 0.25}

        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> 'MLEnsemble':
        """
        Fit all ML models.

        Parameters
        ----------
        X : np.ndarray of shape (N, n_features)
        y : np.ndarray of shape (N,)
        verbose : bool

        Returns
        -------
        self
        """
        if self.use_smote:
            if verbose:
                print("Applying SMOTE oversampling...")
            smote = SMOTE(random_state=42, k_neighbors=3)
            try:
                X, y = smote.fit_resample(X, y)
                if verbose:
                    print(f"  After SMOTE: {X.shape[0]} samples")
            except Exception as e:
                if verbose:
                    print(f"  SMOTE failed ({e}), proceeding without oversampling")

        for name, model in self.models.items():
            if verbose:
                print(f"Fitting {name.upper()}...")
            model.fit(X, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted soft-vote probability predictions.

        Parameters
        ----------
        X : np.ndarray of shape (N, n_features)

        Returns
        -------
        proba : np.ndarray of shape (N, n_classes)
        """
        n_classes = len(self.class_names)
        total_weight = 0.0
        proba = np.zeros((len(X), n_classes))

        for name, model in self.models.items():
            w = self._weights.get(name, 1.0 / len(self.models))
            proba += w * model.predict_proba(X)
            total_weight += w

        return proba / total_weight

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate the ensemble.

        Returns
        -------
        results : dict with accuracy, f1_macro, f1_weighted, report
        """
        preds = self.predict(X)
        return {
            'accuracy': accuracy_score(y, preds),
            'f1_macro': f1_score(y, preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(y, preds, average='weighted', zero_division=0),
            'report': classification_report(
                y, preds,
                target_names=self.class_names,
                zero_division=0,
            ),
        }

    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        joblib.dump({
            'models': self.models,
            'weights': self._weights,
            'class_names': self.class_names,
            'include_dst': self.include_dst,
        }, path)
        print(f"ML Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'MLEnsemble':
        """Load ensemble from disk."""
        data = joblib.load(path)
        obj = cls(class_names=data['class_names'], include_dst=data.get('include_dst', True))
        obj.models = data['models']
        obj._weights = data['weights']
        obj.is_fitted = True
        return obj
