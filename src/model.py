"""
Baseline ML Model for Infant Cry Recognition System.

This module implements a Support Vector Machine (SVM) with RBF kernel
for multi-class cry classification.

Why SVM with RBF Kernel?
========================

1. **Kernel Trick**: The RBF (Radial Basis Function) kernel implicitly maps
   input features into an infinite-dimensional Hilbert space where linear
   separation is possible. Instead of computing φ(x) explicitly, we compute
   K(x, x') = exp(-γ||x - x'||²), which is computationally tractable.

2. **Margin Maximization**: SVM finds the hyperplane that maximizes the margin
   (distance to nearest training points = support vectors). This geometric
   objective provides strong generalization guarantees via VC theory.

3. **Non-linear Decision Boundaries**: Acoustic feature spaces are inherently
   non-linear — cry states with similar spectral properties (e.g., pain vs.
   discomfort) require non-linear boundaries to separate.

4. **Robustness to Moderate Datasets**: SVMs perform well with limited data
   (our ~1000 samples) because the decision boundary depends only on support
   vectors, not the entire dataset. This makes SVM more appropriate than
   deep learning for our data scale.

5. **Interpretability**: Unlike deep networks, we can inspect support vectors,
   analyze per-class performance, and understand failure modes through
   the confusion matrix.

Hyperparameter Tuning:
- C (regularization): Controls bias-variance tradeoff
- γ (kernel width): Controls decision boundary complexity
- GridSearchCV with stratified k-fold for reliable estimation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline

from src.utils import CLASS_LABELS, set_plot_style


# ============================================================
# Data Preparation
# ============================================================

def prepare_data(df, test_size=0.2, random_state=42):
    """
    Prepare data for training: encode labels, split, and scale.

    We use stratified splitting to preserve class proportions in both
    train and test sets, which is critical given our severe class imbalance
    (hungry: 382 vs lonely: 11).

    StandardScaler is used because SVM with RBF kernel is sensitive to
    feature scales — the Euclidean distance in K(x,x') = exp(-γ||x-x'||²)
    would be dominated by large-magnitude features without scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with 'label' column.
    test_size : float
        Fraction for test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Split and scaled data.
    le : LabelEncoder
        Fitted label encoder.
    scaler : StandardScaler
        Fitted scaler.
    feature_names : list
        Names of feature columns.
    """
    # Identify feature columns (exclude metadata)
    meta_cols = ['label', 'source', 'filename']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)

    # Stratified split preserves class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features — critical for SVM with RBF kernel
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, le, scaler, feature_cols


# ============================================================
# Model Training
# ============================================================

def train_svm(X_train, y_train, param_grid=None, cv=5):
    """
    Train SVM with RBF kernel using GridSearchCV.

    We search over:
    - C ∈ {0.1, 1, 10, 100}: regularization strength
      - Small C → wide margin, more misclassifications (high bias)
      - Large C → narrow margin, fewer misclassifications (high variance)
    - γ ∈ {0.001, 0.01, 0.1, 1}: kernel bandwidth
      - Small γ → smooth boundary, looks at distant points (underfitting risk)
      - Large γ → complex boundary, focuses on nearby points (overfitting risk)

    Stratified K-Fold ensures each fold preserves class distribution.

    Parameters
    ----------
    X_train : np.ndarray
        Training features (scaled).
    y_train : np.ndarray
        Training labels (encoded).
    param_grid : dict, optional
        Hyperparameter grid for GridSearchCV.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    best_model : SVC
        Best SVM model from grid search.
    cv_results : dict
        Cross-validation results.
    """
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['rbf']
        }

    svm = SVC(class_weight='balanced', random_state=42)

    # Use F1-macro as scoring metric because of class imbalance
    # Macro-average treats all classes equally regardless of size
    grid_search = GridSearchCV(
        svm, param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV F1-score (macro): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.cv_results_


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, X_test, y_test, le):
    """
    Comprehensive model evaluation with multiple metrics.

    We report:
    - Accuracy: overall correctness (can be misleading with imbalance)
    - Precision: of predicted positives, how many are correct (important
      to avoid false alarms in clinical settings)
    - Recall: of actual positives, how many were found (critical for
      not missing pain/distress states)
    - F1-score: harmonic mean of precision and recall (our primary metric
      because it balances both concerns under class imbalance)

    Parameters
    ----------
    model : SVC
        Trained SVM model.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    le : LabelEncoder
        Label encoder for class names.

    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics.
    report : str
        Full classification report.
    cm : np.ndarray
        Confusion matrix.
    """
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    return metrics, report, cm


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix',
                          normalize=False, ax=None):
    """
    Plot a publication-quality confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix from sklearn.
    class_names : list
        Class label names.
    title : str
        Plot title.
    normalize : bool
        If True, show percentages instead of counts.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = title + ' (Normalized)'
    else:
        cm_display = cm
        fmt = 'd'

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor='gray')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return ax
