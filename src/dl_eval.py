"""
Evaluation & Interpretability Module for CryNet — Phase 2.

This module provides comprehensive model evaluation tools:

1. **Standard Metrics**: Accuracy, Precision, Recall, F1 (macro/weighted),
   per-class classification report, multi-class ROC/AUC.

2. **Grad-CAM** (Selvaraju et al., 2017): Gradient-weighted Class Activation
   Mapping visualizes which spectro-temporal regions the CNN attends to.
   For mel-spectrograms, this reveals which frequency bands and time periods
   are most important for each cry classification.

3. **Ablation Studies**: Systematic removal of architectural components
   (SE blocks, attention, BiLSTM, skip connections) to quantify each
   component's contribution to overall performance.

4. **Robustness Testing**: Evaluates model performance under varying
   levels of injected noise, simulating real-world deployment conditions.

5. **Embedding Extraction**: Extracts penultimate-layer representations
   for t-SNE/UMAP visualization of the learned feature space.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize


# ============================================================
# Comprehensive Evaluation
# ============================================================

@torch.no_grad()
def full_evaluation(model, data_loader, class_names, device=None):
    """
    Run comprehensive evaluation on a dataset.

    Returns
    -------
    results : dict
        Contains: metrics, report, confusion_matrix, all_preds,
        all_labels, all_probs
    """
    if device is None:
        device = torch.device('cpu')

    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Core metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds,
                                           average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds,
                                     average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds,
                             average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds,
                                average='weighted', zero_division=0),
    }

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )
    report_str = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'metrics': metrics,
        'report': report,
        'report_str': report_str,
        'confusion_matrix': cm,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs,
    }


# ============================================================
# ROC / AUC Computation
# ============================================================

def compute_roc_curves(all_labels, all_probs, class_names):
    """
    Compute per-class and micro/macro-averaged ROC curves.

    Parameters
    ----------
    all_labels : np.ndarray
        True labels (integer encoded).
    all_probs : np.ndarray
        Predicted probabilities of shape (N, n_classes).
    class_names : list of str

    Returns
    -------
    roc_data : dict
        Contains fpr, tpr, auc for each class and averages.
    """
    n_classes = len(class_names)
    # Binarize labels for OVR ROC
    y_bin = label_binarize(all_labels, classes=list(range(n_classes)))

    # Handle edge case where some classes might not appear in test set
    if y_bin.shape[1] != n_classes:
        # Pad missing classes
        padded = np.zeros((len(all_labels), n_classes))
        padded[:, :y_bin.shape[1]] = y_bin
        y_bin = padded

    roc_data = {}

    # Per-class ROC
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            # Skip classes with no positive samples
            roc_data[name] = {'fpr': [0, 1], 'tpr': [0, 1], 'auc': 0.5}
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), all_probs.ravel())
    roc_data['micro'] = {
        'fpr': fpr_micro, 'tpr': tpr_micro,
        'auc': auc(fpr_micro, tpr_micro)
    }

    # Macro-average (average of per-class)
    all_fpr = np.unique(np.concatenate([
        roc_data[name]['fpr'] for name in class_names
        if name in roc_data
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for name in class_names:
        if name in roc_data:
            mean_tpr += np.interp(all_fpr, roc_data[name]['fpr'],
                                  roc_data[name]['tpr'])
    mean_tpr /= n_classes
    roc_data['macro'] = {
        'fpr': all_fpr, 'tpr': mean_tpr,
        'auc': auc(all_fpr, mean_tpr)
    }

    return roc_data


# ============================================================
# Precision-Recall Curves
# ============================================================

def compute_pr_curves(all_labels, all_probs, class_names):
    """
    Compute per-class Precision-Recall curves.

    PR curves are more informative than ROC for imbalanced datasets
    because they focus on positive predictions rather than negatives.

    Returns
    -------
    pr_data : dict
        Contains precision, recall, AP for each class.
    """
    n_classes = len(class_names)
    y_bin = label_binarize(all_labels, classes=list(range(n_classes)))

    if y_bin.shape[1] != n_classes:
        padded = np.zeros((len(all_labels), n_classes))
        padded[:, :y_bin.shape[1]] = y_bin
        y_bin = padded

    pr_data = {}
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            pr_data[name] = {'precision': [1], 'recall': [0], 'ap': 0.0}
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        ap = average_precision_score(y_bin[:, i], all_probs[:, i])
        pr_data[name] = {'precision': prec, 'recall': rec, 'ap': ap}

    return pr_data


# ============================================================
# Grad-CAM Implementation
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Grad-CAM uses the gradients flowing into the last convolutional
    layer to produce a coarse localization map highlighting the
    important regions in the input for the predicted class.

    For mel-spectrograms:
    - Horizontal highlights → important time segments
    - Vertical highlights → important frequency bands
    - Combined → specific time-frequency patterns (e.g., a formant
      at a particular moment)

    Algorithm:
    1. Forward pass: store target conv layer activations A^k
    2. Backward pass: compute gradients ∂y^c/∂A^k
    3. Global average pool gradients: α_k = (1/Z) Σ_i Σ_j ∂y^c/∂A^k_{ij}
    4. Weighted combination: L = ReLU(Σ_k α_k · A^k)
    5. Upsample to input size

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
    Deep Networks" — ICCV 2017
    """

    def __init__(self, model, target_layer):
        """
        Parameters
        ----------
        model : nn.Module
            Trained CNN model.
        target_layer : nn.Module
            Target convolutional layer (usually the last one).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for an input.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Single input of shape (1, C, H, W).
        target_class : int, optional
            Class to generate CAM for. If None, uses predicted class.

        Returns
        -------
        cam : np.ndarray
            Grad-CAM heatmap of same spatial size as conv output,
            normalized to [0, 1].
        predicted_class : int
            The class used for gradient computation.
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target = output[0, target_class]
        target.backward()

        # Get stored gradients and activations
        gradients = self.gradients[0]    # (C, H, W)
        activations = self.activations[0] # (C, H, W)

        # Global average pooling of gradients → channel weights
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — we only care about positive influence
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), target_class


# ============================================================
# Embedding Extraction for t-SNE/UMAP
# ============================================================

@torch.no_grad()
def extract_embeddings(model, data_loader, device=None):
    """
    Extract penultimate-layer embeddings from the model.

    These embeddings represent the model's learned feature space.
    Visualizing them with t-SNE or UMAP reveals:
    - Whether classes form distinct clusters
    - Which classes are confused (overlapping clusters)
    - The quality of the learned representation

    Parameters
    ----------
    model : nn.Module
        Trained model with get_embeddings() method.
    data_loader : DataLoader
    device : torch.device

    Returns
    -------
    embeddings : np.ndarray
        Shape (N, embed_dim).
    labels : np.ndarray
        Shape (N,).
    """
    if device is None:
        device = torch.device('cpu')

    model.eval()
    model = model.to(device)

    all_embeddings = []
    all_labels = []

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        emb = model.get_embeddings(inputs)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.concatenate(all_embeddings, axis=0), np.array(all_labels)


# ============================================================
# Robustness Testing
# ============================================================

@torch.no_grad()
def robustness_test(model, data_loader, noise_levels, device=None):
    """
    Test model robustness under varying noise levels.

    Adds Gaussian noise of different magnitudes to the input
    spectrograms and measures how performance degrades. A robust
    model should maintain reasonable accuracy even with moderate noise.

    Parameters
    ----------
    model : nn.Module
    data_loader : DataLoader
    noise_levels : list of float
        Standard deviations of Gaussian noise to inject.
    device : torch.device

    Returns
    -------
    results : dict
        Mapping noise_level → {'accuracy': ..., 'f1_macro': ...}
    """
    if device is None:
        device = torch.device('cpu')

    model.eval()
    model = model.to(device)
    results = {}

    for noise_std in noise_levels:
        all_preds = []
        all_labels = []

        for inputs, labels in data_loader:
            # Add noise
            noisy_inputs = inputs + torch.randn_like(inputs) * noise_std
            noisy_inputs = noisy_inputs.to(device)

            outputs = model(noisy_inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        results[noise_std] = {'accuracy': acc, 'f1_macro': f1}

    return results
