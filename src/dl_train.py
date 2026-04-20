"""
Training Module for CryNet — Infant Cry Recognition Phase 2.

This module implements the complete training pipeline including:

1. **Focal Loss** (Lin et al., 2017): A modified cross-entropy that
   down-weights easy examples and focuses training on hard, misclassified
   samples. Critical for our imbalanced dataset where "hungry" (382 samples)
   would otherwise dominate gradient updates.

   FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

   Where γ controls the focusing strength:
   - γ=0: standard cross-entropy
   - γ=2: strongly focuses on hard examples (our choice)

2. **Cosine Annealing with Warmup**: Learning rate schedule that:
   - Linearly warms up from 0 to lr_max over first few epochs
   - Then decays following a cosine curve to near-zero
   This avoids the instability of high LR at initialization while
   allowing fine-grained convergence at the end.

3. **Gradient Clipping**: Caps gradient norms to prevent exploding
   gradients, especially important for LSTM components.

4. **Early Stopping**: Monitors validation loss and stops training
   when performance plateaus, preventing overfitting.

5. **Training Metrics Tracking**: Records per-epoch loss, accuracy,
   F1-score, learning rate, and gradient norms for visualization.
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

from src.dl_data import mixup_data, mixup_criterion


# ============================================================
# Focal Loss Implementation
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Standard cross-entropy treats all examples equally, which causes
    the model to be overwhelmed by easy, majority-class examples.
    Focal Loss adds a modulating factor (1-p_t)^γ that reduces the
    loss contribution from well-classified examples.

    For infant cry with 8 classes of vastly different sizes:
    - Easy examples (e.g., correctly classified "hungry" — largest class)
      get their loss reduced by (1-0.95)^2 ≈ 0.0025
    - Hard examples (e.g., confused "lonely" vs "scared" — tiny classes)
      retain nearly full loss contribution

    Parameters
    ----------
    alpha : torch.Tensor or None
        Per-class weights of shape (n_classes,). If None, uniform weights.
    gamma : float
        Focusing parameter. Higher γ = more focus on hard examples.
        γ=0 → standard CE, γ=2 → recommended (our choice).
    reduction : str
        'mean' or 'sum'.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.FloatTensor(alpha)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Compute focal loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw logits of shape (B, C).
        targets : torch.Tensor
            Ground truth labels of shape (B,).

        Returns
        -------
        loss : torch.Tensor
            Scalar focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # p_t = probability of the correct class
        p_t = torch.exp(-ce_loss)
        # Focal modulation
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================
# Learning Rate Scheduler: Cosine Annealing with Warmup
# ============================================================

class CosineWarmupScheduler:
    """
    Cosine Annealing LR Schedule with Linear Warmup.

    Schedule:
    - Epochs [0, warmup): LR linearly increases from 0 to base_lr
    - Epochs [warmup, total): LR follows cosine decay to min_lr

    Cosine decay formula:
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * t'))
        where t' = (epoch - warmup) / (total - warmup)

    Why Cosine over StepLR?
    - Smoother decay avoids sudden loss spikes at step boundaries
    - Gradual convergence reaches better local minima
    - No hyperparameter tuning for step sizes/gamma

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_epochs : int
    total_epochs : int
    base_lr : float
    min_lr : float
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 base_lr=1e-3, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        """Update learning rate for current epoch."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                       max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training when it hasn't improved
    for `patience` consecutive epochs. Saves the best model weights.

    Parameters
    ----------
    patience : int
        Number of epochs to wait for improvement.
    min_delta : float
        Minimum change to qualify as improvement.
    """

    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def load_best_model(self, model):
        """Restore best model weights."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# ============================================================
# Training History Tracker
# ============================================================

class TrainingHistory:
    """Track and store training metrics per epoch."""

    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.train_f1 = []
        self.val_f1 = []
        self.learning_rates = []
        self.grad_norms = []
        self.epoch_times = []

    def record(self, train_loss, val_loss, train_acc, val_acc,
               train_f1, val_f1, lr, grad_norm, epoch_time):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.train_f1.append(train_f1)
        self.val_f1.append(val_f1)
        self.learning_rates.append(lr)
        self.grad_norms.append(grad_norm)
        self.epoch_times.append(epoch_time)

    def to_dict(self):
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_acc': self.train_acc,
            'val_acc': self.val_acc,
            'train_f1': self.train_f1,
            'val_f1': self.val_f1,
            'learning_rates': self.learning_rates,
            'grad_norms': self.grad_norms,
            'epoch_times': self.epoch_times
        }


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer,
                    device, use_mixup=True, mixup_alpha=0.4,
                    max_grad_norm=5.0):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    criterion : loss function
    optimizer : optimizer
    device : torch.device
    use_mixup : bool
        Whether to apply MixUp augmentation.
    mixup_alpha : float
        MixUp interpolation strength.
    max_grad_norm : float
        Maximum gradient norm for clipping.

    Returns
    -------
    avg_loss : float
    accuracy : float
    macro_f1 : float
    avg_grad_norm : float
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    grad_norms = []

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_mixup and np.random.random() > 0.5:
            # Apply MixUp with 50% probability
            mixed_inputs, targets_a, targets_b, lam = mixup_data(
                inputs, labels, alpha=mixup_alpha
            )
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # For metrics, use the dominant label
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets_a.cpu().numpy())
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTM
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        )
        grad_norms.append(grad_norm.item())

        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_grad_norm = np.mean(grad_norms)

    return avg_loss, accuracy, macro_f1, avg_grad_norm


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    Evaluate model on validation set.

    Parameters
    ----------
    model : nn.Module
    val_loader : DataLoader
    criterion : loss function
    device : torch.device

    Returns
    -------
    avg_loss : float
    accuracy : float
    macro_f1 : float
    all_preds : np.ndarray
    all_labels : np.ndarray
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        running_loss += loss.item() * inputs.size(0)

    avg_loss = running_loss / len(val_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, macro_f1, np.array(all_preds), np.array(all_labels)


# ============================================================
# Full Training Pipeline
# ============================================================

def train_model(model, train_loader, val_loader, class_weights=None,
                n_epochs=80, base_lr=1e-3, warmup_epochs=5,
                patience=15, use_mixup=True, device=None,
                save_path=None, verbose=True):
    """
    Complete training pipeline for CryNet.

    Parameters
    ----------
    model : nn.Module
        CryNet or ablation variant.
    train_loader : DataLoader
    val_loader : DataLoader
    class_weights : torch.Tensor, optional
        Per-class weights for focal loss.
    n_epochs : int
        Maximum training epochs.
    base_lr : float
        Base learning rate.
    warmup_epochs : int
        LR warmup epochs.
    patience : int
        Early stopping patience.
    use_mixup : bool
        Whether to use MixUp augmentation.
    device : torch.device
    save_path : str, optional
        Path to save best model checkpoint.
    verbose : bool

    Returns
    -------
    model : nn.Module
        Trained model with best weights.
    history : TrainingHistory
        Training metrics history.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else 'cpu')

    model = model.to(device)

    # Focal Loss with class weights
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # AdamW optimizer with weight decay for L2 regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=warmup_epochs,
        total_epochs=n_epochs, base_lr=base_lr
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # History tracking
    history = TrainingHistory()

    if verbose:
        print(f"Training on device: {device}")
        total, trainable = model.count_parameters() if hasattr(model, 'count_parameters') else (0, 0)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"{'='*70}")
        print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
              f"{'Train Acc':>9} | {'Val Acc':>9} | {'Val F1':>8} | {'LR':>10}")
        print(f"{'-'*70}")

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc, train_f1, grad_norm = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=use_mixup
        )

        # Validate
        val_loss, val_acc, val_f1, _, _ = validate(
            model, val_loader, criterion, device
        )

        # Update LR
        current_lr = scheduler.step()

        # Record history
        epoch_time = time.time() - start_time
        history.record(
            train_loss, val_loss, train_acc, val_acc,
            train_f1, val_f1, current_lr, grad_norm, epoch_time
        )

        if verbose and (epoch % 5 == 0 or epoch <= 3 or epoch == n_epochs):
            print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:10.4f} | "
                  f"{train_acc:9.4f} | {val_acc:9.4f} | {val_f1:8.4f} | "
                  f"{current_lr:10.6f}")

        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.should_stop:
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Load best model
    early_stopping.load_best_model(model)

    if verbose:
        print(f"{'='*70}")
        print(f"Best validation loss: {early_stopping.best_loss:.4f}")

    # Save model
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history.to_dict(),
            'class_weights': class_weights
        }, save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    return model, history
