#!/usr/bin/env python3
"""
12 — CryNetV2 Hybrid Training & Full System Evaluation

This script trains the complete hybrid ML+DL infant cry recognition system:

  Phase 1: ML Ensemble (SVM + RF + XGBoost + LR)
  Phase 2: CryNetV2 (Dual-stream DST×Mel cross-attention)
  Phase 3: Full Ensemble Evaluation (ML + CryNet + CryNetV2)
  Phase 4: Predicted Output Demo on one real cry sample

Usage:
    python notebooks/12_hybrid_training.py

Requirements:
    pip install torch librosa scikit-learn xgboost imbalanced-learn joblib
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings('ignore')

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves to files without blocking
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)

from src.utils import set_plot_style, CLEANED_DIR, NOISY_DIR, CLASS_LABELS
from src.dl_data import discover_audio_for_dl
from src.dl_model import CryNet
from src.dl_train import FocalLoss, CosineWarmupScheduler, EarlyStopping

from src.hybrid_data import create_hybrid_loaders, hybrid_mixup_data
from src.hybrid_model import CryNetV2
from src.ml_models import MLEnsemble, build_feature_matrix
from src.dst_features import audio_to_dst_spectrogram, DST_CONFIG

set_plot_style()
DEVICE = torch.device('cpu')
print(f"Device: {DEVICE}")

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATA DISCOVERY & SPLIT
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 1: Data Discovery")
print("="*70)

file_paths, labels, label_names = discover_audio_for_dl([CLEANED_DIR, NOISY_DIR])
print(f"Total samples: {len(file_paths)}")
print(f"Classes: {label_names}")

from collections import Counter
dist = Counter(labels)
for i, name in enumerate(label_names):
    print(f"  {name:15s}: {dist[i]:4d}")

# Stratified 80/20 split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"\nTrain: {len(train_paths)}  |  Val: {len(val_paths)}")

# ─────────────────────────────────────────────
# 2. ML ENSEMBLE TRAINING
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 2: ML Ensemble Training (SVM + RF + XGBoost + LR)")
print("="*70)

ML_FEATURES_PATH = os.path.join(MODELS_DIR, 'ml_features_train.npz')
ML_FEATURES_VAL_PATH = os.path.join(MODELS_DIR, 'ml_features_val.npz')

if os.path.exists(ML_FEATURES_PATH):
    print("Loading cached ML features...")
    data = np.load(ML_FEATURES_PATH)
    X_train_ml, y_train_ml = data['X'], data['y']
    data = np.load(ML_FEATURES_VAL_PATH)
    X_val_ml, y_val_ml = data['X'], data['y']
else:
    print("Extracting training features (this may take ~10 minutes)...")
    X_train_ml, y_train_ml = build_feature_matrix(
        train_paths, train_labels, include_dst=True, verbose=True
    )
    np.savez(ML_FEATURES_PATH, X=X_train_ml, y=y_train_ml)

    print("Extracting validation features...")
    X_val_ml, y_val_ml = build_feature_matrix(
        val_paths, val_labels, include_dst=True, verbose=True
    )
    np.savez(ML_FEATURES_VAL_PATH, X=X_val_ml, y=y_val_ml)

print(f"Feature vector shape: {X_train_ml.shape}")

# Fit ML Ensemble
ml_ensemble = MLEnsemble(use_smote=True, include_dst=True, class_names=label_names)
ml_ensemble.fit(X_train_ml, y_train_ml, verbose=True)

ml_results = ml_ensemble.evaluate(X_val_ml, y_val_ml)
print(f"\nML Ensemble Validation:")
print(f"  Accuracy:  {ml_results['accuracy']:.4f}")
print(f"  F1 Macro:  {ml_results['f1_macro']:.4f}")
print(f"  F1 Weighted: {ml_results['f1_weighted']:.4f}")
print("\n" + ml_results['report'])

ml_ensemble.save(os.path.join(MODELS_DIR, 'ml_ensemble.pkl'))

# ─────────────────────────────────────────────
# 3. DUAL-STREAM DATALOADERS
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3: Building Dual-Stream DataLoaders")
print("="*70)

train_loader, val_loader, train_ds, val_ds = create_hybrid_loaders(
    train_paths, train_labels, val_paths, val_labels,
    class_names=label_names, batch_size=32, num_workers=0,
)
class_weights = train_ds.get_class_weights().to(DEVICE)
print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# ─────────────────────────────────────────────
# 4. CryNetV2 TRAINING
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 4: CryNetV2 Training (Dual-Stream DST×Mel)")
print("="*70)

model = CryNetV2(
    n_classes=len(label_names),
    n_mels=128,
    n_dst_freqs=DST_CONFIG['n_freqs'],
    dropout=0.3,
    num_transformer_layers=2,
).to(DEVICE)

total, trainable = model.count_parameters()
print(f"Parameters: {total:,}  (trainable: {trainable:,})")

criterion = FocalLoss(alpha=class_weights, gamma=2.0)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)
)
scheduler = CosineWarmupScheduler(
    optimizer, warmup_epochs=2, total_epochs=5, base_lr=1e-3
)
early_stopper = EarlyStopping(patience=3)

N_EPOCHS = 5
best_val_f1 = 0.0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
           'train_f1': [], 'val_f1': [], 'lr': []}

print(f"\n{'Epoch':>5} | {'TrLoss':>8} | {'VlLoss':>8} | {'TrAcc':>7} | "
      f"{'VlAcc':>7} | {'VlF1':>7} | {'LR':>10}")
print("-" * 70)

for epoch in range(1, N_EPOCHS + 1):
    # ---------- TRAIN ----------
    model.train()
    tr_loss, tr_preds, tr_labels_list = 0.0, [], []

    for mel, dst, lbl in train_loader:
        mel, dst, lbl = mel.to(DEVICE), dst.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()

        if np.random.random() > 0.5:
            mel_m, dst_m, ya, yb, lam = hybrid_mixup_data(mel, dst, lbl, alpha=0.4)
            out = model(mel_m, dst_m)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
            preds = out.argmax(1)
            tr_preds.extend(preds.cpu().numpy())
            tr_labels_list.extend(ya.cpu().numpy())
        else:
            out = model(mel, dst)
            loss = criterion(out, lbl)
            preds = out.argmax(1)
            tr_preds.extend(preds.cpu().numpy())
            tr_labels_list.extend(lbl.cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        tr_loss += loss.item() * mel.size(0)

    tr_loss /= len(train_ds)
    tr_acc = accuracy_score(tr_labels_list, tr_preds)
    tr_f1 = f1_score(tr_labels_list, tr_preds, average='macro', zero_division=0)

    # ---------- VALIDATE ----------
    model.eval()
    vl_loss, vl_preds, vl_labels_list = 0.0, [], []

    with torch.no_grad():
        for mel, dst, lbl in val_loader:
            mel, dst = mel.to(DEVICE), dst.to(DEVICE)
            out = model(mel, dst)
            loss = criterion(out, lbl.to(DEVICE))
            preds = out.argmax(1)
            vl_preds.extend(preds.cpu().numpy())
            vl_labels_list.extend(lbl.numpy())
            vl_loss += loss.item() * mel.size(0)

    vl_loss /= len(val_ds)
    vl_acc = accuracy_score(vl_labels_list, vl_preds)
    vl_f1 = f1_score(vl_labels_list, vl_preds, average='macro', zero_division=0)

    lr = scheduler.step()
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(vl_acc)
    history['train_f1'].append(tr_f1)
    history['val_f1'].append(vl_f1)
    history['lr'].append(lr)

    if epoch % 5 == 0 or epoch <= 3 or epoch == N_EPOCHS:
        print(f"{epoch:5d} | {tr_loss:8.4f} | {vl_loss:8.4f} | "
              f"{tr_acc:7.4f} | {vl_acc:7.4f} | {vl_f1:7.4f} | {lr:10.6f}")

    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_crynetv2.pth'))

    early_stopper(vl_loss, model)
    if early_stopper.should_stop:
        print(f"\nEarly stopping at epoch {epoch}")
        break

early_stopper.load_best_model(model)
print(f"\nBest Val F1: {best_val_f1:.4f}")

# ─────────────────────────────────────────────
# 5. LOAD ORIGINAL CryNet + BUILD ENSEMBLE
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 5: Loading CryNet + Building Full Hybrid Ensemble")
print("="*70)

crynet_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'best_crynet.pth')
if not os.path.exists(crynet_path):
    crynet_path = os.path.join(os.path.dirname(__file__), '..', 'best_crynet.pth')

crynet = CryNet(n_classes=len(label_names), n_mels=128, dropout=0.3)
if os.path.exists(crynet_path):
    ckpt = torch.load(crynet_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        crynet.load_state_dict(ckpt['model_state_dict'])
    else:
        crynet.load_state_dict(ckpt)
    print(f"CryNet loaded from {crynet_path}")
else:
    print(f"Warning: CryNet checkpoint not found at {crynet_path}. Using untrained CryNet.")
crynet = crynet.to(DEVICE).eval()

# ─────────────────────────────────────────────
# 6. FULL ENSEMBLE EVALUATION
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 6: Full Hybrid Ensemble Evaluation")
print("="*70)

model.eval()

all_mel, all_dst, all_lbl = [], [], []
for mel, dst, lbl in val_loader:
    all_mel.append(mel)
    all_dst.append(dst)
    all_lbl.extend(lbl.numpy())

all_mel = torch.cat(all_mel, dim=0).to(DEVICE)
all_dst = torch.cat(all_dst, dim=0).to(DEVICE)
all_lbl = np.array(all_lbl)

with torch.no_grad():
    # CryNet predictions (Mel only)
    crynet_logits = crynet(all_mel)
    crynet_proba = F.softmax(crynet_logits, dim=1).cpu().numpy()

    # CryNetV2 predictions (Mel + DST)
    v2_logits = model(all_mel, all_dst)
    v2_proba = F.softmax(v2_logits, dim=1).cpu().numpy()

# ML predictions
ml_proba = ml_ensemble.predict_proba(X_val_ml)

# Weighted ensemble
ensemble_proba = 0.20 * ml_proba + 0.30 * crynet_proba + 0.50 * v2_proba
ensemble_preds = np.argmax(ensemble_proba, axis=1)
crynetv2_preds = np.argmax(v2_proba, axis=1)
crynet_preds = np.argmax(crynet_proba, axis=1)

def metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"{name:25s}: Acc={acc:.4f}  F1-Macro={f1:.4f}")
    return acc, f1

print("\nModel Comparison:")
print("-" * 60)
m_results = {}
m_results['ML Ensemble'] = metrics(all_lbl, ml_ensemble.predict(X_val_ml), 'ML Ensemble')
m_results['CryNet (DL)'] = metrics(all_lbl, crynet_preds, 'CryNet (DL)')
m_results['CryNetV2 (DL)'] = metrics(all_lbl, crynetv2_preds, 'CryNetV2 (DL)')
m_results['Full Ensemble'] = metrics(all_lbl, ensemble_preds, 'Full Hybrid Ensemble')

print("\nFull Ensemble Classification Report:")
print(classification_report(all_lbl, ensemble_preds, target_names=label_names, zero_division=0))

# ─────────────────────────────────────────────
# 7. TRAINING CURVES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CryNetV2 Training History', fontsize=16, fontweight='bold')

axes[0].plot(history['train_loss'], label='Train', color='#3498db')
axes[0].plot(history['val_loss'], label='Validation', color='#e74c3c')
axes[0].set_title('Loss (Focal)')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_f1'], label='Train', color='#2ecc71')
axes[1].plot(history['val_f1'], label='Validation', color='#e74c3c')
axes[1].axhline(best_val_f1, color='gold', linestyle='--', label=f'Best={best_val_f1:.3f}')
axes[1].set_title('Macro F1-Score')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(history['lr'], color='#9b59b6')
axes[2].set_title('Learning Rate Schedule')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('LR')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'crynetv2_training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 8. CONFUSION MATRIX
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Confusion Matrices: CryNetV2 vs Full Ensemble', fontsize=14, fontweight='bold')

for ax, preds, title in [
    (axes[0], crynetv2_preds, 'CryNetV2 (DL Only)'),
    (axes[1], ensemble_preds, 'Full Hybrid Ensemble (ML+DL)'),
]:
    cm = confusion_matrix(all_lbl, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax,
                vmin=0, vmax=1, linewidths=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 9. MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────
models_list = list(m_results.keys())
accs = [m_results[m][0] for m in models_list]
f1s = [m_results[m][1] for m in models_list]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models_list))
w = 0.35
bars1 = ax.bar(x - w/2, accs, w, label='Accuracy', color='#3498db', alpha=0.85)
bars2 = ax.bar(x + w/2, f1s, w, label='Macro F1', color='#e74c3c', alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=15)
ax.set_ylim(0, 1.0)
ax.set_title('Model Comparison: Accuracy & Macro F1-Score', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 10. PREDICTED OUTPUT DEMO — One Real Cry
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 10: Predicted Output Demo — Single Cry Sample")
print("="*70)

import librosa

# Pick one validation sample
demo_path = val_paths[0]
demo_true_label = val_labels[0]
demo_class_name = label_names[demo_true_label]

print(f"Demo file: {os.path.basename(demo_path)}")
print(f"True class: {demo_class_name}")

# Load and process
y_demo, _ = librosa.load(demo_path, sr=16000, mono=True, duration=3.0)
target_len = 3 * 16000
if len(y_demo) < target_len:
    y_demo = np.pad(y_demo, (0, target_len - len(y_demo)))

from src.dl_data import audio_to_mel_spectrogram, pad_or_truncate_spectrogram, MEL_CONFIG, MAX_TIME_FRAMES
from src.dst_features import stockwell_transform, DST_CONFIG
from src.ml_models import extract_full_feature_vector

# Mel
mel_demo = audio_to_mel_spectrogram(y_demo, sr=16000)
mel_demo = pad_or_truncate_spectrogram(mel_demo, MAX_TIME_FRAMES)
m, s = mel_demo.mean(), mel_demo.std()
if s > 0:
    mel_demo = (mel_demo - m) / s

# DST
dst_demo = stockwell_transform(y_demo, sr=16000,
    n_freqs=DST_CONFIG['n_freqs'], fmin=DST_CONFIG['fmin'],
    fmax=DST_CONFIG['fmax'], hop_length=DST_CONFIG['hop_length'],
    max_frames=MAX_TIME_FRAMES)

# ML features
ml_feat_demo = extract_full_feature_vector(y_demo, include_dst=True)

# Tensors
mel_t = torch.FloatTensor(mel_demo).unsqueeze(0).unsqueeze(0).to(DEVICE)
dst_t = torch.FloatTensor(dst_demo).unsqueeze(0).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    cn_prob = F.softmax(crynet(mel_t), dim=1).cpu().numpy()[0]
    v2_prob = F.softmax(model(mel_t, dst_t), dim=1).cpu().numpy()[0]

ml_prob = ml_ensemble.predict_proba(ml_feat_demo.reshape(1, -1))[0]
ensemble_prob = 0.20 * ml_prob + 0.30 * cn_prob + 0.50 * v2_prob

predicted_class = label_names[np.argmax(ensemble_prob)]
confidence = ensemble_prob.max() * 100

print(f"\n{'─'*50}")
print(f"  PREDICTION: {predicted_class.upper()}")
print(f"  Confidence: {confidence:.1f}%")
print(f"  True Label: {demo_class_name.upper()}")
print(f"  Correct:    {'✓ YES' if predicted_class == demo_class_name else '✗ NO'}")
print(f"{'─'*50}")

# Visualization
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# Waveform
ax_wave = fig.add_subplot(gs[0, :])
t_axis = np.linspace(0, 3, len(y_demo))
ax_wave.plot(t_axis, y_demo, color='#2c3e50', linewidth=0.7, alpha=0.8)
ax_wave.fill_between(t_axis, y_demo, alpha=0.15, color='#3498db')
ax_wave.set_title(f'Waveform — True: {demo_class_name} | Predicted: {predicted_class}',
                  fontsize=12, fontweight='bold')
ax_wave.set_xlabel('Time (s)')
ax_wave.set_ylabel('Amplitude')
ax_wave.grid(True, alpha=0.2)

# Mel-Spectrogram
ax_mel = fig.add_subplot(gs[1, 0])
img1 = ax_mel.imshow(mel_demo, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(img1, ax=ax_mel, fraction=0.046)
ax_mel.set_title('Mel-Spectrogram', fontsize=10)
ax_mel.set_xlabel('Time Frames')
ax_mel.set_ylabel('Mel Bins')

# DST-Spectrogram
ax_dst = fig.add_subplot(gs[1, 1])
img2 = ax_dst.imshow(dst_demo, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(img2, ax=ax_dst, fraction=0.046)
ax_dst.set_title('DST Spectrogram (Stockwell)', fontsize=10)
ax_dst.set_xlabel('Time Frames')
ax_dst.set_ylabel('DST Freq Bins (log)')

# Class Probabilities — Full Ensemble
ax_proba = fig.add_subplot(gs[1, 2])
colors = ['#2ecc71' if i == np.argmax(ensemble_prob) else '#3498db'
          for i in range(len(label_names))]
bars = ax_proba.barh(label_names, ensemble_prob * 100, color=colors, alpha=0.85)
ax_proba.axvline(50, color='red', linestyle='--', alpha=0.5)
ax_proba.set_title('Ensemble Confidence (%)', fontsize=10)
ax_proba.set_xlabel('Probability (%)')
ax_proba.set_xlim(0, 100)
for bar, prob in zip(bars, ensemble_prob):
    ax_proba.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                  f'{prob*100:.1f}%', va='center', fontsize=8)

# Per-model comparison
ax_cmp = fig.add_subplot(gs[2, :])
x_pos = np.arange(len(label_names))
w = 0.25
ax_cmp.bar(x_pos - w, ml_prob * 100,   w, label='ML Ensemble', color='#9b59b6', alpha=0.8)
ax_cmp.bar(x_pos,     cn_prob * 100,   w, label='CryNet',       color='#e74c3c', alpha=0.8)
ax_cmp.bar(x_pos + w, v2_prob * 100,   w, label='CryNetV2',     color='#2ecc71', alpha=0.8)
ax_cmp.axvline(demo_true_label, color='black', linestyle=':', linewidth=2, label='True Class')
ax_cmp.set_xticks(x_pos)
ax_cmp.set_xticklabels(label_names, rotation=30, ha='right')
ax_cmp.set_title('Per-Model Probability Breakdown', fontsize=11, fontweight='bold')
ax_cmp.set_ylabel('Probability (%)')
ax_cmp.set_ylim(0, 110)
ax_cmp.legend(loc='upper right')
ax_cmp.grid(True, axis='y', alpha=0.2)

plt.suptitle('CryNetV2 Hybrid Inference — Predicted Output Demo',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(os.path.join(MODELS_DIR, 'prediction_demo.png'),
            dpi=150, bbox_inches='tight')
plt.close()

print(f"\nAll outputs saved to: {MODELS_DIR}")
print("Training complete!")
