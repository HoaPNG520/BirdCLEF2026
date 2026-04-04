"""
train.py
========
Trains XGBoost on Perch embeddings extracted by extract_feature.py.

KEY FIX — 206 vs 234 classes:
  The old code used LabelEncoder.fit_transform(y_raw_labels) which only
  sees the 206 species present in training data. Zero-shot species get
  no index and the model outputs 206 dimensions instead of 234.

  The fix: use label2idx from EDA artifacts which covers all 234 taxonomy
  species. Zero-shot species just never appear in training but they exist
  in the mapping so the model can output a 0.0 probability for them.

GPU:
  XGBoost uses device='cuda' automatically.
  TensorFlow (Perch) uses GPU automatically if configured.
"""

import os
import json
import pickle
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, average_precision_score

from configs.config import BASE_DIR_ARTIFACT, BASE_DIR_MODELS, N_CLASSES
from data.dataset import load_label2idx

EMB_DIR  = Path("/kaggle/working/birdclef-embeddings")
SAVE_DIR = BASE_DIR_MODELS
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def padded_cmap(y_true_bin, y_pred_prob, padding=5):
    """BirdCLEF competition metric — mean AP with padding."""
    scores = []
    for i in range(y_true_bin.shape[1]):
        yt = np.concatenate([y_true_bin[:, i], np.zeros(padding)])
        yp = np.concatenate([y_pred_prob[:, i], np.zeros(padding)])
        if yt.sum() > 0:
            scores.append(average_precision_score(yt, yp))
    return float(np.mean(scores)) if scores else 0.0


def labels_to_binary_matrix(y_labels_str, label2idx):
    """
    Convert string label array to binary matrix of shape (N, 234).
    Each row has 1.0 at the species index, 0.0 everywhere else.
    Zero-shot species columns are always 0.0 in training.
    """
    n = len(y_labels_str)
    n_classes = len(label2idx)
    matrix = np.zeros((n, n_classes), dtype=np.float32)
    for i, label in enumerate(y_labels_str):
        if str(label) in label2idx:
            matrix[i, label2idx[str(label)]] = 1.0
    return matrix


def train_xgboost():
    # ── load embeddings ───────────────────────────────────────
    print("Loading embeddings...")
    try:
        X = np.load(EMB_DIR / "X_embeddings.npy")
        y_labels = np.load(EMB_DIR / "y_labels.npy")
    except FileNotFoundError:
        print("Embeddings not found. Run extract_feature.py first.")
        return

    print(f"X shape : {X.shape}")   # (N, 1280)
    print(f"y shape : {y_labels.shape}")  # (N,)

    # ── load label2idx from EDA artifacts — covers all 234 species ──
    # CRITICAL: do NOT use LabelEncoder here.
    # LabelEncoder only sees 206 training species → wrong output size.
    # label2idx covers all 234 taxonomy species → correct output size.
    label2idx = load_label2idx()
    idx2label = {v: k for k, v in label2idx.items()}

    print(f"label2idx covers : {len(label2idx)} species (should be 234)")

    # convert string labels to integer indices using label2idx
    y_encoded = np.array([
        label2idx.get(str(l), 0) for l in y_labels
    ], dtype=np.int32)

    unique_classes = np.unique(y_encoded)
    print(f"Unique classes in training: {len(unique_classes)} (of 234 taxonomy)")

    # ── stratified K-fold — same seed as data team ────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold}")
        print(f"{'='*50}")

        X_train, X_val   = X[train_idx], X[val_idx]
        y_train, y_val   = y_encoded[train_idx], y_encoded[val_idx]
        y_val_str        = y_labels[val_idx]

        # ── XGBoost ───────────────────────────────────────────
        # num_class MUST be 234 — matches submission CSV columns
        # Even though training only has 206 classes, XGBoost needs
        # to output 234 probabilities for the full taxonomy
        clf = xgb.XGBClassifier(
            objective        = 'multi:softprob',
            num_class        = N_CLASSES,       # 234 — not len(unique_classes)
            n_estimators     = 300,
            max_depth        = 6,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            tree_method      = 'hist',
            device           = 'cuda',          # GPU
            random_state     = 42,
            eval_metric      = 'mlogloss',
        )

        clf.fit(
            X_train, y_train,
            eval_set    = [(X_val, y_val)],
            verbose     = 50,
        )

        # ── evaluate with padded-cMAP ─────────────────────────
        val_probs     = clf.predict_proba(X_val)  # (N_val, 234)
        y_val_bin     = labels_to_binary_matrix(y_val_str, label2idx)

        cmap  = padded_cmap(y_val_bin, val_probs)
        ll    = log_loss(y_val, val_probs, labels=list(range(N_CLASSES)))

        print(f"Fold {fold} — cMAP: {cmap:.4f}  log_loss: {ll:.4f}")
        fold_scores.append(cmap)

        # save each fold model
        fold_path = SAVE_DIR / f"xgb_fold{fold}.json"
        clf.save_model(fold_path)
        print(f"Saved: {fold_path}")

    print(f"\nMean OOF cMAP: {np.mean(fold_scores):.4f}")
    print(f"Per-fold      : {[round(s,4) for s in fold_scores]}")

    # ── save label2idx alongside model ────────────────────────
    # inference needs this to build the submission CSV correctly
    with open(SAVE_DIR / "label2idx.json", "w") as f:
        json.dump(label2idx, f, indent=2)
    print(f"\nSaved label2idx.json alongside models")
    print(f"All files in {SAVE_DIR}:")
    for f in sorted(SAVE_DIR.iterdir()):
        print(f"  {f.name}  {f.stat().st_size/1024:.1f} KB")

    return fold_scores


if __name__ == "__main__":
    train_xgboost()