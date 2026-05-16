"""
train.py
========
Trains EfficientNet-B3 on mel spectrograms for BirdCLEF+ 2026.
Supports 5-fold cross-validation, AMP, gradient accumulation, early stopping.

Usage:
    python train.py              # trains all 5 folds
    python train.py --fold 0     # trains one specific fold
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from pathlib import Path
from tqdm import tqdm

from configs.config import (
    BASE_DIR_MODELS,
    N_CLASSES,
    BATCH_SIZE,
    NUM_WORKERS,
    ACCUMULATION_STEPS,
)
from data.dataset import load_df_clean, load_label2idx, BirdDataset
from data.folds import make_folds, get_fold
from data.augment import get_spec_augment, mixup
from models.efficientnet import EfficientNetClassifier

# ── Metric ─────────────────────────────────────────────────────────────────────


def padded_cmap(y_true, y_pred, padding=5):
    """BirdCLEF padded mean AP. Binarizes soft labels for sklearn compatibility."""
    scores = []
    for i in range(y_true.shape[1]):
        yt = (y_true[:, i] > 0).astype(int)
        yp = y_pred[:, i]
        yt = np.concatenate([yt, np.zeros(padding)])
        yp = np.concatenate([yp, np.zeros(padding)])
        if yt.sum() > 0:
            scores.append(average_precision_score(yt, yp))
    return float(np.mean(scores)) if scores else 0.0


# ── Training ───────────────────────────────────────────────────────────────────


def train_fold(
    fold, epochs=20, lr=2e-3, mixup_prob=0.5, pos_weight=100, save_dir=BASE_DIR_MODELS
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}\nFOLD {fold} | Device: {device}\n{'='*50}")

    # ── Data ──────────────────────────────────────────────────────────────────
    label2idx = load_label2idx()
    df = load_df_clean()
    if "fold" not in df.columns:
        df = make_folds(df, n_folds=5)

    train_df, val_df = get_fold(df, fold)
    from torch.utils.data import WeightedRandomSampler

    y_train_prim = np.array(
        [label2idx.get(str(l), 0) for l in train_df["primary_label"]]
    )
    class_counts = np.bincount(y_train_prim, minlength=N_CLASSES)
    class_weights = np.zeros(N_CLASSES, dtype=np.float32)
    class_weights[class_counts > 0] = 1.0 / class_counts[class_counts > 0]
    sample_weights = class_weights[y_train_prim]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_dataset = BirdDataset(
        train_df, label2idx, augment=get_spec_augment(), mode="train"
    )
    val_dataset = BirdDataset(val_df, label2idx, augment=None, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EfficientNetClassifier(n_classes=N_CLASSES).to(device)
    # ~1-2 positives out of 234 classes → ratio ~117:1
    pos_weight = torch.ones(N_CLASSES, device=device) * pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler()

    best_cmap = 0.0
    patience_counter = 0  # ← fixed: initialize before loop
    save_path = save_dir / f"effnet_fold{fold}_best.pth"  # ← fixed: use actual fold

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for i, (mels, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        ):
            mels, labels = mels.to(device), labels.to(device)

            # Mixup
            if np.random.rand() < mixup_prob:
                idx = torch.randperm(mels.size(0)).to(device)
                mels, labels = mixup(mels, labels, mels[idx], labels[idx])

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(mels)
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUMULATION_STEPS

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for mels, labels in val_loader:
                mels = mels.to(device)
                preds = torch.sigmoid(model(mels)).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(labels.numpy())

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        cmap = padded_cmap(val_targets, val_preds)

        print(
            f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Val cMAP: {cmap:.4f}"
        )

        if cmap > best_cmap:
            best_cmap = cmap
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (cMAP={best_cmap:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print("Early stopping — saving GPU quota.")
                break

    print(f"Fold {fold} best cMAP: {best_cmap:.4f}")
    return best_cmap


# ── Entry point ────────────────────────────────────────────────────────────────


def train_all_folds(epochs=20):
    label2idx = load_label2idx()

    fold_scores = []
    for fold in range(5):
        score = train_fold(fold=fold, epochs=epochs)
        fold_scores.append(score)

    print(f"\nMean OOF cMAP: {np.mean(fold_scores):.4f}")
    print(f"Per-fold:      {[f'{s:.4f}' for s in fold_scores]}")

    # Save label2idx alongside models
    with open(Path(BASE_DIR_MODELS) / "label2idx.json", "w") as f:
        json.dump(label2idx, f, indent=2)

    return fold_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", type=int, default=-1, help="Single fold to train (-1 = all)"
    )
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    if args.fold == -1:
        train_all_folds(epochs=args.epochs)
    else:
        train_fold(fold=args.fold, epochs=args.epochs)
