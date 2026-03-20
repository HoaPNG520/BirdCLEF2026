import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from configs.config import (
    TRAIN_CSV, TAXONOMY, N_CLASSES, BATCH_SIZE, NUM_WORKERS
)
from data.dataset import BirdDataset
from data.folds import make_folds, get_fold
from data.augment import get_spec_augment


# ── helpers ───────────────────────────────────────────────

def get_label2idx(tax):
    """Build label→index mapping from taxonomy. Covers all 234 species."""
    labels = sorted(tax['primary_label'].astype(str).unique())
    return {lbl: i for i, lbl in enumerate(labels)}


def padded_cmap(labels, preds, padding=5):
    """
    Padded competition metric — mean average precision per class.
    Padding adds dummy negatives to avoid inflated scores on rare classes.
    Uses sklearn's average_precision_score internally.
    """
    from sklearn.metrics import average_precision_score
    scores = []
    for i in range(labels.shape[1]):
        y_true = np.concatenate([labels[:, i], np.zeros(padding)])
        y_pred = np.concatenate([preds[:, i],  np.zeros(padding)])
        if y_true.sum() > 0:
            scores.append(average_precision_score(y_true, y_pred))
    return float(np.mean(scores)) if scores else 0.0


# ── train / val loops ─────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, augment=None):
    model.train()
    total_loss = 0

    for mel, label in tqdm(loader, desc="train", leave=False):
        mel, label = mel.to(device), label.to(device)

        if augment is not None:
            mel = torch.stack([augment(m) for m in mel])

        optimizer.zero_grad()
        logits = model(mel)
        loss   = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for mel, label in tqdm(loader, desc="val  ", leave=False):
        mel, label = mel.to(device), label.to(device)
        logits     = model(mel)
        loss       = criterion(logits, label)
        total_loss += loss.item()

        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(label.cpu().numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return total_loss / len(loader), preds, labels


# ── main entry point ──────────────────────────────────────

def train_fold(fold=0, epochs=20, batch_size=32, lr=1e-3,
               save_dir='/kaggle/working/', max_batches=None, device=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  |  Fold: {fold}  |  Epochs: {epochs}")

    # ── data ──────────────────────────────────────────────
    df  = pd.read_csv(TRAIN_CSV)
    tax = pd.read_csv(TAXONOMY)

    # drop clips too short to be useful
    df = df[df['filename'].notna()].copy()

    df        = make_folds(df)
    label2idx = get_label2idx(tax)
    train_df, val_df = get_fold(df, fold)

    augment  = get_spec_augment()
    train_ds = BirdDataset(train_df, label2idx, augment=augment)
    val_ds   = BirdDataset(val_df,   label2idx, augment=None)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── model — EfficientNet-B0 via timm ──────────────────
    import timm
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=True,
        in_chans=1,           # single-channel mel spectrogram
        num_classes=N_CLASSES
    )
    model = model.to(device)

    # ── optimizer + scheduler + loss ──────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.BCEWithLogitsLoss()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_cmap = 0.0

    # ── training loop ─────────────────────────────────────
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, augment
        )
        val_loss, preds, labels = validate(
            model, val_loader, criterion, device
        )
        cmap = padded_cmap(labels, preds)
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"cMAP: {cmap:.4f}")

        if cmap > best_cmap:
            best_cmap = cmap
            torch.save(
                model.state_dict(),
                save_dir / f"best_fold{fold}.pth"
            )
            print(f"  ↳ saved best model (cMAP={cmap:.4f})")

    print(f"\nFold {fold} done — best cMAP: {best_cmap:.4f}")
    return best_cmap


# ── smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke',  action='store_true')
    parser.add_argument('--fold',   type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    if args.smoke:
        print("Running smoke test on CPU...")
        train_fold(fold=0, epochs=1, batch_size=4,
                   max_batches=2, device='cpu')
        print("Smoke test passed ✓")
    else:
        train_fold(fold=args.fold, epochs=args.epochs)