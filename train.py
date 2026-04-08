"""
train.py (UPDATED with Mixup)
========
Trains a PyTorch Neural Network on Perch embeddings + Mixup augmentation.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path

from configs.config import BASE_DIR_MODELS, N_CLASSES
from data.dataset import load_label2idx
from augment import mixup          # ← NEW: import your mixup function

EMB_DIR = Path("/kaggle/input/datasets/haphngngcgia/birdclef2026-embeddings")
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
    """Convert string label array to binary matrix of shape (N, 234)."""
    n = len(y_labels_str)
    n_classes = len(label2idx)
    matrix = np.zeros((n, n_classes), dtype=np.float32)
    for i, label in enumerate(y_labels_str):
        if str(label) in label2idx:
            matrix[i, label2idx[str(label)]] = 1.0
    return matrix


class PerchDataset(Dataset):
    """Simple PyTorch dataset for pre-extracted embeddings."""
    def __init__(self, X, y_bin):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_bin, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BirdMLP(nn.Module):
    """Multi-Layer Perceptron for 1D Perch embeddings."""
    def __init__(self, input_dim=1280, num_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_pytorch(mixup_prob: float = 0.5, mixup_alpha: float = 0.4):
    print("Loading embeddings...")
    try:
        X = np.load(EMB_DIR / "X_embeddings.npy")
        y_labels_str = np.load(EMB_DIR / "y_labels.npy")
    except FileNotFoundError:
        print("Embeddings not found. Run extract_feature.py first.")
        return

    label2idx = load_label2idx()
    y_bin = labels_to_binary_matrix(y_labels_str, label2idx)

    y_primary_idx = np.array([label2idx.get(str(l), 0) for l in y_labels_str])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_primary_idx)):
        print(f"\n{'='*50}\nFOLD {fold}\n{'='*50}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train_bin, y_val_bin = y_bin[train_idx], y_bin[val_idx]
        y_train_prim = y_primary_idx[train_idx]

        train_dataset = PerchDataset(X_train, y_train_bin)
        val_dataset   = PerchDataset(X_val, y_val_bin)

        # WeightedRandomSampler (already in your original code)
        class_counts = np.bincount(y_train_prim, minlength=N_CLASSES)
        class_weights = np.zeros(N_CLASSES, dtype=np.float32)
        valid_classes = class_counts > 0
        class_weights[valid_classes] = 1.0 / class_counts[valid_classes]
        sample_weights = class_weights[y_train_prim]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
        val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False)

        model = BirdMLP().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        best_cmap = 0.0
        epochs = 15

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # ====================== MIXUP ======================
                if np.random.rand() < mixup_prob:                    # ← 50% chance
                    # Randomly pick another sample from the same batch
                    idx2 = torch.randperm(inputs.size(0)).to(device)
                    inputs2, targets2 = inputs[idx2], targets[idx2]

                    # Apply mixup (uses your augment.py function)
                    inputs, targets = mixup(inputs, targets, inputs2, targets2, alpha=mixup_alpha)
                # ===================================================

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # === Evaluation (unchanged) ===
            model.eval()
            val_preds = []
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(device)
                    outputs = torch.sigmoid(model(inputs))
                    val_preds.append(outputs.cpu().numpy())

            val_preds = np.vstack(val_preds)
            cmap = padded_cmap(y_val_bin, val_preds)

            if cmap > best_cmap:
                best_cmap = cmap
                torch.save(model.state_dict(), SAVE_DIR / f"pt_fold{fold}.pth")

            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val cMAP: {cmap:.4f}")

        print(f"Fold {fold} Best OOF cMAP: {best_cmap:.4f}")
        fold_scores.append(best_cmap)

    print(f"\nMean OOF cMAP: {np.mean(fold_scores):.4f}")

    with open(SAVE_DIR / "label2idx.json", "w") as f:
        json.dump(label2idx, f, indent=2)

    return fold_scores


if __name__ == "__main__":
    train_pytorch(mixup_prob=0.5, mixup_alpha=0.4)   # ← you can change these values here