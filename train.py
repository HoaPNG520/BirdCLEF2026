import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from pathlib import Path
from tqdm import tqdm

from configs.config import BASE_DIR_MODELS, N_CLASSES, BATCH_SIZE, NUM_WORKERS
from data.dataset import load_df_clean, load_label2idx, BirdDataset
from data.folds import make_folds, get_fold
from data.augment import get_spec_augment, mixup
from models.efficientnet import EfficientNetClassifier

def padded_cmap(y_true_bin, y_pred_prob, padding=5):
    """
    Padded competition metric.
    Binarizes soft labels (0.5 -> 1) for Scikit-learn compatibility.
    """
    from sklearn.metrics import average_precision_score
    import numpy as np

    scores = []
    for i in range(y_true_bin.shape[1]):
        # 1. Extract the column for this species
        yt = y_true_bin[:, i]
        yp = y_pred_prob[:, i]

        # 2. Binarize ground truth: Anything > 0 (0.5 or 1.0) becomes 1
        yt_binary = (yt > 0).astype(int)

        # 3. Apply the competition padding
        yt_padded = np.concatenate([yt_binary, np.zeros(padding)])
        yp_padded = np.concatenate([yp, np.zeros(padding)])

        # 4. Calculate AP if there is at least one positive instance
        if yt_padded.sum() > 0:
            scores.append(average_precision_score(yt_padded, yp_padded))
            
    return float(np.mean(scores)) if scores else 0.0

def train_fold(fold, epochs=20, lr=1e-3, mixup_prob=0.5, save_dir=BASE_DIR_MODELS):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | Fold {fold}")

    # 1. Load Data
    label2idx = load_label2idx()
    df = load_df_clean()
    if 'fold' not in df.columns:
        df = make_folds(df, n_folds=5)
    
    train_df, val_df = get_fold(df, fold)

    # 2. Datasets & Loaders
    train_dataset = BirdDataset(train_df, label2idx, augment=get_spec_augment(), mode="train")
    val_dataset   = BirdDataset(val_df, label2idx, augment=None, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 3. Model & Optimiser
    model = EfficientNetClassifier(n_classes=N_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_cmap = 0.0

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for mels, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            mels, labels = mels.to(device), labels.to(device)

            if np.random.rand() < mixup_prob:
                indices = torch.randperm(mels.size(0)).to(device)
                mels, labels = mixup(mels, labels, mels[indices], labels[indices])

            optimizer.zero_grad()
            outputs = model(mels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # 5. Validation Loop
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for mels, labels in val_loader:
                mels = mels.to(device)
                outputs = torch.sigmoid(model(mels))
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(labels.numpy())

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        cmap = padded_cmap(val_targets, val_preds)

        print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Val cMAP: {cmap:.4f}")

        if cmap > best_cmap:
            best_cmap = cmap
            torch.save(model.state_dict(), save_dir / f"effnet_fold{fold}.pth")
            print(f"--> Saved new best model (cMAP: {best_cmap:.4f})")

    with open(save_dir / "label2idx.json", "w") as f:
        json.dump(label2idx, f, indent=2)

if __name__ == "__main__":
    train_fold(fold=0)