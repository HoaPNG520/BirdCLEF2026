"""
infer.py
========
Runs inference on test soundscapes using Perch + PyTorch MLP.
"""

import json
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from tqdm import tqdm

from configs.config import (
    BASE_DIR_COMPETITION,
    TEST_DIR,
    SAMPLE_RATE,
    DURATION,
    N_CLASSES,
    PERCH_URL,
)

AUDIO_LENGTH = int(SAMPLE_RATE * DURATION)


class BirdMLP(nn.Module):
    """Identical architecture to the one in train.py."""

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


def setup_gpus():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch using: {device}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Prevent TF from hogging all VRAM, allowing PyTorch to coexist
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TF GPU: {len(gpus)} device(s)")
    return device


def load_models(model_dir, device, n_folds=5):
    """Load all fold PyTorch models for ensembling."""
    model_dir = Path(model_dir)
    models = []
    for fold in range(n_folds):
        path = model_dir / f"pt_fold{fold}.pth"
        if path.exists():
            m = BirdMLP().to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
            print(f"Loaded fold {fold}: {path.name}")
    return models


def chunk_audio(y, chunk_len):
    chunks, end_times = [], []
    n_chunks = max(1, len(y) // chunk_len)
    for i in range(n_chunks):
        chunk = y[i * chunk_len : (i + 1) * chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk.astype(np.float32))
        end_times.append((i + 1) * 5)
    return chunks, end_times


def run_inference(
    model_dir="/kaggle/input/birdclef-models",
    output_path="/kaggle/working/submission.csv",
    n_folds=5,
):
    device = setup_gpus()

    with open(Path(model_dir) / "label2idx.json") as f:
        label2idx = json.load(f)

    models = load_models(model_dir, device, n_folds)
    if not models:
        raise RuntimeError("No .pth models found. Check model_dir path.")

    print("Loading Perch model...")
    perch = hub.load(PERCH_URL)

    sample_sub = pd.read_csv(BASE_DIR_COMPETITION / "sample_submission.csv")
    test_dir = Path(TEST_DIR)
    test_files = sorted(test_dir.glob("*.ogg"))

    results = {}
    for audio_path in tqdm(test_files, desc="inference"):
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        stem = audio_path.stem
        chunks, end_times = chunk_audio(y, AUDIO_LENGTH)

        for chunk, end_time in zip(chunks, end_times):
            y_tf = tf.expand_dims(chunk, axis=0)
            _, emb = perch.infer_tf(y_tf)
            X_chunk = emb.numpy()

            # PyTorch Inference
            X_tensor = torch.tensor(X_chunk, dtype=torch.float32).to(device)

            fold_probs = np.zeros((1, N_CLASSES), dtype=np.float32)
            with torch.no_grad():
                for model in models:
                    logits = model(X_tensor)
                    # Apply sigmoid to convert raw logits to probabilities 0.0 - 1.0
                    probs = torch.sigmoid(logits).cpu().numpy()
                    fold_probs += probs

            fold_probs /= len(models)

            row_id = f"{stem}_{end_time}"
            results[row_id] = fold_probs[0]

    rows = []
    for row_id, probs in results.items():
        row = {"row_id": row_id}
        for label, idx in label2idx.items():
            row[label] = round(float(probs[idx]), 6)
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_df = pred_df.reindex(columns=sample_sub.columns, fill_value=0.0)
    pred_df.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}")


if __name__ == "__main__":
    run_inference()
