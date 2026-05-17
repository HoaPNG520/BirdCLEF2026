"""
infer.py
========
Full inference pipeline for BirdCLEF+ 2026.
Ensembles 5-fold EfficientNet models + existing Perch+MLP models.
Writes submission.csv compatible with competition format.

Usage (in Kaggle notebook):
    from infer import run_inference
    run_inference()
"""

import json
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from configs.config import (
    TEST_DIR,
    BASE_DIR_MODELS,
    SAMPLE_RATE,
    DURATION,
    N_CLASSES,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
)
from models.efficientnet import EfficientNetClassifier

CHUNK_LEN = SAMPLE_RATE * DURATION


# ── Model loading ──────────────────────────────────────────────────────────────


def load_effnet_models(model_dir, device, n_folds=5):
    """Load all 5 EfficientNet fold models."""
    model_dir = Path(model_dir)
    models = []
    for fold in range(n_folds):
        path = model_dir / f"effnet_fold{fold}_best.pth"
        if path.exists():
            m = EfficientNetClassifier(n_classes=N_CLASSES, pretrained=False).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
            print(f"  Loaded: {path.name}")
        else:
            print(f"  Warning: Missing {path.name}")
    return models


def load_perch_models(perch_model_dir, mlp_model_dir, device, n_folds=5):
    """Load Perch TF model + 5-fold MLP heads."""
    import tensorflow as tf

    perch = tf.saved_model.load(str(perch_model_dir))
    print("  Perch loaded ✓")

    class BirdMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1280, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, N_CLASSES),
            )

        def forward(self, x):
            return self.net(x)

    mlp_models = []
    mlp_model_dir = Path(mlp_model_dir)
    for fold in range(n_folds):
        path = mlp_model_dir / f"pt_fold{fold}.pth"
        if path.exists():
            m = BirdMLP().to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            mlp_models.append(m)
            print(f"  Loaded: {path.name}")
        else:
            print(f"  Warning: Missing {path.name}")

    return perch, mlp_models


# ── Audio processing ───────────────────────────────────────────────────────────


def audio_to_mel_tensor(chunk):
    """Convert a 5s numpy audio chunk to (1, N_MELS, T) tensor."""
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=16000,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(mel, dtype=torch.float32).unsqueeze(0)


def iter_chunks(y):
    """Yield (chunk, end_sec) for non-overlapping 5s windows."""
    n_chunks = max(1, len(y) // CHUNK_LEN)
    for i in range(n_chunks):
        chunk = y[i * CHUNK_LEN : (i + 1) * CHUNK_LEN]
        if len(chunk) < CHUNK_LEN:
            chunk = np.pad(chunk, (0, CHUNK_LEN - len(chunk)))
        yield chunk, (i + 1) * DURATION


# ── Inference ──────────────────────────────────────────────────────────────────


def run_inference(
    effnet_model_dir=BASE_DIR_MODELS,
    perch_model_dir=None,  # e.g. Path('/kaggle/input/birdclef-perch-model')
    mlp_model_dir=None,  # e.g. Path('/kaggle/input/birdclef-models')
    thresholds_path=None,  # e.g. Path('/kaggle/input/birdclef-analysis-artifacts/thresholds.json')
    output_path="/kaggle/working/submission.csv",
    effnet_weight=0.5,  # ensemble weight for EfficientNet
    perch_weight=0.5,  # ensemble weight for Perch+MLP
):
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    # ── Load label mapping ─────────────────────────────────────────────────────
    label2idx_path = Path(effnet_model_dir) / "label2idx.json"
    with open(label2idx_path) as f:
        label2idx = json.load(f)
    idx2label = {v: k for k, v in label2idx.items()}

    # ── Load thresholds (optional) ─────────────────────────────────────────────
    if thresholds_path and Path(thresholds_path).exists():
        with open(thresholds_path) as f:
            thresholds = json.load(f)
        print(f"Loaded {len(thresholds)} per-class thresholds")
    else:
        thresholds = {label: 0.5 for label in label2idx}
        print("Using default threshold 0.5 for all classes")

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading EfficientNet models...")
    effnet_models = load_effnet_models(effnet_model_dir, device)

    use_perch = perch_model_dir is not None and mlp_model_dir is not None
    if use_perch:
        import tensorflow as tf

        print("\nLoading Perch + MLP models...")
        perch, mlp_models = load_perch_models(perch_model_dir, mlp_model_dir, device)
    else:
        print("\nRunning EfficientNet only (no Perch paths provided)")
        effnet_weight = 1.0

    # ── Load test files ────────────────────────────────────────────────────────
    sample_sub = pd.read_csv(
        Path("/kaggle/input/competitions/birdclef-2026/sample_submission.csv")
    )
    test_files = sorted(Path(TEST_DIR).glob("*.ogg"))
    print(f"\nProcessing {len(test_files)} test soundscapes...")

    if len(test_files) == 0:
        print("No test files found — writing dummy submission")
        dummy = sample_sub.copy()
        for col in label2idx:
            if col in dummy.columns:
                dummy[col] = 0.0
        dummy.to_csv(output_path, index=False)
        print(f"Dummy submission saved → {output_path}")
        return

    # ── Main inference loop ────────────────────────────────────────────────────
    results = {}

    for audio_path in tqdm(test_files, desc="Soundscapes"):
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        stem = audio_path.stem

        for chunk, end_sec in iter_chunks(y):
            row_id = f"{stem}_{end_sec}"
            probs = np.zeros(N_CLASSES, dtype=np.float32)

            # EfficientNet ensemble
            if effnet_models:
                mel_t = audio_to_mel_tensor(chunk).unsqueeze(0).to(device)
                with torch.no_grad():
                    eff_probs = sum(
                        torch.sigmoid(m(mel_t)).cpu().numpy()[0] for m in effnet_models
                    ) / len(effnet_models)
                probs += effnet_weight * eff_probs

            # Perch + MLP ensemble
            if use_perch and mlp_models:
                import tensorflow as tf

                chunk_tf = tf.expand_dims(chunk.astype(np.float32), 0)
                _, emb = perch.infer_tf(chunk_tf)
                X = (
                    torch.tensor(emb.numpy()[0], dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.no_grad():
                    mlp_probs = sum(
                        torch.sigmoid(m(X)).cpu().numpy()[0] for m in mlp_models
                    ) / len(mlp_models)
                probs += perch_weight * mlp_probs

            # Apply per-class thresholds
            row = {"row_id": row_id}
            for label, idx in label2idx.items():
                p = float(probs[idx])
                row[label] = p if p >= thresholds.get(label, 0.5) else 0.0

            results[row_id] = row

    # ── Write submission ───────────────────────────────────────────────────────
    sub_df = pd.DataFrame.from_dict(results, orient="index")
    sub_df = sub_df.reindex(columns=sample_sub.columns, fill_value=0.0)
    sub_df.to_csv(output_path, index=False)

    print(f"\n✅ Submission saved → {output_path}")
    print(f"   Shape: {sub_df.shape}")


if __name__ == "__main__":
    run_inference()
