"""
infer.py
========
Runs inference on test soundscapes and writes submission.csv.

Pipeline:
  test .ogg → 5s chunks → Perch embeddings → XGBoost → probabilities → CSV

Run with internet OFF for submission.
All inputs must be Kaggle datasets — no GitHub cloning.
"""

import json
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import xgboost as xgb
from pathlib import Path
from tqdm import tqdm

from configs.config import (
    BASE_DIR_COMPETITION, TEST_DIR,
    SAMPLE_RATE, DURATION, N_CLASSES, PERCH_URL
)

AUDIO_LENGTH = int(SAMPLE_RATE * DURATION)


def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TF GPU: {len(gpus)} device(s)")
    else:
        print("TF: no GPU found, using CPU")


def load_models(model_dir, n_folds=5):
    """Load all fold XGBoost models for ensembling."""
    model_dir = Path(model_dir)
    models = []
    for fold in range(n_folds):
        path = model_dir / f"xgb_fold{fold}.json"
        if path.exists():
            m = xgb.XGBClassifier()
            m.load_model(path)
            models.append(m)
            print(f"Loaded fold {fold}: {path.name}")
        else:
            print(f"Warning: fold {fold} not found at {path}")
    print(f"Loaded {len(models)} models")
    return models


def chunk_audio(y, chunk_len):
    """Split audio into fixed-length chunks. Pad the last one."""
    chunks, end_times = [], []
    n_chunks = max(1, len(y) // chunk_len)
    for i in range(n_chunks):
        chunk = y[i*chunk_len:(i+1)*chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk.astype(np.float32))
        end_times.append((i + 1) * 5)
    return chunks, end_times


def run_inference(
    model_dir     = "/kaggle/input/birdclef-models",
    artifact_dir  = "/kaggle/input/birdclef-eda-artifacts",
    output_path   = "/kaggle/working/submission.csv",
    n_folds       = 5,
):
    setup_gpu()

    # ── load label mapping ────────────────────────────────────
    with open(Path(model_dir) / "label2idx.json") as f:
        label2idx = json.load(f)
    print(f"label2idx: {len(label2idx)} species")
    assert len(label2idx) == N_CLASSES, \
        f"Expected {N_CLASSES} species, got {len(label2idx)}"

    # ── load models ───────────────────────────────────────────
    models = load_models(model_dir, n_folds)
    if not models:
        raise RuntimeError("No models found. Check model_dir path.")

    # ── load Perch for embedding extraction ───────────────────
    print("Loading Perch model...")
    perch = hub.load(PERCH_URL)
    print("Perch loaded ✓")

    # ── load submission template ──────────────────────────────
    sample_sub = pd.read_csv(BASE_DIR_COMPETITION / "sample_submission.csv")
    test_dir   = Path(TEST_DIR)
    test_files = sorted(test_dir.glob("*.ogg"))
    chunk_len  = AUDIO_LENGTH

    print(f"Test soundscapes: {len(test_files)}")

    results = {}

    for audio_path in tqdm(test_files, desc="inference"):
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        stem  = audio_path.stem
        chunks, end_times = chunk_audio(y, chunk_len)

        for chunk, end_time in zip(chunks, end_times):
            # extract Perch embedding
            y_tf       = tf.expand_dims(chunk, axis=0)
            _, emb     = perch.infer_tf(y_tf)
            X_chunk    = emb.numpy()              # (1, 1280)

            # ensemble: average probabilities across all folds
            fold_probs = np.zeros((1, N_CLASSES), dtype=np.float32)
            for model in models:
                fold_probs += model.predict_proba(X_chunk)
            fold_probs /= len(models)             # (1, 234)

            row_id = f"{stem}_{end_time}"
            results[row_id] = fold_probs[0]       # (234,)

    # ── build submission DataFrame ────────────────────────────
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
    print(f"Shape : {pred_df.shape}")
    print(f"Rows  : {len(pred_df):,}  (one per 5s window)")
    print(pred_df.head(3))

    # verify no missing species columns
    missing = set(sample_sub.columns) - set(pred_df.columns)
    extra   = set(pred_df.columns) - set(sample_sub.columns)
    print(f"\nMissing columns : {len(missing)}  (should be 0)")
    print(f"Extra columns   : {len(extra)}    (should be 0)")

    return pred_df


if __name__ == "__main__":
    run_inference()