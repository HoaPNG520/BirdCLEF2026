"""
extract_feature.py
==================
Extracts Perch embeddings from all training clips and saves them as .npy files.
Run this ONCE before training. Takes ~2-3 hours on GPU.

Perch is Google's bird vocalization classifier pretrained on a large audio dataset.
Using it as a feature extractor gives much better embeddings than mel spectrograms
for bird sounds specifically.

GPU setup: TensorFlow automatically uses GPU if available.
Check with: print(tf.config.list_physical_devices('GPU'))
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from pathlib import Path

from configs.config import (
    BASE_DIR_ARTIFACT, AUDIO_DIR,
    SAMPLE_RATE, DURATION, PERCH_URL
)
from data.dataset import load_df_clean, load_label2idx

AUDIO_LENGTH = int(SAMPLE_RATE * DURATION)
SAVE_DIR     = Path("/kaggle/working/birdclef-embeddings")


def setup_gpu():
    """Configure TensorFlow to use GPU with memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU enabled: {len(gpus)} GPU(s) found")
    else:
        print("No GPU found — running on CPU (will be slow)")
    return len(gpus) > 0


def load_perch():
    """Load Perch model from TF Hub. Cached after first download."""
    print("Loading Perch model...")
    model = hub.load(PERCH_URL)
    print("Perch model loaded ✓")
    return model


def load_and_preprocess(file_path):
    """Load audio, pad/crop to exactly 5s, return float32 array."""
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(y) < AUDIO_LENGTH:
        y = np.pad(y, (0, AUDIO_LENGTH - len(y)))
    else:
        y = y[:AUDIO_LENGTH]
    return y.astype(np.float32)


def extract_embeddings(df, perch_model, batch_size=32):
    """
    Extract Perch embeddings for all clips in df.

    Perch returns:
      logits     : (1, 9736) — Perch's own species predictions
      embeddings : (1, 1280) — the feature vector we want

    We use embeddings as input to XGBoost.
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    all_embeddings = []
    all_labels     = []
    errors         = []

    for i, (_, row) in enumerate(df.iterrows()):
        fpath = AUDIO_DIR / row['filename']
        try:
            y = load_and_preprocess(fpath)

            # Perch expects shape (batch, samples)
            y_batched = tf.expand_dims(y, axis=0)
            logits, embeddings = perch_model.infer_tf(y_batched)

            all_embeddings.append(embeddings.numpy()[0])  # (1280,)
            all_labels.append(str(row['primary_label']))

        except Exception as e:
            errors.append((row['filename'], str(e)))

        if (i + 1) % 500 == 0:
            print(f"  {i+1:,} / {len(df):,} processed  errors={len(errors)}")

    embeddings_arr = np.array(all_embeddings, dtype=np.float32)
    labels_arr     = np.array(all_labels)

    print(f"\nEmbedding shape : {embeddings_arr.shape}")
    print(f"Labels shape    : {labels_arr.shape}")
    print(f"Errors          : {len(errors)}")

    return embeddings_arr, labels_arr


def run_extraction():
    setup_gpu()

    df        = load_df_clean()
    label2idx = load_label2idx()

    # check if already partially done
    emb_path = SAVE_DIR / "X_embeddings.npy"
    lbl_path = SAVE_DIR / "y_labels.npy"

    if emb_path.exists() and lbl_path.exists():
        existing = np.load(emb_path)
        print(f"Already have {len(existing):,} embeddings. Re-extracting all to be safe.")

    perch_model = load_perch()
    X, y        = extract_embeddings(df, perch_model)

    np.save(emb_path, X)
    np.save(lbl_path, y)

    size_mb = (emb_path.stat().st_size + lbl_path.stat().st_size) / 1e6
    print(f"\nSaved to {SAVE_DIR}")
    print(f"  X_embeddings.npy : {X.shape}  ({size_mb:.0f} MB)")
    print(f"  y_labels.npy     : {y.shape}")
    print("Next step: run train.py")

    return X, y


if __name__ == "__main__":
    run_extraction()