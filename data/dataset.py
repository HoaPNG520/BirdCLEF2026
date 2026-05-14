import ast
import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path

from configs.config import (
    AUDIO_DIR,
    SAMPLE_RATE,
    DURATION,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    N_CLASSES,
)
from data.augment import add_background_noise, gain_and_loudness_norm

# ── Artifact loaders ───────────────────────────────────────────────────────────


def load_label2idx(path=None):
    import json
    from configs.config import BASE_DIR_ARTIFACT

    path = path or BASE_DIR_ARTIFACT / "label2idx.json"
    with open(path) as f:
        label2idx = json.load(f)
    assert len(label2idx) == 234, f"Expected 234 species, got {len(label2idx)}"
    return label2idx


def load_df_clean(path=None):
    import pandas as pd
    from configs.config import BASE_DIR_ARTIFACT

    path = path or BASE_DIR_ARTIFACT / "df_clean.csv"
    df = pd.read_csv(path)
    df["primary_label"] = df["primary_label"].astype(str)

    # Re-parse secondary labels from string → list
    if "secondary_labels" in df.columns:
        df["sec_parsed"] = df["secondary_labels"].apply(
            lambda x: (
                ast.literal_eval(x)
                if isinstance(x, str) and x not in ["[]", ""]
                else []
            )
        )
    else:
        df["sec_parsed"] = [[] for _ in range(len(df))]

    print(
        f"Loaded df_clean: {len(df):,} clips, {df['primary_label'].nunique()} species"
    )
    return df


# ── Dataset ────────────────────────────────────────────────────────────────────


class BirdDataset(Dataset):
    """
    PyTorch Dataset for BirdCLEF+ 2026 — EfficientNet pipeline.

    Returns:
        mel_tensor : (1, N_MELS, T) float32
        label      : (N_CLASSES,)  float32  — 1.0 primary, 0.5 secondary
    """

    def __init__(self, df, label2idx, augment=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.augment = augment
        self.mode = mode
        self.chunk_len = SAMPLE_RATE * DURATION

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            fpath = AUDIO_DIR / row["filename"]

            # ── Load audio ────────────────────────────────────────────────────
            y, _ = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)

            # ── Waveform augmentation (training only) ─────────────────────────
            if self.mode == "train":
                y = add_background_noise(y, sr=SAMPLE_RATE, prob=0.5)
                y = gain_and_loudness_norm(y, prob=0.6)

            # ── Crop / pad to exactly 5s ──────────────────────────────────────
            if len(y) < self.chunk_len:
                y = np.pad(y, (0, self.chunk_len - len(y)))
            else:
                if self.mode == "train":
                    start = np.random.randint(0, len(y) - self.chunk_len + 1)
                else:
                    start = (len(y) - self.chunk_len) // 2
                y = y[start : start + self.chunk_len]

            # ── Log-mel spectrogram → (1, N_MELS, T) ─────────────────────────
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                fmin=20,
                fmax=16000,
            )
            mel = librosa.power_to_db(mel, ref=np.max)
            mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

            # ── SpecAugment (training only) ───────────────────────────────────
            if self.augment is not None and self.mode == "train":
                mel_tensor = self.augment(mel_tensor)

            return mel_tensor, self.get_labels(row)

        except Exception:
            # On any load error, randomly sample another clip
            return self.__getitem__(random.randint(0, len(self.df) - 1))

    def get_labels(self, row):
        """Encode primary (1.0) and secondary (0.5) labels into a multi-hot vector."""
        label = torch.zeros(N_CLASSES, dtype=torch.float32)

        primary = str(row["primary_label"])
        if primary in self.label2idx:
            label[self.label2idx[primary]] = 1.0

        sec_parsed = row["sec_parsed"] if "sec_parsed" in row.index else []
        if not isinstance(sec_parsed, list):
            sec_parsed = []
        for sec in sec_parsed:
            if str(sec) in self.label2idx:
                label[self.label2idx[str(sec)]] = 0.5

        return label
