import json
import ast
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path

from configs.config import AUDIO_DIR, SAMPLE_RATE, DURATION, N_FFT, HOP_LENGTH, N_MELS


# ── Artifact loading ──────────────────────────────────────────
# These functions load the files saved by the EDA notebook.
# The ML team should never need to rebuild these from scratch.

ARTIFACT_DIR = Path("/kaggle/input/birdclef-eda-artifacts")


def load_label2idx(path=None):
    """
    Load species → index mapping saved by EDA notebook.
    Always 234 entries — includes zero-shot species.
    """
    path = path or ARTIFACT_DIR / "label2idx.json"
    with open(path) as f:
        label2idx = json.load(f)
    print(f"Loaded label2idx: {len(label2idx)} species")
    assert len(label2idx) == 234, f"Expected 234 species, got {len(label2idx)}"
    return label2idx


def load_df_clean(path=None):
    """
    Load the filtered training DataFrame saved by EDA notebook.
    Already filtered: duration >= 1s, iNat kept, XC rating >= 3.
    Already has sample_weight, sec_parsed, n_secondary columns.
    """
    import pandas as pd

    path = path or ARTIFACT_DIR / "df_clean.csv"
    df = pd.read_csv(path)
    df["primary_label"] = df["primary_label"].astype(str)

    # re-parse sec_parsed from string back to list
    if "secondary_labels" in df.columns:
        df["sec_parsed"] = df["secondary_labels"].apply(
            lambda x: (
                ast.literal_eval(x)
                if isinstance(x, str) and x not in ["[]", ""]
                else []
            )
        )

    print(
        f"Loaded df_clean: {len(df):,} clips, "
        f"{df['primary_label'].nunique()} species"
    )
    return df


def load_zero_shot(path=None):
    """
    Load list of zero-shot species labels (no training audio).
    These are insect sonotypes + 3 Amphibia species.
    """
    path = path or ARTIFACT_DIR / "zero_shot_labels.json"
    with open(path) as f:
        zero_shot = json.load(f)
    print(f"Loaded zero_shot_labels: {len(zero_shot)} species")
    return zero_shot


# ── Dataset class ─────────────────────────────────────────────


class BirdDataset(Dataset):
    """
    PyTorch Dataset for BirdCLEF+ 2026.

    Loads one audio clip per __getitem__ call:
      - random 5s crop for clips longer than 5s
      - zero-pad for clips shorter than 5s
      - converts to log-mel spectrogram shape (1, N_MELS, T)
      - encodes primary_label as 1.0, secondary_labels as 0.5

    Args:
        df        : df_clean from EDA artifacts
        label2idx : label2idx from EDA artifacts
        augment   : optional callable applied to mel tensor
        mode      : 'train' (random crop) or 'val' (center crop)
    """

    def __init__(self, df, label2idx, augment=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.augment = augment
        self.mode = mode
        self.n_classes = len(label2idx)
        self.chunk_len = SAMPLE_RATE * DURATION

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fpath = AUDIO_DIR / row["filename"]

        # ── load audio ────────────────────────────────────────
        y, _ = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)

        # === Background noise (we added last time) ===
        from augment import add_background_noise, gain_and_loudness_norm

        y = add_background_noise(y, sr=SAMPLE_RATE, prob=0.6)

        # === NEW: Gain + Loudness Normalization ===
        y = gain_and_loudness_norm(y, prob=0.7)
        # ===============================================================

        # ── crop or pad to exactly 5 seconds ──────────────────
        if len(y) < self.chunk_len:
            y = np.pad(y, (0, self.chunk_len - len(y)))
        else:
            if self.mode == "train":
                start = np.random.randint(0, len(y) - self.chunk_len + 1)
            else:
                start = (len(y) - self.chunk_len) // 2
            y = y[start : start + self.chunk_len]

        # ── mel spectrogram → (1, N_MELS, T) ─────────────────
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
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        # ── augmentation ──────────────────────────────────────
        if self.augment is not None:
            mel = self.augment(mel)

        # ... rest of label code unchanged ...
