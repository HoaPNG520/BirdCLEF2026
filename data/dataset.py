import os
import json
import ast  # Required to parse the secondary_labels string
import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
from pathlib import Path

# Ensure these are imported from your config
from configs.config import (
    AUDIO_DIR,
    SAMPLE_RATE,
    DURATION,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    N_CLASSES,
)


def load_label2idx(path=None):
    import json
    from configs.config import BASE_DIR_ARTIFACT

    path = path or BASE_DIR_ARTIFACT / "label2idx.json"
    with open(path) as f:
        return json.load(f)


def load_df_clean(path=None):
    import pandas as pd
    from configs.config import BASE_DIR_ARTIFACT

    path = path or BASE_DIR_ARTIFACT / "df_clean.csv"
    return pd.read_csv(path)


class BirdDataset(Dataset):
    def __init__(self, df, label2idx, augment=None, mode="train", model_type="effnet"):
        self.df = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.augment = augment
        self.mode = mode
        self.model_type = model_type  # New: track which model we're feeding
        self.chunk_len_32k = SAMPLE_RATE * DURATION

        # BEATs resampler (32kHz -> 16kHz)
        self.resampler = T.Resample(32000, 16000)

    def get_labels(self, row):
        label = torch.zeros(N_CLASSES, dtype=torch.float32)
        primary = str(row["primary_label"])
        if primary in self.label2idx:
            label[self.label2idx[primary]] = 1.0
        for sec in row.get("sec_parsed", []):
            if str(sec) in self.label2idx:
                label[self.label2idx[str(sec)]] = 0.5
        return label

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            fpath = AUDIO_DIR / row["filename"]
            y, _ = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)

            # 1. Basic 5s alignment at 32kHz
            if len(y) < self.chunk_len_32k:
                y = np.pad(y, (0, self.chunk_len_32k - len(y)))
            else:
                start = (
                    np.random.randint(0, len(y) - self.chunk_len_32k + 1)
                    if self.mode == "train"
                    else (len(y) - self.chunk_len_32k) // 2
                )
                y = y[start : start + self.chunk_len_32k]

            # 2. Branching Logic
            if self.model_type == "beats":
                # --- BEATs Path: Raw Waveform at 16kHz ---
                y_tensor = torch.from_numpy(y).float()
                y_16k = self.resampler(y_tensor)
                # Normalize audio to [-1, 1] range for transformers
                y_16k = y_16k / (torch.max(torch.abs(y_16k)) + 1e-8)
                return y_16k, self.get_labels(row)
            else:
                # --- EfficientNet Path: Mel Spectrogram ---
                mel = librosa.feature.melspectrogram(
                    y=y,
                    sr=SAMPLE_RATE,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    n_mels=N_MELS,
                )
                mel = librosa.power_to_db(mel, ref=np.max)
                mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
                if self.augment:
                    mel_tensor = self.augment(mel_tensor)
                return mel_tensor, self.get_labels(row)

        except Exception as e:
            return self.__getitem__(random.randint(0, len(self.df) - 1))
