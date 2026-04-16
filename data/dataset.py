import json
import ast
import numpy as np
import librosa
import torch
import random
from torch.utils.data import Dataset
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
        try:
            row = self.df.iloc[idx]
            fpath = AUDIO_DIR / row["filename"]

            # 1. Load audio
            y, _ = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)

            # 2. Raw audio augs (Numpy)
            y = add_background_noise(y, sr=SAMPLE_RATE, prob=0.6)
            y = gain_and_loudness_norm(y, prob=0.7)

            # 3. Time alignment (5 seconds)
            if len(y) < self.chunk_len:
                y = np.pad(y, (0, self.chunk_len - len(y)))
            else:
                if self.mode == "train":
                    start = np.random.randint(0, len(y) - self.chunk_len + 1)
                else:
                    start = (len(y) - self.chunk_len) // 2
                y = y[start : start + self.chunk_len]

            # 4. Mel Spectrogram
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

            # 5. Tensor augs
            if self.augment is not None:
                mel_tensor = self.augment(mel_tensor)

            # 6. Labels
            label = self.get_labels(row)  # Use your existing label logic here

            # THE FIX: Always return the data!
            return mel_tensor, label

        except Exception as e:
            # THE INSURANCE: If anything goes wrong, just try another file
            print(f"Skipping bad file {fpath}: {e}")
            return self.__getitem__(random.randint(0, len(self.df) - 1))

    def get_labels(self, row):
        """
        Converts primary and secondary labels into a multi-label vector.
        Primary: 1.0 weight | Secondary: 0.5 weight
        """
        labels = np.zeros(N_CLASSES, dtype=np.float32)

        # 1. Primary label
        primary = row["primary_label"]
        if primary in self.label2idx:
            labels[self.label2idx[primary]] = 1.0

        # 2. Secondary labels
        if "secondary_labels" in row:
            sec_labels = row["secondary_labels"]
            # Parse string representation of list: "['bird1', 'bird2']" -> ['bird1', 'bird2']
            if isinstance(sec_labels, str) and sec_labels.startswith("["):
                try:
                    sec_labels = ast.literal_eval(sec_labels)
                except:
                    sec_labels = []

            if isinstance(sec_labels, list):
                for sl in sec_labels:
                    if sl in self.label2idx:
                        # Soft labeling: secondary birds get half weight
                        labels[self.label2idx[sl]] = 0.5

        return torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            fpath = AUDIO_DIR / row["filename"]

            # 1. Load audio
            y, _ = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)

            # 2. Audio-level augmentations
            if self.mode == "train":
                from data.augment import add_background_noise, gain_and_loudness_norm

                y = add_background_noise(y, sr=SAMPLE_RATE, prob=0.6)
                y = gain_and_loudness_norm(y, prob=0.7)

            # 3. 5-second alignment
            if len(y) < self.chunk_len:
                y = np.pad(y, (0, self.chunk_len - len(y)))
            else:
                if self.mode == "train":
                    start = np.random.randint(0, len(y) - self.chunk_len + 1)
                else:
                    start = (len(y) - self.chunk_len) // 2
                y = y[start : start + self.chunk_len]

            # 4. Mel Spectrogram
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

            if self.augment is not None:
                mel_tensor = self.augment(mel_tensor)

            # 5. Get labels
            label_tensor = self.get_labels(row)

            return mel_tensor, label_tensor

        except Exception as e:
            # Recursive recovery: if a file is bad, grab another one
            new_idx = random.randint(0, len(self.df) - 1)
            return self.__getitem__(new_idx)
