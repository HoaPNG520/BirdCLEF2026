import os
import json
import ast  # Required to parse the secondary_labels string
import random
import numpy as np
import librosa
import torch
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


class BirdDataset(Dataset):
    def __init__(self, df, label2idx, audio_dir=AUDIO_DIR, augment=None, mode="train"):
        self.df = df
        self.label2idx = label2idx
        self.audio_dir = audio_dir
        self.chunk_len = SAMPLE_RATE * DURATION
        self.augment = augment
        self.mode = mode

    def __len__(self):
        return len(self.df)

    # ── NEW: The missing label encoding method ───────────────────
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
            fpath = self.audio_dir / row["filename"]

            y, _ = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)

            # Audio-level augmentations
            if self.mode == "train":
                from data.augment import add_background_noise, gain_and_loudness_norm

                y = add_background_noise(y, sr=SAMPLE_RATE, prob=0.6)
                y = gain_and_loudness_norm(y, prob=0.7)

            # 5-second windowing
            if len(y) < self.chunk_len:
                y = np.pad(y, (0, self.chunk_len - len(y)))
            else:
                if self.mode == "train":
                    start = np.random.randint(0, len(y) - self.chunk_len + 1)
                else:
                    start = (len(y) - self.chunk_len) // 2
                y = y[start : start + self.chunk_len]

            # Mel Spectrogram
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

            # Get labels using the new method
            label_tensor = self.get_labels(row)

            return mel_tensor, label_tensor

        except Exception as e:
            # Recursive recovery: pick a different random file
            new_idx = random.randint(0, len(self.df) - 1)
            return self.__getitem__(new_idx)
