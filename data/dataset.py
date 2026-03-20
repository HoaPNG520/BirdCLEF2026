import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
from configs.config import AUDIO_DIR, SAMPLE_RATE, DURATION, N_FFT, HOP_LENGTH, N_MELS

class BirdDataset(Dataset):
    def __init__(self, df, label2idx, augment=None):
        self.df       = df.reset_index(drop=True)
        self.label2idx = label2idx
        self.augment  = augment
        self.sr       = SAMPLE_RATE
        self.duration = DURATION
        self.chunk_len = self.sr * self.duration

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        fpath = AUDIO_DIR / row['filename']

        # load audio
        y, _ = librosa.load(fpath, sr=self.sr, mono=True)

        # pad if shorter than 5s, else take a random 5s crop
        if len(y) < self.chunk_len:
            y = np.pad(y, (0, self.chunk_len - len(y)))
        else:
            start = np.random.randint(0, len(y) - self.chunk_len + 1)
            y = y[start:start + self.chunk_len]

        # mel spectrogram → (1, N_MELS, T)
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS,
            fmin=20, fmax=16000
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        # label vector — multilabel, shape (N_CLASSES,)
        label = torch.zeros(len(self.label2idx), dtype=torch.float32)
        if str(row['primary_label']) in self.label2idx:
            label[self.label2idx[str(row['primary_label'])]] = 1.0

        return mel, label