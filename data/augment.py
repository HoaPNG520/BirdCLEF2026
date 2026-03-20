import numpy as np
import torch

class SpecAugment:
    """
    Randomly mask time and frequency bands on a mel spectrogram tensor.
    Input shape: (1, N_MELS, T)
    """
    def __init__(self, freq_mask=20, time_mask=40, n_freq=2, n_time=2):
        self.freq_mask = freq_mask  
        self.time_mask = time_mask
        self.n_freq    = n_freq
        self.n_time    = n_time

    def __call__(self, mel):
        mel = mel.clone()
        _, n_mels, n_frames = mel.shape

        # frequency masking
        for _ in range(self.n_freq):
            f = np.random.randint(0, self.freq_mask)
            f0 = np.random.randint(0, max(1, n_mels - f))
            mel[:, f0:f0 + f, :] = mel.mean()

        # time masking
        for _ in range(self.n_time):
            t = np.random.randint(0, self.time_mask)
            t0 = np.random.randint(0, max(1, n_frames - t))
            mel[:, :, t0:t0 + t] = mel.mean()

        return mel


def mixup(mel1, label1, mel2, label2, alpha=0.4):
    """
    Blend two spectrograms and their labels.
    alpha controls blend strength — higher = more mixing.
    """
    lam = np.random.beta(alpha, alpha)
    mel   = lam * mel1 + (1 - lam) * mel2
    label = lam * label1 + (1 - lam) * label2
    return mel, label