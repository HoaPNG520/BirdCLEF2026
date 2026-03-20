import numpy as np
import torch
import torchaudio.transforms as T


def get_spec_augment(freq_mask=20, time_mask=40):
    """
    SpecAugment using torchaudio — masks random frequency and time bands.
    Applied twice each for stronger regularization.
    Input shape: (1, N_MELS, T)
    """
    return torch.nn.Sequential(
        T.FrequencyMasking(freq_mask_param=freq_mask),
        T.TimeMasking(time_mask_param=time_mask),
        T.FrequencyMasking(freq_mask_param=freq_mask),
        T.TimeMasking(time_mask_param=time_mask),
    )


def mixup(mel1, label1, mel2, label2, alpha=0.4):
    """
    Blend two spectrograms and their labels.
    alpha controls blend strength — higher = more mixing.
    No library does this for multilabel audio, so stays custom.
    """
    lam   = np.random.beta(alpha, alpha)
    mel   = lam * mel1   + (1 - lam) * mel2
    label = lam * label1 + (1 - lam) * label2
    return mel, label