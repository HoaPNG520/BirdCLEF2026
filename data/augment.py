import numpy as np
import torch
import torchaudio.transforms as T


def get_spec_augment(freq_mask=20, time_mask=40):
    """SpecAugment using torchaudio — masks random frequency and time bands."""
    return torch.nn.Sequential(
        T.FrequencyMasking(freq_mask_param=freq_mask),
        T.TimeMasking(time_mask_param=time_mask),
        T.FrequencyMasking(freq_mask_param=freq_mask),
        T.TimeMasking(time_mask_param=time_mask),
    )


def mixup(mel1, label1, mel2, label2, alpha=0.4):
    """
    PyTorch-native Mixup. Blends two spectrogram tensors and their labels.
    Both inputs must be PyTorch tensors on the same device.
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    mel = lam * mel1 + (1 - lam) * mel2
    label = lam * label1 + (1 - lam) * label2
    return mel, label


def add_background_noise(
    y: np.ndarray,
    sr: int = 32000,
    prob: float = 0.6,
    min_snr_db: float = 3.0,
    max_snr_db: float = 30.0,
) -> np.ndarray:
    """
    Add realistic background noise (pink noise) with random SNR.
    This is one of the strongest augmentations in BirdCLEF.
    Applied to raw audio before mel or Perch.
    """
    if np.random.rand() > prob:
        return y  # skip with probability

    # Generate pink noise (1/f spectrum — sounds very natural for nature recordings)
    noise = np.random.randn(len(y)).astype(np.float32)
    noise = np.cumsum(noise)  # cumulative sum → pink noise
    noise = noise / (np.max(np.abs(noise)) + 1e-8)

    # Random SNR (signal-to-noise ratio)
    snr_db = np.random.uniform(min_snr_db, max_snr_db)
    snr = 10 ** (snr_db / 20.0)

    # Scale noise to desired SNR
    signal_power = np.mean(y**2)
    noise_power = np.mean(noise**2)
    noise_scale = np.sqrt(signal_power / (noise_power * snr + 1e-8))
    noise = noise * noise_scale

    # Mix
    y_noisy = y + noise
    y_noisy = np.clip(y_noisy, -1.0, 1.0)  # prevent clipping

    return y_noisy


def gain_and_loudness_norm(
    y: np.ndarray,
    prob: float = 0.7,
    gain_db_range: tuple = (-10.0, 10.0),
    target_rms: float = 0.1,
) -> np.ndarray:
    """
    Gain augmentation + loudness normalization.
    Very effective for BirdCLEF — makes model robust to volume differences.
    """
    if np.random.rand() > prob:
        return y

    # === Random Gain ===
    gain_db = np.random.uniform(gain_db_range[0], gain_db_range[1])
    gain = 10 ** (gain_db / 20.0)
    y = y * gain

    # === Loudness Normalization (to fixed RMS) ===
    rms = np.sqrt(np.mean(y**2))
    if rms > 1e-8:  # avoid division by zero
        y = y * (target_rms / rms)

    # Prevent clipping
    y = np.clip(y, -1.0, 1.0)

    return y
