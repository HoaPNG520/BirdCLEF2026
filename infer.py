"""
infer.py
========
PyTorch inference pipeline. Slices test audio, computes Mel spectrograms, 
ensembles PyTorch EfficientNet models, and writes submission.csv.
"""

import json
import numpy as np
import pandas as pd
import librosa
import torch
from pathlib import Path
from tqdm import tqdm

from configs.config import (
    BASE_DIR_COMPETITION, TEST_DIR, BASE_DIR_MODELS,
    SAMPLE_RATE, DURATION, N_CLASSES, HOP_LENGTH, N_FFT, N_MELS
)
from models.efficientnet import EfficientNetClassifier

AUDIO_LENGTH = int(SAMPLE_RATE * DURATION)

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    return device

def load_pytorch_models(model_dir, device, n_folds=5):
    """Load PyTorch EfficientNet folds."""
    model_dir = Path(model_dir)
    models = []
    for fold in range(n_folds):
        path = model_dir / f"effnet_fold{fold}.pth"
        if path.exists():
            model = EfficientNetClassifier(n_classes=N_CLASSES).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models.append(model)
            print(f"Loaded: {path.name}")
        else:
            print(f"Warning: Missing {path.name}")
    return models

def chunk_audio_to_mels(y, chunk_len):
    """Splits audio, pads, and converts directly to PyTorch Mel tensors."""
    chunks, end_times = [], []
    n_chunks = max(1, len(y) // chunk_len)
    
    for i in range(n_chunks):
        chunk = y[i*chunk_len:(i+1)*chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
            
        # Replicate dataset.py logic exactly
        mel = librosa.feature.melspectrogram(
            y=chunk, sr=SAMPLE_RATE, n_fft=N_FFT, 
            hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=20, fmax=16000
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # Shape: (1, N_MELS, T) to match EfficientNet expected input
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        chunks.append(mel_tensor)
        end_times.append((i + 1) * 5)
        
    return chunks, end_times

def run_inference():
    device = setup_device()
    model_dir = BASE_DIR_MODELS

    # 1. Load label mapping
    with open(model_dir / "label2idx.json") as f:
        label2idx = json.load(f)
    
    # 2. Load PyTorch Ensemble
    models = load_pytorch_models(model_dir, device)
    if not models:
        raise RuntimeError("No models found!")

    # 3. Setup submission
    sample_sub = pd.read_csv(BASE_DIR_COMPETITION / "sample_submission.csv")
    test_files = sorted(Path(TEST_DIR).glob("*.ogg"))
    results = {}

    # 4. Inference Loop
    for audio_path in tqdm(test_files, desc="Inference"):
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        mels, end_times = chunk_audio_to_mels(y, AUDIO_LENGTH)

        for mel_tensor, end_time in zip(mels, end_times):
            # Add batch dimension: (1, 1, 128, T)
            x = mel_tensor.unsqueeze(0).to(device)
            
            fold_probs = np.zeros((1, N_CLASSES), dtype=np.float32)
            with torch.no_grad():
                for model in models:
                    logits = model(x)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    fold_probs += probs
            
            fold_probs /= len(models)
            
            row_id = f"{audio_path.stem}_{end_time}"
            results[row_id] = fold_probs[0]

    # 5. Format CSV
    rows = []
    for row_id, probs in results.items():
        row = {"row_id": row_id}
        for label, idx in label2idx.items():
            row[label] = round(float(probs[idx]), 6)
        rows.append(row)

    pred_df = pd.DataFrame(rows).reindex(columns=sample_sub.columns, fill_value=0.0)
    pred_df.to_csv("/kaggle/working/submission.csv", index=False)
    print("\nSubmission saved successfully.")

if __name__ == "__main__":
    run_inference()