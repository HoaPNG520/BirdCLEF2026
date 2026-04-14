# BirdCLEF+ 2026

Multilabel bioacoustic species classification from passive soundscape recordings.
Competition hosted by Cornell Lab of Ornithology on Kaggle.

**Team:** Data Scientist · ML Engineer  
**Timeline:** 8 weeks  
**Objective:** Maximize padded-cMAP across 234 species (Aves, Amphibia, Insecta, Mammalia, Reptilia)

---

## Table of contents

1. [Competition overview](#competition-overview)
2. [Repository structure](#repository-structure)
3. [File descriptions](#file-descriptions)
4. [Setup](#setup)
5. [Workflow](#workflow)
6. [Kaggle pipeline (3-Notebooks)](#kaggle-pipeline-3-notebooks)
7. [Team responsibilities](#team-responsibilities)
8. [Key EDA findings](#key-eda-findings)
9. [Experiment log](#experiment-log)

---

## Competition overview

The model receives long passive soundscape recordings collected across the Pantanal wetlands. Each recording is segmented into 5-second windows at inference time. For each window the model must output a probability score for all 234 species defined in `taxonomy.csv`.

The evaluation metric is **padded-cMAP** (padded mean Average Precision across classes). Padding adds dummy negative examples per class to prevent artificially inflated scores on rare species.

Key challenge: 28 species appear in `taxonomy.csv` but have no corresponding training audio. Of these, 25 are anonymous insect sonotypes (`son01`–`son25`) requiring unsupervised detection strategies.

---

## Repository structure

```text
birdclef-2026/
│
├── configs/
│   ├── __init__.py            # package marker — empty, do not modify
│   └── config.py              # shared constants: paths, audio params, model params
│
├── data/
│   ├── __init__.py            # package marker — empty, do not modify
│   ├── dataset.py             # PyTorch Dataset — audio loading, chunking, mel spectrogram
│   ├── augment.py             # SpecAugment (torchaudio) + PyTorch-native Mixup
│   └── folds.py               # Stratified K-fold split — shared by both team members
│
├── models/
│   ├── __init__.py            # package marker — empty, do not modify
│   ├── efficientnet.py        # EfficientNet backbone (timm) — Phase 2 Primary Model
│   └── beats.py               # BEATs audio transformer — Phase 3 Model
│
├── train.py                   # end-to-end PyTorch training, validation, cMAP metric
├── infer.py                   # offline PyTorch inference pipeline for hidden test set
│
├── experiments.csv            # shared experiment log — updated after every run
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## File descriptions

### `configs/config.py`
**Owner:** both team members  
**Status:** complete

Central configuration file. All file paths, audio processing parameters (`N_MELS`, `HOP_LENGTH`), and model hyperparameters are defined here. No other file should hardcode constants.

```python
from configs.config import AUDIO_DIR, SAMPLE_RATE, N_MELS, N_CLASSES
```

---

### `data/dataset.py`
**Owner:** data scientist  
**Status:** complete

Implements `BirdDataset`. Processes audio directly into PyTorch Mel spectrogram tensors on the CPU to feed the GPU training loop. 

**Responsibilities:**
- Loads `.ogg` audio via `librosa.load()` at 32kHz mono.
- Applies random 5s crop (train) or center pad/crop (val).
- Converts waveform to a log-scaled mel spectrogram tensor of shape `(1, N_MELS, T)`.
- Encodes `primary_label` (1.0) and `secondary_labels` (0.5) as soft targets.

**What is custom vs library:**

| Component | Source |
|-----------|--------|
| Audio loading + resampling | `librosa.load()` |
| Mel spectrogram computation | `librosa.feature.melspectrogram()` |
| Dataset base class | `torch.utils.data.Dataset` |

---

### `data/augment.py`
**Owner:** data scientist  
**Status:** complete

Provides two augmentation strategies. 

**`get_spec_augment()`**: Uses `torchaudio.transforms` for frequency/time masking.
**`mixup(mel1, label1, mel2, label2)`**: PyTorch-native implementation. Blends two spectrogram tensors and their multilabel vectors using a Beta distribution directly on the GPU.

---

### `models/efficientnet.py`
**Owner:** ML engineer  
**Status:** complete (Phase 2)

Implements `EfficientNetClassifier`. Adapts `timm` image models for 1-channel audio spectrograms.

**What is custom vs library:**

| Component | Source |
|-----------|--------|
| Backbone architecture | `timm.create_model(in_chans=1)` |
| Classification Head | Custom — AdaptiveAvgPool2d + Dropout + Linear |

---

### `models/beats.py`
**Owner:** ML engineer  
**Status:** complete (Phase 3 prep)

Wraps Microsoft's BEATs pretrained audio transformer. **Crucial difference:** BEATs requires raw 1D waveforms `(batch, time)`, *not* 2D Mel spectrograms. `dataset.py` will need a mode flag when we switch to this.

---

### `train.py`
**Owner:** both team members  
**Status:** complete (Phase 2)

End-to-end PyTorch training engine. Bypasses the old feature extraction step.

**Responsibilities:**
- Initializes EfficientNet and `AdamW` optimizer.
- Applies batch-level PyTorch Mixup dynamically.
- Computes `padded_cmap` during validation.
- Saves `effnet_fold{fold}.pth` and `label2idx.json` artifacts.

**Usage:**
```python
from train import train_fold
train_fold(fold=0, epochs=20, lr=1e-3, save_dir='/kaggle/working/model_artifacts/')
```

---

### `infer.py`
**Owner:** both team members  
**Status:** complete

The Kaggle submission script. Designed to run offline.

**Responsibilities:**
- Replicates the exact `librosa` Mel spectrogram generation logic from `dataset.py` to prevent train/test domain gap.
- Slices 5s windows, runs them through the EfficientNet ensemble.
- Averages probabilities across folds and outputs `submission.csv`.

---

## Setup

**Local environment (Windows)**
```powershell
cd birdclef-2026
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install -r requirements.txt
```

---

## Workflow

```text
1.  git pull origin main               # sync teammate's latest changes
2.  git checkout feat/new-model        # work on your own branch
3.  edit .py files in VSCode
4.  python train.py --smoke            # verify locally before pushing
5.  git add . && git commit -m "..."
6.  git push origin feat/new-model
7.  merge to main when stable
8.  Run Kaggle 3-notebook pipeline
9.  record val cMAP in experiments.csv
10. git add experiments.csv && git commit && git push
```

---

## Kaggle pipeline (3-Notebooks)

We use a strict 3-notebook architecture to handle the competition's offline inference rules. **Do not mix training and inference in the same notebook.**

### 1. EDA Notebook (`birdclef2026-eda.ipynb`)
* **Internet:** ON | **GPU:** OFF
* **Input:** Competition Data
* **Action:** Run cells 1–15 top to bottom. Cell 15 saves `df_clean.csv` and `label2idx.json` to `/kaggle/working/eda_artifacts/`.
* **Output:** Save version output and publish as a Kaggle Dataset named: **`birdclef-eda-artifacts`**

### 2. Training Notebook (`birdclef2026-training.ipynb`)
* **Internet:** ON | **GPU:** ON (T4x2)
* **Inputs:** Competition Data + `birdclef-eda-artifacts`

**Cell 1 — clone latest code**
```python
import sys, subprocess, shutil, os
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
clone_dir = "/kaggle/working/birdclef2026"
if os.path.exists(clone_dir): shutil.rmtree(clone_dir)

subprocess.run(["git", "clone", "--quiet", "--branch", "main", 
                f"https://{token}@[github.com/HoaPNG520/BirdCLEF2026.git](https://github.com/HoaPNG520/BirdCLEF2026.git)", clone_dir], check=True)
sys.path.insert(0, clone_dir)
```

**Cell 2 — run training**
```python
from train import train_fold
import os

out_dir = "/kaggle/working/model_artifacts"
os.makedirs(out_dir, exist_ok=True)

# Train folds
train_fold(fold=0, epochs=15, lr=1e-3, save_dir=out_dir)
```

**Cell 3 — package codebase (CRITICAL)**
```python
import shutil

# Copy codebase alongside models for offline use
for d in ["configs", "data", "models"]:
    shutil.copytree(os.path.join(clone_dir, d), os.path.join(out_dir, d), dirs_exist_ok=True)
shutil.copy2(os.path.join(clone_dir, "infer.py"), out_dir)
```
* **Output:** Save version output and publish as a Kaggle Dataset named: **`birdclef-models-v1`**

### 3. Inference Notebook (`birdclef2026-real-inference.ipynb`)
* **Internet:** **OFF** | **GPU:** ON
* **Inputs:** Competition Data + `birdclef-eda-artifacts` + `birdclef-models-v1`

**Cell 1 — run offline inference**
```python
import sys, os

# 1. Mount packaged codebase from our models dataset
MODEL_DIR = "/kaggle/input/birdclef-models-v1"
sys.path.insert(0, MODEL_DIR)

# 2. Import and run script directly from the dataset
from infer import run_inference
run_inference()

if os.path.exists("/kaggle/working/submission.csv"):
    print("Success! Ready for submission.")
```
* **Output:** Submit to competition!

---

## Team responsibilities

| Task | Data Scientist | ML Engineer |
|------|---------------|-------------|
| EDA and data exploration | owner | reviewer |
| `data/dataset.py` | owner | consumer |
| `data/augment.py` | owner | consumer |
| `data/folds.py` | owner | consumer |
| `configs/config.py` | owner | contributor |
| `train.py` | contributor | owner |
| `models/efficientnet.py` | — | owner |
| `models/beats.py` | — | owner |
| `infer.py` | contributor | owner |
| Kaggle Notebook orchestration | both | both |
| Ensemble design | contributor | owner |
| `experiments.csv` | both | both |

---

## Key EDA findings

| Finding | Value | Implication |
|---------|-------|-------------|
| Total training clips | 35,549 | — |
| Species with training audio | 206 | — |
| Species in taxonomy | 234 | Model must output 234 classes |
| Zero-shot species | 28 | No training audio — requires special handling |
| Zero-shot breakdown | 25 Insecta sonotypes + 3 Amphibia | Clustering strategy needed for Insecta in Phase 4 |
| All files at | 32,000 Hz | No resampling inconsistencies |
| Corrupt files | 0 | Clean dataset |
| Clips under 1 second | 370 | Drop before training |
| Mean clip duration | 34.9s | ~6–7 chunks per clip on average |
| Max clip duration | 6,881s | Random crop strategy handles this cleanly |
| `rating = 0.0` for iNat clips | — | Do not filter iNat clips by rating |
| Data sources | XC (Xeno-Canto) + iNat | Different quality profiles per collection |
| Test soundscapes | Hidden until submission | Train/test domain gap is the primary modelling challenge |

---

## Experiment log

| date | who | branch | model | val_cMAP | LB_score | notes |