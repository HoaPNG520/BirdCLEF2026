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
6. [Kaggle notebook](#kaggle-notebook)
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

```
birdclef-2026/
│
├── configs/
│   ├── __init__.py            # package marker — empty, do not modify
│   └── config.py              # shared constants: paths, audio params, model params
│
├── data/
│   ├── __init__.py            # package marker — empty, do not modify
│   ├── dataset.py             # PyTorch Dataset — audio loading, chunking, mel spectrogram
│   ├── augment.py             # SpecAugment (torchaudio) + Mixup (custom)
│   └── folds.py               # Stratified K-fold split — shared by both team members
│
├── models/
│   ├── __init__.py            # package marker — empty, do not modify
│   ├── efficientnet.py        # EfficientNet backbone (ML Engineer) — Week 3–4
│   └── beats.py               # BEATs audio transformer (ML Engineer) — Week 4–5
│
├── train.py                   # training loop, validation, cMAP metric, checkpointing
├── infer.py                   # inference pipeline for test soundscapes — Week 7–8
│
├── notebooks/
│   └── eda.ipynb              # exploratory data analysis (read-only after Phase 1)
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

Central configuration file. All file paths, audio processing parameters, and model hyperparameters are defined here. No other file should hardcode constants — every module imports from this file.

```python
from configs.config import AUDIO_DIR, SAMPLE_RATE, N_MELS, N_CLASSES
```

Changing a parameter here propagates automatically to the entire pipeline. This ensures consistency between the data scientist's preprocessing and the ML engineer's training loop.

---

### `configs/__init__.py` · `data/__init__.py` · `models/__init__.py`
**Owner:** neither — do not modify  
**Status:** intentionally empty

These files are Python package markers. Their sole purpose is to allow cross-directory imports such as `from data.dataset import BirdDataset`. They contain no logic and should never be edited. Every directory containing importable `.py` modules requires one.

---

### `data/dataset.py`
**Owner:** data scientist  
**Status:** complete

Implements `BirdDataset`, a subclass of `torch.utils.data.Dataset`. This is the primary interface between raw audio files and the training loop.

**Responsibilities:**
- Reads file paths and labels from the training DataFrame
- Loads `.ogg` audio via `librosa.load()` at 32kHz mono
- Applies a random 5-second crop for clips longer than 5 seconds
- Zero-pads clips shorter than 5 seconds to reach the required length
- Converts the audio waveform to a log-scaled mel spectrogram of shape `(1, 128, 313)`
- Encodes `primary_label` as a one-hot vector of length 234
- Encodes `secondary_labels` as soft targets with weight 0.5 to partially supervise co-occurring species

**Usage:**
```python
from data.dataset import BirdDataset
from data.augment import get_spec_augment

dataset = BirdDataset(df=train_df, label2idx=label2idx,
                      augment=get_spec_augment())
mel, label = dataset[0]
# mel.shape   → (1, 128, 313)
# label.shape → (234,)
```

**What is custom vs library:**

| Component | Source |
|-----------|--------|
| File path resolution | Custom — BirdCLEF directory structure |
| Audio loading + resampling | `librosa.load()` |
| 5-second crop / padding | Custom — competition-specific chunking strategy |
| Mel spectrogram computation | `librosa.feature.melspectrogram()` |
| dB conversion | `librosa.power_to_db()` |
| Label encoding | Custom — BirdCLEF multilabel schema |
| Secondary label soft targets | Custom — competition-specific label strategy |
| Dataset base class | `torch.utils.data.Dataset` |

---

### `data/folds.py`
**Owner:** data scientist  
**Status:** complete

Provides reproducible stratified train/validation splits using `sklearn.model_selection.StratifiedKFold`. Both team members must use this module exclusively — never construct splits manually — to guarantee that validation scores are directly comparable across experiments.

**Functions:**

`make_folds(df, n_folds=5, seed=42)` — adds a `fold` column (integer 0–4) to the DataFrame using stratified sampling on `primary_label`. The random seed is fixed to ensure reproducibility.

`get_fold(df, fold)` — returns `(train_df, val_df)` for a given fold index.

**Usage:**
```python
from data.folds import make_folds, get_fold

df = make_folds(df, n_folds=5, seed=42)
train_df, val_df = get_fold(df, fold=0)
```

**What is custom vs library:** `StratifiedKFold` from scikit-learn performs the split. The wrapper adds DataFrame integration and enforces a fixed seed contract between team members.

---

### `data/augment.py`
**Owner:** data scientist  
**Status:** complete

Provides two augmentation strategies applied during training to improve generalisation and address the train/test domain gap.

**`get_spec_augment(freq_mask=20, time_mask=40)`**  
Returns a `torch.nn.Sequential` pipeline that applies frequency masking and time masking twice each to a mel spectrogram tensor. Implemented entirely via `torchaudio.transforms.FrequencyMasking` and `torchaudio.transforms.TimeMasking` — no custom masking logic.

**`mixup(mel1, label1, mel2, label2, alpha=0.4)`**  
Blends two spectrogram tensors and their corresponding multilabel vectors using a Beta-distributed interpolation coefficient λ. Custom implementation — no library provides multilabel audio Mixup.

**Usage:**
```python
from data.augment import get_spec_augment, mixup

augment  = get_spec_augment()
mel_aug  = augment(mel)

mel_mix, lbl_mix = mixup(mel_a, lbl_a, mel_b, lbl_b)
```

**What is custom vs library:**

| Component | Source |
|-----------|--------|
| Frequency masking | `torchaudio.transforms.FrequencyMasking` |
| Time masking | `torchaudio.transforms.TimeMasking` |
| Mixup blending | Custom — multilabel audio mixup has no library equivalent |

---

### `train.py`
**Owner:** both team members  
**Status:** complete (baseline)

Entry point for all training experiments. Orchestrates the full training pipeline: data loading, model construction, optimisation loop, validation, metric computation, and checkpointing.

**Key functions:**

`get_label2idx(tax)` — constructs a deterministic `{label_string: index}` mapping from the taxonomy DataFrame covering all 234 species including zero-shot classes.

`train_one_epoch(model, loader, optimizer, criterion, device, augment)` — single training epoch with optional SpecAugment applied per batch.

`validate(model, loader, criterion, device)` — full validation pass returning loss, raw predictions, and ground-truth labels.

`padded_cmap(labels, preds, padding=5)` — computes the competition evaluation metric. Uses `sklearn.metrics.average_precision_score` per class with padding dummy negatives. Custom implementation — the padding strategy is specific to BirdCLEF scoring.

`train_fold(fold, epochs, batch_size, lr, save_dir)` — main entry point. Constructs the full pipeline and runs training, printing loss and cMAP after each epoch. Saves the best checkpoint by validation cMAP.

**Usage from Kaggle notebook:**
```python
from train import train_fold

train_fold(fold=0, epochs=20, batch_size=32, lr=1e-3,
           save_dir='/kaggle/working/checkpoints/')
```

**Smoke test (local CPU, no GPU required):**
```powershell
python train.py --smoke
```

**What is custom vs library:**

| Component | Source |
|-----------|--------|
| Model architecture | `timm.create_model()` |
| Optimiser | `torch.optim.AdamW` |
| Learning rate schedule | `torch.optim.lr_scheduler.CosineAnnealingLR` |
| Loss function | `torch.nn.BCEWithLogitsLoss` |
| Progress bar | `tqdm` |
| Average precision | `sklearn.metrics.average_precision_score` |
| Padded-cMAP metric | Custom — competition-specific padding strategy |
| Training loop structure | Custom boilerplate — standard PyTorch pattern |

---

### `models/efficientnet.py`
**Owner:** ML engineer  
**Status:** not yet implemented — Week 3–4

Will contain a custom `EfficientNetClassifier` wrapping `timm.create_model('efficientnet_b3')` with a competition-specific classification head. Until this file is written, `train.py` calls `timm.create_model()` directly as a working baseline.

---

### `models/beats.py`
**Owner:** ML engineer  
**Status:** not yet implemented — Week 4–5

Will implement `BEATsClassifier`, wrapping Microsoft's BEATs pretrained audio transformer. Unlike EfficientNet which was pretrained on ImageNet images, BEATs was pretrained on AudioSet — a large-scale audio dataset covering bird calls, insect sounds, and amphibian vocalisations. This distinction makes BEATs significantly stronger on non-Aves classes.

BEATs accepts raw waveforms rather than mel spectrograms. When integrated, `BirdDataset` will require a mode flag to return raw audio in addition to mel spectrograms.

---

### `infer.py`
**Owner:** both team members  
**Status:** not yet implemented — Week 7–8

Will implement the full submission pipeline: loading test soundscapes, segmenting into 5-second windows, running the trained model or ensemble on each window, and writing a `submission.csv` conforming to the format specified in `sample_submission.csv`.

---

### `experiments.csv`
**Owner:** both team members  
**Status:** fill in after every run

Shared experiment log. Every training run must be recorded here immediately after completion and committed to the repository.

**Columns:** `date · who · branch · change · val_cMAP · LB_score · notes`

---

## Setup

**Local environment (Windows)**
```powershell
cd birdclef-2026
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Verify installation**
```powershell
python train.py --smoke
# expected output: "Smoke test passed ✓"
```

---

## Workflow

```
1.  git pull origin main               # sync teammate's latest changes
2.  git checkout feat/data-pipeline    # work on your own branch
3.  edit .py files in VSCode
4.  python train.py --smoke            # verify locally before pushing
5.  git add . && git commit -m "..."
6.  git push origin feat/data-pipeline
7.  merge to main when stable
8.  Kaggle → restart kernel → Run All
9.  record val cMAP in experiments.csv
10. git add experiments.csv && git commit && git push
```

---

## Kaggle notebook

**Cell 1 — clone latest code**
```python
import subprocess, sys, shutil, os
from kaggle_secrets import UserSecretsClient

token     = UserSecretsClient().get_secret("GITHUB_TOKEN")
clone_dir = "/kaggle/working/birdclef2026"

if os.path.exists(clone_dir):
    shutil.rmtree(clone_dir)

subprocess.run([
    "git", "clone", "--quiet", "--branch", "main",
    f"https://{token}@github.com/HoaPNG520/BirdCLEF2026.git",
    clone_dir
], check=True)

sys.path.insert(0, clone_dir)
print("Repo cloned ✓")
```

**Cell 2 — imports**
```python
import sys
for mod in list(sys.modules.keys()):
    if any(mod.startswith(x) for x in ['configs', 'data', 'train']):
        del sys.modules[mod]

from configs.config import *
from data.dataset import BirdDataset
from data.augment import get_spec_augment
from train import train_fold
print("Code loaded ✓")
```

**Cell 3 — run training**
```python
train_fold(
    fold=0, epochs=20, batch_size=32,
    lr=1e-3, save_dir='/kaggle/working/checkpoints/'
)
```

> After every `git push`: restart kernel → Run All cells in order.

---

## EDA notebook
Lives on Kaggle — search "birdclef2026-eda" in your Kaggle notebooks.
Run cells 1–15 top to bottom.
Cell 15 saves EDA artifacts to /kaggle/working/eda_artifacts/.
Publish those artifacts as a Kaggle dataset named "birdclef-eda-artifacts".

## Training notebook
Lives on Kaggle — 3 cells only:
  Cell 1: clone latest code from GitHub
  Cell 2: import modules + verify EDA artifacts are mounted
  Cell 3: call train_fold()

Add these two datasets as input before running:
  - BirdCLEF+ 2026 competition data
  - birdclef-eda-artifacts (from EDA notebook output)

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
| Threshold tuning | owner | — |
| Error analysis by class | owner | — |
| Insect sonotype clustering | owner | — |
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