from pathlib import Path

# ── paths ──────────────────────────────────────────────────────
BASE_DIR_COMPETITION = Path("/kaggle/input/competitions/birdclef-2026")
AUDIO_DIR            = BASE_DIR_COMPETITION / "train_audio"
TEST_DIR             = BASE_DIR_COMPETITION / "test_soundscapes"
TRAIN_CSV            = BASE_DIR_COMPETITION / "train.csv"
TAXONOMY             = BASE_DIR_COMPETITION / "taxonomy.csv"

# EDA artifacts — produced by data science team EDA notebook
BASE_DIR_ARTIFACT    = Path("/kaggle/input/datasets/haphngngcgia/birdclef-eda-artifacts")

# Model outputs
BASE_DIR_MODELS      = Path("/kaggle/working/birdclef-models")

# ── audio ──────────────────────────────────────────────────────
SAMPLE_RATE = 32_000
DURATION    = 5
HOP_LENGTH  = 512
N_FFT       = 1024
N_MELS      = 128

# ── model ──────────────────────────────────────────────────────
# CRITICAL: always 234 — covers ALL taxonomy species including zero-shot
# Never use len(train_species) or LabelEncoder.classes_ here
N_CLASSES   = 234
BATCH_SIZE  = 16      # Reduced from 1024 to fit in 16GB VRAM
NUM_WORKERS = 2       # Re-enabled for speed (with persistent_workers)
ACCUMULATION_STEPS = 8

MODEL_TYPE = "beats" # Options: "effnet" or "beats"
TARGET_SR  = 16000   # BEATs specifically requires 16kHz

# ── Perch TF Hub ───────────────────────────────────────────────
PERCH_URL   = "https://tfhub.dev/google/bird-vocalization-classifier/4"