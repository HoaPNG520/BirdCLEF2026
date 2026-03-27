from pathlib import Path

# ── paths ──────────────────────────────────────────────────
BASE_DIR_COMPETITION  = Path("/kaggle/input/competitions/birdclef-2026")
AUDIO_DIR = BASE_DIR_COMPETITION / "train_audio"
TEST_DIR  = BASE_DIR_COMPETITION / "test_soundscapes"
TRAIN_CSV = BASE_DIR_COMPETITION / "train.csv"
TAXONOMY  = BASE_DIR_COMPETITION / "taxonomy.csv"

BASE_DIR_ARTIFACT = Path("/kaggle/input/datasets/haphngngcgia/birdclef-eda-artifacts")

# ── audio ──────────────────────────────────────────────────
SAMPLE_RATE = 32_000
DURATION    = 5
HOP_LENGTH  = 512
N_FFT       = 1024
N_MELS      = 128

# ── model ──────────────────────────────────────────────────
N_CLASSES   = 234
BATCH_SIZE  = 32
NUM_WORKERS = 2