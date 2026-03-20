from pathlib import Path

# ── paths ──────────────────────────────────────────────────
BASE_DIR  = Path("/kaggle/input/competitions/birdclef-2026")
AUDIO_DIR = BASE_DIR / "train_audio"
TEST_DIR  = BASE_DIR / "test_soundscapes"
TRAIN_CSV = BASE_DIR / "train.csv"
TAXONOMY  = BASE_DIR / "taxonomy.csv"

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