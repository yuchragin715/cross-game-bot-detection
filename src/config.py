from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RE_DATA_ROOT = DATA_DIR / "red_eclipse"
LOL_DATA_ROOT = DATA_DIR / "lol2"

# Reproducibility seed
RNG_SEED = 42

# Stitch bot (block bootstrap)
SEGMENT_MS = 2000              # segment length (ms)
MIN_EVENTS = 10              # min events per segment
TARGET_DURATION_MS = 180000  # ~3 min synthetic game (matches one RE game)

# LoL window size
LOL_WINDOW_MIN = 3           # minutes per LoL slice (matches RE game length)
