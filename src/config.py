from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RE_DATA_ROOT = DATA_DIR / "red_eclipse"
LOL_DATA_ROOT = DATA_DIR / "lol2"
LOL_DERIVED_ROOT = LOL_DATA_ROOT / "derived"
CSGO_DATA_ROOT = DATA_DIR / "CSGO" / "data"

# Reproducibility seed
RNG_SEED = 42

# Stitch bot (block bootstrap)
SEGMENT_MS = 2000              # legacy center (docs); cutting uses SEGMENT_MS_RANGE
SEGMENT_MS_RANGE = (1500, 2500)  # randomized chunk length (avoids 2s periodicity)
MIN_EVENTS = 10              # min events per segment
TARGET_DURATION_MS = 180000  # ~3 min synthetic game (matches one RE game)

# LoL: match-aligned window (game clock minutes), not file-start
LOL_WINDOW_MIN = 3           # legacy alias (= end - start); kept for old notebooks
LOL_MATCH_WINDOW_START_MIN = 10
LOL_MATCH_WINDOW_END_MIN = 13
LOL_MATCH_MIN_EVENTS = 100   # drop sparse 10–13 windows
LOL_MOUSE_WINDOWS_CACHE_VERSION = "v1"
LOL_TIMESTAMP_PARSE = "ms_rjust_v1"

# CSGO window size (align with ~3 min RE game)
CSGO_WINDOW_MIN = 3          # minutes per CSGO slice (from converted mouse time)

# Segment-level detection (used by main_segment; harmless defaults here)
WINDOW_MS = 10_000
WINDOW_MIN_EVENTS = 30

# VAE bot (fixed-length MLP; assemble via stitch_bot_game)
VAE_SEG_LEN = 64
VAE_Z_DIM = 12
VAE_HIDDEN = 128
VAE_BETA = 0.5
VAE_EPOCHS = 50
VAE_BATCH_SIZE = 256
VAE_LR = 1e-3
