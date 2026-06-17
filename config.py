"""Central configuration — reads from environment variables, falls back to project-relative defaults."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent)).resolve()
DATA_DIR     = Path(os.environ.get("DATA_DIR",     PROJECT_ROOT / "data")).resolve()
RESULTS_DIR  = Path(os.environ.get("RESULTS_DIR",  PROJECT_ROOT / "results")).resolve()

SEED = 42