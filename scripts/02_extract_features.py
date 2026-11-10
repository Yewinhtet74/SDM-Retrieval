"""Step 2: Extract and cache frozen CLIP 512-dimensional features (★ foundation ★)."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.features import extract_and_cache


if __name__ == "__main__":
    extract_and_cache(load_config())