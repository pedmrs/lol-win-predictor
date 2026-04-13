from __future__ import annotations

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = SRC_DIR / "model"
MODEL_PATH = MODEL_DIR / "lol_model.pkl"
SELECTED_FEATURES_PATH = MODEL_DIR / "selected_features.json"
