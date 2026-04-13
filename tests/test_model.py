from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "high_diamond_ranked_10min.csv"
MODEL_PATH = PROJECT_ROOT / "ml" / "artifacts" / "lol_model.pkl"
SELECTED_FEATURES_PATH = PROJECT_ROOT / "ml" / "artifacts" / "selected_features.json"
TARGET_COLUMN = "blueWins"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MINIMUM_ACCURACY = 0.70

def test_exported_model_accuracy_is_deployable() -> None:
    model = joblib.load(MODEL_PATH)
    selected_features = json.loads(SELECTED_FEATURES_PATH.read_text(encoding="utf-8"))
    data = pd.read_csv(DATA_PATH)

    x = data[selected_features]
    y = data[TARGET_COLUMN]
    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= MINIMUM_ACCURACY, (
        f"Model accuracy {accuracy:.3f} is below the required "
        f"{MINIMUM_ACCURACY:.2f} threshold."
    )
