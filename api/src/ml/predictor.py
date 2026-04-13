from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from sklearn.base import BaseEstimator

def load_selected_features(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))

def load_model(path: Path) -> BaseEstimator:
    return joblib.load(path)

def predict_winner(
    model: BaseEstimator,
    selected_features: list[str],
    payload: dict[str, Any],
) -> dict[str, int | str]:
    feature_frame = pd.DataFrame([{feature: payload[feature] for feature in selected_features}])
    prediction = int(model.predict(feature_frame)[0])
    result = "Blue team likely wins" if prediction == 1 else "Red team likely wins"
    return {"prediction": prediction, "result": result}
