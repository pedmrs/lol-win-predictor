from __future__ import annotations
from fastapi.testclient import TestClient
from src.core.state import app_state
from src.main import app

VALID_PREDICTION_PAYLOAD = {
    "blueKills": 6,
    "redKills": 4,
    "blueGoldDiff": 1200,
    "blueExperienceDiff": 900,
    "blueDragons": 1,
    "redDragons": 0,
    "blueHeralds": 1,
    "redHeralds": 0,
}

def test_health_endpoint_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint_returns_prediction() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PREDICTION_PAYLOAD)

    body = response.json()
    assert response.status_code == 200
    assert body["prediction"] in [0, 1]
    assert body["result"] in ["Blue team likely wins", "Red team likely wins"]

def test_predict_endpoint_rejects_invalid_payload() -> None:
    payload = {**VALID_PREDICTION_PAYLOAD, "unexpected": 123}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_predict_endpoint_returns_503_when_model_is_not_loaded() -> None:
    with TestClient(app) as client:
        loaded_state = app_state.copy()
        app_state.clear()
        try:
            response = client.post("/predict", json=VALID_PREDICTION_PAYLOAD)
        finally:
            app_state.update(loaded_state)

    assert response.status_code == 503
    assert response.json() == {"detail": "Model is not loaded."}
