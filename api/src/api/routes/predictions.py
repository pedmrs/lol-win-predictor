from __future__ import annotations
from fastapi import APIRouter, HTTPException
from src.api.schemas import PredictionRequest, PredictionResponse
from src.core.state import app_state
from src.ml.predictor import predict_winner

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = app_state.get("model")
    selected_features = app_state.get("selected_features")
    if model is None or selected_features is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    prediction = predict_winner(
        model=model,
        selected_features=selected_features,
        payload=request.model_dump(),
    )
    return PredictionResponse(**prediction)
