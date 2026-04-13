from __future__ import annotations
from pydantic import BaseModel, ConfigDict

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blueKills: float
    redKills: float
    blueGoldDiff: float
    blueExperienceDiff: float
    blueDragons: float
    redDragons: float
    blueHeralds: float
    redHeralds: float

class PredictionResponse(BaseModel):
    prediction: int
    result: str
