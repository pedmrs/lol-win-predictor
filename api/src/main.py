from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.core.config import MODEL_PATH, SELECTED_FEATURES_PATH
from src.core.state import app_state
from src.ml.predictor import load_model, load_selected_features

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if not MODEL_PATH.exists() or not SELECTED_FEATURES_PATH.exists():
        raise RuntimeError(
            "Model artifacts are missing. Run `uv run python ../ml/train_model.py` "
            "from the api directory first."
        )
    app_state["model"] = load_model(MODEL_PATH)
    app_state["selected_features"] = load_selected_features(SELECTED_FEATURES_PATH)
    try:
        yield
    finally:
        app_state.clear()


app = FastAPI(title="League of Legends Win Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
