from __future__ import annotations

from fastapi import APIRouter

from src.api.routes.health import router as health_router
from src.api.routes.predictions import router as predictions_router

router = APIRouter()
router.include_router(health_router)
router.include_router(predictions_router)
