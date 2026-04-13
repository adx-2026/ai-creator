"""
Model listing route.
"""

from fastapi import APIRouter, Depends

from app.config import AVAILABLE_MODELS
from app.auth import get_current_user

router = APIRouter()


@router.get("/api/models")
def get_models(curr: dict = Depends(get_current_user)):
    return AVAILABLE_MODELS
