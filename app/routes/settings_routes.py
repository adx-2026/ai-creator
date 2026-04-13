"""
User settings routes.
"""

from fastapi import APIRouter, Request, Depends

from app.auth import get_current_user
from app.settings import load_user_settings, save_user_settings_sync

router = APIRouter()


@router.get("/api/settings")
def get_settings(curr: dict = Depends(get_current_user)):
    return load_user_settings(curr["username"])


@router.post("/api/settings")
async def update_settings(request: Request, curr: dict = Depends(get_current_user)):
    data = await request.json()
    settings = load_user_settings(curr["username"])
    settings.update(data)
    save_user_settings_sync(curr["username"], settings)
    return settings
