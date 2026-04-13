"""
File serving routes: images and videos.
"""

import os

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, FileResponse

from app.auth import get_current_user

router = APIRouter()


@router.get("/api/images/{username}/{filename}")
async def serve_img(username: str, filename: str, curr: dict = Depends(get_current_user)):
    if curr["username"] != username and not curr["is_admin"]:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    path = os.path.join(f"users/{username}/outputs", filename)
    return FileResponse(path) if os.path.exists(path) else JSONResponse(status_code=404, content={"error": "Not found"})


@router.get("/api/videos/{username}/{filename}")
async def serve_video(username: str, filename: str, curr: dict = Depends(get_current_user)):
    if curr["username"] != username and not curr["is_admin"]:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    path = os.path.join(f"users/{username}/outputs", filename)
    return (
        FileResponse(path, media_type="video/mp4")
        if os.path.exists(path)
        else JSONResponse(status_code=404, content={"error": "Not found"})
    )
