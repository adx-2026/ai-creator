"""
Authentication routes: login, logout, current user.
"""

import uuid

from fastapi import APIRouter, Form, Depends
from fastapi.responses import JSONResponse

from app.config import USERS
from app.auth import SESSIONS, get_current_user, oauth2_scheme

router = APIRouter()


@router.post("/api/login")
def login(username: str = Form(...), password: str = Form(...)):
    if USERS.get(username, {}).get("password") == password:
        token = str(uuid.uuid4())
        SESSIONS[token] = username
        return {
            "access_token": token,
            "username": username,
            "is_admin": USERS[username].get("is_admin", False),
        }
    return JSONResponse(status_code=401, content={"error": "Invalid credentials"})


@router.get("/api/me")
def get_me(curr: dict = Depends(get_current_user)):
    return curr


@router.post("/api/logout")
def logout(token: str = Depends(oauth2_scheme)):
    if token in SESSIONS:
        del SESSIONS[token]
    return {"success": True}
