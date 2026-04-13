"""
User authentication: session management and dependency injection.
"""

from fastapi import Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer

from app.config import USERS

SESSIONS: dict = {}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login", auto_error=False)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    query_token: str = Query(None, alias="token"),
):
    actual_token = token or query_token
    if not actual_token or actual_token not in SESSIONS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_info = USERS.get(SESSIONS[actual_token], {})
    return {
        "username": SESSIONS[actual_token],
        "is_admin": user_info.get("is_admin", False),
    }
