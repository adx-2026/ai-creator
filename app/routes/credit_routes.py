"""
Credit management routes.
"""

import json

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from app.config import CONFIG_FILE, raw_config
from app.auth import get_current_user
from app.credits import get_user_credit, _credit_lock

router = APIRouter()


@router.get("/api/credit")
def get_credit(curr: dict = Depends(get_current_user)):
    return {"credit": get_user_credit(curr["username"])}


@router.post("/api/credit")
async def set_credit(request: Request, curr: dict = Depends(get_current_user)):
    if not curr.get("is_admin"):
        return JSONResponse(status_code=403, content={"error": "仅管理员可操作"})
    data = await request.json()
    target_user = data.get("username", curr["username"])
    amount = float(data.get("credit", 0))
    async with _credit_lock:
        raw_config.setdefault("users", {}).setdefault(target_user, {})["credit"] = round(amount, 4)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(raw_config, f, ensure_ascii=False, indent=4)
    return {"username": target_user, "credit": round(amount, 4)}
