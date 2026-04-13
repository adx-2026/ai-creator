"""
Prompt template CRUD routes.
"""

import os
import json

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from app.config import PUBLIC_TEMPLATES_FILE
from app.auth import get_current_user

router = APIRouter()


@router.get("/api/templates")
def get_templates(curr: dict = Depends(get_current_user)):
    pub = json.load(open(PUBLIC_TEMPLATES_FILE, "r", encoding="utf-8")) if os.path.exists(PUBLIC_TEMPLATES_FILE) else []
    priv_path = f"users/{curr['username']}/templates.json"
    priv = json.load(open(priv_path, "r", encoding="utf-8")) if os.path.exists(priv_path) else []
    return {"public": pub, "private": priv}


@router.post("/api/templates")
async def save_template(request: Request, curr: dict = Depends(get_current_user)):
    data = await request.json()
    user, is_pub = curr["username"], data.get("is_public", False)
    target = PUBLIC_TEMPLATES_FILE if is_pub else f"users/{user}/templates.json"
    items = json.load(open(target, "r", encoding="utf-8")) if os.path.exists(target) else []
    new_item = {
        "name": data["name"],
        "content": data["content"],
        "negative_prompt": data.get("negative_prompt", ""),
        "author": user,
    }
    idx = next((i for i, x in enumerate(items) if x["name"] == data["name"]), -1)
    if idx >= 0:
        if is_pub and items[idx].get("author") != user and not curr["is_admin"]:
            return JSONResponse(status_code=403, content={"error": "无权修改他人的模板"})
        items[idx] = new_item
    else:
        items.append(new_item)
    json.dump(items, open(target, "w", encoding="utf-8"), ensure_ascii=False)
    return {"success": True}


@router.delete("/api/templates/{scope}/{name}")
def delete_template(scope: str, name: str, curr: dict = Depends(get_current_user)):
    user = curr["username"]
    target = PUBLIC_TEMPLATES_FILE if scope == "public" else f"users/{curr['username']}/templates.json"
    if not os.path.exists(target):
        return {"success": True}
    items = json.load(open(target, "r", encoding="utf-8"))
    idx = next((i for i, x in enumerate(items) if x["name"] == name), -1)
    if idx >= 0:
        if scope == "public" and items[idx].get("author") != user and not curr["is_admin"]:
            return JSONResponse(status_code=403, content={"error": "无权删除"})
        items.pop(idx)
        json.dump(items, open(target, "w", encoding="utf-8"), ensure_ascii=False)
    return {"success": True}
