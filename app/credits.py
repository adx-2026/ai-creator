"""
Credit billing system: pricing tables, cost estimation, and deduction.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional

from app.config import CONFIG_FILE, raw_config

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CREDITS_PER_YUAN = 10

# Image generation price per image (CNY)
_IMAGE_PRICE_PER_IMAGE: Dict[str, float] = {
    "qwen-image-2.0-pro": 0.5,
    "qwen-image-2.0-pro-2026-03-03": 0.5,
    "qwen-image-2.0": 0.2,
    "qwen-image-2.0-2026-03-03": 0.2,
    "qwen-image-max": 0.5,
    "qwen-image-max-2025-12-30": 0.5,
    "qwen-image-plus": 0.2,
    "qwen-image-plus-2026-01-09": 0.2,
    "qwen-image": 0.25,
    "qwen-image-edit-max": 0.5,
    "qwen-image-edit-max-2026-01-16": 0.5,
    "qwen-image-edit-plus": 0.2,
    "qwen-image-edit-plus-2025-12-15": 0.2,
    "qwen-image-edit-plus-2025-10-30": 0.2,
    "qwen-image-edit": 0.3,
}
_DEFAULT_IMAGE_PRICE = 0.5

# VL model token price (CNY / million tokens)
_VL_TOKEN_PRICE: Dict[str, Dict[str, float]] = {
    "qwen3-vl-plus": {"input": 1.0, "output": 10.0},
    "qwen3-vl-flash": {"input": 0.15, "output": 1.5},
    "qwen-vl-max": {"input": 3.0, "output": 9.0},
    "qwen-vl-plus": {"input": 1.5, "output": 4.5},
}
_VL_DEFAULT_PRICE = {"input": 1.0, "output": 10.0}
_VL_EST_INPUT_TOKENS = 3000
_VL_EST_OUTPUT_TOKENS = 500

# Wan video price per second (CNY)
_VIDEO_PRICE_PER_SECOND: Dict[str, float] = {
    "wan2.6-t2v": 0.6,
    "wan2.6-t2v-us": 0.733924,
    "wan2.5-t2v-preview": 0.6,
    "wan2.2-t2v-plus": 0.14,
    "wanx2.1-t2v-turbo": 0.24,
    "wanx2.1-t2v-plus": 0.70,
    "wan2.6-i2v-flash": 0.3,
    "wan2.6-i2v-plus": 0.6,
    "wan2.6-i2v-turbo": 0.3,
    "wanx2.1-i2v-turbo": 0.24,
    "wanx2.1-i2v-plus": 0.70,
}
_DEFAULT_VIDEO_PRICE_PER_SECOND = 0.5

# Text model token price (CNY / million tokens)
_TEXT_TOKEN_PRICE: Dict[str, Dict[str, float]] = {
    "qwen3-235b-a22b": {"input": 1.0, "output": 8.0},
    "qwen3-30b-a3b": {"input": 0.22, "output": 0.88},
    "qwen3-8b": {"input": 0.05, "output": 0.2},
}
_TEXT_DEFAULT_PRICE = {"input": 1.0, "output": 8.0}
_TEXT_EST_INPUT_TOKENS = 400
_TEXT_EST_OUTPUT_TOKENS = 400

_credit_lock = asyncio.Lock()

# ------------------------------------------------------------------
# Cost calculation helpers
# ------------------------------------------------------------------

def get_image_credit_per_image(model_id: str) -> float:
    price = _IMAGE_PRICE_PER_IMAGE.get(model_id, _DEFAULT_IMAGE_PRICE)
    return round(price * CREDITS_PER_YUAN, 4)


def get_vl_credit_per_call(model_id: str) -> float:
    prices = _VL_TOKEN_PRICE.get(model_id, _VL_DEFAULT_PRICE)
    cost_yuan = (
        _VL_EST_INPUT_TOKENS / 1_000_000 * prices["input"]
        + _VL_EST_OUTPUT_TOKENS / 1_000_000 * prices["output"]
    )
    return round(cost_yuan * CREDITS_PER_YUAN, 4)


def get_text_credit_per_call(model_id: str) -> float:
    prices = _TEXT_TOKEN_PRICE.get(model_id, _TEXT_DEFAULT_PRICE)
    cost_yuan = (
        _TEXT_EST_INPUT_TOKENS / 1_000_000 * prices["input"]
        + _TEXT_EST_OUTPUT_TOKENS / 1_000_000 * prices["output"]
    )
    return round(cost_yuan * CREDITS_PER_YUAN, 4)


def get_video_credit_per_second(model_id: str) -> float:
    price = _VIDEO_PRICE_PER_SECOND.get(model_id, _DEFAULT_VIDEO_PRICE_PER_SECOND)
    return round(price * CREDITS_PER_YUAN, 4)


def estimate_job_credits(
    model_id: str,
    mode: str,
    subtasks: List[Dict],
    video_params: Optional[Dict] = None,
    batch_size: int = 1,
) -> float:
    if not is_aliyun_model(model_id):
        return 0.0
    if mode in ("video", "multi_video"):
        duration = int((video_params or {}).get("duration", 5))
        return round(get_video_credit_per_second(model_id) * duration * len(subtasks), 4)
    if mode == "extract":
        vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
        vl_credit = get_vl_credit_per_call(vl_model)
        img_credit = get_image_credit_per_image(model_id)
        return round(len(subtasks) * (vl_credit + batch_size * img_credit), 4)
    if mode == "ecommerce":
        return round(len(subtasks) * get_image_credit_per_image(model_id), 4)
    return round(len(subtasks) * get_image_credit_per_image(model_id), 4)


def get_user_credit(username: str) -> float:
    return raw_config.get("users", {}).get(username, {}).get("credit", 0.0)


async def deduct_credits(username: str, amount: float):
    if amount <= 0:
        return
    async with _credit_lock:
        users = raw_config.setdefault("users", {})
        user_cfg = users.setdefault(username, {})
        user_cfg["credit"] = round(user_cfg.get("credit", 0) - amount, 4)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(raw_config, f, ensure_ascii=False, indent=4)


def is_aliyun_model(model_id: str) -> bool:
    return model_id.startswith("qwen") or model_id.startswith("wan") or model_id.startswith("qwen3-vl")
