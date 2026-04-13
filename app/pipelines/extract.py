"""
Pattern extraction pipeline: Qwen VL visual understanding -> prompt generation.
"""

import os
import base64
import asyncio

from dashscope import MultiModalConversation

from app.rate_limiter import aliyun_rate_limiter

_PATTERN_EXTRACT_PROMPT = (
    "Please look at the pattern or design printed/embroidered on this product. "
    "Write a detailed English prompt describing only the pattern/design itself — "
    "do NOT mention the product type, material, or shape. "
    "Focus on: subject, style, colors, composition, texture, and mood of the pattern. "
    "Reply with the prompt text only, no extra explanation."
)


async def extract_pattern_prompt(image_path: str) -> str:
    vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
    await aliyun_rate_limiter.wait(vl_model)
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".").replace("jpg", "jpeg") or "jpeg"
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/{ext};base64,{b64_data}"},
                {"text": _PATTERN_EXTRACT_PROMPT},
            ],
        }
    ]
    rsp = await asyncio.to_thread(MultiModalConversation.call, model=vl_model, messages=messages)
    if rsp.status_code != 200:
        raise Exception(f"Qwen VL 图案提取失败: {rsp.message}")
    for item in rsp.output.choices[0].message.content:
        if "text" in item:
            return item["text"].strip()
    raise Exception("Qwen VL 未返回文字描述")
