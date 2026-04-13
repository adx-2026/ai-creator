"""
3D embroidery conversion pipeline:
  Step 1: Qwen VL pattern understanding
  Step 2: Generate embroidery prompt from description
"""

import os
import base64
import asyncio

from dashscope import MultiModalConversation

from app.rate_limiter import aliyun_rate_limiter

_THREED_UNDERSTAND_PROMPT = (
    "Analyze this image and describe the pattern, design, or graphic content visible in it. "
    "Provide a concise, detailed English description of: the main subject or motif, colors, artistic style, "
    "shapes, compositional elements, and any notable textures or visual details. "
    "Focus on what would help recreate this as an embroidery design. "
    "Reply with the description only, no extra explanation."
)

THREED_PROMPT_TEMPLATE = (
    "A highly detailed realistic embroidery of {description}, stitched with colorful threads on a clean, "
    "plain knitted/woven fabric background. The design features intricate thread work with visible stitch patterns, "
    "3D raised embroidery effect, satin stitch and fill stitch techniques, precise needlework details. "
    "The embroidery faithfully reproduces the original flat illustration with textured yarn and thread, "
    "showing realistic fiber texture and slight dimensional relief. Clean and minimal background with no other objects, "
    "no clutter, no wrinkles, no stains. Centered composition, natural soft lighting, close-up macro photography style, studio shot."
)


async def understand_threed_pattern(image_path: str) -> str:
    """Call Qwen VL to analyze image pattern content and return an English description."""
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
                {"text": _THREED_UNDERSTAND_PROMPT},
            ],
        }
    ]
    rsp = await asyncio.to_thread(MultiModalConversation.call, model=vl_model, messages=messages)
    if rsp.status_code != 200:
        raise Exception(f"Qwen VL 图案分析失败: {rsp.message}")
    for item in rsp.output.choices[0].message.content:
        if "text" in item:
            return item["text"].strip()
    raise Exception("Qwen VL 未返回图案描述")
