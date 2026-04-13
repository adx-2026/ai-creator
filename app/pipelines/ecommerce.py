"""
E-commerce scene generation pipeline:
  Step 1: Qwen VL product understanding
  Step 2: Qwen3 text model scene prompt generation
"""

import os
import re
import json
import base64
import asyncio
from typing import List

from dashscope import MultiModalConversation, Generation

from app.rate_limiter import aliyun_rate_limiter

# ------------------------------------------------------------------
# Step 1: Product understanding
# ------------------------------------------------------------------
_PRODUCT_UNDERSTAND_PROMPT = (
    "Analyze this product image for an e-commerce listing. "
    "Provide a detailed English description including: product category and type, "
    "primary colors and materials/textures, shape and key visual characteristics, "
    "notable design elements or features, any visible branding or text. "
    "Focus ONLY on the product itself — do not describe the background or setting. "
    "Reply with the description only, no extra explanation."
)


async def understand_product(image_path: str) -> str:
    """Call Qwen VL to analyze a product image and return an English description."""
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
                {"text": _PRODUCT_UNDERSTAND_PROMPT},
            ],
        }
    ]
    rsp = await asyncio.to_thread(MultiModalConversation.call, model=vl_model, messages=messages)
    if rsp.status_code != 200:
        raise Exception(f"Qwen VL 产品理解失败: {rsp.message}")
    for item in rsp.output.choices[0].message.content:
        if "text" in item:
            return item["text"].strip()
    raise Exception("Qwen VL 未返回产品描述")


# ------------------------------------------------------------------
# Step 2: Scene prompt generation
# ------------------------------------------------------------------
_SCENE_GENERATE_SYSTEM_PROMPT = (
    "You are an expert e-commerce product photographer and creative director. "
    "Generate diverse, practical photography scene descriptions for product listings. "
    "Each scene should be realistic, enhance the product's visual appeal, and suit different buyer contexts."
)


def _build_scene_prompt(product_description: str, scene_count: int) -> str:
    return (
        f"Based on this product description, generate {scene_count} distinct e-commerce photography scene prompts.\n\n"
        f"Product: {product_description}\n\n"
        f"Requirements:\n"
        f"- Cover different visual contexts: pure studio shot, lifestyle scene, flat lay, seasonal/themed, outdoor, etc.\n"
        f"- Each scene must naturally showcase the product as the hero subject\n"
        f"- Vary lighting styles, backgrounds, and compositions\n"
        f"- Keep each prompt concise (40-80 words) and ready to use for image generation\n\n"
        f"Output a valid JSON array of exactly {scene_count} English strings.\n"
        f"Format: [\"scene 1 prompt\", \"scene 2 prompt\", ...]\n"
        f"Output the JSON array only, no other text."
    )


def _parse_scenes_from_response(raw: str, expected_count: int) -> List[str]:
    """Parse scene prompts from model response (JSON array or fallback line split)."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    match = re.search(r"\[[\s\S]*?\]", cleaned)
    if match:
        try:
            scenes = json.loads(match.group())
            if isinstance(scenes, list):
                return [str(s).strip() for s in scenes if str(s).strip()][:expected_count]
        except json.JSONDecodeError:
            pass
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    scenes = []
    for line in lines:
        cleaned_line = re.sub(r"^[\d\.\-\*\s]+", "", line).strip("\"'")
        if cleaned_line and len(cleaned_line) > 10:
            scenes.append(cleaned_line)
    return scenes[:expected_count]


async def call_qwen_text_model(prompt: str, system_prompt: str = "") -> str:
    """Call Qwen3 text model and return the text response."""
    model_id = os.getenv("QWEN_TEXT_MODEL", "qwen3-235b-a22b")
    await aliyun_rate_limiter.wait(model_id)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    rsp = await asyncio.to_thread(
        Generation.call,
        model=model_id,
        messages=messages,
        result_format="message",
        enable_thinking=False,
    )
    if rsp.status_code != 200:
        raise Exception(f"Qwen 文本模型调用失败: {rsp.message}")
    return rsp.output.choices[0].message.content.strip()


async def generate_scenes(product_description: str, scene_count: int) -> List[str]:
    """Generate e-commerce scene prompts from a product description."""
    scene_prompt = _build_scene_prompt(product_description, scene_count)
    raw = await call_qwen_text_model(scene_prompt, _SCENE_GENERATE_SYSTEM_PROMPT)
    return _parse_scenes_from_response(raw, scene_count)
