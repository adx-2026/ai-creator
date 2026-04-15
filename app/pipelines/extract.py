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

_TITLE_GEN_PROMPT_TEMPLATE = (
    "你是电商产品标题生成专家。请仔细识别这张图片中的主体内容（动物、植物、物品、人物、场景等）。"
    "然后基于以下关键词模板，将图片中识别到的内容融入模板中，生成一个完整的电商产品标题。\n"
    "关键词模板：{template}\n"
    "要求：\n"
    "1. 直接输出最终标题文本\n"
    "2. 不要任何解释或前缀\n"
    "3. 标题应自然流畅，突出产品卖点\n"
    "4. 将模板中的占位内容替换为图片实际识别到的内容"
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


async def generate_product_title(image_path: str, keyword_template: str) -> str:
    """Analyze a generated image with VL and produce a product title using the keyword template."""
    vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
    await aliyun_rate_limiter.wait(vl_model)
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".").replace("jpg", "jpeg") or "jpeg"
    prompt = _TITLE_GEN_PROMPT_TEMPLATE.format(template=keyword_template)
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/{ext};base64,{b64_data}"},
                {"text": prompt},
            ],
        }
    ]
    rsp = await asyncio.to_thread(MultiModalConversation.call, model=vl_model, messages=messages)
    if rsp.status_code != 200:
        raise Exception(f"Qwen VL 标题生成失败: {rsp.message}")
    for item in rsp.output.choices[0].message.content:
        if "text" in item:
            return item["text"].strip()
    raise Exception("Qwen VL 未返回标题文本")
