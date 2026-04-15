"""
Product dimension annotation pipeline:
  Step 1: Qwen VL analyzes product position and shape
  Step 2: Build engineering annotation prompt with user-provided dimensions
  Step 3: qwen-image-edit overlays dimension lines on the product image
"""

import os
import uuid
import base64
import asyncio
from typing import Optional

import requests
from dashscope import MultiModalConversation

from app.rate_limiter import aliyun_rate_limiter


_VL_BBOX_PROMPT = (
    "Analyze this product image. Briefly describe in 2-3 sentences: "
    "1) the product type and main shape (e.g., rectangular box, cylinder, bottle), "
    "2) which side is the length (longest horizontal span), which is the height (vertical span), "
    "and whether depth is visible, "
    "3) if the product is circular/cylindrical, confirm that and note its proportions. "
    "Focus on spatial layout information that would help place engineering dimension annotation lines accurately. "
    "Reply in English only."
)


def _build_annotation_prompt(
    product_analysis: str,
    shape: str,
    length: Optional[float],
    width: Optional[float],
    height: Optional[float],
    diameter: Optional[float],
) -> str:
    """Build the image-edit prompt for adding engineering dimension annotations."""
    if shape == "circle":
        dims_parts = []
        if diameter:
            dims_parts.append(f"Diameter: {diameter}cm")
        dims_str = ", ".join(dims_parts) if dims_parts else "diameter (estimate from image)"
    else:
        dims_parts = []
        if length:
            dims_parts.append(f"Length: {length}cm")
        if width:
            dims_parts.append(f"Width: {width}cm")
        if height:
            dims_parts.append(f"Height: {height}cm")
        dims_str = ", ".join(dims_parts) if dims_parts else "dimensions (estimate from image)"

    return (
        f"Add professional engineering-style dimension annotations to this product image. "
        f"Product context: {product_analysis} "
        f"Draw precise dimension lines with double-headed arrows clearly showing: {dims_str}. "
        f"Place readable dimension labels (e.g. '10cm') next to each arrow line. "
        f"Use thin black lines with arrowheads and short extension lines from the product edges. "
        f"Place horizontal dimensions above or below the product; vertical dimensions to the left or right. "
        f"Follow standard engineering technical drawing conventions. "
        f"Keep the original product image completely unchanged — only add the annotation overlay on top."
    )


async def analyze_product_bbox(image_path: str) -> str:
    """Call Qwen VL to analyze product shape and position for dimension annotation."""
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
                {"text": _VL_BBOX_PROMPT},
            ],
        }
    ]
    rsp = await asyncio.to_thread(MultiModalConversation.call, model=vl_model, messages=messages)
    if rsp.status_code != 200:
        raise Exception(f"Qwen VL 产品识别失败: {rsp.message}")
    for item in rsp.output.choices[0].message.content:
        if "text" in item:
            return item["text"].strip()
    return "a product on a white background"


async def annotate_with_dimensions(
    image_path: str,
    edit_model: str,
    prompt: str,
    user_dir: str,
    size: str,
) -> str:
    """Call qwen-image-edit to overlay dimension annotations and save result."""
    await aliyun_rate_limiter.wait(edit_model)
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().replace(".", "").replace("jpg", "jpeg") or "jpeg"

    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/{ext};base64,{b64_data}"},
                {"text": prompt},
            ],
        }
    ]
    rsp = await asyncio.to_thread(
        MultiModalConversation.call,
        model=edit_model,
        messages=messages,
        n=1,
        size=size,
    )
    if rsp.status_code != 200:
        raise Exception(f"qwen-image-edit 标注图生成失败: {rsp.message}")

    for item in rsp.output.choices[0].message.content:
        if "image" in item:
            img_data = await asyncio.to_thread(requests.get, item["image"])
            filename = f"dim_{uuid.uuid4().hex[:8]}.png"
            out_path = os.path.join(user_dir, filename)
            with open(out_path, "wb") as f:
                f.write(img_data.content)
            return filename

    raise Exception("qwen-image-edit 未返回标注图片")
