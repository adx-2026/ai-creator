"""
Size annotation pipeline:
  Step 1: Qwen VL identifies product outline/contours
  Step 2: Qwen image-edit overlays measurement lines, arrows, and text
"""

import os
import uuid
import base64
import asyncio

import requests
from dashscope import MultiModalConversation

from app.rate_limiter import aliyun_rate_limiter

_VL_OUTLINE_PROMPT = (
    "Briefly describe the main product shape and key visual boundaries/edges "
    "in this image. Focus on the overall shape (corners, curves, proportions). "
    "Keep it under 60 words. English only."
)


async def understand_product_outline(image_path: str) -> str:
    """Call Qwen VL to get a brief contour description of the product."""
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
                {"text": _VL_OUTLINE_PROMPT},
            ],
        }
    ]
    rsp = await asyncio.to_thread(MultiModalConversation.call, model=vl_model, messages=messages)
    if rsp.status_code != 200:
        return ""
    for item in rsp.output.choices[0].message.content:
        if "text" in item:
            return item["text"].strip()
    return ""


def _build_annotation_prompt(shape: str, dimensions: dict, outline: str) -> str:
    """Build the image-edit instruction prompt for size annotation."""
    if shape == "circle":
        d = dimensions.get("diameter", "")
        dim_text = f"diameter {d} cm" if d else "diameter as shown"
        shape_desc = "circular"
    else:
        parts = []
        if dimensions.get("length"):
            parts.append(f"length {dimensions['length']} cm")
        if dimensions.get("width"):
            parts.append(f"width {dimensions['width']} cm")
        if dimensions.get("height"):
            parts.append(f"height {dimensions['height']} cm")
        dim_text = ", ".join(parts) if parts else "dimensions as shown"
        shape_desc = "rectangular"

    prompt = (
        f"Add professional product dimension annotation lines to this product image. "
        f"The product is {shape_desc}. "
        f"Dimensions to annotate: {dim_text}. "
        f"Instructions: draw thin straight dimension lines with small arrowheads at both ends, "
        f"positioned along the product edges. Place the numeric dimension value (e.g. '25cm') "
        f"centered on each dimension line. Use clean dark lines on a light/white area, or light "
        f"lines on dark areas, so they contrast clearly. Keep the original product fully visible "
        f"and undistorted. Style: clean technical product sheet annotation."
    )
    if outline:
        prompt += f" Product shape reference: {outline}."
    return prompt


async def generate_size_annotation(
    image_path: str,
    shape: str,
    dimensions: dict,
    model_id: str,
    output_size: str,
    user_dir: str,
    user: str,
) -> tuple:
    """
    Full pipeline: VL outline → image-edit annotation.
    Returns (result_path, display_url).
    """
    outline = await understand_product_outline(image_path)
    edit_prompt = _build_annotation_prompt(shape, dimensions, outline)

    await aliyun_rate_limiter.wait(model_id)
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().replace(".", "").replace("jpg", "jpeg") or "jpeg"

    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/{ext};base64,{b64_data}"},
                {"text": edit_prompt},
            ],
        }
    ]
    kwargs = {
        "model": model_id,
        "messages": messages,
        "n": 1,
        "size": output_size,
    }
    rsp = await asyncio.to_thread(MultiModalConversation.call, **kwargs)
    if rsp.status_code != 200:
        raise Exception(f"Qwen image-edit 失败: {rsp.message}")

    for item in rsp.output.choices[0].message.content:
        if "image" in item:
            img_data = await asyncio.to_thread(requests.get, item["image"])
            filename = f"annot_{uuid.uuid4().hex[:8]}.png"
            out_path = os.path.join(user_dir, filename)
            with open(out_path, "wb") as f:
                f.write(img_data.content)
            return out_path, f"/api/images/{user}/{filename}"

    raise Exception("Qwen image-edit 未返回标注图片")
