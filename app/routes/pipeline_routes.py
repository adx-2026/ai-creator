"""
Pipeline routes: e-commerce product understanding/scene generation, 3D conversion,
and product dimension annotation.
"""

import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, Request, Depends
from fastapi.responses import JSONResponse

from app.auth import get_current_user
from app.credits import get_vl_credit_per_call, get_text_credit_per_call, get_image_credit_per_image, deduct_credits
from app.pipelines.ecommerce import understand_product, generate_scenes
from app.pipelines.threed import understand_threed_pattern
from app.pipelines.dimension import analyze_product_bbox, annotate_with_dimensions, _build_annotation_prompt

router = APIRouter()


@router.post("/api/ecommerce/understand")
async def ecommerce_understand(
    images: Optional[List[UploadFile]] = File(None),
    curr: dict = Depends(get_current_user),
):
    """Step 1: Qwen VL analyzes each product image, returns product description."""
    user = curr["username"]
    vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
    user_dir = f"users/{user}/outputs"
    os.makedirs(user_dir, exist_ok=True)

    if not images:
        return JSONResponse(status_code=400, content={"error": "请上传至少一张产品图片"})

    items = []
    for img in images:
        if not img or not getattr(img, "filename", None):
            continue
        fname = f"{uuid.uuid4().hex[:8]}_{img.filename}"
        path = os.path.join(user_dir, fname)
        img_data = await img.read()
        with open(path, "wb") as f:
            f.write(img_data)

        display_url = f"/api/images/{user}/{fname}"
        try:
            description = await understand_product(path)
            credit_cost = get_vl_credit_per_call(vl_model)
            await deduct_credits(user, credit_cost)
            items.append({
                "image_path": path,
                "image_name": img.filename,
                "display_url": display_url,
                "description": description,
                "credit_cost": credit_cost,
            })
        except Exception as e:
            items.append({
                "image_path": path,
                "image_name": img.filename,
                "display_url": display_url,
                "description": "",
                "error": str(e),
            })

    return {"items": items}


@router.post("/api/ecommerce/scenes")
async def ecommerce_scenes(request: Request, curr: dict = Depends(get_current_user)):
    """Step 2: Qwen3 text model generates scene prompts for each product description."""
    user = curr["username"]
    data = await request.json()
    items_in = data.get("items", [])
    scene_count = max(1, min(int(data.get("scene_count", 3)), 20))
    text_model = os.getenv("QWEN_TEXT_MODEL", "qwen3-235b-a22b")

    items_out = []
    for item in items_in:
        desc = (item.get("description") or "").strip()
        image_path = item.get("image_path", "")
        image_name = item.get("image_name", "")
        display_url = item.get("display_url", "")
        if not desc:
            items_out.append({
                "image_path": image_path, "image_name": image_name,
                "display_url": display_url, "description": desc,
                "scenes": [], "error": "产品描述为空，无法生成场景",
            })
            continue
        try:
            scenes = await generate_scenes(desc, scene_count)
            credit_cost = get_text_credit_per_call(text_model)
            await deduct_credits(user, credit_cost)
            items_out.append({
                "image_path": image_path, "image_name": image_name,
                "display_url": display_url, "description": desc,
                "scenes": scenes, "credit_cost": credit_cost,
            })
        except Exception as e:
            items_out.append({
                "image_path": image_path, "image_name": image_name,
                "display_url": display_url, "description": desc,
                "scenes": [], "error": str(e),
            })

    return {"items": items_out}


@router.post("/api/dimension/annotate")
async def dimension_annotate(
    image: UploadFile = File(...),
    shape: str = Form("rectangle"),
    length: Optional[float] = Form(None),
    width: Optional[float] = Form(None),
    height: Optional[float] = Form(None),
    diameter: Optional[float] = Form(None),
    output_width: int = Form(1024),
    output_height: int = Form(1024),
    edit_model: str = Form("qwen-image-edit-plus"),
    curr: dict = Depends(get_current_user),
):
    """
    Product dimension annotation pipeline:
      1. Qwen VL identifies product shape and bounding area.
      2. Build engineering annotation prompt with user-provided dimension values.
      3. qwen-image-edit overlays dimension lines and labels onto the product image.
    """
    user = curr["username"]
    vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
    user_dir = f"users/{user}/outputs"
    os.makedirs(user_dir, exist_ok=True)

    if not image or not getattr(image, "filename", None):
        return JSONResponse(status_code=400, content={"error": "请上传产品图片"})

    fname = f"{uuid.uuid4().hex[:8]}_{image.filename}"
    path = os.path.join(user_dir, fname)
    img_bytes = await image.read()
    with open(path, "wb") as f:
        f.write(img_bytes)

    source_url = f"/api/images/{user}/{fname}"

    # Step 1: Qwen VL — identify product shape and spatial layout
    try:
        product_analysis = await analyze_product_bbox(path)
        await deduct_credits(user, get_vl_credit_per_call(vl_model))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"AI 产品识别失败: {str(e)}"})

    # Step 2: Build annotation prompt
    prompt = _build_annotation_prompt(
        product_analysis=product_analysis,
        shape=shape,
        length=length,
        width=width,
        height=height,
        diameter=diameter,
    )

    # Step 3: qwen-image-edit — generate annotated image
    size_str = f"{output_width}*{output_height}"
    try:
        result_filename = await annotate_with_dimensions(
            image_path=path,
            edit_model=edit_model,
            prompt=prompt,
            user_dir=user_dir,
            size=size_str,
        )
        await deduct_credits(user, get_image_credit_per_image(edit_model))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"尺寸图生成失败: {str(e)}"})

    return {
        "source_url": source_url,
        "result_url": f"/api/images/{user}/{result_filename}",
        "product_analysis": product_analysis,
    }


@router.post("/api/threed/understand")
async def threed_understand(
    images: Optional[List[UploadFile]] = File(None),
    curr: dict = Depends(get_current_user),
):
    """Step 1: Qwen VL analyzes each image pattern, returns pattern description."""
    user = curr["username"]
    vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
    user_dir = f"users/{user}/outputs"
    os.makedirs(user_dir, exist_ok=True)

    if not images:
        return JSONResponse(status_code=400, content={"error": "请上传至少一张图片"})

    items = []
    for img in images:
        if not img or not getattr(img, "filename", None):
            continue
        fname = f"{uuid.uuid4().hex[:8]}_{img.filename}"
        path = os.path.join(user_dir, fname)
        img_data = await img.read()
        with open(path, "wb") as f:
            f.write(img_data)

        display_url = f"/api/images/{user}/{fname}"
        try:
            description = await understand_threed_pattern(path)
            credit_cost = get_vl_credit_per_call(vl_model)
            await deduct_credits(user, credit_cost)
            items.append({
                "image_path": path,
                "image_name": img.filename,
                "display_url": display_url,
                "description": description,
                "credit_cost": credit_cost,
            })
        except Exception as e:
            items.append({
                "image_path": path,
                "image_name": img.filename,
                "display_url": display_url,
                "description": "",
                "error": str(e),
            })

    return {"items": items}
