"""
Job management routes: create, list, retry, delete, download.
"""

import os
import io
import re
import json
import uuid
import zipfile
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.auth import get_current_user
from app.job_queue import job_queue
from app.pipelines.threed import THREED_PROMPT_TEMPLATE
from fastapi import HTTPException

router = APIRouter()


@router.post("/api/jobs")
async def create_job(
    prompts: str = Form(""),
    negative_prompt: str = Form(""),
    mode: str = Form(...),
    template_name: str = Form(""),
    model_id: str = Form(""),
    batch_size: int = Form(1),
    target_ratio: str = Form(""),
    video_size: str = Form("1280*720"),
    video_duration: int = Form(5),
    video_shot_type: str = Form("single"),
    video_audio: bool = Form(True),
    video_watermark: bool = Form(False),
    ecommerce_data: str = Form(""),
    threed_data: str = Form(""),
    images: Optional[List[UploadFile]] = File(None),
    curr: dict = Depends(get_current_user),
):
    # ── E-commerce scene mode ──
    if mode == "ecommerce":
        if not ecommerce_data:
            return JSONResponse(status_code=400, content={"error": "ecommerce mode requires ecommerce_data"})
        try:
            ec_items = json.loads(ecommerce_data)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "ecommerce_data JSON 格式错误"})
        source_paths_ec, prompts_ec = [], []
        for item in ec_items:
            for scene in item.get("scenes", []):
                scene = scene.strip()
                if scene:
                    wrapped = (
                        f"Place the reference product (from the provided image) into this scene: {scene} "
                        "CRITICAL: The product's appearance must be IDENTICAL to the reference image — "
                        "preserve its exact shape, design, colors, branding, and surface details. "
                        "Only the background, environment, lighting angle, and composition may change."
                    )
                    source_paths_ec.append(item["image_path"])
                    prompts_ec.append(wrapped)
        if not prompts_ec:
            return JSONResponse(status_code=400, content={"error": "没有有效的场景数据"})
        job = await job_queue.add_job(
            curr["username"], "ecommerce", prompts_ec, source_paths_ec,
            template_name, model_id, negative_prompt, 1, target_ratio, video_params=None,
        )
        return job

    # ── 3D conversion mode ──
    if mode == "threed":
        if not threed_data:
            return JSONResponse(status_code=400, content={"error": "threed mode requires threed_data"})
        try:
            td_items = json.loads(threed_data)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "threed_data JSON 格式错误"})
        prompts_td = []
        for item in td_items:
            desc = (item.get("description") or "").strip()
            if desc:
                final_prompt = THREED_PROMPT_TEMPLATE.format(description=desc)
                prompts_td.append(final_prompt)
        if not prompts_td:
            return JSONResponse(status_code=400, content={"error": "没有有效的图案描述数据"})
        job = await job_queue.add_job(
            curr["username"], "threed", prompts_td, None,
            template_name, model_id, negative_prompt, 1, target_ratio, video_params=None,
        )
        return job

    prompt_str = prompts.strip()

    if mode == "fission" and not prompt_str:
        prompt_str = "Generate a high-quality, stylistically similar variation of the provided reference image.it should incorporate a certain degree of variation and be different to the original."
    elif mode == "convert" and not prompt_str:
        prompt_str = "保持原图主体结构和风格不变，将画面自然延展或重绘以适应设定的新比例尺寸，边缘过渡自然。"
    elif mode == "extract":
        prompt_str = ""
    elif mode == "multi_t2i":
        if not prompt_str:
            return {"error": "请至少输入一行提示词"}
    elif mode == "multi_video":
        if not prompt_str:
            return {"error": "视频生成需要填写提示词"}
    elif mode != "video" and not prompt_str:
        return {"error": "No prompt provided"}
    elif mode == "video" and not prompt_str:
        return {"error": "视频生成需要填写提示词"}

    source_paths = []
    if images:
        os.makedirs(f"users/{curr['username']}/outputs", exist_ok=True)
        for img in images:
            if img and getattr(img, "filename", None):
                path = os.path.join(
                    f"users/{curr['username']}/outputs",
                    f"{uuid.uuid4().hex[:8]}_{img.filename}",
                )
                img_data = await img.read()
                with open(path, "wb") as f:
                    f.write(img_data)
                source_paths.append(path)

    video_params = None
    if mode in ("video", "multi_video"):
        video_params = {
            "size": video_size,
            "duration": video_duration,
            "shot_type": video_shot_type,
            "audio": video_audio,
            "watermark": video_watermark,
        }

    if mode in ("multi_t2i", "multi_video"):
        prompts_list = [p.strip() for p in prompt_str.split("\n") if p.strip()]
    else:
        prompts_list = [prompt_str]

    job = await job_queue.add_job(
        curr["username"], mode, prompts_list, source_paths,
        template_name, model_id, negative_prompt, batch_size, target_ratio,
        video_params=video_params,
    )
    return job


@router.get("/api/jobs")
def get_jobs(curr: dict = Depends(get_current_user)):
    return sorted(
        [j for j in job_queue.jobs.values() if j.get("user") == curr["username"]],
        key=lambda x: x["created_at"],
        reverse=True,
    )


@router.post("/api/jobs/{job_id}/retry-failed")
async def retry_failed_job_tasks(job_id: str, curr: dict = Depends(get_current_user)):
    job = job_queue.jobs.get(job_id)
    if not job or (job["user"] != curr["username"] and not curr["is_admin"]):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    try:
        return await job_queue.retry_failed_subtasks(job_id)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})


@router.post("/api/jobs/{job_id}/retry-task/{task_id}")
async def retry_single_job_task(job_id: str, task_id: str, curr: dict = Depends(get_current_user)):
    job = job_queue.jobs.get(job_id)
    if not job or (job["user"] != curr["username"] and not curr["is_admin"]):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    try:
        return await job_queue.retry_failed_subtasks(job_id, [task_id])
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})


@router.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, curr: dict = Depends(get_current_user)):
    job = job_queue.jobs.get(job_id)
    if not job or (job["user"] != curr["username"] and not curr["is_admin"]):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    del job_queue.jobs[job_id]
    job_queue.sync_user_jobs(job["user"])
    return {"success": True}


@router.get("/api/jobs/{job_id}/download")
def download_job_files(job_id: str, curr: dict = Depends(get_current_user)):
    job = job_queue.jobs.get(job_id)
    if not job or (job["user"] != curr["username"] and not curr["is_admin"]):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for res in job.get("results", []):
            for img in res.get("images", []):
                filename = img["url"].split("/")[-1]
                file_path = os.path.join(f"users/{job['user']}/outputs", filename)
                if os.path.exists(file_path):
                    unique_prefix = filename.split("_")[1].split(".")[0] if "_" in filename else uuid.uuid4().hex[:6]
                    zip_name = f"{unique_prefix}_{img.get('download_name', filename)}"
                    zip_file.write(file_path, arcname=zip_name)
            for vid in res.get("videos", []):
                filename = vid["url"].split("/")[-1]
                file_path = os.path.join(f"users/{job['user']}/outputs", filename)
                if os.path.exists(file_path):
                    zip_file.write(file_path, arcname=vid.get("download_name", filename))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=job_{job_id}_images.zip"},
    )
