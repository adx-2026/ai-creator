"""
Job queue: JobQueue class and the background process_queue consumer loop.
"""

import os
import re
import json
import time
import uuid
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import HTTPException

from app.config import SUBTASK_CONCURRENCY
from app.models import (
    build_subtasks, normalize_job, refresh_job_progress, upsert_task_result, make_subtask,
)
from app.credits import (
    estimate_job_credits, get_image_credit_per_image, get_vl_credit_per_call,
    get_video_credit_per_second, deduct_credits,
)
from app.rate_limiter import run_with_retries
from app.providers import get_provider_for_model
from app.providers.wan_video import WanVideoProvider
from app.pipelines.extract import extract_pattern_prompt


class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self.queue: asyncio.Queue = asyncio.Queue()

    def sync_user_jobs(self, user: str):
        user_dir = f"users/{user}"
        os.makedirs(user_dir, exist_ok=True)
        with open(f"{user_dir}/jobs.json", "w", encoding="utf-8") as f:
            json.dump(
                [j for j in self.jobs.values() if j.get("user") == user],
                f,
                ensure_ascii=False,
            )

    def load_jobs(self):
        if not os.path.exists("users"):
            return
        for user_dir in os.listdir("users"):
            jobs_file = os.path.join("users", user_dir, "jobs.json")
            if os.path.exists(jobs_file):
                try:
                    with open(jobs_file, "r", encoding="utf-8") as f:
                        for j in json.load(f):
                            normalize_job(j)
                            if j["status"] in ["queued", "processing"]:
                                j["status"] = "failed"
                                j["results"].append({"error": "服务曾被重启，该任务已中断", "status": "error"})
                                for subtask in j.get("subtasks", []):
                                    if subtask.get("status") == "pending":
                                        subtask["status"] = "error"
                                        subtask["attempts"] = max(1, subtask.get("attempts", 0))
                                        upsert_task_result(j, subtask, {
                                            "prompt": subtask.get("prompt", ""),
                                            "source_img": os.path.basename(subtask["source_img"]).split("_", 1)[-1] if subtask.get("source_img") else None,
                                            "error": "服务曾被重启，该子任务已中断",
                                            "status": "error",
                                            "attempts": subtask["attempts"],
                                        })
                                refresh_job_progress(j)
                            self.jobs[j["id"]] = j
                except Exception:
                    pass

    async def add_job(
        self,
        user: str,
        mode: str,
        prompts: List[str],
        source_image_paths: Optional[List[str]] = None,
        template_name: str = "",
        model_id: str = "",
        negative_prompt: str = "",
        batch_size: int = 1,
        target_ratio: str = "",
        video_params: Optional[Dict] = None,
    ) -> dict:
        job_id = str(uuid.uuid4())
        subtasks = build_subtasks(mode, prompts, source_image_paths, batch_size)
        estimated_credits = estimate_job_credits(model_id, mode, subtasks, video_params, batch_size)
        self.jobs[job_id] = {
            "id": job_id, "user": user, "mode": mode, "model_id": model_id,
            "status": "queued", "total": 0, "completed": 0, "failed": 0,
            "results": [], "created_at": time.time(), "eta": None,
            "template_name": template_name, "negative_prompt": negative_prompt,
            "target_ratio": target_ratio, "video_params": video_params or {},
            "batch_size": batch_size,
            "subtasks": subtasks,
            "estimated_credits": estimated_credits,
        }
        refresh_job_progress(self.jobs[job_id])
        self.sync_user_jobs(user)
        await self.queue.put({"job_id": job_id, "subtask_ids": None})
        return self.jobs[job_id]

    async def retry_failed_subtasks(self, job_id: str, task_ids: Optional[List[str]] = None) -> dict:
        job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] in ["queued", "processing"]:
            raise HTTPException(status_code=409, detail="Job is already queued or processing")

        candidates = [t for t in job.get("subtasks", []) if t.get("status") == "error"]
        if task_ids is not None:
            wanted = set(task_ids)
            candidates = [t for t in candidates if t["id"] in wanted]
        if not candidates:
            raise HTTPException(status_code=400, detail="No failed subtasks to retry")

        for subtask in candidates:
            subtask["status"] = "pending"
        refresh_job_progress(job)
        job["status"] = "queued"
        job["eta"] = None
        self.sync_user_jobs(job["user"])
        await self.queue.put({"job_id": job_id, "subtask_ids": [t["id"] for t in candidates]})
        return job


# Singleton
job_queue = JobQueue()


async def process_queue():
    """Background consumer: pulls jobs from queue and processes subtasks."""
    while True:
        job_data = await job_queue.queue.get()
        if isinstance(job_data, dict):
            job_id = job_data["job_id"]
        else:
            job_id = job_data[0]

        job = job_queue.jobs[job_id]
        normalize_job(job)
        user = job["user"]
        tpl_name = job.get("template_name", "")
        model_id = job.get("model_id", "")
        negative_prompt = job.get("negative_prompt", "")
        target_ratio = job.get("target_ratio", "")
        job["status"] = "processing"
        user_dir = f"users/{user}/outputs"
        os.makedirs(user_dir, exist_ok=True)

        # ── Pattern extraction mode ──
        if job["mode"] == "extract":
            await _process_extract(job, model_id, target_ratio, user, user_dir)
            job_queue.queue.task_done()
            continue

        # ── Video generation mode ──
        if job["mode"] in ("video", "multi_video"):
            await _process_video(job, model_id, user, user_dir)
            job_queue.queue.task_done()
            continue

        # ── Image generation mode (concurrent) ──
        await _process_images(job, model_id, negative_prompt, target_ratio, tpl_name, user, user_dir)
        job_queue.queue.task_done()


async def _process_extract(job: dict, model_id: str, target_ratio: str, user: str, user_dir: str):
    batch_size = job.get("batch_size", 1)
    provider = get_provider_for_model(model_id)
    tasks = [t for t in job.get("subtasks", []) if t.get("status") == "pending"]

    sem = asyncio.Semaphore(SUBTASK_CONCURRENCY)
    completed_times: List[float] = []

    async def run_subtask(subtask):
        async with sem:
            start_time = time.time()
            img_path = subtask.get("source_img")
            subtask["attempts"] = subtask.get("attempts", 0) + 1
            try:
                extracted_prompt = await extract_pattern_prompt(img_path)
                all_images = []
                for j in range(batch_size):
                    dl_base_name = re.sub(
                        r'[\\/*?:"<>|]', "",
                        f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{j + 1}_{uuid.uuid4().hex[:4]}",
                    )
                    imgs, _ = await run_with_retries(
                        lambda p=extracted_prompt, b=dl_base_name: provider.generate(
                            model_id, p, "", img_path, user_dir, b, user, target_ratio
                        ),
                        model_id=model_id,
                    )
                    all_images.extend(imgs)
                    if j < batch_size - 1:
                        await asyncio.sleep(0.2)

                subtask["status"] = "success"
                vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
                credit_cost = round(
                    get_vl_credit_per_call(vl_model) + len(all_images) * get_image_credit_per_image(model_id), 4
                )
                await deduct_credits(user, credit_cost)
                upsert_task_result(job, subtask, {
                    "prompt": extracted_prompt,
                    "extracted_prompt": extracted_prompt,
                    "source_img": os.path.basename(img_path).split("_", 1)[-1] if img_path else None,
                    "images": all_images,
                    "status": "success",
                    "attempts": subtask["attempts"],
                    "credit_cost": credit_cost,
                })
            except Exception as e:
                subtask["status"] = "error"
                upsert_task_result(job, subtask, {
                    "prompt": "",
                    "source_img": os.path.basename(img_path).split("_", 1)[-1] if img_path else None,
                    "error": str(e),
                    "status": "error",
                    "attempts": subtask["attempts"],
                })
            elapsed = time.time() - start_time
            completed_times.append(elapsed)
            avg_time = sum(completed_times) / len(completed_times)
            pending_count = sum(1 for t in job.get("subtasks", []) if t.get("status") == "pending")
            job["eta"] = max(0, (pending_count / SUBTASK_CONCURRENCY) * avg_time)
            refresh_job_progress(job)
            job_queue.sync_user_jobs(user)

    await asyncio.gather(*[run_subtask(t) for t in tasks])
    job["status"] = "completed"
    job["eta"] = 0
    job_queue.sync_user_jobs(user)


async def _process_video(job: dict, model_id: str, user: str, user_dir: str):
    pending_tasks = [t for t in job.get("subtasks", []) if t.get("status") == "pending"]
    if not pending_tasks:
        job["status"] = "completed"
        job["eta"] = 0
        job_queue.sync_user_jobs(user)
        return
    vp = job.get("video_params", {})
    wan_provider = WanVideoProvider()
    total_video = len(pending_tasks)
    for idx, subtask in enumerate(pending_tasks):
        prompt = subtask.get("prompt", "")
        dl_base_name = re.sub(r'[\\/*?:"<>|]', "", f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx + 1}")
        try:
            subtask["attempts"] = subtask.get("attempts", 0) + 1
            videos, _ = await run_with_retries(
                lambda p=prompt, st=subtask, b=dl_base_name: wan_provider.generate(
                    model_id=model_id,
                    prompt=p,
                    reference_paths=[st["source_img"]] if st.get("source_img") else [],
                    user_dir=user_dir,
                    dl_base_name=b,
                    user=user,
                    size=vp.get("size", "1280*720"),
                    duration=int(vp.get("duration", 5)),
                    shot_type=vp.get("shot_type", "single"),
                    audio=bool(vp.get("audio", True)),
                    watermark=bool(vp.get("watermark", False)),
                ),
                model_id=model_id,
            )
            subtask["status"] = "success"
            duration_sec = int(vp.get("duration", 5))
            credit_cost = round(get_video_credit_per_second(model_id) * duration_sec, 4)
            await deduct_credits(user, credit_cost)
            upsert_task_result(job, subtask, {
                "prompt": prompt,
                "source_img": os.path.basename(subtask["source_img"]).split("_", 1)[-1] if subtask.get("source_img") else None,
                "videos": videos, "images": [], "status": "success",
                "attempts": subtask["attempts"], "credit_cost": credit_cost,
            })
        except Exception as e:
            subtask["status"] = "error"
            upsert_task_result(job, subtask, {
                "prompt": prompt,
                "source_img": os.path.basename(subtask["source_img"]).split("_", 1)[-1] if subtask.get("source_img") else None,
                "error": str(e), "status": "error", "attempts": subtask["attempts"],
            })
        refresh_job_progress(job)
        remaining = total_video - (idx + 1)
        job["eta"] = remaining * 300
        job_queue.sync_user_jobs(user)
    job["status"] = "completed"
    job["eta"] = 0
    job_queue.sync_user_jobs(user)


async def _process_images(
    job: dict, model_id: str, negative_prompt: str, target_ratio: str,
    tpl_name: str, user: str, user_dir: str,
):
    provider = get_provider_for_model(model_id)
    tasks = [t for t in job.get("subtasks", []) if t.get("status") == "pending"]

    sem = asyncio.Semaphore(SUBTASK_CONCURRENCY)
    completed_times: List[float] = []

    async def run_subtask(subtask):
        async with sem:
            start_time = time.time()
            prompt = subtask.get("prompt", "")
            img_path = subtask.get("source_img")
            try:
                base_src = os.path.splitext(os.path.basename(img_path).split("_", 1)[-1])[0] if img_path else "t2i"
                dl_base_name = re.sub(
                    r'[\\/*?:"<>|]', "",
                    f"{base_src}_{tpl_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
                    if tpl_name
                    else f"{base_src}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}",
                )
                subtask["attempts"] = subtask.get("attempts", 0) + 1
                generated_images, final_text = await run_with_retries(
                    lambda: provider.generate(
                        model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user, target_ratio
                    ),
                    model_id=model_id,
                )
                subtask["status"] = "success"
                credit_cost = round(len(generated_images) * get_image_credit_per_image(model_id), 4)
                await deduct_credits(user, credit_cost)
                upsert_task_result(job, subtask, {
                    "prompt": prompt,
                    "source_img": os.path.basename(img_path).split("_", 1)[-1] if img_path else None,
                    "result": final_text.strip(),
                    "images": generated_images,
                    "status": "success",
                    "attempts": subtask["attempts"],
                    "credit_cost": credit_cost,
                })
            except Exception as e:
                subtask["status"] = "error"
                upsert_task_result(job, subtask, {
                    "prompt": prompt,
                    "source_img": os.path.basename(img_path).split("_", 1)[-1] if img_path else None,
                    "error": str(e),
                    "status": "error",
                    "attempts": subtask["attempts"],
                })
            elapsed = time.time() - start_time
            completed_times.append(elapsed)
            avg_time = sum(completed_times) / len(completed_times)
            pending_count = sum(1 for t in job.get("subtasks", []) if t.get("status") == "pending")
            job["eta"] = max(0, (pending_count / SUBTASK_CONCURRENCY) * avg_time)
            refresh_job_progress(job)
            job_queue.sync_user_jobs(user)

    await asyncio.gather(*[run_subtask(t) for t in tasks])
    job["status"] = "completed"
    job["eta"] = 0
    job_queue.sync_user_jobs(user)
