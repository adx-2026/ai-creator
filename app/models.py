"""
Job / subtask data structures and helper functions.
"""

import uuid
from typing import Any, Dict, List, Optional


def make_subtask(prompt: str, source_img: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:12],
        "prompt": prompt,
        "source_img": source_img,
        "status": "pending",
        "attempts": 0,
        "result_index": None,
    }


def build_subtasks(
    mode: str,
    prompts: List[str],
    source_image_paths: Optional[List[str]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    if mode == "video":
        return [make_subtask(prompts[0] if prompts else "", source_image_paths[0] if source_image_paths else None)]

    if mode == "multi_video":
        ref = source_image_paths[0] if source_image_paths else None
        return [make_subtask(p, ref) for p in prompts if p.strip()]

    if mode == "ecommerce":
        return [make_subtask(p, img) for p, img in zip(prompts, source_image_paths or [])]

    if mode == "extract":
        if source_image_paths:
            return [make_subtask("", img_path) for img_path in source_image_paths]
        return []

    subtasks = []
    for _ in range(batch_size):
        if mode in ["i2i", "fission", "convert"] and source_image_paths:
            for img_path in source_image_paths:
                for prompt in prompts:
                    subtasks.append(make_subtask(prompt, img_path))
        else:
            for prompt in prompts:
                subtasks.append(make_subtask(prompt, None))
    return subtasks


def refresh_job_progress(job: Dict[str, Any]):
    subtasks = job.get("subtasks", [])
    job["total"] = len(subtasks)
    job["completed"] = sum(1 for t in subtasks if t.get("status") == "success")
    job["failed"] = sum(1 for t in subtasks if t.get("status") == "error")


def upsert_task_result(job: Dict[str, Any], subtask: Dict[str, Any], payload: Dict[str, Any]):
    payload["task_id"] = subtask["id"]
    idx = subtask.get("result_index")
    if idx is None or idx >= len(job["results"]):
        job["results"].append(payload)
        subtask["result_index"] = len(job["results"]) - 1
    else:
        job["results"][idx] = payload


def normalize_job(job: Dict[str, Any]):
    job.setdefault("results", [])
    job.setdefault("template_name", "")
    job.setdefault("negative_prompt", "")
    job.setdefault("target_ratio", "")
    job.setdefault("video_params", {})
    job.setdefault("batch_size", 1)
    job.setdefault("eta", None)
    if not job.get("subtasks"):
        subtasks = []
        for res in job.get("results", []):
            subtask = make_subtask(res.get("prompt", ""), res.get("source_img"))
            subtask["id"] = res.get("task_id") or subtask["id"]
            subtask["status"] = "success" if res.get("status") == "success" else "error"
            subtask["attempts"] = max(1, res.get("attempts", 1 if subtask["status"] != "pending" else 0))
            subtasks.append(subtask)
            res["task_id"] = subtask["id"]
        job["subtasks"] = subtasks
        for idx, subtask in enumerate(job["subtasks"]):
            subtask["result_index"] = idx if idx < len(job["results"]) else None
    else:
        for idx, subtask in enumerate(job["subtasks"]):
            subtask.setdefault("id", uuid.uuid4().hex[:12])
            subtask.setdefault("prompt", "")
            subtask.setdefault("source_img", None)
            subtask.setdefault("status", "pending")
            subtask.setdefault("attempts", 0)
            subtask.setdefault("result_index", idx if idx < len(job["results"]) else None)
            if subtask["result_index"] is not None and subtask["result_index"] < len(job["results"]):
                job["results"][subtask["result_index"]]["task_id"] = subtask["id"]
    refresh_job_progress(job)
