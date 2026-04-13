"""
Alibaba Wan video generation provider (DashScope).
"""

import os
import uuid
import asyncio
from typing import List, Tuple

import requests

from app.rate_limiter import aliyun_rate_limiter


async def upload_to_dashscope(file_path: str, model_id: str) -> str:
    """Upload a local file to DashScope OSS and return the oss:// temporary URL."""
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    filename = os.path.basename(file_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    params = {"action": "getPolicy", "model": model_id}

    policy_resp = await asyncio.to_thread(
        requests.get,
        "https://dashscope.aliyuncs.com/api/v1/uploads",
        headers=headers,
        params=params,
    )
    if policy_resp.status_code != 200:
        raise Exception(f"DashScope 上传策略获取失败 ({policy_resp.status_code}): {policy_resp.text}")

    policy_data = policy_resp.json().get("data", {})
    key = f"{policy_data.get('upload_dir', '')}/{filename}"

    with open(file_path, "rb") as f:
        file_content = f.read()

    files = {
        "OSSAccessKeyId": (None, policy_data.get("oss_access_key_id")),
        "Signature": (None, policy_data.get("signature")),
        "policy": (None, policy_data.get("policy")),
        "key": (None, key),
        "success_action_status": (None, "200"),
    }

    if "x_oss_object_acl" in policy_data:
        files["x-oss-object-acl"] = (None, policy_data["x_oss_object_acl"])
    if "x_oss_forbid_overwrite" in policy_data:
        files["x-oss-forbid-overwrite"] = (None, policy_data["x_oss_forbid_overwrite"])

    files["file"] = (filename, file_content)

    oss_resp = await asyncio.to_thread(
        requests.post, policy_data.get("upload_host"), files=files
    )

    if oss_resp.status_code not in [200, 204]:
        raise Exception(f"OSS 上传失败 (HTTP {oss_resp.status_code}): {oss_resp.text[:200]}")

    return f"oss://{key}"


class WanVideoProvider:
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.create_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        self.query_base = "https://dashscope.aliyuncs.com/api/v1/tasks"

    async def generate(
        self,
        model_id: str,
        prompt: str,
        reference_paths: List[str],
        user_dir: str,
        dl_base_name: str,
        user: str,
        size: str = "1280*720",
        duration: int = 5,
        shot_type: str = "single",
        audio: bool = True,
        watermark: bool = False,
    ) -> Tuple[List[dict], str]:
        await aliyun_rate_limiter.wait(model_id)
        reference_urls = []
        reference_video_urls = []
        for path in reference_paths:
            if path and os.path.exists(path):
                url = await upload_to_dashscope(path, model_id)
                ext = os.path.splitext(path)[1].lower()
                if ext in (".mp4", ".mov", ".avi", ".webm"):
                    reference_video_urls.append(url)
                else:
                    reference_urls.append(url)

        print(f"[WAN] model={model_id} reference_urls={reference_urls} reference_video_urls={reference_video_urls}")

        payload: dict = {
            "model": model_id,
            "input": {"prompt": prompt},
            "parameters": {
                "size": size,
                "duration": duration,
                "shot_type": shot_type,
                "watermark": watermark,
            },
        }
        if reference_urls:
            payload["input"]["reference_urls"] = reference_urls
        if reference_video_urls:
            payload["input"]["reference_video_urls"] = reference_video_urls
        if "flash" in model_id:
            payload["parameters"]["audio"] = audio

        headers = {
            "X-DashScope-Async": "enable",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-OssResourceResolve": "enable",
        }

        create_resp = await asyncio.to_thread(
            requests.post, self.create_url, headers=headers, json=payload
        )
        create_data = create_resp.json()
        if create_resp.status_code != 200:
            raise Exception(f"创建万相视频任务失败: {create_data}")

        task_id = create_data.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"未获取到 task_id: {create_data}")

        # Poll for result (up to 20 minutes)
        query_headers = {"Authorization": f"Bearer {self.api_key}"}
        for _ in range(80):
            await asyncio.sleep(15)
            query_resp = await asyncio.to_thread(
                requests.get, f"{self.query_base}/{task_id}", headers=query_headers
            )
            query_data = query_resp.json()
            status = query_data.get("output", {}).get("task_status", "UNKNOWN")

            if status == "SUCCEEDED":
                video_url = query_data["output"]["video_url"]
                video_data = await asyncio.to_thread(requests.get, video_url)
                filename = f"vid_{uuid.uuid4().hex[:8]}.mp4"
                filepath = os.path.join(user_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(video_data.content)
                return [{"url": f"/api/videos/{user}/{filename}", "download_name": f"{dl_base_name}.mp4", "type": "video"}], ""
            elif status == "FAILED":
                code = query_data["output"].get("code", "")
                msg = query_data["output"].get("message", "Unknown error")
                raise Exception(f"万相视频生成失败 [{code}]: {msg}")
            elif status == "UNKNOWN":
                raise Exception(f"任务 {task_id} 不存在 (UNKNOWN)")
            # PENDING / RUNNING: continue polling

        raise Exception(f"视频生成超时（task_id: {task_id}）")
