"""
ByteDance Doubao (Seedream) image generation provider.
"""

import os
import uuid
import base64
import asyncio

import requests

from app.providers import ImageProvider


class DoubaoProvider(ImageProvider):
    async def generate(self, model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user, target_ratio=""):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('ARK_API_KEY')}",
        }

        if target_ratio:
            prompt += f"\n\n[要求：请严格按照 {target_ratio} 的横纵比例生成或处理图像。]"

        final_prompt = prompt + (f"\n\n请尽量避免出现以下元素：{negative_prompt}" if negative_prompt else "")
        payload = {
            "model": model_id,
            "prompt": final_prompt,
            "response_format": "url",
            "size": "2K",
            "stream": False,
            "watermark": False,
        }

        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(img_path)[1].lower().replace(".", "").replace("jpg", "jpeg")
            payload["image"] = f"data:image/{ext or 'jpeg'};base64,{b64_data}"

        resp = await asyncio.to_thread(
            requests.post,
            "https://ark.cn-beijing.volces.com/api/v3/images/generations",
            headers=headers,
            json=payload,
        )
        resp_json = resp.json()
        if "error" in resp_json:
            raise Exception(f"Doubao Error: {resp_json['error'].get('message', str(resp_json['error']))}")

        generated_images = []
        for item in resp_json.get("data", []):
            if "url" in item:
                img_data = await asyncio.to_thread(requests.get, item["url"])
                filename = f"gen_{uuid.uuid4().hex[:8]}.png"
                with open(os.path.join(user_dir, filename), "wb") as f:
                    f.write(img_data.content)
                generated_images.append({
                    "url": f"/api/images/{user}/{filename}",
                    "download_name": f"{dl_base_name}.png",
                })
            elif "b64_json" in item:
                filename = f"gen_{uuid.uuid4().hex[:8]}.png"
                with open(os.path.join(user_dir, filename), "wb") as f:
                    f.write(base64.b64decode(item["b64_json"]))
                generated_images.append({
                    "url": f"/api/images/{user}/{filename}",
                    "download_name": f"{dl_base_name}.png",
                })
        return generated_images, ""
