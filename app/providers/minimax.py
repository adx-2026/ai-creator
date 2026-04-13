"""
MiniMax image generation provider.
"""

import os
import uuid
import base64
import asyncio

import requests

from app.providers import ImageProvider


class MinimaxProvider(ImageProvider):
    async def generate(self, model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user, target_ratio=""):
        mm_model = model_id.replace("minimax-", "") if model_id.startswith("minimax-") else model_id
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('MINIMAX_API_KEY')}",
        }

        if target_ratio:
            prompt += f"\n\n[要求：请严格按照 {target_ratio} 的横纵比例生成或处理图像。]"

        final_prompt = prompt + (f"\n\n请尽量避免出现以下元素：{negative_prompt}" if negative_prompt else "")
        payload = {
            "model": mm_model,
            "prompt": final_prompt,
            "response_format": "base64",
            "n": 1,
            "prompt_optimizer": True,
        }

        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(img_path)[1].lower().replace(".", "").replace("jpg", "jpeg")
            payload["subject_reference"] = [
                {"type": "character", "image_file": f"data:image/{ext or 'jpeg'};base64,{b64_data}"}
            ]

        resp = await asyncio.to_thread(
            requests.post, "https://api.minimaxi.com/v1/image_generation", headers=headers, json=payload
        )
        resp_json = resp.json()
        if resp_json.get("base_resp", {}).get("status_code") != 0:
            raise Exception(f"MiniMax Error: {resp_json.get('base_resp', {}).get('status_msg')}")

        generated_images = []
        for b64_img in resp_json.get("data", {}).get("image_base64", []):
            filename = f"gen_{uuid.uuid4().hex[:8]}.png"
            with open(os.path.join(user_dir, filename), "wb") as f:
                f.write(base64.b64decode(b64_img))
            generated_images.append({
                "url": f"/api/images/{user}/{filename}",
                "download_name": f"{dl_base_name}.png",
            })
        return generated_images, ""
