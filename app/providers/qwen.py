"""
Alibaba Qwen image generation provider (DashScope).
"""

import os
import uuid
import base64
import asyncio

import requests
import dashscope
from dashscope import MultiModalConversation

from app.providers import ImageProvider
from app.config import map_ratio_to_size
from app.rate_limiter import aliyun_rate_limiter


class QwenProvider(ImageProvider):
    def __init__(self):
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")

    async def generate(self, model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user, target_ratio=""):
        await aliyun_rate_limiter.wait(model_id)
        size_param = "1024*1024"

        if target_ratio:
            if model_id == "qwen-image-2.0-pro":
                size_param = map_ratio_to_size(target_ratio)
            else:
                prompt += f"\n\n[要求：请严格按照 {target_ratio} 的横纵比例生成或处理图像。]"

        content_list = []
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(img_path)[1].lower().replace(".", "").replace("jpg", "jpeg")
            content_list.append({"image": f"data:image/{ext or 'jpeg'};base64,{b64_data}"})
        content_list.append({"text": prompt})

        kwargs = {
            "model": model_id,
            "messages": [{"role": "user", "content": content_list}],
            "n": 1,
            "size": size_param,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        rsp = await asyncio.to_thread(MultiModalConversation.call, **kwargs)
        if rsp.status_code != 200:
            raise Exception(f"Qwen Error: {rsp.message}")

        generated_images = []
        for item in rsp.output.choices[0].message.content:
            if "image" in item:
                img_data = await asyncio.to_thread(requests.get, item["image"])
                filename = f"gen_{uuid.uuid4().hex[:8]}.png"
                with open(os.path.join(user_dir, filename), "wb") as f:
                    f.write(img_data.content)
                generated_images.append({
                    "url": f"/api/images/{user}/{filename}",
                    "download_name": f"{dl_base_name}.png",
                })
        return generated_images, ""
