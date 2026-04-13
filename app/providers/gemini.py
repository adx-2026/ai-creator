"""
Google Gemini image generation provider.
"""

import os
import re
import uuid
import base64
import asyncio

import google.generativeai as genai
import PIL.Image

from app.providers import ImageProvider


class GeminiProvider(ImageProvider):
    def __init__(self):
        genai.configure(
            api_key=os.getenv("GENAI_API_KEY", ""),
            transport="rest",
            client_options={
                "api_endpoint": os.getenv("GENAI_API_ENDPOINT", "http://127.0.0.1:8045")
            },
        )

    async def generate(self, model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user, target_ratio=""):
        model = genai.GenerativeModel(model_id)

        if target_ratio:
            prompt += f"\n\n[要求：请严格按照 {target_ratio} 的横纵比例生成或处理图像。]"

        final_prompt = prompt + (f"\n\nNegative Constraints: {negative_prompt}" if negative_prompt else "")
        contents = []
        if img_path and os.path.exists(img_path):
            try:
                contents.extend([PIL.Image.open(img_path), f"Reference image attached. Instruction: {final_prompt}"])
            except Exception:
                contents.append(final_prompt)
        else:
            contents.append(final_prompt)

        resp = await asyncio.to_thread(model.generate_content, contents)
        generated_images, final_text = [], ""

        if hasattr(resp, "candidates") and resp.candidates:
            for cand in resp.candidates:
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    for part in cand.content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            data = (
                                base64.b64decode(part.inline_data.data)
                                if isinstance(part.inline_data.data, str)
                                else part.inline_data.data
                            )
                            ext = (part.inline_data.mime_type or "image/png").split("/")[-1].replace("jpeg", "jpg")
                            filename = f"gen_{uuid.uuid4().hex[:8]}.{ext}"
                            with open(os.path.join(user_dir, filename), "wb") as f:
                                f.write(data)
                            generated_images.append({
                                "url": f"/api/images/{user}/{filename}",
                                "download_name": f"{dl_base_name}.{ext}",
                            })
        if not generated_images and hasattr(resp, "text") and resp.text:
            text = resp.text
            base64_patterns = re.findall(r"!\[.*?\]\((data:image/([^;]+);base64,([^)]+))\)", text)
            if base64_patterns:
                for idx, (_, ext, b64_data) in enumerate(base64_patterns):
                    ext = ext.replace("jpeg", "jpg")
                    filename = f"gen_{uuid.uuid4().hex[:8]}.{ext}"
                    with open(os.path.join(user_dir, filename), "wb") as f:
                        f.write(base64.b64decode(b64_data))
                    generated_images.append({
                        "url": f"/api/images/{user}/{filename}",
                        "download_name": f"{dl_base_name}_{idx + 1}.{ext}",
                    })
                final_text = re.sub(r"!\[.*?\]\((data:image/[^;]+;base64,[^)]+)\)", "", text)
            else:
                final_text = text
        return generated_images, final_text
