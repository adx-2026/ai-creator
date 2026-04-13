"""
AI image/video provider abstraction and factory.
"""

from typing import Tuple, List
from app.config import AVAILABLE_MODELS


class ImageProvider:
    async def generate(
        self,
        model_id: str,
        prompt: str,
        negative_prompt: str,
        img_path: str,
        user_dir: str,
        dl_base_name: str,
        user: str,
        target_ratio: str = "",
    ) -> Tuple[List[dict], str]:
        raise NotImplementedError()


def get_provider_for_model(model_id: str) -> ImageProvider:
    from app.providers.gemini import GeminiProvider
    from app.providers.qwen import QwenProvider
    from app.providers.minimax import MinimaxProvider
    from app.providers.doubao import DoubaoProvider

    for group in AVAILABLE_MODELS:
        for m in group["models"]:
            if m["id"] == model_id:
                if m["prefix"] == "gemini":
                    return GeminiProvider()
                if m["prefix"] == "qwen":
                    return QwenProvider()
                if m["prefix"] == "minimax":
                    return MinimaxProvider()
                if m["prefix"] == "doubao":
                    return DoubaoProvider()
    return GeminiProvider()  # Default fallback
