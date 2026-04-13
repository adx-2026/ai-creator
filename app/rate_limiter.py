"""
Aliyun API rate limiter with per-model throttling and retry penalty.
"""

import asyncio
import time
from typing import Dict

from app.config import ALIYUN_QWEN_STRICT_MODELS
from app.credits import is_aliyun_model


def get_aliyun_rate_rule(model_id: str) -> Dict[str, float]:
    if model_id in ALIYUN_QWEN_STRICT_MODELS:
        return {"min_interval": 31.0, "retry_penalty": 35.0}
    if model_id.startswith("wan"):
        return {"min_interval": 1.2, "retry_penalty": 5.0}
    if model_id.startswith("qwen"):
        return {"min_interval": 1.2, "retry_penalty": 8.0}
    return {"min_interval": 0.0, "retry_penalty": 0.0}


def is_rate_limit_error(message: str) -> bool:
    msg = (message or "").lower()
    keywords = [
        "429", "rate limit", "throttl", "too quickly", "requests limit",
        "quota", "request rate increased", "allocated quota exceeded",
    ]
    return any(k in msg for k in keywords)


class AliyunRateLimiter:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._next_allowed_at: Dict[str, float] = {}

    async def wait(self, model_id: str):
        if not is_aliyun_model(model_id):
            return
        delay = 0.0
        async with self._lock:
            now = time.monotonic()
            next_allowed = self._next_allowed_at.get(model_id, now)
            delay = max(0.0, next_allowed - now)
            interval = get_aliyun_rate_rule(model_id)["min_interval"]
            self._next_allowed_at[model_id] = max(now, next_allowed) + interval
        if delay > 0:
            await asyncio.sleep(delay)

    async def penalize(self, model_id: str, attempt: int = 0):
        if not is_aliyun_model(model_id):
            return
        rule = get_aliyun_rate_rule(model_id)
        penalty = rule["retry_penalty"] * max(1, attempt + 1)
        async with self._lock:
            now = time.monotonic()
            self._next_allowed_at[model_id] = max(
                self._next_allowed_at.get(model_id, now), now + penalty
            )


# Singleton instance
aliyun_rate_limiter = AliyunRateLimiter()


async def run_with_retries(callable_factory, model_id: str, max_retries: int = 5):
    last_error = None
    for attempt in range(max_retries):
        try:
            return await callable_factory()
        except Exception as e:
            last_error = e
            if attempt >= max_retries - 1 or not is_rate_limit_error(str(e)):
                raise
            await aliyun_rate_limiter.penalize(model_id, attempt)
    if last_error:
        raise last_error
