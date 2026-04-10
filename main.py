# main.py
import asyncio
import time
import uuid
import os
import re
import base64
import json
import requests
import io
import zipfile
import dashscope
from dashscope import MultiModalConversation, Generation
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
import google.generativeai as genai
import PIL.Image

# ==========================================
# 0. 尺寸映射工具 (专为 qwen-image-2.0-pro 使用)
# ==========================================
def map_ratio_to_size(ratio_str: str) -> str:
    """
    将常见比例字符串转换为 qwen-image-2.0-pro 推荐的标准分辨率字符串
    """
    if not ratio_str:
        return "1024*1024"
        
    ratio_str = ratio_str.replace('：', ':').strip()
    mapping = {
        "1:1": "1024*1024",
        "2:3": "1024*1536",
        "3:2": "1536*1024",
        "3:4": "1080*1440",
        "4:3": "1440*1080",
        "9:16": "1080*1920",
        "16:9": "1920*1080",
        "21:9": "2048*872"
    }
    return mapping.get(ratio_str, "1024*1024")

# ==========================================
# 1. 基础配置与初始化
# ==========================================
config_file = "config.json"
example_file = "config_example.json"
PUBLIC_TEMPLATES_FILE = "public_templates.json"

if not os.path.exists(example_file):
    with open(example_file, "w", encoding="utf-8") as f:
        json.dump({"users": {"admin": {"password": "admin_secure_pass_2026", "is_admin": True}, "test": {"password": "123456", "is_admin": False}}}, f, indent=4)
if not os.path.exists(config_file):
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump({"users": {"admin": {"password": "123456", "is_admin": True}, "test": {"password": "123456", "is_admin": False}}}, f, indent=4)
if not os.path.exists(PUBLIC_TEMPLATES_FILE):
    with open(PUBLIC_TEMPLATES_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

with open(config_file, "r", encoding="utf-8") as f:
    raw_config = json.load(f)

USERS = {}
for k, v in raw_config.get("users", {}).items():
    if isinstance(v, str): USERS[k] = {"password": v, "is_admin": (k == "admin")}
    else: USERS[k] = v

SUBTASK_CONCURRENCY = max(1, int(raw_config.get("subtask_concurrency", 3)))

SESSIONS = {}
app = FastAPI()
os.makedirs("users", exist_ok=True)
os.makedirs("static", exist_ok=True)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login", auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme), query_token: str = Query(None, alias="token")):
    actual_token = token or query_token
    if not actual_token or actual_token not in SESSIONS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_info = USERS.get(SESSIONS[actual_token], {})
    return {"username": SESSIONS[actual_token], "is_admin": user_info.get("is_admin", False)}

# ==========================================
# 2. 动态加载模型配置
# ==========================================
def parse_models(env_str: str, provider_name: str, prefix: str, model_type: str = "image"):
    if not env_str: return None
    models = []
    for item in env_str.split(','):
        if not item.strip(): continue
        parts = item.split(':', 1)
        m_id = parts[0].strip()
        m_name = parts[1].strip() if len(parts) > 1 else m_id
        models.append({"id": m_id, "name": m_name, "prefix": prefix, "type": model_type})
    return {"provider": provider_name, "models": models} if models else None

AVAILABLE_MODELS = []
for env_key, prov_name, prefix, model_type in [
    ("GEMINI_MODELS", "Google Gemini", "gemini", "image"),
    ("QWEN_MODELS", "Alibaba 通义千问", "qwen", "image"),
    ("MINIMAX_MODELS", "MiniMax 稀宇科技", "minimax", "image"),
    ("DOUBAO_MODELS", "ByteDance 豆包", "doubao", "image"),
    ("WAN_MODELS", "Alibaba 万相视频", "wan", "video"),
]:
    parsed = parse_models(os.getenv(env_key, ""), prov_name, prefix, model_type)
    if parsed: AVAILABLE_MODELS.append(parsed)

ALIYUN_QWEN_STRICT_MODELS = {"qwen-image-2.0-pro", "qwen-image-2.0-pro-2026-03-03"}

# ==========================================
# 积分计费系统 (Credit System)
# 1元 = 10积分
# ==========================================
CREDITS_PER_YUAN = 10

# 图像生成模型单价（元/张，中国内地）
_IMAGE_PRICE_PER_IMAGE: Dict[str, float] = {
    "qwen-image-2.0-pro": 0.5,
    "qwen-image-2.0-pro-2026-03-03": 0.5,
    "qwen-image-2.0": 0.2,
    "qwen-image-2.0-2026-03-03": 0.2,
    "qwen-image-max": 0.5,
    "qwen-image-max-2025-12-30": 0.5,
    "qwen-image-plus": 0.2,
    "qwen-image-plus-2026-01-09": 0.2,
    "qwen-image": 0.25,
    "qwen-image-edit-max": 0.5,
    "qwen-image-edit-max-2026-01-16": 0.5,
    "qwen-image-edit-plus": 0.2,
    "qwen-image-edit-plus-2025-12-15": 0.2,
    "qwen-image-edit-plus-2025-10-30": 0.2,
    "qwen-image-edit": 0.3,
}
_DEFAULT_IMAGE_PRICE = 0.5  # 未知图像模型默认价格（元/张）

# VL 模型单价（元/百万 Token，中国内地）
_VL_TOKEN_PRICE: Dict[str, Dict[str, float]] = {
    "qwen3-vl-plus": {"input": 1.0, "output": 10.0},
    "qwen3-vl-flash": {"input": 0.15, "output": 1.5},
    "qwen-vl-max": {"input": 3.0, "output": 9.0},
    "qwen-vl-plus": {"input": 1.5, "output": 4.5},
}
_VL_DEFAULT_PRICE = {"input": 1.0, "output": 10.0}
_VL_EST_INPUT_TOKENS = 3000   # 图像+提示词预估 token 数
_VL_EST_OUTPUT_TOKENS = 500   # 输出描述预估 token 数

# 万相视频模型单价（元/秒，720P / 默认档位）
_VIDEO_PRICE_PER_SECOND: Dict[str, float] = {
    "wan2.6-t2v": 0.6,
    "wan2.6-t2v-us": 0.733924,
    "wan2.5-t2v-preview": 0.6,
    "wan2.2-t2v-plus": 0.14,
    "wanx2.1-t2v-turbo": 0.24,
    "wanx2.1-t2v-plus": 0.70,
    "wan2.6-i2v-flash": 0.3,
    "wan2.6-i2v-plus": 0.6,
    "wan2.6-i2v-turbo": 0.3,
    "wanx2.1-i2v-turbo": 0.24,
    "wanx2.1-i2v-plus": 0.70,
}
_DEFAULT_VIDEO_PRICE_PER_SECOND = 0.5  # 未知视频模型默认价格（元/秒）

# 文本推理模型单价（元/百万 Token，中国内地）
_TEXT_TOKEN_PRICE: Dict[str, Dict[str, float]] = {
    "qwen3-235b-a22b": {"input": 1.0, "output": 8.0},
    "qwen3-30b-a3b": {"input": 0.22, "output": 0.88},
    "qwen3-8b": {"input": 0.05, "output": 0.2},
}
_TEXT_DEFAULT_PRICE = {"input": 1.0, "output": 8.0}
_TEXT_EST_INPUT_TOKENS = 400   # 场景生成 prompt 预估 token 数
_TEXT_EST_OUTPUT_TOKENS = 400  # 输出场景描述预估 token 数

_credit_lock = asyncio.Lock()


def get_image_credit_per_image(model_id: str) -> float:
    """获取图像模型每张图片的积分成本（积分）"""
    price = _IMAGE_PRICE_PER_IMAGE.get(model_id, _DEFAULT_IMAGE_PRICE)
    return round(price * CREDITS_PER_YUAN, 4)


def get_vl_credit_per_call(model_id: str) -> float:
    """估算 VL 模型每次调用的积分成本（积分）"""
    prices = _VL_TOKEN_PRICE.get(model_id, _VL_DEFAULT_PRICE)
    cost_yuan = (
        _VL_EST_INPUT_TOKENS / 1_000_000 * prices["input"] +
        _VL_EST_OUTPUT_TOKENS / 1_000_000 * prices["output"]
    )
    return round(cost_yuan * CREDITS_PER_YUAN, 4)


def get_text_credit_per_call(model_id: str) -> float:
    """估算文本推理模型每次调用的积分成本（积分）"""
    prices = _TEXT_TOKEN_PRICE.get(model_id, _TEXT_DEFAULT_PRICE)
    cost_yuan = (
        _TEXT_EST_INPUT_TOKENS / 1_000_000 * prices["input"] +
        _TEXT_EST_OUTPUT_TOKENS / 1_000_000 * prices["output"]
    )
    return round(cost_yuan * CREDITS_PER_YUAN, 4)


def get_video_credit_per_second(model_id: str) -> float:
    """获取视频模型每秒的积分成本（积分）"""
    price = _VIDEO_PRICE_PER_SECOND.get(model_id, _DEFAULT_VIDEO_PRICE_PER_SECOND)
    return round(price * CREDITS_PER_YUAN, 4)


def estimate_job_credits(model_id: str, mode: str, subtasks: List[Dict], video_params: Optional[Dict] = None, batch_size: int = 1) -> float:
    """估算整个任务的积分消耗（任务提交前调用）"""
    if not is_aliyun_model(model_id):
        return 0.0
    if mode in ("video", "multi_video"):
        duration = int((video_params or {}).get("duration", 5))
        return round(get_video_credit_per_second(model_id) * duration * len(subtasks), 4)
    if mode == "extract":
        vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
        vl_credit = get_vl_credit_per_call(vl_model)
        img_credit = get_image_credit_per_image(model_id)
        return round(len(subtasks) * (vl_credit + batch_size * img_credit), 4)
    if mode == "ecommerce":
        # VL 和文本模型费用在 step API 中实时扣除，此处只估算图像生成费用
        return round(len(subtasks) * get_image_credit_per_image(model_id), 4)
    # t2i / i2i / fission / convert：每个 subtask 生成 1 张
    return round(len(subtasks) * get_image_credit_per_image(model_id), 4)


def get_user_credit(username: str) -> float:
    """获取用户当前积分"""
    return raw_config.get("users", {}).get(username, {}).get("credit", 0.0)


async def deduct_credits(username: str, amount: float):
    """扣除用户积分并持久化至 config.json"""
    if amount <= 0:
        return
    async with _credit_lock:
        users = raw_config.setdefault("users", {})
        user_cfg = users.setdefault(username, {})
        user_cfg["credit"] = round(user_cfg.get("credit", 0) - amount, 4)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(raw_config, f, ensure_ascii=False, indent=4)


def is_aliyun_model(model_id: str) -> bool:
    return model_id.startswith("qwen") or model_id.startswith("wan") or model_id.startswith("qwen3-vl")


def get_aliyun_rate_rule(model_id: str) -> Dict[str, float]:
    if model_id in ALIYUN_QWEN_STRICT_MODELS:
        # 阿里云文档中该模型任务提交限制为 2 次/分钟，连续任务按 31 秒间隔处理更稳妥。
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
        "quota", "request rate increased", "allocated quota exceeded"
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
            self._next_allowed_at[model_id] = max(self._next_allowed_at.get(model_id, now), now + penalty)


aliyun_rate_limiter = AliyunRateLimiter()


def make_subtask(prompt: str, source_img: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:12],
        "prompt": prompt,
        "source_img": source_img,
        "status": "pending",
        "attempts": 0,
        "result_index": None,
    }


def build_subtasks(mode: str, prompts: List[str], source_image_paths: Optional[List[str]], batch_size: int) -> List[Dict[str, Any]]:
    if mode == "video":
        return [make_subtask(prompts[0] if prompts else "", source_image_paths[0] if source_image_paths else None)]

    # multi_video：每行提示词对应一个视频子任务，共享同一张参考图（若有）
    if mode == "multi_video":
        ref = source_image_paths[0] if source_image_paths else None
        return [make_subtask(p, ref) for p in prompts if p.strip()]

    # ecommerce 模式：prompts 与 source_image_paths 一一对应，逐对创建子任务
    if mode == "ecommerce":
        return [make_subtask(p, img) for p, img in zip(prompts, source_image_paths or [])]

    # extract 模式：每张图片对应一个 pipeline 子任务，batch_size 控制每张图生成几张输出
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

# ==========================================
# 3. 供应商策略模式封装 (Provider Pattern)
# ==========================================
class ImageProvider:
    async def generate(self, model_id: str, prompt: str, negative_prompt: str, img_path: str, user_dir: str, dl_base_name: str, user: str, target_ratio: str = "") -> Tuple[List[dict], str]:
        raise NotImplementedError()

class GeminiProvider(ImageProvider):
    def __init__(self):
        genai.configure(api_key=os.getenv("GENAI_API_KEY", ""), transport='rest', client_options={'api_endpoint': os.getenv("GENAI_API_ENDPOINT", "http://127.0.0.1:8045")})
    
    async def generate(self, model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user, target_ratio=""):
        model = genai.GenerativeModel(model_id)
        
        # 将比例要求写入提示词
        if target_ratio:
            prompt += f"\n\n[要求：请严格按照 {target_ratio} 的横纵比例生成或处理图像。]"
            
        final_prompt = prompt + (f"\n\nNegative Constraints: {negative_prompt}" if negative_prompt else "")
        contents = []
        if img_path and os.path.exists(img_path):
            try: contents.extend([PIL.Image.open(img_path), f"Reference image attached. Instruction: {final_prompt}"])
            except Exception: contents.append(final_prompt)
        else:
            contents.append(final_prompt)
            
        resp = await asyncio.to_thread(model.generate_content, contents)
        generated_images, final_text = [], ""
        
        if hasattr(resp, 'candidates') and resp.candidates:
            for cand in resp.candidates:
                if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                    for part in cand.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            data = base64.b64decode(part.inline_data.data) if isinstance(part.inline_data.data, str) else part.inline_data.data
                            ext = (part.inline_data.mime_type or "image/png").split('/')[-1].replace('jpeg', 'jpg')
                            filename = f"gen_{uuid.uuid4().hex[:8]}.{ext}"
                            with open(os.path.join(user_dir, filename), "wb") as f: f.write(data)
                            generated_images.append({"url": f"/api/images/{user}/{filename}", "download_name": f"{dl_base_name}.{ext}"})
        if not generated_images and hasattr(resp, 'text') and resp.text:
            text = resp.text
            base64_patterns = re.findall(r'!\[.*?\]\((data:image/([^;]+);base64,([^)]+))\)', text)
            if base64_patterns:
                for idx, (_, ext, b64_data) in enumerate(base64_patterns):
                    ext = ext.replace('jpeg', 'jpg')
                    filename = f"gen_{uuid.uuid4().hex[:8]}.{ext}"
                    with open(os.path.join(user_dir, filename), "wb") as f: f.write(base64.b64decode(b64_data))
                    generated_images.append({"url": f"/api/images/{user}/{filename}", "download_name": f"{dl_base_name}_{idx+1}.{ext}"})
                final_text = re.sub(r'!\[.*?\]\((data:image/[^;]+;base64,[^)]+)\)', '', text)
            else: final_text = text
        return generated_images, final_text

class MinimaxProvider(ImageProvider):
    async def generate(self, model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user, target_ratio=""):
        mm_model = model_id.replace("minimax-", "") if model_id.startswith("minimax-") else model_id
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('MINIMAX_API_KEY')}"}
        
        # 将比例要求写入提示词
        if target_ratio:
            prompt += f"\n\n[要求：请严格按照 {target_ratio} 的横纵比例生成或处理图像。]"
            
        final_prompt = prompt + (f"\n\n请尽量避免出现以下元素：{negative_prompt}" if negative_prompt else "")
        payload = {"model": mm_model, "prompt": final_prompt, "response_format": "base64", "n": 1, "prompt_optimizer": True}
        
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f: b64_data = base64.b64encode(f.read()).decode('utf-8')
            ext = os.path.splitext(img_path)[1].lower().replace('.', '').replace('jpg', 'jpeg')
            payload["subject_reference"] = [{"type": "character", "image_file": f"data:image/{ext or 'jpeg'};base64,{b64_data}"}]
            
        resp = await asyncio.to_thread(requests.post, "https://api.minimaxi.com/v1/image_generation", headers=headers, json=payload)
        resp_json = resp.json()
        if resp_json.get("base_resp", {}).get("status_code") != 0:
            raise Exception(f"MiniMax Error: {resp_json.get('base_resp', {}).get('status_msg')}")
            
        generated_images = []
        for b64_img in resp_json.get("data", {}).get("image_base64", []):
            filename = f"gen_{uuid.uuid4().hex[:8]}.png"
            with open(os.path.join(user_dir, filename), "wb") as f: f.write(base64.b64decode(b64_img))
            generated_images.append({"url": f"/api/images/{user}/{filename}", "download_name": f"{dl_base_name}.png"})
        return generated_images, ""

class QwenProvider(ImageProvider):
    def __init__(self): dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
    
    async def generate(self, model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user, target_ratio=""):
        await aliyun_rate_limiter.wait(model_id)
        size_param = "1024*1024"
        
        # 尺寸策略：qwen-image-2.0-pro 传参数，其他模型改提示词
        if target_ratio:
            if model_id == "qwen-image-2.0-pro":
                size_param = map_ratio_to_size(target_ratio)
            else:
                prompt += f"\n\n[要求：请严格按照 {target_ratio} 的横纵比例生成或处理图像。]"

        content_list = []
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f: b64_data = base64.b64encode(f.read()).decode('utf-8')
            ext = os.path.splitext(img_path)[1].lower().replace('.', '').replace('jpg', 'jpeg')
            content_list.append({"image": f"data:image/{ext or 'jpeg'};base64,{b64_data}"})
        content_list.append({"text": prompt})
        
        kwargs = {"model": model_id, "messages": [{"role": "user", "content": content_list}], "n": 1, "size": size_param}
        if negative_prompt: kwargs["negative_prompt"] = negative_prompt
        
        rsp = await asyncio.to_thread(MultiModalConversation.call, **kwargs)
        if rsp.status_code != 200: raise Exception(f"Qwen Error: {rsp.message}")
        
        generated_images = []
        for item in rsp.output.choices[0].message.content:
            if 'image' in item:
                img_data = await asyncio.to_thread(requests.get, item['image'])
                filename = f"gen_{uuid.uuid4().hex[:8]}.png"
                with open(os.path.join(user_dir, filename), "wb") as f: f.write(img_data.content)
                generated_images.append({"url": f"/api/images/{user}/{filename}", "download_name": f"{dl_base_name}.png"})
        return generated_images, ""

class DoubaoProvider(ImageProvider):
    async def generate(self, model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user, target_ratio=""):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('ARK_API_KEY')}"}
        
        # 将比例要求写入提示词
        if target_ratio:
            prompt += f"\n\n[要求：请严格按照 {target_ratio} 的横纵比例生成或处理图像。]"
            
        final_prompt = prompt + (f"\n\n请尽量避免出现以下元素：{negative_prompt}" if negative_prompt else "")
        payload = {"model": model_id, "prompt": final_prompt, "response_format": "url", "size": "2K", "stream": False, "watermark": False}
        
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f: b64_data = base64.b64encode(f.read()).decode('utf-8')
            ext = os.path.splitext(img_path)[1].lower().replace('.', '').replace('jpg', 'jpeg')
            payload["image"] = f"data:image/{ext or 'jpeg'};base64,{b64_data}"
            
        resp = await asyncio.to_thread(requests.post, "https://ark.cn-beijing.volces.com/api/v3/images/generations", headers=headers, json=payload)
        resp_json = resp.json()
        if "error" in resp_json: raise Exception(f"Doubao Error: {resp_json['error'].get('message', str(resp_json['error']))}")
        
        generated_images = []
        for item in resp_json.get("data", []):
            if "url" in item:
                img_data = await asyncio.to_thread(requests.get, item['url'])
                filename = f"gen_{uuid.uuid4().hex[:8]}.png"
                with open(os.path.join(user_dir, filename), "wb") as f: f.write(img_data.content)
                generated_images.append({"url": f"/api/images/{user}/{filename}", "download_name": f"{dl_base_name}.png"})
            elif "b64_json" in item:
                filename = f"gen_{uuid.uuid4().hex[:8]}.png"
                with open(os.path.join(user_dir, filename), "wb") as f: f.write(base64.b64decode(item["b64_json"]))
                generated_images.append({"url": f"/api/images/{user}/{filename}", "download_name": f"{dl_base_name}.png"})
        return generated_images, ""

def get_provider_for_model(model_id: str) -> ImageProvider:
    for group in AVAILABLE_MODELS:
        for m in group["models"]:
            if m["id"] == model_id:
                if m["prefix"] == "gemini": return GeminiProvider()
                if m["prefix"] == "qwen": return QwenProvider()
                if m["prefix"] == "minimax": return MinimaxProvider()
                if m["prefix"] == "doubao": return DoubaoProvider()
    return GeminiProvider() # Default fallback

# ==========================================
# 图案提取 - Qwen VL 视觉理解
# ==========================================
_PATTERN_EXTRACT_PROMPT = (
    "Please look at the pattern or design printed/embroidered on this product. "
    "Write a detailed English prompt describing only the pattern/design itself — "
    "do NOT mention the product type, material, or shape. "
    "Focus on: subject, style, colors, composition, texture, and mood of the pattern. "
    "Reply with the prompt text only, no extra explanation."
)

async def extract_pattern_prompt(image_path: str) -> str:
    vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
    await aliyun_rate_limiter.wait(vl_model)
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".").replace("jpg", "jpeg") or "jpeg"
    messages = [{"role": "user", "content": [
        {"image": f"data:image/{ext};base64,{b64_data}"},
        {"text": _PATTERN_EXTRACT_PROMPT},
    ]}]
    rsp = await asyncio.to_thread(MultiModalConversation.call, model=vl_model, messages=messages)
    if rsp.status_code != 200:
        raise Exception(f"Qwen VL 图案提取失败: {rsp.message}")
    for item in rsp.output.choices[0].message.content:
        if "text" in item:
            return item["text"].strip()
    raise Exception("Qwen VL 未返回文字描述")

# ==========================================
# 电商场景生图 - 三步 Pipeline
# ==========================================
_PRODUCT_UNDERSTAND_PROMPT = (
    "Analyze this product image for an e-commerce listing. "
    "Provide a detailed English description including: product category and type, "
    "primary colors and materials/textures, shape and key visual characteristics, "
    "notable design elements or features, any visible branding or text. "
    "Focus ONLY on the product itself — do not describe the background or setting. "
    "Reply with the description only, no extra explanation."
)

_SCENE_GENERATE_SYSTEM_PROMPT = (
    "You are an expert e-commerce product photographer and creative director. "
    "Generate diverse, practical photography scene descriptions for product listings. "
    "Each scene should be realistic, enhance the product's visual appeal, and suit different buyer contexts."
)


def _build_scene_prompt(product_description: str, scene_count: int) -> str:
    return (
        f"Based on this product description, generate {scene_count} distinct e-commerce photography scene prompts.\n\n"
        f"Product: {product_description}\n\n"
        f"Requirements:\n"
        f"- Cover different visual contexts: pure studio shot, lifestyle scene, flat lay, seasonal/themed, outdoor, etc.\n"
        f"- Each scene must naturally showcase the product as the hero subject\n"
        f"- Vary lighting styles, backgrounds, and compositions\n"
        f"- Keep each prompt concise (40-80 words) and ready to use for image generation\n\n"
        f"Output a valid JSON array of exactly {scene_count} English strings.\n"
        f"Format: [\"scene 1 prompt\", \"scene 2 prompt\", ...]\n"
        f"Output the JSON array only, no other text."
    )


def _parse_scenes_from_response(raw: str, expected_count: int) -> List[str]:
    """从模型响应中解析场景 prompt 列表，支持 JSON 数组或降级换行分割"""
    # 去除 <think>...</think> 思考块
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', raw, flags=re.IGNORECASE).strip()
    # 尝试提取 JSON 数组
    match = re.search(r'\[[\s\S]*?\]', cleaned)
    if match:
        try:
            scenes = json.loads(match.group())
            if isinstance(scenes, list):
                return [str(s).strip() for s in scenes if str(s).strip()][:expected_count]
        except json.JSONDecodeError:
            pass
    # 降级：按行分割并清理编号
    lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
    scenes = []
    for line in lines:
        cleaned_line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip('"\'')
        if cleaned_line and len(cleaned_line) > 10:
            scenes.append(cleaned_line)
    return scenes[:expected_count]


async def understand_product(image_path: str) -> str:
    """调用 Qwen VL 模型分析产品图片，返回英文产品描述"""
    vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
    await aliyun_rate_limiter.wait(vl_model)
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".").replace("jpg", "jpeg") or "jpeg"
    messages = [{"role": "user", "content": [
        {"image": f"data:image/{ext};base64,{b64_data}"},
        {"text": _PRODUCT_UNDERSTAND_PROMPT},
    ]}]
    rsp = await asyncio.to_thread(MultiModalConversation.call, model=vl_model, messages=messages)
    if rsp.status_code != 200:
        raise Exception(f"Qwen VL 产品理解失败: {rsp.message}")
    for item in rsp.output.choices[0].message.content:
        if "text" in item:
            return item["text"].strip()
    raise Exception("Qwen VL 未返回产品描述")


# ==========================================
# 3D图片转换 - 两步 Pipeline
# ==========================================
_THREED_UNDERSTAND_PROMPT = (
    "Analyze this image and describe the pattern, design, or graphic content visible in it. "
    "Provide a concise, detailed English description of: the main subject or motif, colors, artistic style, "
    "shapes, compositional elements, and any notable textures or visual details. "
    "Focus on what would help recreate this as an embroidery design. "
    "Reply with the description only, no extra explanation."
)

_THREED_PROMPT_TEMPLATE = (
    "A highly detailed realistic embroidery of {description}, stitched with colorful threads on a clean, "
    "plain knitted/woven fabric background. The design features intricate thread work with visible stitch patterns, "
    "3D raised embroidery effect, satin stitch and fill stitch techniques, precise needlework details. "
    "The embroidery faithfully reproduces the original flat illustration with textured yarn and thread, "
    "showing realistic fiber texture and slight dimensional relief. Clean and minimal background with no other objects, "
    "no clutter, no wrinkles, no stains. Centered composition, natural soft lighting, close-up macro photography style, studio shot."
)


async def understand_threed_pattern(image_path: str) -> str:
    """调用 Qwen VL 模型分析图片图案内容，返回英文图案描述"""
    vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
    await aliyun_rate_limiter.wait(vl_model)
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".").replace("jpg", "jpeg") or "jpeg"
    messages = [{"role": "user", "content": [
        {"image": f"data:image/{ext};base64,{b64_data}"},
        {"text": _THREED_UNDERSTAND_PROMPT},
    ]}]
    rsp = await asyncio.to_thread(MultiModalConversation.call, model=vl_model, messages=messages)
    if rsp.status_code != 200:
        raise Exception(f"Qwen VL 图案分析失败: {rsp.message}")
    for item in rsp.output.choices[0].message.content:
        if "text" in item:
            return item["text"].strip()
    raise Exception("Qwen VL 未返回图案描述")


async def call_qwen_text_model(prompt: str, system_prompt: str = "") -> str:
    """调用 Qwen3 文本推理模型，返回文本响应"""
    model_id = os.getenv("QWEN_TEXT_MODEL", "qwen3-235b-a22b")
    await aliyun_rate_limiter.wait(model_id)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    rsp = await asyncio.to_thread(
        Generation.call,
        model=model_id,
        messages=messages,
        result_format="message",
        enable_thinking=False,  # 关闭思考模式，减少 token 消耗
    )
    if rsp.status_code != 200:
        raise Exception(f"Qwen 文本模型调用失败: {rsp.message}")
    return rsp.output.choices[0].message.content.strip()


# ==========================================
# 用户设置持久化
# ==========================================
def _user_settings_path(username: str) -> str:
    return f"users/{username}/settings.json"


def load_user_settings(username: str) -> dict:
    path = _user_settings_path(username)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_user_settings_sync(username: str, settings: dict):
    path = _user_settings_path(username)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

# ==========================================
# 万相视频生成 - DashScope OSS 文件上传
# ==========================================
async def upload_to_dashscope(file_path: str, model_id: str) -> str:
    """将本地文件上传至 DashScope OSS，返回 oss:// 临时 URL"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    filename = os.path.basename(file_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # 修复点 1：根据官方文档，getPolicy 的必需参数为 action 和 model。
    params = {
        "action": "getPolicy",
        "model": model_id
    }
    
    policy_resp = await asyncio.to_thread(
        requests.get,
        "https://dashscope.aliyuncs.com/api/v1/uploads",
        headers=headers,
        params=params
    )
    if policy_resp.status_code != 200:
        raise Exception(f"DashScope 上传策略获取失败 ({policy_resp.status_code}): {policy_resp.text}")

    policy_data = policy_resp.json().get("data", {})
    key = f"{policy_data.get('upload_dir', '')}/{filename}"

    with open(file_path, 'rb') as f:
        file_content = f.read()

    # 修复点 2：严格按照官方文档字段拼接 OSS 上传表单
    files = {
        'OSSAccessKeyId': (None, policy_data.get('oss_access_key_id')),
        'Signature': (None, policy_data.get('signature')),
        'policy': (None, policy_data.get('policy')),
        'key': (None, key),
        'success_action_status': (None, '200'),
    }
    
    if 'x_oss_object_acl' in policy_data:
        files['x-oss-object-acl'] = (None, policy_data['x_oss_object_acl'])
    if 'x_oss_forbid_overwrite' in policy_data:
        files['x-oss-forbid-overwrite'] = (None, policy_data['x_oss_forbid_overwrite'])
        
    files['file'] = (filename, file_content)

    oss_resp = await asyncio.to_thread(
        requests.post, 
        policy_data.get('upload_host'),
        files=files
    )
    
    if oss_resp.status_code not in [200, 204]:
        raise Exception(f"OSS 上传失败 (HTTP {oss_resp.status_code}): {oss_resp.text[:200]}")

    return f"oss://{key}"


class WanVideoProvider:
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.create_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        self.query_base = "https://dashscope.aliyuncs.com/api/v1/tasks"

    async def generate(self, model_id: str, prompt: str, reference_paths: List[str],
                       user_dir: str, dl_base_name: str, user: str,
                       size: str = "1280*720", duration: int = 5,
                       shot_type: str = "single", audio: bool = True,
                       watermark: bool = False) -> Tuple[List[dict], str]:
        # 上传参考文件，获取 OSS 临时 URL（图片→reference_urls，视频→reference_video_urls）
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
            }
        }
        if reference_urls:
            payload["input"]["reference_urls"] = reference_urls
        if reference_video_urls:
            payload["input"]["reference_video_urls"] = reference_video_urls
        # audio 参数仅 flash 模型支持
        if "flash" in model_id:
            payload["parameters"]["audio"] = audio

        headers = {
            "X-DashScope-Async": "enable",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # 修复点 4：使用 OSS 临时地址进行推理时，必须补充启用资源解析 Header
            "X-DashScope-OssResourceResolve": "enable"
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

        # 轮询结果（最多等待 20 分钟）
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
            # PENDING / RUNNING: 继续等待

        raise Exception(f"视频生成超时（task_id: {task_id}）")

# ==========================================
# 4. 任务队列调度
# ==========================================
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


class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.queue = asyncio.Queue()

    def sync_user_jobs(self, user):
        user_dir = f"users/{user}"
        os.makedirs(user_dir, exist_ok=True)
        with open(f"{user_dir}/jobs.json", "w", encoding="utf-8") as f:
            json.dump([j for j in self.jobs.values() if j.get("user") == user], f, ensure_ascii=False)

    def load_jobs(self):
        if not os.path.exists("users"): return
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
                                            "source_img": os.path.basename(subtask["source_img"]).split('_', 1)[-1] if subtask.get("source_img") else None,
                                            "error": "服务曾被重启，该子任务已中断",
                                            "status": "error",
                                            "attempts": subtask["attempts"],
                                        })
                                refresh_job_progress(j)
                            self.jobs[j["id"]] = j
                except Exception: pass

    async def add_job(self, user, mode, prompts, source_image_paths=None, template_name="", model_id="", negative_prompt="", batch_size=1, target_ratio="", video_params=None):
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

    async def retry_failed_subtasks(self, job_id: str, task_ids: Optional[List[str]] = None):
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

job_queue = JobQueue()

async def process_queue():
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

        # ==========================================
        # 图案提取生图模式 (Qwen VL + 图像模型)
        # ==========================================
        if job["mode"] == "extract":
            batch_size = job.get("batch_size", 1)
            provider = get_provider_for_model(model_id)
            tasks = [t for t in job.get("subtasks", []) if t.get("status") == "pending"]

            sem_extract = asyncio.Semaphore(SUBTASK_CONCURRENCY)
            completed_times_extract: List[float] = []

            async def run_extract_subtask(subtask):
                async with sem_extract:
                    start_time = time.time()
                    img_path = subtask.get("source_img")
                    subtask["attempts"] = subtask.get("attempts", 0) + 1
                    try:
                        # 步骤 1：VL 提取图案描述
                        extracted_prompt = await extract_pattern_prompt(img_path)

                        # 步骤 2：使用提取的 prompt + 原图生成 batch_size 张图片
                        all_images = []
                        for j in range(batch_size):
                            dl_base_name = re.sub(r'[\\/*?:"<>|]', "", f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{j+1}_{uuid.uuid4().hex[:4]}")
                            imgs, _ = await run_with_retries(
                                lambda p=extracted_prompt, b=dl_base_name: provider.generate(model_id, p, "", img_path, user_dir, b, user, target_ratio),
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
                    completed_times_extract.append(elapsed)
                    avg_time = sum(completed_times_extract) / len(completed_times_extract)
                    pending_count = sum(1 for t in job.get("subtasks", []) if t.get("status") == "pending")
                    job["eta"] = max(0, (pending_count / SUBTASK_CONCURRENCY) * avg_time)
                    refresh_job_progress(job)
                    job_queue.sync_user_jobs(user)

            await asyncio.gather(*[run_extract_subtask(t) for t in tasks])

            job["status"] = "completed"
            job["eta"] = 0
            job_queue.sync_user_jobs(user)
            job_queue.queue.task_done()
            continue

        # ==========================================
        # 视频生成模式 (万相 WAN) — 单视频 & 多视频顺序执行
        # ==========================================
        if job["mode"] in ("video", "multi_video"):
            pending_tasks = [t for t in job.get("subtasks", []) if t.get("status") == "pending"]
            if not pending_tasks:
                job["status"] = "completed"
                job["eta"] = 0
                job_queue.sync_user_jobs(user)
                job_queue.queue.task_done()
                continue
            vp = job.get("video_params", {})
            wan_provider = WanVideoProvider()
            total_video = len(pending_tasks)
            for idx, subtask in enumerate(pending_tasks):
                prompt = subtask.get("prompt", "")
                dl_base_name = re.sub(r'[\\/*?:"<>|]', "", f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx+1}")
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
                        "source_img": os.path.basename(subtask["source_img"]).split('_', 1)[-1] if subtask.get("source_img") else None,
                        "videos": videos, "images": [], "status": "success",
                        "attempts": subtask["attempts"], "credit_cost": credit_cost,
                    })
                except Exception as e:
                    subtask["status"] = "error"
                    upsert_task_result(job, subtask, {
                        "prompt": prompt,
                        "source_img": os.path.basename(subtask["source_img"]).split('_', 1)[-1] if subtask.get("source_img") else None,
                        "error": str(e), "status": "error", "attempts": subtask["attempts"],
                    })
                refresh_job_progress(job)
                remaining = total_video - (idx + 1)
                job["eta"] = remaining * 300  # 粗略预估每视频约 5 分钟
                job_queue.sync_user_jobs(user)
            job["status"] = "completed"
            job["eta"] = 0
            job_queue.sync_user_jobs(user)
            job_queue.queue.task_done()
            continue

        # ==========================================
        # 图像生成模式 (并发执行)
        # ==========================================
        provider = get_provider_for_model(model_id)
        tasks = [t for t in job.get("subtasks", []) if t.get("status") == "pending"]

        sem = asyncio.Semaphore(SUBTASK_CONCURRENCY)
        completed_times: List[float] = []

        async def run_image_subtask(subtask):
            async with sem:
                start_time = time.time()
                prompt = subtask.get("prompt", "")
                img_path = subtask.get("source_img")
                try:
                    base_src = os.path.splitext(os.path.basename(img_path).split('_', 1)[-1])[0] if img_path else "t2i"
                    dl_base_name = re.sub(r'[\\/*?:"<>|]', "", f"{base_src}_{tpl_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}" if tpl_name else f"{base_src}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}")
                    subtask["attempts"] = subtask.get("attempts", 0) + 1
                    generated_images, final_text = await run_with_retries(
                        lambda: provider.generate(model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user, target_ratio),
                        model_id=model_id,
                    )
                    subtask["status"] = "success"
                    credit_cost = round(len(generated_images) * get_image_credit_per_image(model_id), 4)
                    await deduct_credits(user, credit_cost)
                    upsert_task_result(job, subtask, {
                        "prompt": prompt,
                        "source_img": os.path.basename(img_path).split('_', 1)[-1] if img_path else None,
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
                        "source_img": os.path.basename(img_path).split('_', 1)[-1] if img_path else None,
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

        await asyncio.gather(*[run_image_subtask(t) for t in tasks])

        job["status"] = "completed"
        job["eta"] = 0
        job_queue.sync_user_jobs(user)
        job_queue.queue.task_done()

@app.on_event("startup")
async def startup_event():
    job_queue.load_jobs()
    asyncio.create_task(process_queue())

# ==========================================
# 5. API 路由定义
# ==========================================
@app.post("/api/login")
def login(username: str = Form(...), password: str = Form(...)):
    if USERS.get(username, {}).get("password") == password:
        token = str(uuid.uuid4())
        SESSIONS[token] = username
        return {"access_token": token, "username": username, "is_admin": USERS[username].get("is_admin", False)}
    return JSONResponse(status_code=401, content={"error": "Invalid credentials"})

@app.get("/api/me")
def get_me(curr: dict = Depends(get_current_user)): return curr

@app.get("/api/credit")
def get_credit(curr: dict = Depends(get_current_user)):
    return {"credit": get_user_credit(curr["username"])}

@app.post("/api/credit")
async def set_credit(request: Request, curr: dict = Depends(get_current_user)):
    if not curr.get("is_admin"):
        return JSONResponse(status_code=403, content={"error": "仅管理员可操作"})
    data = await request.json()
    target_user = data.get("username", curr["username"])
    amount = float(data.get("credit", 0))
    async with _credit_lock:
        raw_config.setdefault("users", {}).setdefault(target_user, {})["credit"] = round(amount, 4)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(raw_config, f, ensure_ascii=False, indent=4)
    return {"username": target_user, "credit": round(amount, 4)}
@app.post("/api/logout")
def logout(token: str = Depends(oauth2_scheme)):
    if token in SESSIONS: del SESSIONS[token]
    return {"success": True}
@app.get("/api/models")
def get_models(curr: dict = Depends(get_current_user)): return AVAILABLE_MODELS
@app.get("/api/templates")
def get_templates(curr: dict = Depends(get_current_user)):
    pub = json.load(open(PUBLIC_TEMPLATES_FILE, "r", encoding="utf-8")) if os.path.exists(PUBLIC_TEMPLATES_FILE) else []
    priv_path = f"users/{curr['username']}/templates.json"
    priv = json.load(open(priv_path, "r", encoding="utf-8")) if os.path.exists(priv_path) else []
    return {"public": pub, "private": priv}
@app.post("/api/templates")
async def save_template(request: Request, curr: dict = Depends(get_current_user)):
    data = await request.json()
    user, is_pub = curr["username"], data.get("is_public", False)
    target = PUBLIC_TEMPLATES_FILE if is_pub else f"users/{user}/templates.json"
    items = json.load(open(target, "r", encoding="utf-8")) if os.path.exists(target) else []
    new_item = {"name": data["name"], "content": data["content"], "negative_prompt": data.get("negative_prompt", ""), "author": user}
    idx = next((i for i, x in enumerate(items) if x["name"] == data["name"]), -1)
    if idx >= 0:
        if is_pub and items[idx].get("author") != user and not curr["is_admin"]: return JSONResponse(status_code=403, content={"error": "无权修改他人的模板"})
        items[idx] = new_item
    else: items.append(new_item)
    json.dump(items, open(target, "w", encoding="utf-8"), ensure_ascii=False)
    return {"success": True}
@app.delete("/api/templates/{scope}/{name}")
def delete_template(scope: str, name: str, curr: dict = Depends(get_current_user)):
    user, target = curr["username"], PUBLIC_TEMPLATES_FILE if scope == "public" else f"users/{curr['username']}/templates.json"
    if not os.path.exists(target): return {"success": True}
    items = json.load(open(target, "r", encoding="utf-8"))
    idx = next((i for i, x in enumerate(items) if x["name"] == name), -1)
    if idx >= 0:
        if scope == "public" and items[idx].get("author") != user and not curr["is_admin"]: return JSONResponse(status_code=403, content={"error": "无权删除"})
        items.pop(idx)
        json.dump(items, open(target, "w", encoding="utf-8"), ensure_ascii=False)
    return {"success": True}

@app.get("/api/settings")
def get_settings(curr: dict = Depends(get_current_user)):
    return load_user_settings(curr["username"])


@app.post("/api/settings")
async def update_settings(request: Request, curr: dict = Depends(get_current_user)):
    data = await request.json()
    settings = load_user_settings(curr["username"])
    settings.update(data)
    save_user_settings_sync(curr["username"], settings)
    return settings


@app.post("/api/ecommerce/understand")
async def ecommerce_understand(
    images: Optional[List[UploadFile]] = File(None),
    curr: dict = Depends(get_current_user)
):
    """Step 1：Qwen VL 分析每张产品图，返回产品描述"""
    user = curr["username"]
    vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
    user_dir = f"users/{user}/outputs"
    os.makedirs(user_dir, exist_ok=True)

    if not images:
        return JSONResponse(status_code=400, content={"error": "请上传至少一张产品图片"})

    items = []
    for img in images:
        if not img or not getattr(img, "filename", None):
            continue
        fname = f"{uuid.uuid4().hex[:8]}_{img.filename}"
        path = os.path.join(user_dir, fname)
        img_data = await img.read()
        with open(path, "wb") as f:
            f.write(img_data)

        display_url = f"/api/images/{user}/{fname}"
        try:
            description = await understand_product(path)
            credit_cost = get_vl_credit_per_call(vl_model)
            await deduct_credits(user, credit_cost)
            items.append({
                "image_path": path,
                "image_name": img.filename,
                "display_url": display_url,
                "description": description,
                "credit_cost": credit_cost,
            })
        except Exception as e:
            items.append({
                "image_path": path,
                "image_name": img.filename,
                "display_url": display_url,
                "description": "",
                "error": str(e),
            })

    return {"items": items}


@app.post("/api/ecommerce/scenes")
async def ecommerce_scenes(request: Request, curr: dict = Depends(get_current_user)):
    """Step 2：Qwen3 文本模型为每个产品描述生成电商场景 prompt 列表"""
    user = curr["username"]
    data = await request.json()
    items_in = data.get("items", [])
    scene_count = max(1, min(int(data.get("scene_count", 3)), 20))
    text_model = os.getenv("QWEN_TEXT_MODEL", "qwen3-235b-a22b")

    items_out = []
    for item in items_in:
        desc = (item.get("description") or "").strip()
        image_path = item.get("image_path", "")
        image_name = item.get("image_name", "")
        display_url = item.get("display_url", "")
        if not desc:
            items_out.append({
                "image_path": image_path, "image_name": image_name,
                "display_url": display_url, "description": desc,
                "scenes": [], "error": "产品描述为空，无法生成场景",
            })
            continue
        try:
            scene_prompt = _build_scene_prompt(desc, scene_count)
            raw = await call_qwen_text_model(scene_prompt, _SCENE_GENERATE_SYSTEM_PROMPT)
            scenes = _parse_scenes_from_response(raw, scene_count)
            credit_cost = get_text_credit_per_call(text_model)
            await deduct_credits(user, credit_cost)
            items_out.append({
                "image_path": image_path, "image_name": image_name,
                "display_url": display_url, "description": desc,
                "scenes": scenes, "credit_cost": credit_cost,
            })
        except Exception as e:
            items_out.append({
                "image_path": image_path, "image_name": image_name,
                "display_url": display_url, "description": desc,
                "scenes": [], "error": str(e),
            })

    return {"items": items_out}


@app.post("/api/threed/understand")
async def threed_understand(
    images: Optional[List[UploadFile]] = File(None),
    curr: dict = Depends(get_current_user)
):
    """Step 1：Qwen VL 分析每张图片的图案内容，返回图案描述"""
    user = curr["username"]
    vl_model = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")
    user_dir = f"users/{user}/outputs"
    os.makedirs(user_dir, exist_ok=True)

    if not images:
        return JSONResponse(status_code=400, content={"error": "请上传至少一张图片"})

    items = []
    for img in images:
        if not img or not getattr(img, "filename", None):
            continue
        fname = f"{uuid.uuid4().hex[:8]}_{img.filename}"
        path = os.path.join(user_dir, fname)
        img_data = await img.read()
        with open(path, "wb") as f:
            f.write(img_data)

        display_url = f"/api/images/{user}/{fname}"
        try:
            description = await understand_threed_pattern(path)
            credit_cost = get_vl_credit_per_call(vl_model)
            await deduct_credits(user, credit_cost)
            items.append({
                "image_path": path,
                "image_name": img.filename,
                "display_url": display_url,
                "description": description,
                "credit_cost": credit_cost,
            })
        except Exception as e:
            items.append({
                "image_path": path,
                "image_name": img.filename,
                "display_url": display_url,
                "description": "",
                "error": str(e),
            })

    return {"items": items}


@app.post("/api/jobs")
async def create_job(
    prompts: str = Form(""), negative_prompt: str = Form(""), mode: str = Form(...),
    template_name: str = Form(""), model_id: str = Form(""), batch_size: int = Form(1),
    target_ratio: str = Form(""),
    video_size: str = Form("1280*720"), video_duration: int = Form(5),
    video_shot_type: str = Form("single"), video_audio: bool = Form(True),
    video_watermark: bool = Form(False),
    ecommerce_data: str = Form(""), threed_data: str = Form(""),
    images: Optional[List[UploadFile]] = File(None), curr: dict = Depends(get_current_user)
):
    # ── 电商场景模式：直接构建子任务并提交，提前返回 ──
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
            curr['username'], "ecommerce", prompts_ec, source_paths_ec,
            template_name, model_id, negative_prompt, 1, target_ratio, video_params=None
        )
        return job

    # ── 3D图片转换模式：构建最终提示词并提交纯文生图子任务 ──
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
                final_prompt = _THREED_PROMPT_TEMPLATE.format(description=desc)
                prompts_td.append(final_prompt)
        if not prompts_td:
            return JSONResponse(status_code=400, content={"error": "没有有效的图案描述数据"})
        job = await job_queue.add_job(
            curr['username'], "threed", prompts_td, None,
            template_name, model_id, negative_prompt, 1, target_ratio, video_params=None
        )
        return job

    prompt_str = prompts.strip()

    if mode == "fission" and not prompt_str:
        prompt_str = "Generate a high-quality, stylistically similar variation of the provided reference image.it should incorporate a certain degree of variation and be different to the original."
    elif mode == "convert" and not prompt_str:
        prompt_str = "保持原图主体结构和风格不变，将画面自然延展或重绘以适应设定的新比例尺寸，边缘过渡自然。"
    elif mode == "extract":
        prompt_str = ""  # prompt 由 VL 模型自动生成，前端无需填写
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
                path = os.path.join(f"users/{curr['username']}/outputs", f"{uuid.uuid4().hex[:8]}_{img.filename}")
                img_data = await img.read()
                with open(path, "wb") as f:
                    f.write(img_data)
                source_paths.append(path)

    video_params = None
    if mode in ("video", "multi_video"):
        video_params = {
            "size": video_size, "duration": video_duration,
            "shot_type": video_shot_type, "audio": video_audio, "watermark": video_watermark
        }

    # multi 模式：按换行拆分为多个子任务
    if mode in ("multi_t2i", "multi_video"):
        prompts_list = [p.strip() for p in prompt_str.split('\n') if p.strip()]
    else:
        prompts_list = [prompt_str]

    job = await job_queue.add_job(
        curr['username'], mode, prompts_list, source_paths,
        template_name, model_id, negative_prompt, batch_size, target_ratio,
        video_params=video_params
    )
    return job

@app.get("/api/jobs")
def get_jobs(curr: dict = Depends(get_current_user)):
    return sorted([j for j in job_queue.jobs.values() if j.get("user") == curr["username"]], key=lambda x: x['created_at'], reverse=True)

@app.post("/api/jobs/{job_id}/retry-failed")
async def retry_failed_job_tasks(job_id: str, curr: dict = Depends(get_current_user)):
    job = job_queue.jobs.get(job_id)
    if not job or (job["user"] != curr["username"] and not curr["is_admin"]):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    try:
        return await job_queue.retry_failed_subtasks(job_id)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

@app.post("/api/jobs/{job_id}/retry-task/{task_id}")
async def retry_single_job_task(job_id: str, task_id: str, curr: dict = Depends(get_current_user)):
    job = job_queue.jobs.get(job_id)
    if not job or (job["user"] != curr["username"] and not curr["is_admin"]):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    try:
        return await job_queue.retry_failed_subtasks(job_id, [task_id])
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, curr: dict = Depends(get_current_user)):
    job = job_queue.jobs.get(job_id)
    if not job or (job["user"] != curr["username"] and not curr["is_admin"]): return JSONResponse(status_code=403, content={"error": "Forbidden"})
    del job_queue.jobs[job_id]
    job_queue.sync_user_jobs(job["user"])
    return {"success": True}

@app.get("/api/images/{username}/{filename}")
async def serve_img(username: str, filename: str, curr: dict = Depends(get_current_user)):
    if curr["username"] != username and not curr["is_admin"]: return JSONResponse(status_code=403, content={"error": "Forbidden"})
    path = os.path.join(f"users/{username}/outputs", filename)
    return FileResponse(path) if os.path.exists(path) else JSONResponse(status_code=404, content={"error": "Not found"})

@app.get("/api/videos/{username}/{filename}")
async def serve_video(username: str, filename: str, curr: dict = Depends(get_current_user)):
    if curr["username"] != username and not curr["is_admin"]: return JSONResponse(status_code=403, content={"error": "Forbidden"})
    path = os.path.join(f"users/{username}/outputs", filename)
    return FileResponse(path, media_type="video/mp4") if os.path.exists(path) else JSONResponse(status_code=404, content={"error": "Not found"})

@app.get("/api/jobs/{job_id}/download")
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
                    unique_prefix = filename.split('_')[1].split('.')[0] if '_' in filename else uuid.uuid4().hex[:6]
                    zip_name = f"{unique_prefix}_{img.get('download_name', filename)}"
                    zip_file.write(file_path, arcname=zip_name)
            for vid in res.get("videos", []):
                filename = vid["url"].split("/")[-1]
                file_path = os.path.join(f"users/{job['user']}/outputs", filename)
                if os.path.exists(file_path):
                    zip_file.write(file_path, arcname=vid.get('download_name', filename))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=job_{job_id}_images.zip"}
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")
