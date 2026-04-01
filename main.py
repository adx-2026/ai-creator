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
from dashscope import MultiModalConversation
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

load_dotenv()
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
        # 上传参考文件，获取 OSS 临时 URL
        await aliyun_rate_limiter.wait(model_id)
        reference_urls = []
        for path in reference_paths:
            if path and os.path.exists(path):
                # 修复点 3：传入对应的 model_id 获取专属于该模型的临时上传凭证
                url = await upload_to_dashscope(path, model_id)
                reference_urls.append(url)

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
        self.jobs[job_id] = {
            "id": job_id, "user": user, "mode": mode, "model_id": model_id,
            "status": "queued", "total": 0, "completed": 0, "failed": 0,
            "results": [], "created_at": time.time(), "eta": None,
            "template_name": template_name, "negative_prompt": negative_prompt,
            "target_ratio": target_ratio, "video_params": video_params or {},
            "batch_size": batch_size,
            "subtasks": build_subtasks(mode, prompts, source_image_paths, batch_size)
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
                        upsert_task_result(job, subtask, {
                            "prompt": extracted_prompt,
                            "extracted_prompt": extracted_prompt,
                            "source_img": os.path.basename(img_path).split("_", 1)[-1] if img_path else None,
                            "images": all_images,
                            "status": "success",
                            "attempts": subtask["attempts"],
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
        # 视频生成模式 (万相 WAN)
        # ==========================================
        if job["mode"] == "video":
            subtask = next((t for t in job.get("subtasks", []) if t.get("status") == "pending"), None)
            if not subtask:
                job["status"] = "completed"
                job["eta"] = 0
                job_queue.sync_user_jobs(user)
                job_queue.queue.task_done()
                continue
            prompt = subtask.get("prompt", "")
            dl_base_name = re.sub(r'[\\/*?:"<>|]', "", f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            vp = job.get("video_params", {})
            try:
                subtask["attempts"] = subtask.get("attempts", 0) + 1
                wan_provider = WanVideoProvider()
                videos, _ = await run_with_retries(
                    lambda: wan_provider.generate(
                        model_id=model_id,
                        prompt=prompt,
                        reference_paths=[subtask["source_img"]] if subtask.get("source_img") else [],
                        user_dir=user_dir,
                        dl_base_name=dl_base_name,
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
                upsert_task_result(job, subtask, {"prompt": prompt, "source_img": os.path.basename(subtask["source_img"]).split('_', 1)[-1] if subtask.get("source_img") else None, "videos": videos, "images": [], "status": "success", "attempts": subtask["attempts"]})
            except Exception as e:
                subtask["status"] = "error"
                upsert_task_result(job, subtask, {"prompt": prompt, "source_img": os.path.basename(subtask["source_img"]).split('_', 1)[-1] if subtask.get("source_img") else None, "error": str(e), "status": "error", "attempts": subtask["attempts"]})
            refresh_job_progress(job)
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
                    upsert_task_result(job, subtask, {
                        "prompt": prompt,
                        "source_img": os.path.basename(img_path).split('_', 1)[-1] if img_path else None,
                        "result": final_text.strip(),
                        "images": generated_images,
                        "status": "success",
                        "attempts": subtask["attempts"],
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

@app.post("/api/jobs")
async def create_job(
    prompts: str = Form(""), negative_prompt: str = Form(""), mode: str = Form(...),
    template_name: str = Form(""), model_id: str = Form(""), batch_size: int = Form(1),
    target_ratio: str = Form(""),
    video_size: str = Form("1280*720"), video_duration: int = Form(5),
    video_shot_type: str = Form("single"), video_audio: bool = Form(True),
    video_watermark: bool = Form(False),
    images: Optional[List[UploadFile]] = File(None), curr: dict = Depends(get_current_user)
):
    prompt_str = prompts.strip()

    if mode == "fission" and not prompt_str:
        prompt_str = "Generate a high-quality, stylistically similar variation of the provided reference image.it should incorporate a certain degree of variation and be different to the original."
    elif mode == "convert" and not prompt_str:
        prompt_str = "保持原图主体结构和风格不变，将画面自然延展或重绘以适应设定的新比例尺寸，边缘过渡自然。"
    elif mode == "extract":
        prompt_str = ""  # prompt 由 VL 模型自动生成，前端无需填写
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
    if mode == "video":
        video_params = {
            "size": video_size, "duration": video_duration,
            "shot_type": video_shot_type, "audio": video_audio, "watermark": video_watermark
        }

    job = await job_queue.add_job(
        curr['username'], mode, [prompt_str], source_paths,
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
