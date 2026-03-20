import asyncio
import time
import uuid
import os
import re
import base64
import json
import requests
import io
import dashscope
from dashscope import MultiModalConversation
from dotenv import load_dotenv
from typing import List, Optional, Tuple
from datetime import datetime

load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
import google.generativeai as genai
import PIL.Image

# ==========================================
# 0. 新增：图像比例处理核心工具
# ==========================================
def crop_to_aspect_ratio(image: PIL.Image.Image, target_ratio_str: str) -> PIL.Image.Image:
    """
    将图片居中裁剪到指定的比例 (例如 '16:9', '3:2', '1:1')
    """
    try:
        # 支持中英文冒号兼容
        w_ratio, h_ratio = map(int, target_ratio_str.replace('：', ':').split(':'))
        if w_ratio <= 0 or h_ratio <= 0: return image
        target_aspect = w_ratio / h_ratio
    except Exception:
        # 如果解析失败（例如没传比例），原样返回
        return image

    original_w, original_h = image.size
    original_aspect = original_w / original_h

    # 误差在 0.01 以内认为比例已经正确，直接返回
    if abs(original_aspect - target_aspect) < 0.01:
        return image

    if original_aspect > target_aspect:
        # 原图更宽，按高度作为基准，左右两边裁剪
        new_w = int(original_h * target_aspect)
        left = (original_w - new_w) / 2
        right = left + new_w
        return image.crop((left, 0, right, original_h))
    else:
        # 原图更高，按宽度作为基准，上下两边裁剪
        new_h = int(original_w / target_aspect)
        top = (original_h - new_h) / 2
        bottom = top + new_h
        return image.crop((0, top, original_w, bottom))

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
def parse_models(env_str: str, provider_name: str, prefix: str):
    if not env_str: return None
    models = []
    for item in env_str.split(','):
        if not item.strip(): continue
        parts = item.split(':', 1)
        m_id = parts[0].strip()
        m_name = parts[1].strip() if len(parts) > 1 else m_id
        models.append({"id": m_id, "name": m_name, "prefix": prefix})
    return {"provider": provider_name, "models": models} if models else None

AVAILABLE_MODELS = []
for env_key, prov_name, prefix in [
    ("GEMINI_MODELS", "Google Gemini", "gemini"),
    ("QWEN_MODELS", "Alibaba 通义千问", "qwen"),
    ("MINIMAX_MODELS", "MiniMax 稀宇科技", "minimax"),
    ("DOUBAO_MODELS", "ByteDance 豆包", "doubao")
]:
    parsed = parse_models(os.getenv(env_key, ""), prov_name, prefix)
    if parsed: AVAILABLE_MODELS.append(parsed)

# ==========================================
# 3. 供应商策略模式封装 (Provider Pattern)
# ==========================================
class ImageProvider:
    async def generate(self, model_id: str, prompt: str, negative_prompt: str, img_path: str, user_dir: str, dl_base_name: str, user: str) -> Tuple[List[dict], str]:
        raise NotImplementedError()

class GeminiProvider(ImageProvider):
    def __init__(self):
        genai.configure(api_key=os.getenv("GENAI_API_KEY", ""), transport='rest', client_options={'api_endpoint': os.getenv("GENAI_API_ENDPOINT", "http://127.0.0.1:8045")})
    
    async def generate(self, model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user):
        model = genai.GenerativeModel(model_id)
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
    async def generate(self, model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user):
        mm_model = model_id.replace("minimax-", "") if model_id.startswith("minimax-") else model_id
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('MINIMAX_API_KEY')}"}
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
    
    async def generate(self, model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user):
        content_list = []
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f: b64_data = base64.b64encode(f.read()).decode('utf-8')
            ext = os.path.splitext(img_path)[1].lower().replace('.', '').replace('jpg', 'jpeg')
            content_list.append({"image": f"data:image/{ext or 'jpeg'};base64,{b64_data}"})
        content_list.append({"text": prompt})
        
        kwargs = {"model": model_id, "messages": [{"role": "user", "content": content_list}], "n": 1, "size": "1024*1024"}
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
    async def generate(self, model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('ARK_API_KEY')}"}
        final_prompt = prompt + (f"\n\n请尽量避免出现以下元素：{negative_prompt}" if negative_prompt else "")
        payload = {"model": model_id, "prompt": final_prompt, "response_format": "url", "size": "2K", "stream": False, "watermark": True}
        
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
# 4. 任务队列调度
# ==========================================
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
                            if j["status"] in ["queued", "processing"]:
                                j["status"] = "failed"
                                j["results"].append({"error": "服务曾被重启，该任务已中断", "status": "error"})
                            self.jobs[j["id"]] = j
                except Exception: pass

    # 修改：加入 target_ratio 参数
    async def add_job(self, user, mode, prompts, source_image_paths=None, template_name="", model_id="", negative_prompt="", batch_size=1, target_ratio=""):
        job_id = str(uuid.uuid4())
        total_tasks = len(prompts) * max(1, len(source_image_paths) if source_image_paths else 1) * batch_size
        self.jobs[job_id] = {
            "id": job_id, "user": user, "mode": mode, "model_id": model_id,
            "status": "queued", "total": total_tasks, "completed": 0, "failed": 0,
            "results": [], "created_at": time.time(), "eta": None,
            "template_name": template_name, "negative_prompt": negative_prompt,
            "target_ratio": target_ratio # 记录比例信息
        }
        self.sync_user_jobs(user)
        await self.queue.put((job_id, user, prompts, source_image_paths, template_name, model_id, negative_prompt, batch_size, target_ratio))
        return self.jobs[job_id]

job_queue = JobQueue()

async def process_queue():
    while True:
        # 解包时加入 target_ratio
        job_data = await job_queue.queue.get()
        job_id, user, prompts, source_image_paths, tpl_name, model_id, negative_prompt, batch_size, target_ratio = job_data
        
        job = job_queue.jobs[job_id]
        job["status"] = "processing"
        user_dir = f"users/{user}/outputs"
        os.makedirs(user_dir, exist_ok=True)
        provider = get_provider_for_model(model_id)
        
        tasks = []
        for _ in range(batch_size):
            if job["mode"] in ['i2i', 'fission'] and source_image_paths:
                tasks.extend([(p, img_path) for img_path in source_image_paths for p in prompts])
            else:
                tasks.extend([(p, None) for p in prompts])
                
        avg_time = 5.0
        for i, (prompt, img_path) in enumerate(tasks):
            start_time = time.time()
            try:
                base_src = os.path.splitext(os.path.basename(img_path).split('_', 1)[-1])[0] if img_path else "t2i"
                dl_base_name = re.sub(r'[\\/*?:"<>|]', "", f"{base_src}_{tpl_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if tpl_name else f"{base_src}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                generated_images, final_text = await provider.generate(model_id, prompt, negative_prompt, img_path, user_dir, dl_base_name, user)
                
                # --- 新增：强制应用目标比例（生成后的后处理裁剪） ---
                if target_ratio and generated_images:
                    for img_info in generated_images:
                        filename = img_info["url"].split('/')[-1]
                        filepath = os.path.join(user_dir, filename)
                        if os.path.exists(filepath):
                            try:
                                with PIL.Image.open(filepath) as img:
                                    cropped = crop_to_aspect_ratio(img, target_ratio)
                                    cropped.save(filepath)
                            except Exception as e:
                                print(f"Error applying ratio {target_ratio}: {e}")
                # ----------------------------------------------------

                job["results"].append({
                    "prompt": prompt, "source_img": os.path.basename(img_path).split('_', 1)[-1] if img_path else None,
                    "result": final_text.strip(), "images": generated_images, "status": "success"
                })
                job["completed"] += 1
            except Exception as e:
                job["results"].append({"prompt": prompt, "error": str(e), "status": "error"})
                job["failed"] += 1
                
            avg_time = (avg_time * i + (time.time() - start_time)) / (i + 1)
            job["eta"] = max(0, (job["total"] - job["completed"] - job["failed"]) * avg_time)
            job_queue.sync_user_jobs(user)
            
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
    data, user, is_pub = await request.json(), curr["username"], data.get("is_public", False)
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
    target_ratio: str = Form(""), # 新增：接收前端传来的比例参数
    images: Optional[List[UploadFile]] = File(None), curr: dict = Depends(get_current_user)
):
    user, prompt_str = curr["username"], prompts.strip()
    if mode == "fission" and not prompt_str: prompt_str = "Generate a high-quality, stylistically similar variation of the provided reference image."
    elif not prompt_str: return {"error": "No prompt provided"}
    
    source_paths = []
    if mode in ["i2i", "fission"] and images:
        os.makedirs(f"users/{user}/outputs", exist_ok=True)
        for img in images:
            if img and getattr(img, "filename", None):
                path = os.path.join(f"users/{user}/outputs", f"{uuid.uuid4().hex[:8]}_{img.filename}")
                img_data = await img.read()
                
                # 新增：如果在图生图时选择了比例，先把原图按比例裁剪好，再喂给 AI，提高出图准确率
                if target_ratio:
                    try:
                        with PIL.Image.open(io.BytesIO(img_data)) as pil_img:
                            cropped_img = crop_to_aspect_ratio(pil_img, target_ratio)
                            cropped_img.save(path)
                    except Exception:
                        with open(path, "wb") as f: f.write(img_data)
                else:
                    with open(path, "wb") as f: f.write(img_data)
                    
                source_paths.append(path)
                
    job = await job_queue.add_job(user, mode, [prompt_str], source_paths, template_name, model_id, negative_prompt, batch_size, target_ratio)
    return job

# 新增：独立的小工具接口，如果用户只想单纯剪裁图片不跑AI，可以调这个
@app.post("/api/tools/convert_ratio")
async def convert_image_ratio(
    target_ratio: str = Form(...), 
    image: UploadFile = File(...), 
    curr: dict = Depends(get_current_user)
):
    """纯本地裁剪工具接口，秒出结果"""
    try:
        img_data = await image.read()
        with PIL.Image.open(io.BytesIO(img_data)) as pil_img:
            cropped = crop_to_aspect_ratio(pil_img, target_ratio)
            
            user_dir = f"users/{curr['username']}/outputs"
            os.makedirs(user_dir, exist_ok=True)
            ext = image.filename.split('.')[-1] if '.' in image.filename else 'png'
            filename = f"crop_{uuid.uuid4().hex[:8]}.{ext}"
            filepath = os.path.join(user_dir, filename)
            cropped.save(filepath)
            
            return {
                "success": True, 
                "url": f"/api/images/{curr['username']}/{filename}",
                "ratio": target_ratio
            }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"比例转换失败: {str(e)}"})

@app.get("/api/jobs")
def get_jobs(curr: dict = Depends(get_current_user)):
    return sorted([j for j in job_queue.jobs.values() if j.get("user") == curr["username"]], key=lambda x: x['created_at'], reverse=True)
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

app.mount("/", StaticFiles(directory="static", html=True), name="static")