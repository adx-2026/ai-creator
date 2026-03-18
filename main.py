import asyncio
import time
import uuid
import os
import re
import base64
import json
import requests
import dashscope
from dashscope import MultiModalConversation
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
import google.generativeai as genai
import PIL.Image
from datetime import datetime

def get_closest_aspect_ratio(width, height):
    ratio = width / height
    ratios = {"1:1": 1.0, "4:3": 1.333, "3:4": 0.75, "16:9": 1.777, "9:16": 0.5625}
    closest_key = min(ratios.keys(), key=lambda k: abs(ratios[k] - ratio))
    return closest_key

def get_qwen_size(aspect_ratio):
    mapping = {
        "1:1": "1024*1024", "4:3": "1024*768", "3:4": "768*1024",
        "16:9": "1280*720", "9:16": "720*1280"
    }
    return mapping.get(aspect_ratio, "1024*1024")

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
# 兼容旧版配置结构和新版对象结构
for k, v in raw_config.get("users", {}).items():
    if isinstance(v, str):
        USERS[k] = {"password": v, "is_admin": (k == "admin")}
    else:
        USERS[k] = v

SESSIONS = {}

api_key = os.getenv("GENAI_API_KEY", "")
api_endpoint = os.getenv("GENAI_API_ENDPOINT", "http://127.0.0.1:8045")
model_name = os.getenv("GENAI_MODEL_NAME", "gemini-3.1-flash-image")

genai.configure(api_key=api_key, transport='rest', client_options={'api_endpoint': api_endpoint})
image_model = genai.GenerativeModel(model_name)

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")

AVAILABLE_MODELS = [
    {"id": "gemini-3.1-flash-image", "name": "Gemini 3.1 Flash Image"},
    {"id": "qwen-image-2.0-pro", "name": "通义千问 Image 2.0 Pro"},
    {"id": "minimax-image-01", "name": "MiniMax Image-01"},
    {"id": "minimax-image-01-live", "name": "MiniMax Image-01-Live"}
]

app = FastAPI()

os.makedirs("users", exist_ok=True)
os.makedirs("static", exist_ok=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login", auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme), query_token: str = Query(None, alias="token")):
    actual_token = token or query_token
    if not actual_token or actual_token not in SESSIONS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    username = SESSIONS[actual_token]
    user_info = USERS.get(username, {})
    return {
        "username": username,
        "is_admin": user_info.get("is_admin", False)
    }

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.queue = asyncio.Queue()

    def sync_user_jobs(self, user):
        user_dir = f"users/{user}"
        os.makedirs(user_dir, exist_ok=True)
        user_jobs = [j for j in self.jobs.values() if j.get("user") == user]
        with open(f"{user_dir}/jobs.json", "w", encoding="utf-8") as f:
            json.dump(user_jobs, f, ensure_ascii=False)

    def load_jobs(self):
        if not os.path.exists("users"): return
        for user_dir in os.listdir("users"):
            jobs_file = os.path.join("users", user_dir, "jobs.json")
            if os.path.exists(jobs_file):
                try:
                    with open(jobs_file, "r", encoding="utf-8") as f:
                        user_jobs = json.load(f)
                        for j in user_jobs:
                            if j["status"] in ["queued", "processing"]:
                                j["status"] = "failed"
                                j["results"].append({"error": "服务曾被重启，该任务已中断", "status": "error"})
                            self.jobs[j["id"]] = j
                except Exception as e:
                    print(f"Error loading jobs for {user_dir}: {e}")

    async def add_job(self, user, mode, prompts, source_image_paths=None, template_name="", template_content="", model_id="gemini-3.1-flash-image", negative_prompt=""):
        job_id = str(uuid.uuid4())
        total_tasks = len(prompts)
        if mode == 'i2i' and source_image_paths:
            total_tasks = len(prompts) * max(1, len(source_image_paths))
            
        self.jobs[job_id] = {
            "id": job_id, "user": user, "mode": mode, "model_id": model_id,
            "status": "queued", "total": total_tasks, "completed": 0, "failed": 0,
            "results": [], "created_at": time.time(), "started_at": None, "eta": None,
            "template_name": template_name, "negative_prompt": negative_prompt
        }
        self.sync_user_jobs(user)
        await self.queue.put((job_id, user, prompts, source_image_paths, template_name, template_content, model_id, negative_prompt))
        return self.jobs[job_id]

job_queue = JobQueue()

async def process_queue():
    while True:
        job_id, user, prompts, source_image_paths, tpl_name, tpl_content, model_id, negative_prompt = await job_queue.queue.get()
        job = job_queue.jobs[job_id]
        job["status"] = "processing"
        job["started_at"] = time.time()
        job_queue.sync_user_jobs(user)
        
        mode = job["mode"]
        user_dir = f"users/{user}/outputs"
        os.makedirs(user_dir, exist_ok=True)
        
        tasks = []
        if mode == 'i2i' and source_image_paths:
            for img_path in source_image_paths:
                for p in prompts: tasks.append((p, img_path))
        else:
            for p in prompts: tasks.append((p, None))
                
        avg_time = 5.0
        
        for i, (prompt, img_path) in enumerate(tasks):
            start_time = time.time()
            try:
                generated_images = []
                final_text = ""
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_src = "t2i"
                
                target_aspect_ratio = "1:1"
                qwen_target_size = "1024*1024"

                if img_path and os.path.exists(img_path):
                    original_name = os.path.basename(img_path).split('_', 1)[-1]
                    base_src = os.path.splitext(original_name)[0]
                    try:
                        with PIL.Image.open(img_path) as src_img:
                            w, h = src_img.size
                            target_aspect_ratio = get_closest_aspect_ratio(w, h)
                            qwen_target_size = get_qwen_size(target_aspect_ratio)
                    except Exception as e:
                        pass
                    
                dl_base_name = f"{base_src}_{tpl_name}_{ts}" if tpl_name else f"{base_src}_{ts}"
                dl_base_name = re.sub(r'[\\/*?:"<>|]', "", dl_base_name)

                if model_id.startswith("minimax-"):
                    mm_model = model_id.replace("minimax-", "")
                    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {MINIMAX_API_KEY}"}
                    final_prompt = prompt
                    if negative_prompt:
                        final_prompt += f"\n\n请尽量避免出现以下元素：{negative_prompt}"
                        
                    payload = {"model": mm_model, "prompt": final_prompt, "response_format": "base64", "n": 1, "prompt_optimizer": True}
                    mm_ratios = ["1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16"]
                    if mm_model == "image-01": mm_ratios.append("21:9")
                        
                    payload["aspect_ratio"] = target_aspect_ratio if target_aspect_ratio in mm_ratios else "1:1"
                        
                    if img_path and os.path.exists(img_path):
                        with open(img_path, "rb") as f:
                            b64_data = base64.b64encode(f.read()).decode('utf-8')
                        ext = os.path.splitext(img_path)[1].lower().replace('.', '')
                        if ext == 'jpg': ext = 'jpeg'
                        mime = f"image/{ext}" if ext else "image/jpeg"
                        payload["subject_reference"] = [{"type": "character", "image_file": f"data:{mime};base64,{b64_data}"}]

                    response = await asyncio.to_thread(requests.post, "https://api.minimaxi.com/v1/image_generation", headers=headers, json=payload)
                    resp_json = response.json()
                    base_resp = resp_json.get("base_resp", {})
                    
                    if base_resp.get("status_code") == 0:
                        for b64_img in resp_json.get("data", {}).get("image_base64", []):
                            filename = f"gen_{uuid.uuid4().hex[:8]}.png"
                            filepath = os.path.join(user_dir, filename)
                            with open(filepath, "wb") as f:
                                f.write(base64.b64decode(b64_img))
                            generated_images.append({"url": f"/api/images/{user}/{filename}", "download_name": f"{dl_base_name}.png"})
                    else:
                        raise Exception(f"MiniMax Error: {base_resp.get('status_msg')} (Code: {base_resp.get('status_code')})")

                elif model_id == "qwen-image-2.0-pro":
                    content_list = []
                    if img_path and os.path.exists(img_path):
                        try:
                            with open(img_path, "rb") as f:
                                b64_data = base64.b64encode(f.read()).decode('utf-8')
                            ext = os.path.splitext(img_path)[1].lower().replace('.', '')
                            if ext == 'jpg': ext = 'jpeg'
                            mime = f"image/{ext}" if ext else "image/jpeg"
                            content_list.append({"image": f"data:{mime};base64,{b64_data}"})
                        except Exception as e: pass
                            
                    content_list.append({"text": prompt})
                    messages = [{"role": "user", "content": content_list}]
                    kwargs = {"model": "qwen-image-2.0-pro", "messages": messages, "n": 1, "size": qwen_target_size, "prompt_extend": True, "watermark": False}
                    if negative_prompt: kwargs["negative_prompt"] = negative_prompt
                    
                    rsp = await asyncio.to_thread(MultiModalConversation.call, **kwargs)
                    if rsp.status_code == 200:
                        for item in rsp.output.choices[0].message.content:
                            if 'image' in item:
                                img_url = item['image']
                                img_data = await asyncio.to_thread(requests.get, img_url)
                                filename = f"gen_{uuid.uuid4().hex[:8]}.png"
                                filepath = os.path.join(user_dir, filename)
                                with open(filepath, "wb") as f:
                                    f.write(img_data.content)
                                generated_images.append({"url": f"/api/images/{user}/{filename}", "download_name": f"{dl_base_name}.png"})
                    else:
                        raise Exception(f"DashScope Error: {rsp.message} (Code: {rsp.code})")

                else:
                    final_prompt_text = prompt
                    if negative_prompt: final_prompt_text += f"\n\nNegative Constraints (DO NOT INCLUDE): {negative_prompt}"
                    final_prompt_text += f"\n\n[CRITICAL: Output image aspect ratio MUST rigidly be {target_aspect_ratio}. Do NOT force a square output if the requested ratio is different.]"

                    contents = []
                    if img_path and os.path.exists(img_path):
                        try:
                            contents.append(PIL.Image.open(img_path))
                            contents.append(f"Reference image attached. Instruction: {final_prompt_text}")
                        except Exception as e: contents.append(final_prompt_text)
                    else:
                        contents.append(final_prompt_text)
                    
                    response = await asyncio.to_thread(image_model.generate_content, contents)

                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        data = base64.b64decode(part.inline_data.data) if isinstance(part.inline_data.data, str) else part.inline_data.data
                                        mime = part.inline_data.mime_type
                                        ext = mime.split('/')[-1] if mime else 'png'
                                        if ext == 'jpeg': ext = 'jpg'
                                        
                                        filename = f"gen_{uuid.uuid4().hex[:8]}.{ext}"
                                        dl_full_name = f"{dl_base_name}.{ext}"
                                        filepath = os.path.join(user_dir, filename)
                                        with open(filepath, "wb") as f: f.write(data)
                                        generated_images.append({"url": f"/api/images/{user}/{filename}", "download_name": dl_full_name})

                    if not generated_images:
                        try:
                            text = response.text
                            if text:
                                base64_patterns = re.findall(r'!\[.*?\]\((data:image/([^;]+);base64,([^)]+))\)', text)
                                if base64_patterns:
                                    idx_img = 1
                                    for full_data_uri, ext, b64_data in base64_patterns:
                                        if ext == 'jpeg': ext = 'jpg'
                                        filename = f"gen_{uuid.uuid4().hex[:8]}.{ext}"
                                        dl_full_name = f"{dl_base_name}_{idx_img}.{ext}" if idx_img > 1 else f"{dl_base_name}.{ext}"
                                        idx_img += 1
                                        with open(os.path.join(user_dir, filename), "wb") as f: f.write(base64.b64decode(b64_data))
                                        generated_images.append({"url": f"/api/images/{user}/{filename}", "download_name": dl_full_name})
                                    final_text = re.sub(r'!\[.*?\]\((data:image/[^;]+;base64,[^)]+)\)', '', text)
                                else:
                                    final_text = text
                        except Exception: pass
                        
                job["results"].append({
                    "prompt": prompt, "source_img": os.path.basename(img_path).split('_', 1)[-1] if img_path else None,
                    "result": final_text.strip() if final_text else "", "images": generated_images, "status": "success"
                })
                job["completed"] += 1
            except Exception as e:
                job["results"].append({
                    "prompt": prompt, "source_img": os.path.basename(img_path).split('_', 1)[-1] if img_path else None,
                    "error": str(e), "status": "error"
                })
                job["failed"] += 1
                
            elapsed = time.time() - start_time
            avg_time = (avg_time * i + elapsed) / (i + 1)
            job["eta"] = max(0, (job["total"] - job["completed"] - job["failed"]) * avg_time)
            job_queue.sync_user_jobs(user) # 同步进度到磁盘
            
        job["status"] = "completed"
        job["eta"] = 0
        job_queue.sync_user_jobs(user)
        job_queue.queue.task_done()

@app.on_event("startup")
async def startup_event():
    job_queue.load_jobs()
    asyncio.create_task(process_queue())

@app.post("/api/login")
def login(username: str = Form(...), password: str = Form(...)):
    user_info = USERS.get(username)
    if user_info and user_info.get("password") == password:
        token = str(uuid.uuid4())
        SESSIONS[token] = username
        os.makedirs(f"users/{username}/outputs", exist_ok=True)
        return {"access_token": token, "username": username, "is_admin": user_info.get("is_admin", False)}
    return JSONResponse(status_code=401, content={"error": "Invalid credentials"})

@app.get("/api/me")
def get_me(curr: dict = Depends(get_current_user)):
    return {"username": curr["username"], "is_admin": curr["is_admin"]}

@app.post("/api/logout")
def logout(token: str = Depends(oauth2_scheme)):
    if token in SESSIONS: del SESSIONS[token]
    return {"success": True}

@app.get("/api/models")
def get_models(curr: dict = Depends(get_current_user)):
    return AVAILABLE_MODELS

@app.get("/api/templates")
def get_templates(curr: dict = Depends(get_current_user)):
    user = curr["username"]
    pub = []
    if os.path.exists(PUBLIC_TEMPLATES_FILE):
        with open(PUBLIC_TEMPLATES_FILE, "r", encoding="utf-8") as f: pub = json.load(f)
    priv = []
    path = f"users/{user}/templates.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: priv = json.load(f)
    return {"public": pub, "private": priv}

@app.post("/api/templates")
async def save_template(request: Request, curr: dict = Depends(get_current_user)):
    data = await request.json()
    user = curr["username"]
    is_public = data.get("is_public", False)
    target_file = PUBLIC_TEMPLATES_FILE if is_public else f"users/{user}/templates.json"

    items = []
    if os.path.exists(target_file):
        with open(target_file, "r", encoding="utf-8") as f: items = json.load(f)

    new_item = {
        "name": data["name"], "content": data["content"],
        "negative_prompt": data.get("negative_prompt", ""), "author": user
    }

    idx = next((i for i, x in enumerate(items) if x["name"] == data["name"]), -1)
    if idx >= 0:
        if is_public and items[idx].get("author") != user and not curr["is_admin"]:
            return JSONResponse(status_code=403, content={"error": "无权修改他人的公共模板"})
        items[idx] = new_item
    else:
        items.append(new_item)

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    return {"success": True}

@app.delete("/api/templates/{scope}/{name}")
def delete_template(scope: str, name: str, curr: dict = Depends(get_current_user)):
    user = curr["username"]
    target_file = PUBLIC_TEMPLATES_FILE if scope == "public" else f"users/{user}/templates.json"
    
    if not os.path.exists(target_file): return {"success": True}
    with open(target_file, "r", encoding="utf-8") as f: items = json.load(f)

    idx = next((i for i, x in enumerate(items) if x["name"] == name), -1)
    if idx >= 0:
        if scope == "public" and items[idx].get("author") != user and not curr["is_admin"]:
            return JSONResponse(status_code=403, content={"error": "无权删除他人的公共模板"})
        items.pop(idx)
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)

    return {"success": True}

@app.post("/api/jobs")
async def create_job(
    prompts: str = Form(...), negative_prompt: str = Form(""), mode: str = Form(...),
    template_name: str = Form(""), model_id: str = Form("gemini-3.1-flash-image"),
    images: Optional[List[UploadFile]] = File(None), curr: dict = Depends(get_current_user)
):
    user = curr["username"]
    prompt_list = [p.strip() for p in prompts.split("\n") if p.strip()]
    if not prompt_list: return {"error": "No prompts provided"}
        
    source_image_paths = []
    if mode == "i2i" and images:
        user_dir = f"users/{user}/outputs"
        os.makedirs(user_dir, exist_ok=True)
        for img in images:
            if img and getattr(img, "filename", None):
                path = os.path.join(user_dir, f"{uuid.uuid4().hex[:8]}_{img.filename}")
                with open(path, "wb") as f: f.write(await img.read())
                source_image_paths.append(path)
            
    job = await job_queue.add_job(user, mode, prompt_list, source_image_paths, template_name, "", model_id, negative_prompt)
    return job

@app.get("/api/jobs")
def get_jobs(curr: dict = Depends(get_current_user)):
    user = curr["username"]
    user_jobs = [j for j in job_queue.jobs.values() if j.get("user") == user]
    return sorted(user_jobs, key=lambda x: x['created_at'], reverse=True)

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, curr: dict = Depends(get_current_user)):
    user = curr["username"]
    job = job_queue.jobs.get(job_id)
    if not job: return JSONResponse(status_code=404, content={"error": "Not found"})
    if job["user"] != user and not curr["is_admin"]: return JSONResponse(status_code=403, content={"error": "Forbidden"})

    del job_queue.jobs[job_id]
    job_queue.sync_user_jobs(job["user"])
    return {"success": True}

@app.get("/api/images/{username}/{filename}")
async def serve_img(username: str, filename: str, curr: dict = Depends(get_current_user)):
    user = curr["username"]
    if user != username and not curr["is_admin"]: return JSONResponse(status_code=403, content={"error": "Forbidden"})
    path = os.path.join(f"users/{username}/outputs", filename)
    if os.path.exists(path): return FileResponse(path)
    return JSONResponse(status_code=404, content={"error": "Not found"})

app.mount("/", StaticFiles(directory="static", html=True), name="static")