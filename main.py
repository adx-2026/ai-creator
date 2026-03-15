import asyncio
import time
import uuid
import os
import re
import base64
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
import google.generativeai as genai
import PIL.Image
from datetime import datetime

# Setup Config using JSON
config_file = "config.json"
example_file = "config_example.json"

if not os.path.exists(example_file):
    with open(example_file, "w", encoding="utf-8") as f:
        json.dump({
            "users": {
                "admin": "admin_secure_pass_2026",
                "nala": "nala_secure_pass_123"
            }
        }, f, indent=4)

if not os.path.exists(config_file):
    # Create default config.json if not present
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump({
            "users": {
                "admin": "123456",
                "nala": "123456"
            }
        }, f, indent=4)

with open(config_file, "r", encoding="utf-8") as f:
    config = json.load(f)

USERS = config.get("users", {})

SESSIONS = {}

# Init SDK
genai.configure(
    api_key="sk-b5b6a796fe94489987b09ff932cfa7b4",
    transport='rest',
    client_options={'api_endpoint': 'http://127.0.0.1:8045'}
)

image_model = genai.GenerativeModel('gemini-3.1-flash-image')

app = FastAPI()

os.makedirs("users", exist_ok=True)
os.makedirs("static", exist_ok=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login", auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme), query_token: str = Query(None, alias="token")):
    actual_token = token or query_token
    if not actual_token or actual_token not in SESSIONS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return SESSIONS[actual_token]

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.queue = asyncio.Queue()

    async def add_job(self, user, mode, prompts, source_image_paths=None, template_name="", template_content=""):
        job_id = str(uuid.uuid4())
        
        total_tasks = len(prompts)
        if mode == 'i2i' and source_image_paths:
            total_tasks = len(prompts) * max(1, len(source_image_paths))
            
        self.jobs[job_id] = {
            "id": job_id,
            "user": user,
            "mode": mode,
            "status": "queued",
            "total": total_tasks,
            "completed": 0,
            "failed": 0,
            "results": [],
            "created_at": time.time(),
            "started_at": None,
            "eta": None,
            "template_name": template_name
        }
        await self.queue.put((job_id, user, prompts, source_image_paths, template_name, template_content))
        return self.jobs[job_id]

job_queue = JobQueue()

async def process_queue():
    while True:
        job_id, user, prompts, source_image_paths, tpl_name, tpl_content = await job_queue.queue.get()
        job = job_queue.jobs[job_id]
        job["status"] = "processing"
        job["started_at"] = time.time()
        
        mode = job["mode"]
        user_dir = f"users/{user}/outputs"
        os.makedirs(user_dir, exist_ok=True)
        
        tasks = []
        if mode == 'i2i' and source_image_paths:
            for img_path in source_image_paths:
                for p in prompts:
                    tasks.append((p, img_path))
        else:
            for p in prompts:
                tasks.append((p, None))
                
        avg_time = 5.0
        
        for i, (prompt, img_path) in enumerate(tasks):
            start_time = time.time()
            try:
                contents = []
                
                # Content includes prompts directly since UI template syncs to textbox
                if img_path and os.path.exists(img_path):
                    try:
                        img = PIL.Image.open(img_path)
                        enhanced_prompt = f"Reference image attached. Instruction: {prompt}"
                        contents.append(img)
                        contents.append(enhanced_prompt)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        contents.append(prompt)
                else:
                    contents.append(prompt)
                
                response = await asyncio.to_thread(image_model.generate_content, contents)
                
                generated_images = []
                final_text = ""
                
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_src = "t2i"
                if img_path:
                    original_name = os.path.basename(img_path).split('_', 1)[-1]
                    base_src = os.path.splitext(original_name)[0]
                    
                dl_base_name = f"{base_src}_{tpl_name}_{ts}" if tpl_name else f"{base_src}_{ts}"
                dl_base_name = re.sub(r'[\\/*?:"<>|]', "", dl_base_name)

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
                                    with open(filepath, "wb") as f:
                                        f.write(data)
                                    generated_images.append({
                                        "url": f"/api/images/{user}/{filename}",
                                        "download_name": dl_full_name
                                    })

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
                                    suffix = f"_{idx_img}" if idx_img > 1 else ""
                                    dl_full_name = f"{dl_base_name}{suffix}.{ext}"
                                    idx_img += 1
                                    
                                    filepath = os.path.join(user_dir, filename)
                                    with open(filepath, "wb") as f:
                                        f.write(base64.b64decode(b64_data))
                                    generated_images.append({
                                        "url": f"/api/images/{user}/{filename}",
                                        "download_name": dl_full_name
                                    })
                                
                                final_text = re.sub(r'!\[.*?\]\((data:image/[^;]+;base64,[^)]+)\)', '', text)
                            else:
                                final_text = text
                    except Exception:
                        pass
                        
                job["results"].append({
                    "prompt": prompt, 
                    "source_img": os.path.basename(img_path).split('_', 1)[-1] if img_path else None,
                    "result": final_text.strip() if final_text else "",
                    "images": generated_images,
                    "status": "success"
                })
                job["completed"] += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                job["results"].append({
                    "prompt": prompt, 
                    "source_img": os.path.basename(img_path).split('_', 1)[-1] if img_path else None,
                    "error": str(e), 
                    "status": "error"
                })
                job["failed"] += 1
                
            elapsed = time.time() - start_time
            avg_time = (avg_time * i + elapsed) / (i + 1)
            rem = job["total"] - job["completed"] - job["failed"]
            job["eta"] = max(0, rem * avg_time)
            
        job["status"] = "completed"
        job["eta"] = 0
        job_queue.queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

@app.post("/api/login")
def login(username: str = Form(...), password: str = Form(...)):
    if USERS.get(username) == password:
        token = str(uuid.uuid4())
        SESSIONS[token] = username
        os.makedirs(f"users/{username}/outputs", exist_ok=True)
        return {"access_token": token, "username": username}
    return JSONResponse(status_code=401, content={"error": "Invalid credentials"})

@app.get("/api/me")
def get_me(user: str = Depends(get_current_user)):
    return {"username": user}

@app.post("/api/logout")
def logout(token: str = Depends(oauth2_scheme)):
    if token in SESSIONS:
        del SESSIONS[token]
    return {"success": True}

@app.get("/api/templates")
def get_templates(user: str = Depends(get_current_user)):
    path = f"users/{user}/templates.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.post("/api/templates")
async def save_templates(request: Request, user: str = Depends(get_current_user)):
    try:
        data = await request.json()
        path = f"users/{user}/templates.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return {"success": True}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/jobs")
async def create_job(
    prompts: str = Form(...),
    mode: str = Form(...),
    template_name: str = Form(""),
    images: Optional[List[UploadFile]] = File(None),
    user: str = Depends(get_current_user)
):
    prompt_list = [p.strip() for p in prompts.split("\n") if p.strip()]
    if not prompt_list:
        return {"error": "No prompts provided"}
        
    source_image_paths = []
    if mode == "i2i" and images:
        user_dir = f"users/{user}/outputs"
        os.makedirs(user_dir, exist_ok=True)
        for img in images:
            if img and getattr(img, "filename", None):
                path = os.path.join(user_dir, f"{uuid.uuid4().hex[:8]}_{img.filename}")
                with open(path, "wb") as f:
                    f.write(await img.read())
                source_image_paths.append(path)
            
    job = await job_queue.add_job(user, mode, prompt_list, source_image_paths, template_name, "")
    return job

@app.get("/api/jobs")
def get_jobs(user: str = Depends(get_current_user)):
    user_jobs = [j for j in job_queue.jobs.values() if j.get("user") == user]
    return sorted(user_jobs, key=lambda x: x['created_at'], reverse=True)

@app.get("/api/images/{username}/{filename}")
async def serve_img(username: str, filename: str, user: str = Depends(get_current_user)):
    if user != username and user != "admin":
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    path = os.path.join(f"users/{username}/outputs", filename)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"error": "Not found"})

app.mount("/", StaticFiles(directory="static", html=True), name="static")