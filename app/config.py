"""
Application configuration: loads .env, config.json, model lists, and constants.
"""

import os
import json
from dotenv import load_dotenv
import dashscope

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")

# ------------------------------------------------------------------
# Config files
# ------------------------------------------------------------------
CONFIG_FILE = "config.json"
CONFIG_EXAMPLE_FILE = "config_example.json"
PUBLIC_TEMPLATES_FILE = "public_templates.json"

if not os.path.exists(CONFIG_EXAMPLE_FILE):
    with open(CONFIG_EXAMPLE_FILE, "w", encoding="utf-8") as f:
        json.dump({"users": {"admin": {"password": "admin_secure_pass_2026", "is_admin": True}, "test": {"password": "123456", "is_admin": False}}}, f, indent=4)
if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"users": {"admin": {"password": "123456", "is_admin": True}, "test": {"password": "123456", "is_admin": False}}}, f, indent=4)
if not os.path.exists(PUBLIC_TEMPLATES_FILE):
    with open(PUBLIC_TEMPLATES_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    raw_config = json.load(f)

USERS = {}
for k, v in raw_config.get("users", {}).items():
    if isinstance(v, str):
        USERS[k] = {"password": v, "is_admin": (k == "admin")}
    else:
        USERS[k] = v

SUBTASK_CONCURRENCY = max(1, int(raw_config.get("subtask_concurrency", 3)))

# ------------------------------------------------------------------
# Dynamic model list from environment
# ------------------------------------------------------------------
def parse_models(env_str: str, provider_name: str, prefix: str, model_type: str = "image"):
    if not env_str:
        return None
    models = []
    for item in env_str.split(','):
        if not item.strip():
            continue
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
    if parsed:
        AVAILABLE_MODELS.append(parsed)

ALIYUN_QWEN_STRICT_MODELS = {"qwen-image-2.0-pro", "qwen-image-2.0-pro-2026-03-03"}

# ------------------------------------------------------------------
# Size mapping for qwen-image-2.0-pro
# ------------------------------------------------------------------
def map_ratio_to_size(ratio_str: str) -> str:
    if not ratio_str:
        return "1024*1024"
    ratio_str = ratio_str.replace('\uff1a', ':').strip()
    mapping = {
        "1:1": "1024*1024",
        "2:3": "1024*1536",
        "3:2": "1536*1024",
        "3:4": "1080*1440",
        "4:3": "1440*1080",
        "9:16": "1080*1920",
        "16:9": "1920*1080",
        "21:9": "2048*872",
    }
    return mapping.get(ratio_str, "1024*1024")
