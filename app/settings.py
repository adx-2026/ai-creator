"""
User settings persistence (per-user JSON files).
"""

import os
import json


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
