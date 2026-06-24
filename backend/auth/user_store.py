"""用户存储模块。

使用 JSON 文件存储用户信息（轻量方案，适合开发/演示环境）。
生产环境建议切换到 PostgreSQL users 表。
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from backend.auth.jwt_auth import hash_password, verify_password

logger = logging.getLogger(__name__)

USERS_FILE = Path(__file__).resolve().parent.parent / "data" / "users.json"
_lock = threading.Lock()


def _load_users() -> dict:
    """加载用户列表。"""
    if not USERS_FILE.exists():
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict):
    """保存用户列表。"""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def create_user(username: str, password: str) -> Optional[str]:
    """注册新用户。成功返回 user_id，失败返回 None。"""
    with _lock:
        users = _load_users()
        # 检查用户名是否已存在
        for user in users.values():
            if user.get("username") == username:
                logger.warning(f"Username already exists: {username}")
                return None
        user_id = f"user_{len(users) + 1:04d}"
        users[user_id] = {
            "username": username,
            "password_hash": hash_password(password),
        }
        _save_users(users)
        logger.info(f"User created: {user_id} ({username})")
        return user_id


def authenticate_user(username: str, password: str) -> Optional[str]:
    """验证用户名密码。成功返回 user_id，失败返回 None。"""
    users = _load_users()
    for user_id, user_data in users.items():
        if user_data.get("username") == username:
            if verify_password(password, user_data["password_hash"]):
                return user_id
    return None
