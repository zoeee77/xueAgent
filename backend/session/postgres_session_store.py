"""PostgreSQL 持久化会话存储。

提供与内存 SessionStore 兼容的接口，同时支持多会话管理。

架构：LRU 内存缓存 + PostgreSQL 持久化双写
- 读：优先从 LRU 缓存读取，未命中则查 DB 并回填缓存
- 写：同时写入 LRU 缓存和 PostgreSQL（缓存失效 + DB 持久化）
- 淘汰：LRU 层满时自动淘汰最久未访问的 session，DB 作为数据兜底
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from backend.session.lru_session_store import LRUSessionStore

logger = logging.getLogger(__name__)


def _get_db_config() -> dict:
    """从环境变量或 .env 读取 PostgreSQL 配置。"""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "xueAgent"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "root"),
    }


class PostgreSQLSessionStore:
    """PostgreSQL 持久化会话存储 + LRU 内存缓存层。

    接口与内存 SessionStore 兼容：
        - get(session_id) -> list[dict]
        - set(session_id, value)
        - append(session_id, item)

    额外提供多会话管理：
        - create_session(user_id, session_id, title)
        - list_sessions(user_id, limit)
        - delete_session(session_id)
    """

    def __init__(self, db_config: Optional[dict] = None, lru_cache: Optional[LRUSessionStore] = None):
        self._db_config = db_config or _get_db_config()
        self._lru_cache = lru_cache

    @contextmanager
    def _get_conn(self):
        """获取数据库连接的上下文管理器。"""
        conn = psycopg2.connect(**self._db_config)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ─── 缓存层操作 ─────────────────────────────────────────────

    def _get_from_cache(self, session_id: str) -> Optional[list[dict]]:
        """从 LRU 缓存读取。"""
        if self._lru_cache is None:
            return None
        messages = self._lru_cache.get(session_id)
        return messages if messages else None

    def _set_to_cache(self, session_id: str, messages: list[dict]) -> None:
        """写入 LRU 缓存。"""
        if self._lru_cache is not None:
            self._lru_cache.set(session_id, messages)

    def _invalidate_cache(self, session_id: str) -> None:
        """使 LRU 缓存失效。"""
        if self._lru_cache is not None:
            self._lru_cache.delete(session_id)

    # ─── 与 SessionStore 兼容的接口 ─────────────────────────────

    def get(self, session_id: str, limit: int = 20) -> list[dict]:
        """获取某个会话的最近 N 条消息。

        优先从 LRU 缓存读取，未命中则查 DB 并回填缓存。
        """
        # 1. 尝试从缓存读取
        cached = self._get_from_cache(session_id)
        if cached is not None:
            return cached[-limit:] if len(cached) > limit else cached

        # 2. 缓存未命中，从 DB 读取
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT role, content, created_at as timestamp
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (session_id, limit),
                )
                rows = cur.fetchall()

        messages = [
            {"role": row["role"], "content": row["content"], "timestamp": str(row["timestamp"])}
            for row in reversed(rows)
        ]

        # 3. 回填缓存
        self._set_to_cache(session_id, messages)
        return messages

    def set(self, session_id: str, value: list[dict]):
        """替换会话的完整消息列表。

        同时更新 LRU 缓存和 PostgreSQL。
        """
        # 1. 写入 DB
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
                for msg in value:
                    cur.execute(
                        "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)",
                        (session_id, msg.get("role", "user"), msg.get("content", "")),
                    )
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chat_sessions SET updated_at = NOW() WHERE session_id = %s",
                    (session_id,),
                )

        # 2. 更新缓存
        self._set_to_cache(session_id, value)

    def append(self, session_id: str, item: dict):
        """追加一条消息到会话。

        同时追加到 LRU 缓存和 PostgreSQL。
        """
        # 1. 写入 DB
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)",
                    (session_id, item.get("role", "user"), item.get("content", "")),
                )
                cur.execute(
                    "UPDATE chat_sessions SET updated_at = NOW() WHERE session_id = %s",
                    (session_id,),
                )

        # 2. 更新缓存（先失效再追加，保证一致性）
        cached = self._get_from_cache(session_id)
        if cached is not None:
            cached.append(item)
            self._set_to_cache(session_id, cached)

    # ─── 多会话管理 ─────────────────────────────────────────────

    def create_session(self, user_id: str, session_id: str, title: Optional[str] = None) -> bool:
        """创建新会话。如果 session_id 已存在则跳过。"""
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chat_sessions (session_id, user_id, title)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (session_id) DO NOTHING
                        """,
                        (session_id, user_id, title or "新对话"),
                    )
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False

    def list_sessions(self, user_id: str, limit: int = 20) -> list[dict]:
        """获取用户的会话列表，按更新时间倒序。"""
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT session_id, user_id, title, created_at, updated_at
                    FROM chat_sessions
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
        return [
            {
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "title": row["title"],
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其所有消息（外键 CASCADE）。"""
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
                    success = cur.rowcount > 0

            # 同步删除缓存
            if success:
                self._invalidate_cache(session_id)

            return success
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False

    def check_ownership(self, user_id: str, session_id: str) -> bool:
        """检查 session 是否属于指定用户。"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM chat_sessions WHERE session_id = %s AND user_id = %s",
                    (session_id, user_id),
                )
                return cur.fetchone() is not None
