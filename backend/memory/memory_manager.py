"""SQLite-based memory management for multi-agent system."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)


class MemoryManager:
    """Async SQLite-backed memory manager for user profiles, chat history, and preferences."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            backend_dir = Path(__file__).resolve().parent.parent
            db_path = str(backend_dir / "data" / "user_memory.db")
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            db_dir = os.path.dirname(self._db_path)
            os.makedirs(db_dir, exist_ok=True)
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
        return self._db

    async def init(self) -> None:
        """Create tables if they do not exist."""
        db = await self._get_db()
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, key)
            );
        """)
        await db.commit()
        logger.info("Memory tables initialized at %s", self._db_path)

    async def save_profile(self, user_id: str, profile: dict) -> None:
        """Save or update a user profile. Accepts a dict (e.g. UserProfile.model_dump())."""
        db = await self._get_db()
        profile_json = json.dumps(profile, ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """
            INSERT INTO user_profiles (user_id, profile_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at = excluded.updated_at
            """,
            (user_id, profile_json, now, now),
        )
        await db.commit()

    async def get_profile(self, user_id: str) -> Optional[dict]:
        """Return user profile as a dict, or None if not found."""
        db = await self._get_db()
        async with db.execute(
            "SELECT profile_json FROM user_profiles WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row["profile_json"])

    async def add_message(self, user_id: str, session_id: str, role: str, content: str) -> None:
        """Add a chat message to history."""
        db = await self._get_db()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "INSERT INTO chat_history (user_id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, session_id, role, content, now),
        )
        await db.commit()

    async def get_history(self, user_id: str, session_id: str, limit: int = 10) -> list[dict]:
        """Return recent chat messages for a user/session, newest first, capped at *limit*."""
        db = await self._get_db()
        async with db.execute(
            """
            SELECT role, content, timestamp
            FROM chat_history
            WHERE user_id = ? AND session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        # Return in chronological order (oldest first)
        return [
            {"role": row["role"], "content": row["content"], "timestamp": row["timestamp"]}
            for row in reversed(rows)
        ]

    async def set_preference(self, user_id: str, key: str, value: str) -> None:
        """Set a single preference (upsert)."""
        db = await self._get_db()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            """
            INSERT INTO preferences (user_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (user_id, key, value, now),
        )
        await db.commit()

    async def get_preferences(self, user_id: str) -> dict:
        """Return all preferences for a user as a {key: value} dict."""
        db = await self._get_db()
        async with db.execute(
            "SELECT key, value FROM preferences WHERE user_id = ?", (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("Memory database connection closed.")
