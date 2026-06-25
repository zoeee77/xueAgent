"""LRU + TTL 内存会话存储。

提供基于内存的 session 缓存层，支持:
- LRU 淘汰策略（满时淘汰最久未访问的 session）
- TTL 过期策略（超过 TTL 秒自动失效）
- 与 PostgreSQLSessionStore 兼容的接口

使用场景:
- 高频读写的会话消息缓存
- 与 PostgreSQL 持久化层配合，实现缓存 + 持久化双写架构
"""

import logging
import time
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


class LRUSessionStore:
    """带 LRU 淘汰 + TTL 过期的内存会话存储。

    Args:
        max_size: 最大缓存 session 数量，默认 1000
        ttl_seconds: 缓存过期时间（秒），默认 1800（30 分钟）
        evict_ratio: 满时淘汰比例，默认 0.1（即 10%）
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 1800,
        evict_ratio: float = 0.1,
    ):
        # OrderedDict 保证插入顺序，LRU 淘汰时移除最旧的
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._evict_count = max(1, int(max_size * evict_ratio))

    # ─── 核心接口 ───────────────────────────────────────────────

    def get(self, session_id: str) -> list[dict]:
        """获取会话消息列表。

        若 session 不存在或已过期，返回空列表。
        命中缓存时会更新访问时间（LRU 语义）。

        Args:
            session_id: 会话 ID

        Returns:
            消息列表，过期或不存在时返回空列表
        """
        if session_id not in self._cache:
            return []

        entry = self._cache[session_id]
        if time.time() - entry["last_accessed"] >= self._ttl:
            # TTL 过期，移除
            del self._cache[session_id]
            logger.debug("Session %s expired (TTL=%ds)", session_id, self._ttl)
            return []

        # LRU: 命中时移至末尾（最近使用）
        self._cache.move_to_end(session_id)
        entry["last_accessed"] = time.time()
        return entry["messages"]

    def set(self, session_id: str, messages: list[dict]) -> None:
        """设置或替换会话消息列表。

        若缓存已满，先执行 LRU 淘汰。

        Args:
            session_id: 会话 ID
            messages: 消息列表
        """
        self._ensure_capacity(session_id)

        self._cache[session_id] = {
            "messages": messages,
            "last_accessed": time.time(),
        }
        # 新插入的 session 自然位于末尾（最近使用）
        self._cache.move_to_end(session_id)

    def append(self, session_id: str, item: dict) -> None:
        """追加一条消息到会话。

        Args:
            session_id: 会话 ID
            item: 消息条目，如 {"role": "user", "content": "你好"}
        """
        messages = self.get(session_id)
        messages.append(item)
        self.set(session_id, messages)

    def delete(self, session_id: str) -> bool:
        """删除指定会话。

        Args:
            session_id: 会话 ID

        Returns:
            True 表示删除成功，False 表示不存在
        """
        if session_id in self._cache:
            del self._cache[session_id]
            return True
        return False

    # ─── 维护操作 ───────────────────────────────────────────────

    def cleanup(self) -> int:
        """清理所有过期 session。

        Returns:
            清理的 session 数量
        """
        now = time.time()
        expired_keys = [
            sid
            for sid, entry in self._cache.items()
            if now - entry["last_accessed"] >= self._ttl
        ]
        for sid in expired_keys:
            del self._cache[sid]

        if expired_keys:
            logger.info(
                "Cleaned up %d expired sessions (TTL=%ds)",
                len(expired_keys),
                self._ttl,
            )
        return len(expired_keys)

    def evict_lru(self) -> int:
        """手动触发 LRU 淘汰。

        Returns:
            淘汰的 session 数量
        """
        if len(self._cache) <= self._max_size:
            return 0

        to_remove = min(
            self._evict_count, len(self._cache) - self._max_size
        )
        # OrderedDict 头部 = 最久未访问
        removed_keys = list(self._cache.keys())[:to_remove]
        for key in removed_keys:
            del self._cache[key]

        logger.info("LRU evicted %d sessions", len(removed_keys))
        return len(removed_keys)

    # ─── 统计信息 ───────────────────────────────────────────────

    @property
    def size(self) -> int:
        """当前缓存的 session 数量。"""
        return len(self._cache)

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def ttl(self) -> int:
        return self._ttl

    def get_stats(self) -> dict:
        """返回缓存统计信息。"""
        return {
            "total_sessions": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "utilization": round(len(self._cache) / self._max_size * 100, 1),
        }

    # ─── 内部方法 ───────────────────────────────────────────────

    def _ensure_capacity(self, session_id: str) -> None:
        """确保缓存有足够空间存放新 session。

        若缓存已满且 session_id 不存在，触发 LRU 淘汰。
        """
        if session_id in self._cache:
            return

        if len(self._cache) >= self._max_size:
            # OrderedDict 头部 = 最久未访问
            removed = 0
            while len(self._cache) >= self._max_size and removed < self._evict_count:
                oldest_key, _ = self._cache.popitem(last=False)
                removed += 1
                logger.debug("LRU evicted session: %s", oldest_key)
