"""知识库模块：PostgreSQL 数据查询服务，支持内存缓存。"""

import json
import hashlib
import time
import logging
import os
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """知识库查询服务，数据源为 PostgreSQL，支持内存缓存。"""

    def __init__(self, cache_ttl: int = 300):
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple] = {}  # key -> (value, timestamp)

        self._db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "dbname": os.getenv("DB_NAME", "xueAgent"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "root"),
        }

        self._test_connection()

    def _get_conn(self):
        """获取数据库连接（调用方负责 close）。"""
        return psycopg2.connect(**self._db_config)

    def _test_connection(self) -> None:
        """测试数据库连接是否可用。"""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
            logger.info("PostgreSQL 连接成功: %s:%s/%s", self._db_config["host"], self._db_config["port"], self._db_config["dbname"])
        except Exception as e:
            logger.error("PostgreSQL 连接失败: %s，回退到 JSON 数据源", e)
            raise

    # ─── 缓存工具 ────────────────────────────────────────────────

    def _cache_key(self, method: str, **kwargs) -> str:
        raw = f"{method}:{json.dumps(kwargs, sort_keys=True, ensure_ascii=False)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str):
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value) -> None:
        self._cache[key] = (value, time.time())

    # ── 专业查询 ────────────────────────────────────────────────

    def query_major(self, name: str) -> Optional[dict]:
        """查询专业信息。支持模糊匹配。"""
        if not name:
            return None

        cache_key = self._cache_key("major", name=name)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 精确匹配
                cur.execute("SELECT * FROM majors WHERE name = %s", (name,))
                row = cur.fetchone()
                if row:
                    result = self._row_to_major_dict(row)
                    self._set_cached(cache_key, result)
                    return result

                # 模糊匹配
                cur.execute("SELECT * FROM majors WHERE name LIKE %s LIMIT 1", (f"%{name}%",))
                row = cur.fetchone()
                if row:
                    result = self._row_to_major_dict(row)
                    self._set_cached(cache_key, result)
                    return result
        finally:
            conn.close()

        return None

    @staticmethod
    def _row_to_major_dict(row) -> dict:
        """将数据库行转换为与原来 JSON 格式一致的 dict。"""
        return {
            "name": row["name"],
            "description": row["description"] or "",
            "avg_salary": row["avg_salary"],
            "employment_rate": float(row["employment_rate"]) if row["employment_rate"] else None,
            "resource_threshold": row["resource_threshold"] or "",
            "personality_fit": row["personality_fit"] or [],
            "keywords": row["keywords"] or [],
            "courses": row["courses"] or [],
            "industries": row["industries"] or [],
            "career_paths": row["career_paths"] or [],
        }

    # ─── 院校查询 ────────────────────────────────────────────────

    def query_university(self, name: str, province: Optional[str] = None) -> Optional[dict]:
        """查询院校信息。"""
        cache_key = self._cache_key("university", name=name, province=province)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 精确匹配
                cur.execute("SELECT * FROM universities WHERE name = %s", (name,))
                row = cur.fetchone()
                if row:
                    result = self._row_to_university_dict(row)
                    self._set_cached(cache_key, result)
                    return result

                # 带省份的模糊匹配
                if province:
                    cur.execute(
                        "SELECT * FROM universities WHERE province = %s AND name LIKE %s LIMIT 1",
                        (province, f"%{name}%"),
                    )
                    row = cur.fetchone()
                    if row:
                        result = self._row_to_university_dict(row)
                        self._set_cached(cache_key, result)
                        return result

                # 不带省份的模糊匹配
                cur.execute("SELECT * FROM universities WHERE name LIKE %s LIMIT 1", (f"%{name}%",))
                row = cur.fetchone()
                if row:
                    result = self._row_to_university_dict(row)
                    self._set_cached(cache_key, result)
                    return result
        finally:
            conn.close()

        return None

    @staticmethod
    def _row_to_university_dict(row) -> dict:
        return {
            "name": row["name"],
            "province": row["province"] or "",
            "tier": row["tier"] or "",
            "min_score_2025": row["min_score_2025"],
            "avg_score_2025": row["avg_score_2025"],
            "rank_range": row["rank_range"] or "",
            "description": row["description"] or "",
        }

    # ─── 行业查询 ────────────────────────────────────────────────

    def query_industry(self, name: str) -> Optional[dict]:
        """查询行业信息。"""
        cache_key = self._cache_key("industry", name=name)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 精确匹配
                cur.execute("SELECT * FROM industries WHERE name = %s", (name,))
                row = cur.fetchone()
                if row:
                    result = self._row_to_industry_dict(row)
                    self._set_cached(cache_key, result)
                    return result

                # 模糊匹配
                cur.execute("SELECT * FROM industries WHERE name LIKE %s LIMIT 1", (f"%{name}%",))
                row = cur.fetchone()
                if row:
                    result = self._row_to_industry_dict(row)
                    self._set_cached(cache_key, result)
                    return result
        finally:
            conn.close()

        return None

    @staticmethod
    def _row_to_industry_dict(row) -> dict:
        return {
            "name": row["name"],
            "entry_barrier": row["entry_barrier"] or "",
            "family_resource_dependent": bool(row["family_resource_dependent"]),
            "salary_range": row["salary_range"] or {},
            "graduate_distribution": row["graduate_distribution"] or {},
            "top_employers": row["top_employers"] or [],
            "description": row["description"] or "",
        }

    # ─── 决策规则 ────────────────────────────────────────────────

    def _get_decision_rules(self) -> dict:
        """从 decision_rules 表加载规则。"""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT key, value FROM decision_rules")
                rows = cur.fetchall()
            result = {}
            for row in rows:
                key = row["key"]
                value = row["value"]
                if isinstance(value, str):
                    value = json.loads(value)
                result[key] = value
            return result
        finally:
            conn.close()

    def get_score_strategy(self, score: int) -> Optional[str]:
        """根据分数段获取填报策略。"""
        cache_key = self._cache_key("score_strategy", score=score)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        rules = self._get_decision_rules()
        score_ranges = rules.get("score_range_strategies", {})

        def _parse_priority(item):
            range_str = item[0]
            if "+" in range_str:
                return int(range_str.replace("+", ""))
            elif "以下" in range_str:
                return -1
            else:
                parts = range_str.split("-")
                if len(parts) == 2:
                    try:
                        return int(parts[0])
                    except ValueError:
                        return -1
                return -1

        for range_str, strategy in sorted(score_ranges.items(), key=_parse_priority, reverse=True):
            if "+" in range_str:
                min_score = int(range_str.replace("+", ""))
                if score >= min_score:
                    self._set_cached(cache_key, strategy)
                    return strategy
            elif "以下" in range_str:
                parts = range_str.split("以下")
                if parts[0]:
                    try:
                        max_score = int(parts[0])
                        if score < max_score:
                            self._set_cached(cache_key, strategy)
                            return strategy
                    except ValueError:
                        pass
            else:
                parts = range_str.split("-")
                if len(parts) == 2:
                    try:
                        low, high = int(parts[0]), int(parts[1])
                        if low <= score <= high:
                            self._set_cached(cache_key, strategy)
                            return strategy
                    except ValueError:
                        pass
        return None

    def get_priority_rule(self, resource_level: str) -> Optional[str]:
        """根据家庭资源水平获取优先级规则。"""
        mapping = {
            "high": "high_resource_family",
            "medium": "medium_resource_family",
            "low": "low_resource_family",
        }
        key = mapping.get(resource_level)
        if key:
            rules = self._get_decision_rules()
            return rules.get("priority_rules", {}).get(key)
        return None

    # ─── 全量数据（供 DataRetriever 使用） ────────────────────────

    @property
    def all_majors(self) -> dict:
        """返回所有专业 {name: {info}}，带缓存。"""
        cache_key = "all_majors"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM majors ORDER BY name")
                rows = cur.fetchall()
            result = {}
            for row in rows:
                d = self._row_to_major_dict(row)
                result[d["name"]] = d
            self._set_cached(cache_key, result)
            return result
        finally:
            conn.close()

    @property
    def all_universities(self) -> dict:
        """返回所有院校 {name: {info}}。"""
        cache_key = "all_universities"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM universities ORDER BY name")
                rows = cur.fetchall()
            result = {}
            for row in rows:
                d = self._row_to_university_dict(row)
                result[d["name"]] = d
            self._set_cached(cache_key, result)
            return result
        finally:
            conn.close()

    @property
    def all_industries(self) -> dict:
        """返回所有行业 {name: {info}}。"""
        cache_key = "all_industries"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM industries ORDER BY name")
                rows = cur.fetchall()
            result = {}
            for row in rows:
                d = self._row_to_industry_dict(row)
                result[d["name"]] = d
            self._set_cached(cache_key, result)
            return result
        finally:
            conn.close()
