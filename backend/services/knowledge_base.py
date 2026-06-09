"""知识库模块：加载、查询、缓存 JSON 数据。"""

import json
import hashlib
import time
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class KnowledgeBase:
    """知识库查询服务，支持多数据源和内存缓存。"""

    def __init__(self, data_dir: Path = DATA_DIR, cache_ttl: int = 300):
        self._data_dir = data_dir
        self._cache_ttl = cache_ttl
        self._majors: dict = {}
        self._universities: dict = {}
        self._industries: dict = {}
        self._decision_rules: dict = {}
        self._cache: dict[str, tuple] = {}  # key -> (value, timestamp)

        self._load_all()

    def _load_all(self) -> None:
        """加载所有 JSON 数据文件。"""
        self._majors = self._load_json("majors.json")
        self._universities = self._load_json("universities.json")
        self._industries = self._load_json("industries.json")
        self._decision_rules = self._load_json("decision_rules.json")

    def _load_json(self, filename: str) -> dict:
        filepath = self._data_dir / filename
        if not filepath.exists():
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _cache_key(self, method: str, **kwargs) -> str:
        raw = f"{method}:{json.dumps(kwargs, sort_keys=True)}"
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

    def query_major(self, name: str) -> Optional[dict]:
        """查询专业信息。支持模糊匹配。"""
        if not name:
            return None

        cache_key = self._cache_key("major", name=name)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # 精确匹配
        if name in self._majors:
            result = self._majors[name]
            self._set_cached(cache_key, result)
            return result

        # 模糊匹配（包含关键字）
        for key, value in self._majors.items():
            if name in key or key in name:
                result = value
                self._set_cached(cache_key, result)
                return result

        return None

    def query_university(self, name: str, province: Optional[str] = None) -> Optional[dict]:
        """查询院校信息。"""
        cache_key = self._cache_key("university", name=name, province=province)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if name in self._universities:
            result = self._universities[name]
            self._set_cached(cache_key, result)
            return result

        if province:
            for key, value in self._universities.items():
                if value.get("province") == province and name in key:
                    result = value
                    self._set_cached(cache_key, result)
                    return result

        return None

    def query_industry(self, name: str) -> Optional[dict]:
        """查询行业信息。"""
        cache_key = self._cache_key("industry", name=name)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if name in self._industries:
            result = self._industries[name]
            self._set_cached(cache_key, result)
            return result

        for key, value in self._industries.items():
            if name in key or key in name:
                result = value
                self._set_cached(cache_key, result)
                return result

        return None

    def get_score_strategy(self, score: int) -> Optional[str]:
        """根据分数段获取填报策略。"""
        rules = self._decision_rules.get("score_range_strategies", {})

        def _parse_priority(item):
            """解析分数段key，返回排序优先级（越大越优先）。"""
            range_str = item[0]
            if "+" in range_str:
                # e.g. "650+"
                return int(range_str.replace("+", ""))
            elif "以下" in range_str:
                # e.g. "500以下" → 最低优先级
                return -1
            else:
                # e.g. "600-650"
                parts = range_str.split("-")
                if len(parts) == 2:
                    try:
                        return int(parts[0])
                    except ValueError:
                        return -1
                return -1

        for range_str, strategy in sorted(rules.items(), key=_parse_priority, reverse=True):
            if "+" in range_str:
                min_score = int(range_str.replace("+", ""))
                if score >= min_score:
                    return strategy
            elif "以下" in range_str:
                # e.g. "500以下"
                parts = range_str.split("以下")
                if parts[0]:
                    try:
                        max_score = int(parts[0])
                        if score < max_score:
                            return strategy
                    except ValueError:
                        pass
            else:
                parts = range_str.split("-")
                if len(parts) == 2:
                    try:
                        low, high = int(parts[0]), int(parts[1])
                        if low <= score <= high:
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
            return self._decision_rules.get("priority_rules", {}).get(key)
        return None

    @property
    def all_majors(self) -> dict:
        return dict(self._majors)

    @property
    def all_universities(self) -> dict:
        return dict(self._universities)

    @property
    def all_industries(self) -> dict:
        return dict(self._industries)
