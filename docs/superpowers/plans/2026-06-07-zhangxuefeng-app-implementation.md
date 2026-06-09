# 张雪峰 AI 志愿填报顾问 · 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个具备"张雪峰式决策能力"的 AI 志愿填报顾问系统，包含 FastAPI 后端、LangChain AI 链、JSON 知识库、Streamlit 前端、完整测试和 Docker 部署。

**Architecture:** 前后端分离架构。FastAPI 后端提供流式 SSE 聊天接口，内部通过 LangChain 编排 LLM 调用。动态 Prompt 构建器从 JSON 知识库查询数据并注入到模块化 System Prompt。Streamlit 前端通过 HTTP 调用后端 API，实现多轮对话界面。

**Tech Stack:** Python 3.10+, FastAPI, LangChain, langchain-openai, Streamlit, pydantic-settings, pytest, SSE-starlette, Docker

---

## 文件结构总览

```
.
├── backend/
│   ├── __init__.py
│   ├── main.py                      # FastAPI 入口 + 路由 + SSE
│   ├── models/
│   │   ├── __init__.py
│   │   ├── message.py               # ChatRequest, ChatResponse, Message Pydantic 模型
│   │   └── config.py                # AppSettings (BaseSettings)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── knowledge_base.py        # KnowledgeBase 类: 加载/查询 JSON 数据 + 缓存
│   │   ├── prompt_builder.py        # PromptBuilder 类: 加载 prompt 模块 + 动态注入
│   │   ├── llm_chain.py             # create_chain() 函数: LangChain 链构建 + 流式调用
│   │   └── data_updater.py          # add_data/update_data 函数: 增量更新 JSON
│   ├── data/
│   │   ├── majors.json              # 专业数据示例
│   │   ├── universities.json        # 院校数据示例
│   │   ├── industries.json          # 行业数据示例
│   │   └── decision_rules.json      # 分数段策略规则
│   └── prompts/
│       ├── mental_models.txt        # 5个思维模型
│       ├── decision_heuristics.txt  # 8条决策启发式
│       └── expression_dna.txt       # 表达风格 DNA
├── frontend/
│   └── app.py                       # Streamlit: 多轮对话界面 + 示例按钮 + spinner
├── tests/
│   ├── __init__.py
│   ├── test_knowledge_base.py       # KnowledgeBase 查询/边界/缓存测试
│   ├── test_prompt_builder.py       # Prompt 完整性/注入/回归/边界测试
│   ├── test_llm_chain.py            # LLM Chain mock 测试
│   └── test_api.py                  # API 端点测试
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── README.md
```

**执行顺序**: Task 1 (基础模块) → Task 2,3,4 可并行 → Task 5 → Task 6 → Task 7

---

### Task 1: 项目基础结构 + 配置 + 数据模型

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `backend/__init__.py`
- Create: `backend/models/__init__.py`
- Create: `backend/models/config.py`
- Create: `backend/models/message.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

```
# Backend
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
sse-starlette>=2.0.0
httpx>=0.27.0

# AI / LangChain
langchain>=0.3.0
langchain-openai>=0.2.0

# Frontend
streamlit>=1.38.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.24.0
pytest-cov>=5.0.0
```

- [ ] **Step 2: 创建 .env.example**

```
# OpenAI 兼容 API 配置
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
LLM_PROVIDER=openai

# 应用配置
CACHE_TTL_SECONDS=300
MAX_HISTORY_LENGTH=20
BACKEND_URL=http://localhost:8000
```

- [ ] **Step 3: 创建 .gitignore**

```
.env
__pycache__/
*.py[cod]
*.egg-info/
dist/
.venv/
venv/
.pytest_cache/
.coverage
htmlcov/
.DS_Store
*.log
```

- [ ] **Step 4: 创建 backend/__init__.py**

```python
# empty init
```

- [ ] **Step 5: 创建 backend/models/config.py**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """应用配置，通过环境变量或 .env 文件加载。"""

    openai_api_key: str
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    cache_ttl_seconds: int = 300
    max_history_length: int = 20
    backend_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# 全局单例，应用启动时加载
settings = AppSettings()
```

- [ ] **Step 6: 创建 backend/models/message.py**

```python
from pydantic import BaseModel, Field
from typing import Optional


class Message(BaseModel):
    """单条聊天消息。"""

    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    """聊天请求。"""

    message: str
    history: list[Message] = Field(default_factory=list)
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = "ok"
```

- [ ] **Step 7: 创建 backend/models/__init__.py**

```python
from .config import AppSettings, settings
from .message import Message, ChatRequest, HealthResponse

__all__ = [
    "AppSettings",
    "settings",
    "Message",
    "ChatRequest",
    "HealthResponse",
]
```

- [ ] **Step 8: 创建 tests/__init__.py**

```python
# empty init
```

- [ ] **Step 9: 验证基础结构**

Run: `python -c "from backend.models import settings, ChatRequest, HealthResponse; print('OK')"`
Expected: 如果设置了 OPENAI_API_KEY 环境变量则输出 "OK"，否则报错（正常）

---

### Task 2: 知识库模块 (KnowledgeBase + JSON 数据)

**Files:**
- Create: `backend/services/__init__.py`
- Create: `backend/services/knowledge_base.py`
- Create: `backend/data/majors.json`
- Create: `backend/data/universities.json`
- Create: `backend/data/industries.json`
- Create: `backend/data/decision_rules.json`
- Create: `backend/services/data_updater.py`
- Create: `tests/test_knowledge_base.py`

- [ ] **Step 1: 创建 backend/data/majors.json**

```json
{
  "金融学": {
    "employment_rate": 0.72,
    "avg_salary": 8500,
    "top_directions": ["银行柜员", "理财顾问", "财务"],
    "resource_threshold": "high",
    "description": "金融行业门槛高，985/211优先，普通家庭慎选"
  },
  "计算机科学与技术": {
    "employment_rate": 0.91,
    "avg_salary": 12000,
    "top_directions": ["后端开发", "前端开发", "算法工程师", "运维"],
    "resource_threshold": "low",
    "description": "就业面广，靠技术不靠关系，普通家庭首选"
  },
  "法学": {
    "employment_rate": 0.58,
    "avg_salary": 6000,
    "top_directions": ["律师", "法务", "公务员"],
    "resource_threshold": "medium",
    "description": "必须通过法考，5+3+2周期长，普通家庭需慎重"
  },
  "临床医学": {
    "employment_rate": 0.85,
    "avg_salary": 10000,
    "top_directions": ["三甲医院医师", "基层医院医师"],
    "resource_threshold": "low",
    "description": "学制长(5+3)，但就业稳定，社会地位高"
  },
  "新闻学": {
    "employment_rate": 0.65,
    "avg_salary": 5500,
    "top_directions": ["新媒体运营", "记者", "公关"],
    "resource_threshold": "medium",
    "description": "传统媒体萎缩，新媒体竞争激烈，非顶尖院校慎选"
  },
  "电气工程及其自动化": {
    "employment_rate": 0.88,
    "avg_salary": 9000,
    "top_directions": ["国家电网", "电力设计院", "自动化"],
    "resource_threshold": "low",
    "description": "电网系统就业稳定，适合追求稳定的家庭"
  },
  "会计学": {
    "employment_rate": 0.78,
    "avg_salary": 6500,
    "top_directions": ["企业会计", "审计", "税务"],
    "resource_threshold": "low",
    "description": "万金油专业，考CPA是王道"
  },
  "土木工程": {
    "employment_rate": 0.70,
    "avg_salary": 7000,
    "top_directions": ["施工单位", "设计院", "地产"],
    "resource_threshold": "low",
    "description": "行业下行，但基建仍有需求，适合能吃苦的"
  }
}
```

- [ ] **Step 2: 创建 backend/data/universities.json**

```json
{
  "郑州大学": {
    "province": "河南",
    "tier": "211",
    "min_score_2024": 580,
    "avg_score_2024": 610,
    "description": "河南唯一211，省内认可度高"
  },
  "河南大学": {
    "province": "河南",
    "tier": "双一流",
    "min_score_2024": 550,
    "avg_score_2024": 575,
    "description": "百年老校，文科见长"
  },
  "武汉大学": {
    "province": "湖北",
    "tier": "985",
    "min_score_2024": 630,
    "avg_score_2024": 650,
    "description": "中部顶尖985，法学/测绘/水利强势"
  },
  "华中科技大学": {
    "province": "湖北",
    "tier": "985",
    "min_score_2024": 635,
    "avg_score_2024": 655,
    "description": "工科强校，计算机/机械/光电强势"
  },
  "深圳大学": {
    "province": "广东",
    "tier": "普通一本",
    "min_score_2024": 560,
    "avg_score_2024": 590,
    "description": "地理位置优势明显，深圳就业资源好"
  },
  "北京邮电大学": {
    "province": "北京",
    "tier": "211",
    "min_score_2024": 620,
    "avg_score_2024": 640,
    "description": "信息科技强校，互联网就业优势大"
  }
}
```

- [ ] **Step 3: 创建 backend/data/industries.json**

```json
{
  "金融": {
    "entry_barrier": "high",
    "family_resource_dependent": true,
    "top_employers": ["银行", "证券", "保险", "基金"],
    "graduate_distribution": {
      "top_tier": 0.15,
      "mid_tier": 0.35,
      "grassroots": 0.50
    },
    "description": "资源密集型行业，家庭背景影响大"
  },
  "互联网": {
    "entry_barrier": "medium",
    "family_resource_dependent": false,
    "top_employers": ["大厂", "中小厂", "创业公司"],
    "graduate_distribution": {
      "top_tier": 0.25,
      "mid_tier": 0.45,
      "grassroots": 0.30
    },
    "description": "技术导向，凭能力说话，普通家庭友好"
  },
  "医疗": {
    "entry_barrier": "high",
    "family_resource_dependent": false,
    "top_employers": ["三甲医院", "基层医院", "私立医疗"],
    "graduate_distribution": {
      "top_tier": 0.20,
      "mid_tier": 0.50,
      "grassroots": 0.30
    },
    "description": "学制长但就业稳定，社会地位高"
  },
  "制造业": {
    "entry_barrier": "low",
    "family_resource_dependent": false,
    "top_employers": ["国企", "外企", "民企"],
    "graduate_distribution": {
      "top_tier": 0.10,
      "mid_tier": 0.60,
      "grassroots": 0.30
    },
    "description": "就业面广，起薪一般，经验值钱"
  }
}
```

- [ ] **Step 4: 创建 backend/data/decision_rules.json**

```json
{
  "score_range_strategies": {
    "650+": "冲刺985顶尖专业，学校和专业兼顾",
    "600-650": "可以兼顾学校和专业，优选211强势专业",
    "550-600": "优先选城市，其次专业，最后学校",
    "500-550": "专业>城市>学校，学一门手艺最重要",
    "500以下": "优先选实用技能专业，就业导向，城市次之"
  },
  "priority_rules": {
    "high_resource_family": "学校 > 专业 > 城市",
    "low_resource_family": "专业 > 城市 > 学校",
    "medium_resource_family": "专业 > 学校 > 城市"
  }
}
```

- [ ] **Step 5: 创建 backend/services/__init__.py**

```python
# empty init
```

- [ ] **Step 6: 创建 backend/services/knowledge_base.py**

```python
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
        for range_str, strategy in sorted(rules.items(), key=lambda x: -int(x[0].split("-")[0].replace("+", "999"))):
            if "+" in range_str:
                min_score = int(range_str.replace("+", ""))
                if score >= min_score:
                    return strategy
            else:
                parts = range_str.split("-")
                if len(parts) == 2:
                    low, high = int(parts[0]), int(parts[1])
                    if low <= score <= high:
                        return strategy
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
```

- [ ] **Step 7: 创建 backend/services/data_updater.py**

```python
"""知识库增量更新脚本。支持单条新增、批量导入、自动备份。"""

import json
import shutil
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _backup_file(filepath: Path) -> None:
    """备份文件到带时间戳的副本。"""
    backup = filepath.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(filepath, backup)


def add_or_update_data(filename: str, data: dict, merge: bool = True) -> None:
    """新增或更新单条数据。

    Args:
        filename: 数据文件名 (如 majors.json)
        data: 要写入的字典
        merge: True=增量合并, False=完全替换
    """
    filepath = DATA_DIR / filename
    if filepath.exists():
        _backup_file(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    if merge:
        existing.update(data)
    else:
        existing = data

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def validate_major(data: dict) -> list[str]:
    """校验专业数据格式，返回错误列表。"""
    errors = []
    required = ["employment_rate", "avg_salary", "top_directions", "resource_threshold", "description"]
    for field in required:
        if field not in data:
            errors.append(f"缺少必填字段: {field}")
    if "employment_rate" in data and not (0 <= data["employment_rate"] <= 1):
        errors.append("employment_rate 必须在 0-1 之间")
    return errors
```

- [ ] **Step 8: 创建 tests/test_knowledge_base.py**

```python
"""知识库模块测试。"""

import pytest
from pathlib import Path
import tempfile
import json
import sys

# 确保能导入 backend 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.knowledge_base import KnowledgeBase


@pytest.fixture
def temp_kb():
    """创建带测试数据的临时知识库。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # 写入测试数据
        (data_dir / "majors.json").write_text(
            json.dumps({
                "金融学": {
                    "employment_rate": 0.72,
                    "avg_salary": 8500,
                    "top_directions": ["银行柜员"],
                    "resource_threshold": "high",
                    "description": "测试专业"
                },
                "计算机科学与技术": {
                    "employment_rate": 0.91,
                    "avg_salary": 12000,
                    "top_directions": ["后端开发"],
                    "resource_threshold": "low",
                    "description": "测试专业2"
                }
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (data_dir / "universities.json").write_text(
            json.dumps({
                "郑州大学": {
                    "province": "河南",
                    "tier": "211",
                    "min_score_2024": 580,
                    "avg_score_2024": 610,
                    "description": "测试院校"
                }
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (data_dir / "industries.json").write_text(
            json.dumps({
                "金融": {
                    "entry_barrier": "high",
                    "family_resource_dependent": True,
                    "top_employers": ["银行"],
                    "graduate_distribution": {"top_tier": 0.15, "mid_tier": 0.35, "grassroots": 0.50},
                    "description": "测试行业"
                }
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (data_dir / "decision_rules.json").write_text(
            json.dumps({
                "score_range_strategies": {
                    "650+": "冲刺985",
                    "600-650": "兼顾学校和专业",
                    "550-600": "优先选城市",
                    "500-550": "专业>城市>学校",
                    "500以下": "优先实用技能"
                },
                "priority_rules": {
                    "high_resource_family": "学校 > 专业 > 城市",
                    "low_resource_family": "专业 > 城市 > 学校"
                }
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        yield KnowledgeBase(data_dir=data_dir, cache_ttl=60)


class TestKnowledgeBaseQueryMajor:
    """专业查询测试。"""

    def test_exact_match(self, temp_kb):
        result = temp_kb.query_major("金融学")
        assert result is not None
        assert result["employment_rate"] == 0.72
        assert result["resource_threshold"] == "high"

    def test_fuzzy_match(self, temp_kb):
        result = temp_kb.query_major("金融")
        assert result is not None
        assert "employment_rate" in result

    def test_no_match(self, temp_kb):
        result = temp_kb.query_major("不存在的专业")
        assert result is None

    def test_empty_name(self, temp_kb):
        result = temp_kb.query_major("")
        assert result is None


class TestKnowledgeBaseQueryUniversity:
    """院校查询测试。"""

    def test_exact_match(self, temp_kb):
        result = temp_kb.query_university("郑州大学")
        assert result is not None
        assert result["province"] == "河南"
        assert result["tier"] == "211"

    def test_with_province(self, temp_kb):
        result = temp_kb.query_university("郑州", province="河南")
        assert result is not None

    def test_no_match(self, temp_kb):
        result = temp_kb.query_university("不存在的学校")
        assert result is None


class TestKnowledgeBaseQueryIndustry:
    """行业查询测试。"""

    def test_exact_match(self, temp_kb):
        result = temp_kb.query_industry("金融")
        assert result is not None
        assert result["entry_barrier"] == "high"
        assert result["family_resource_dependent"] is True

    def test_fuzzy_match(self, temp_kb):
        result = temp_kb.query_industry("金")
        assert result is not None

    def test_no_match(self, temp_kb):
        result = temp_kb.query_industry("不存在的行业")
        assert result is None


class TestKnowledgeBaseDecisionRules:
    """决策规则测试。"""

    def test_score_strategy_high(self, temp_kb):
        result = temp_kb.get_score_strategy(660)
        assert result is not None
        assert "985" in result

    def test_score_strategy_mid(self, temp_kb):
        result = temp_kb.get_score_strategy(620)
        assert result is not None
        assert "兼顾" in result

    def test_score_strategy_low(self, temp_kb):
        result = temp_kb.get_score_strategy(520)
        assert result is not None
        assert "专业" in result

    def test_score_strategy_very_low(self, temp_kb):
        result = temp_kb.get_score_strategy(480)
        assert result is not None

    def test_priority_rule_low_resource(self, temp_kb):
        result = temp_kb.get_priority_rule("low")
        assert result is not None
        assert "专业" in result

    def test_priority_rule_high_resource(self, temp_kb):
        result = temp_kb.get_priority_rule("high")
        assert result is not None
        assert "学校" in result


class TestKnowledgeBaseCache:
    """缓存测试。"""

    def test_cache_returns_same_result(self, temp_kb):
        result1 = temp_kb.query_major("金融学")
        result2 = temp_kb.query_major("金融学")
        assert result1 == result2

    def test_cache_handles_no_match(self, temp_kb):
        result1 = temp_kb.query_major("不存在")
        result2 = temp_kb.query_major("不存在")
        assert result1 is None
        assert result2 is None
```

- [ ] **Step 9: 运行知识库测试**

Run: `pytest tests/test_knowledge_base.py -v`
Expected: ALL TESTS PASS

---

### Task 3: Prompt 模块 + PromptBuilder

**Files:**
- Create: `backend/prompts/mental_models.txt`
- Create: `backend/prompts/decision_heuristics.txt`
- Create: `backend/prompts/expression_dna.txt`
- Create: `backend/services/prompt_builder.py`
- Create: `tests/test_prompt_builder.py`

- [ ] **Step 1: 创建 backend/prompts/mental_models.txt**

```
# 思维模型 · 张雪峰认知操作系统

## 1. 现实优先原则
我不拍脑袋给建议，我看数据。就业率、薪资中位数、录取分数线——这些才是真的，其他都是扯淡。
判断一个专业好不好，先查三个数据：毕业半年后就业率、薪资中位数、去了什么单位。

## 2. 阶层流动视角
教育是普通家庭孩子跨越阶层最主要的通道。但不是所有专业都能起到这个作用。
选专业就是在选你未来十年的生活方式和收入水平。这不是兴趣问题，是生存问题。

## 3. 就业导向决策
毕业去哪比学什么重要。看一个专业好不好，就看它的毕业生中位数去了哪里。
不是去了高盛就是好专业，去了你家门口银行网点卖理财就不是。

## 4. 家庭资源评估
家里有矿的和家里没矿的，选专业的逻辑完全不同。
资源密集型行业（金融、艺术、传媒）：家里没资源就别碰。
技术密集型行业（计算机、医学、电气）：凭本事吃饭，普通家庭首选。

## 5. 地域杠杆效应
一线城市 > 新一线 > 省会 > 地级市。
大学所在的城市决定了你能接触到什么样的实习机会、什么样的眼界。
同样分数，能去一线去一线，能去省会去省会。
```

- [ ] **Step 2: 创建 backend/prompts/decision_heuristics.txt**

```
# 决策启发式 · 张雪峰认知操作系统

## 核心决策规则

1. **先选城市，再选专业，最后选学校**（普通家庭500-600分段）
   城市决定你能接触到什么机会。同样的学校在不同城市，就业完全不同。

2. **技术类优先**（普通家庭首选）
   计算机、电气、医学——这些专业靠技术不靠关系，毕业就能找到活。

3. **资源密集型专业慎选**
   金融、艺术、传媒——家里没资源就别碰。毕业了你拿什么跟985的抢？

4. **学制长的专业要看家庭承受力**
   医学(5+3)、法学(5+3+2)——读得起就读，读不起就别选。

5. **万金油专业不如一门手艺**
   会计、管理、市场营销——什么都会一点，什么都不精。不如学个具体的技术。

6. **分数段决定策略**
   650+：冲985顶尖专业
   600-650：211强势专业
   550-600：选城市+选好专业
   500-550：学一门手艺
   500以下：实用技能优先

7. **看行业不看名字**
   不要看专业名字好听就选。要看这个行业真实的社会需求。

8. **普通家庭的核心逻辑**
   毕业能养活自己 → 有发展前景 → 社会地位。按这个顺序选。
```

- [ ] **Step 3: 创建 backend/prompts/expression_dna.txt**

```
# 表达风格 DNA · 张雪峰认知操作系统

## 语言风格
- 东北大哥语气，直率、接地气
- 快节奏、段子化表达，不说废话
- 喜欢用"我跟你说"、"停停停"、"千万别"开头
- 善用反问和比喻："你拿什么抢？"、"社会就是一个大筛子"
- 数据+比喻双驱动：先说数据，再用比喻解释

## 回答结构
1. 先用一句话给结论（强观点）
2. 摆数据：就业率、薪资、分数线
3. 分析：为什么这么说
4. 建议：具体可操作的建议
5. 如果不确定，用"这个事我还真不太了解，但按我的经验..."

## 约束
- 不说"如果张雪峰，他可能会..."，直接以张雪峰身份说话
- 不说空话套话，每句话都要有信息量
- 用口语化表达，但数据要准确
- 遇到敏感话题保持客观，用数据说话
```

- [ ] **Step 4: 创建 backend/services/prompt_builder.py**

```python
"""Prompt 构建器：加载 prompt 模块，根据问题类型动态注入数据。"""

from pathlib import Path
from typing import Optional

from .knowledge_base import KnowledgeBase

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptBuilder:
    """构建 System Prompt，支持模块化加载和动态数据注入。"""

    def __init__(self, knowledge_base: KnowledgeBase, prompts_dir: Path = PROMPTS_DIR):
        self._kb = knowledge_base
        self._prompts_dir = prompts_dir
        self._mental_models = self._load_prompt("mental_models.txt")
        self._decision_heuristics = self._load_prompt("decision_heuristics.txt")
        self._expression_dna = self._load_prompt("expression_dna.txt")

    def _load_prompt(self, filename: str) -> str:
        filepath = self._prompts_dir / filename
        if not filepath.exists():
            return ""
        return filepath.read_text(encoding="utf-8")

    @property
    def mental_models(self) -> str:
        return self._mental_models

    @property
    def decision_heuristics(self) -> str:
        return self._decision_heuristics

    @property
    def expression_dna(self) -> str:
        return self._expression_dna

    def classify_question(self, question: str) -> str:
        """根据问题内容分类。

        Returns:
            "major" | "university" | "industry" | "general"
        """
        q = question.lower()
        major_keywords = ["专业", "学什么", "选专业", "就业前景"]
        university_keywords = ["学校", "大学", "院校", "报哪个学校", "分数线"]
        industry_keywords = ["行业", "就业", "工资", "薪资", "去哪", "做什么"]

        if any(kw in q for kw in major_keywords):
            return "major"
        if any(kw in q for kw in university_keywords):
            return "university"
        if any(kw in q for kw in industry_keywords):
            return "industry"
        return "general"

    def _extract_keywords(self, question: str) -> dict:
        """从问题中提取关键信息。"""
        info = {
            "score": None,
            "province": None,
            "major": None,
            "university": None,
            "industry": None,
        }

        # 提取分数（数字+分）
        import re
        score_match = re.search(r"(\d{3})\s*分", question)
        if score_match:
            info["score"] = int(score_match.group(1))

        # 提取省份（常见省份名）
        provinces = ["河南", "河北", "山东", "江苏", "浙江", "广东", "四川", "湖北", "湖南",
                     "安徽", "江西", "福建", "陕西", "山西", "辽宁", "吉林", "黑龙江",
                     "云南", "贵州", "广西", "甘肃", "新疆", "宁夏", "青海", "西藏",
                     "内蒙古", "海南", "北京", "上海", "天津", "重庆"]
        for p in provinces:
            if p in question:
                info["province"] = p
                break

        return info

    def build_system_prompt(self, question: str, history: Optional[list] = None) -> str:
        """构建完整的 System Prompt。

        根据问题类型注入相关数据：
        - 专业咨询 → 注入专业就业率数据 + 对应启发式
        - 院校选择 → 注入分数线数据 + 地域规则
        - 行业前景 → 注入行业就业分布数据
        - 综合规划 → 注入全部思维模型
        """
        question_type = self.classify_question(question)
        keywords = self._extract_keywords(question)

        parts = []

        # 1. 表达风格 DNA（始终注入）
        parts.append("## 表达风格\n")
        parts.append(self._expression_dna)
        parts.append("")

        # 2. 思维模型（始终注入基础框架）
        parts.append("## 思维模型\n")
        parts.append(self._mental_models)
        parts.append("")

        # 3. 决策启发式（始终注入）
        parts.append("## 决策启发式\n")
        parts.append(self._decision_heuristics)
        parts.append("")

        # 4. 动态数据注入
        parts.append("## 相关数据参考\n")
        data_injected = False

        if question_type == "major" or keywords.get("major"):
            # 尝试匹配专业
            major_keywords = ["金融", "计算机", "法学", "医学", "新闻", "电气", "会计", "土木"]
            for m in major_keywords:
                if m in question:
                    data = self._kb.query_major(m)
                    if data:
                        parts.append(f"### {m}专业数据\n")
                        parts.append(f"- 就业率: {data.get('employment_rate', 'N/A')}")
                        parts.append(f"- 平均薪资: {data.get('avg_salary', 'N/A')}")
                        parts.append(f"- 主要去向: {', '.join(data.get('top_directions', []))}")
                        parts.append(f"- 资源门槛: {data.get('resource_threshold', 'N/A')}")
                        parts.append(f"- 说明: {data.get('description', '')}")
                        parts.append("")
                        data_injected = True

        if question_type == "university" or keywords.get("province"):
            strategy = self._kb.get_score_strategy(keywords["score"]) if keywords.get("score") else None
            if strategy:
                parts.append(f"### 分数段策略 ({keywords['score']}分)\n")
                parts.append(strategy)
                parts.append("")
                data_injected = True

        if question_type == "industry":
            industry_keywords = ["金融", "互联网", "医疗", "制造"]
            for ind in industry_keywords:
                if ind in question:
                    data = self._kb.query_industry(ind)
                    if data:
                        parts.append(f"### {ind}行业数据\n")
                        parts.append(f"- 进入门槛: {data.get('entry_barrier', 'N/A')}")
                        parts.append(f"- 家庭资源依赖: {'是' if data.get('family_resource_dependent') else '否'}")
                        parts.append(f"- 主要雇主: {', '.join(data.get('top_employers', []))}")
                        dist = data.get("graduate_distribution", {})
                        if dist:
                            parts.append(f"- 毕业去向: 顶尖{dist.get('top_tier',0)*100:.0f}% 中层{dist.get('mid_tier',0)*100:.0f}% 基层{dist.get('grassroots',0)*100:.0f}%")
                        parts.append(f"- 说明: {data.get('description', '')}")
                        parts.append("")
                        data_injected = True

        if not data_injected:
            parts.append("## 相关数据参考\n")
            parts.append("（用户问题未匹配到具体数据，请基于思维模型和启发式回答）\n")

        # 5. 免责声明（仅首次）
        is_first_turn = not history or len(history) <= 1
        if is_first_turn:
            parts.append("## 重要\n")
            parts.append("首次回复时请说：「我以张雪峰视角和你聊，基于公开言论推断，非本人观点」\n")
            parts.append("后续对话不再重复此声明。\n")

        return "\n".join(parts)

    def get_all_modules(self) -> dict[str, str]:
        """返回所有 prompt 模块（用于测试）。"""
        return {
            "mental_models": self._mental_models,
            "decision_heuristics": self._decision_heuristics,
            "expression_dna": self._expression_dna,
        }
```

- [ ] **Step 5: 创建 tests/test_prompt_builder.py**

```python
"""Prompt 构建器测试。含完整性、动态注入、回归、边界、模块独立性测试。"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.prompt_builder import PromptBuilder
from backend.services.knowledge_base import KnowledgeBase


@pytest.fixture
def builder():
    """创建 PromptBuilder 实例（使用测试数据）。"""
    kb = KnowledgeBase()
    return PromptBuilder(knowledge_base=kb)


class TestPromptModules:
    """模块独立性测试。"""

    def test_mental_models_loaded(self, builder):
        modules = builder.get_all_modules()
        assert "mental_models" in modules
        assert len(modules["mental_models"]) > 0

    def test_decision_heuristics_loaded(self, builder):
        modules = builder.get_all_modules()
        assert "decision_heuristics" in modules
        assert len(modules["decision_heuristics"]) > 0

    def test_expression_dna_loaded(self, builder):
        modules = builder.get_all_modules()
        assert "expression_dna" in modules
        assert len(modules["expression_dna"]) > 0


class TestQuestionClassification:
    """问题分类测试。"""

    def test_classify_major(self, builder):
        assert builder.classify_question("我想学计算机专业") == "major"
        assert builder.classify_question("金融学就业前景怎么样") == "major"

    def test_classify_university(self, builder):
        assert builder.classify_question("560分报哪个学校") == "university"
        assert builder.classify_question("郑州大学分数线") == "university"

    def test_classify_industry(self, builder):
        assert builder.classify_question("金融行业工资高吗") == "industry"
        assert builder.classify_question("互联网就业前景") == "industry"

    def test_classify_general(self, builder):
        assert builder.classify_question("我该怎么选") == "general"


class TestPromptCompleteness:
    """完整性测试：验证 System Prompt 包含全部三个模块。"""

    def test_system_prompt_contains_mental_models(self, builder):
        prompt = builder.build_system_prompt("选什么专业好")
        assert "现实优先" in prompt
        assert "阶层流动" in prompt
        assert "就业导向" in prompt

    def test_system_prompt_contains_decision_heuristics(self, builder):
        prompt = builder.build_system_prompt("选什么专业好")
        assert "先选城市" in prompt
        assert "技术类优先" in prompt

    def test_system_prompt_contains_expression_dna(self, builder):
        prompt = builder.build_system_prompt("选什么专业好")
        assert "东北大哥" in prompt
        assert "接地气" in prompt


class TestDynamicInjection:
    """动态注入测试：验证不同问题类型注入正确数据。"""

    def test_major_injection(self, builder):
        prompt = builder.build_system_prompt("金融学好不好就业")
        assert "金融学" in prompt
        assert "就业率" in prompt
        assert "平均薪资" in prompt

    def test_university_score_injection(self, builder):
        prompt = builder.build_system_prompt("河南560分选什么专业")
        assert "560" in prompt
        assert "优先选城市" in prompt

    def test_industry_injection(self, builder):
        prompt = builder.build_system_prompt("金融行业工资怎么样")
        assert "金融" in prompt
        assert "进入门槛" in prompt
        assert "家庭资源依赖" in prompt


class TestRegressionAnchors:
    """模板回归测试：验证关键锚点字符串存在。"""

    def test_anchor_reality_first(self, builder):
        prompt = builder.build_system_prompt("随便问")
        assert "现实优先原则" in prompt

    def test_anchor_employment_rate(self, builder):
        prompt = builder.build_system_prompt("金融学")
        assert "就业率" in prompt

    def test_anchor_data_driven(self, builder):
        prompt = builder.build_system_prompt("随便问")
        assert "数据" in prompt


class TestBoundaryCases:
    """边界测试。"""

    def test_empty_question(self, builder):
        prompt = builder.build_system_prompt("")
        assert len(prompt) > 0
        assert "现实优先" in prompt

    def test_no_match_data(self, builder):
        prompt = builder.build_system_prompt("我想学一个不存在的专业叫量子按摩学")
        assert len(prompt) > 0
        assert "未匹配到具体数据" in prompt

    def test_history_suppresses_disclaimer(self, builder):
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好啊"},
            {"role": "user", "content": "金融学怎么样"},
        ]
        prompt = builder.build_system_prompt("金融学怎么样", history=history)
        assert "首次回复时请说" not in prompt

    def test_first_turn_shows_disclaimer(self, builder):
        prompt = builder.build_system_prompt("金融学怎么样")
        assert "首次回复时请说" in prompt
```

- [ ] **Step 6: 运行 Prompt 测试**

Run: `pytest tests/test_prompt_builder.py -v`
Expected: ALL TESTS PASS

---

### Task 4: LangChain LLM Chain 模块

**Files:**
- Create: `backend/services/llm_chain.py`
- Create: `tests/test_llm_chain.py`

- [ ] **Step 1: 创建 backend/services/llm_chain.py**

```python
"""LangChain LLM 链模块：构建和管理 LLM 调用。"""

from typing import Optional, AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.models.config import settings


def create_llm() -> ChatOpenAI:
    """创建 LLM 实例，从配置读取参数。"""
    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
        streaming=True,
        temperature=0.7,
    )


def format_history(history: list[dict]) -> list:
    """将历史消息转换为 LangChain 消息格式。"""
    messages = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


async def stream_chat(
    question: str,
    system_prompt: str,
    history: Optional[list[dict]] = None,
) -> AsyncGenerator[str, None]:
    """流式调用 LLM。

    Args:
        question: 用户问题
        system_prompt: 构建好的 System Prompt
        history: 历史消息列表

    Yields:
        流式文本片段
    """
    llm = create_llm()

    messages = [SystemMessage(content=system_prompt)]
    if history:
        messages.extend(format_history(history))
    messages.append(HumanMessage(content=question))

    async for chunk in llm.astream(messages):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content


def chat_sync(
    question: str,
    system_prompt: str,
    history: Optional[list[dict]] = None,
) -> str:
    """同步调用 LLM（用于测试）。"""
    llm = create_llm()

    messages = [SystemMessage(content=system_prompt)]
    if history:
        messages.extend(format_history(history))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)
```

- [ ] **Step 2: 创建 tests/test_llm_chain.py**

```python
"""LLM Chain 测试：使用 mock 测试 LLM 调用。"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.llm_chain import format_history, stream_chat


class TestFormatHistory:
    """历史消息格式化测试。"""

    def test_format_empty(self):
        result = format_history([])
        assert len(result) == 0

    def test_format_single_message(self):
        result = format_history([{"role": "user", "content": "你好"}])
        assert len(result) == 1
        assert result[0].content == "你好"

    def test_format_multiple_messages(self):
        history = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
            {"role": "user", "content": "问题2"},
        ]
        result = format_history(history)
        assert len(result) == 3
        assert result[0].content == "问题1"
        assert result[1].content == "回答1"
        assert result[2].content == "问题2"

    def test_format_unknown_role_treated_as_user(self):
        history = [{"role": "system", "content": "系统消息"}]
        result = format_history(history)
        # system role 不添加到历史（由 system_prompt 处理）
        assert len(result) == 1


@pytest.mark.asyncio
class TestStreamChat:
    """流式聊天测试（mock）。"""

    async def test_stream_yields_content(self):
        """测试流式输出能产生内容（需要 mock LLM）。"""
        mock_chunk = MagicMock()
        mock_chunk.content = "测试回复"

        async def mock_stream(_):
            yield mock_chunk

        with patch("backend.services.llm_chain.create_llm") as mock_create:
            mock_llm = MagicMock()
            mock_llm.astream = mock_stream
            mock_create.return_value = mock_llm

            chunks = []
            async for chunk in stream_chat("测试问题", "测试system prompt"):
                chunks.append(chunk)

            assert len(chunks) > 0
            assert chunks[0] == "测试回复"

    async def test_stream_with_history(self):
        """测试带历史消息的流式调用。"""
        mock_chunk = MagicMock()
        mock_chunk.content = "带历史的回复"

        async def mock_stream(_):
            yield mock_chunk

        with patch("backend.services.llm_chain.create_llm") as mock_create:
            mock_llm = MagicMock()
            mock_llm.astream = mock_stream
            mock_create.return_value = mock_llm

            history = [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好啊"},
            ]
            chunks = []
            async for chunk in stream_chat("继续问", "system", history=history):
                chunks.append(chunk)

            assert len(chunks) > 0
```

- [ ] **Step 3: 运行 LLM Chain 测试**

Run: `pytest tests/test_llm_chain.py -v`
Expected: ALL TESTS PASS

---

### Task 5: FastAPI 后端入口 + API 路由

**Files:**
- Create: `backend/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: 创建 backend/main.py**

```python
"""FastAPI 应用入口 + API 路由。"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from backend.models.config import settings
from backend.models.message import ChatRequest, HealthResponse
from backend.services.knowledge_base import KnowledgeBase
from backend.services.prompt_builder import PromptBuilder
from backend.services.llm_chain import stream_chat


# 全局服务实例
kb: KnowledgeBase
prompt_builder: PromptBuilder


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期管理。"""
    global kb, prompt_builder
    kb = KnowledgeBase(cache_ttl=settings.cache_ttl_seconds)
    prompt_builder = PromptBuilder(knowledge_base=kb)
    yield


app = FastAPI(title="张雪峰 AI 志愿填报顾问", lifespan=lifespan)

# Session 存储：session_id -> history
sessions: dict[str, list[dict]] = {}


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查端点。"""
    return HealthResponse(status="ok")


@app.post("/chat")
async def chat(request: ChatRequest):
    """流式聊天接口。使用 SSE 返回流式输出。

    Request:
        message: 用户输入
        history: 历史消息列表（可选）
        session_id: 会话ID（可选，不提供则自动生成）

    Response:
        Server-Sent Events stream，每个 event 包含 text chunk
    """
    global kb, prompt_builder

    session_id = request.session_id or "default"

    # 获取或创建 session 历史
    if session_id not in sessions:
        sessions[session_id] = []

    # 合并传入历史和 session 历史
    history = request.history if request.history else sessions[session_id]

    # 截断历史消息（防止 token 超出限制）
    max_len = settings.max_history_length
    if len(history) > max_len:
        history = history[-max_len:]

    # 构建 System Prompt
    system_prompt = prompt_builder.build_system_prompt(
        question=request.message,
        history=history,
    )

    # 保存用户消息到 session
    sessions[session_id].append({"role": "user", "content": request.message})

    # 构建完整历史（包含当前请求）
    full_history = history + [{"role": "user", "content": request.message}]

    async def event_generator():
        assistant_response = ""
        async for chunk in stream_chat(
            question=request.message,
            system_prompt=system_prompt,
            history=full_history,
        ):
            assistant_response += chunk
            yield {"event": "message", "data": json.dumps({"chunk": chunk})}

        # 保存助手回复到 session
        sessions[session_id].append({"role": "assistant", "content": assistant_response})
        yield {"event": "done", "data": json.dumps({"complete": True})}

    return EventSourceResponse(event_generator())


@app.post("/chat/sync")
async def chat_sync_endpoint(request: ChatRequest):
    """同步聊天接口（非流式，适用于不支持 SSE 的场景）。"""
    from backend.services.llm_chain import chat_sync

    global kb, prompt_builder

    session_id = request.session_id or "default"
    history = request.history if request.history else sessions.get(session_id, [])

    system_prompt = prompt_builder.build_system_prompt(
        question=request.message,
        history=history,
    )

    response_text = chat_sync(
        question=request.message,
        system_prompt=system_prompt,
        history=history,
    )

    return {"message": response_text}
```

- [ ] **Step 2: 创建 tests/test_api.py**

```python
"""API 端点测试。"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.main import app, sessions


@pytest.fixture
def client():
    """创建测试客户端。"""
    # 在测试中初始化 kb 和 prompt_builder
    from backend.services.knowledge_base import KnowledgeBase
    from backend.services.prompt_builder import PromptBuilder
    import backend.main

    backend.main.kb = KnowledgeBase()
    backend.main.prompt_builder = PromptBuilder(backend.main.kb)

    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    """每个测试后清空 sessions。"""
    yield
    sessions.clear()


class TestHealthEndpoint:
    """健康检查测试。"""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestChatEndpoint:
    """聊天接口测试。"""

    @patch("backend.main.stream_chat")
    def test_chat_accepts_request(self, mock_stream, client):
        """测试聊天接口能接受请求。"""
        async def mock_generator():
            yield {"event": "message", "data": '{"chunk": "测试回复"}'}
            yield {"event": "done", "data": '{"complete": true}'}

        mock_stream.return_value = mock_generator()

        response = client.post(
            "/chat",
            json={"message": "金融学怎么样", "session_id": "test-1"},
        )
        # SSE 响应，status code 应该是 200
        assert response.status_code == 200

    def test_chat_invalid_request(self, client):
        """测试无效请求（缺少 message 字段）。"""
        response = client.post("/chat", json={})
        # pydantic 验证失败返回 422
        assert response.status_code == 422

    @patch("backend.main.stream_chat")
    def test_chat_stores_session(self, mock_stream, client):
        """测试聊天接口存储 session 历史。"""
        async def mock_generator():
            yield {"event": "message", "data": '{"chunk": "回复"}'}
            yield {"event": "done", "data": '{"complete": true}'}

        mock_stream.return_value = mock_generator()

        client.post("/chat", json={"message": "你好", "session_id": "test-session"})

        assert "test-session" in sessions
        assert sessions["test-session"][0]["role"] == "user"
        assert sessions["test-session"][0]["content"] == "你好"

    def test_chat_default_session(self, client):
        """测试不提供 session_id 时使用 default。"""
        with patch("backend.main.stream_chat") as mock_stream:
            async def mock_generator():
                yield {"event": "message", "data": '{"chunk": "回复"}'}
                yield {"event": "done", "data": '{"complete": true}'}
            mock_stream.return_value = mock_generator()

            client.post("/chat", json={"message": "你好"})
            assert "default" in sessions


class TestSyncChatEndpoint:
    """同步聊天接口测试。"""

    @patch("backend.services.llm_chain.chat_sync")
    def test_sync_chat_returns_message(self, mock_chat, client):
        mock_chat.return_value = "测试回复"
        response = client.post("/chat/sync", json={"message": "你好"})
        assert response.status_code == 200
        assert "message" in response.json()
```

- [ ] **Step 3: 运行 API 测试**

Run: `pytest tests/test_api.py -v`
Expected: ALL TESTS PASS

---

### Task 6: Streamlit 前端

**Files:**
- Create: `frontend/__init__.py` (empty)
- Create: `frontend/app.py`

- [ ] **Step 1: 创建 frontend/__init__.py**

```python
# empty init
```

- [ ] **Step 2: 创建 frontend/app.py**

```python
"""Streamlit 前端：多轮对话界面 + 示例问题 + Loading spinner。"""

import json
import os
import uuid

import httpx
import streamlit as st

# 配置
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="张雪峰 AI 志愿填报顾问",
    page_icon="🎓",
    layout="wide",
)

# CSS 样式优化
st.markdown(
    """
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}
.assistant-message {
    background-color: #f5f5f5;
    border-left: 4px solid #4caf50;
}
.role-label {
    font-weight: bold;
    margin-bottom: 0.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# 示例问题
SAMPLE_QUESTIONS = [
    "河南560分想学金融，你怎么看？",
    "计算机和临床医学选哪个？",
    "普通家庭孩子选什么专业好？",
    "新闻学专业就业前景怎么样？",
    "600分在河南能上什么好学校？",
    "电气工程及其自动化好不好就业？",
]


def init_session():
    """初始化 Streamlit session state。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def render_message(role: str, content: str):
    """渲染单条聊天消息。"""
    css_class = "user-message" if role == "user" else "assistant-message"
    label = "🧑 你" if role == "user" else "🎓 张雪峰"
    st.markdown(
        f"""
<div class="chat-message {css_class}">
    <div class="role-label">{label}</div>
    <div>{content}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def call_backend(message: str, history: list[dict]) -> str:
    """调用后端同步聊天接口。"""
    url = f"{BACKEND_URL}/chat/sync"
    payload = {
        "message": message,
        "history": history,
        "session_id": st.session_state.session_id,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("message", "未收到有效回复")
    except httpx.ConnectError:
        return "⚠️ 无法连接到后端服务，请确认后端已启动（http://localhost:8000）"
    except httpx.HTTPStatusError as e:
        return f"⚠️ 后端返回错误: {e.response.status_code}"
    except Exception as e:
        return f"⚠️ 请求失败: {str(e)}"


def main():
    init_session()

    # 标题
    st.title("🎓 张雪峰 AI 志愿填报顾问")
    st.caption(
        "基于张雪峰思维操作系统的数据驱动志愿填报建议。"
        "我以张雪峰视角和你聊，基于公开言论推断，非本人观点。"
    )

    # 示例问题按钮（一行6个）
    st.markdown("### 💡 示例问题")
    cols = st.columns(3)
    for i, question in enumerate(SAMPLE_QUESTIONS):
        col = cols[i % 3]
        if col.button(question, key=f"sample_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            # 触发自动回复
            st.session_state.trigger_reply = question

    # 历史聊天记录（可折叠）
    if st.session_state.messages:
        with st.expander(f"📜 历史对话 ({len(st.session_state.messages)} 条)", expanded=False):
            for i, msg in enumerate(st.session_state.messages):
                render_message(msg["role"], msg["content"])

    # 聊天历史展示
    st.markdown("### 💬 对话")
    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"])

    # 输入框
    user_input = st.chat_input("输入你的问题，例如：河南560分选什么专业好？")

    # 处理用户输入
    message = user_input or st.session_state.get("trigger_reply")

    if message:
        # 清除触发标记
        if "trigger_reply" in st.session_state:
            del st.session_state.trigger_reply

        # 如果不是示例按钮触发的（已经在上面添加了），添加用户消息
        if message != st.session_state.get("trigger_reply") or message not in [m["content"] for m in st.session_state.messages if m["role"] == "user"]:
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != message:
                st.session_state.messages.append({"role": "user", "content": message})

        # 调用后端
        with st.spinner("🎓 张雪峰正在查看数据并思考中..."):
            # 构建历史（不含最新消息）
            history = st.session_state.messages[:-1] if st.session_state.messages else []
            response = call_backend(message, history)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # 刷新页面显示最新回复
        st.rerun()


if __name__ == "__main__":
    main()
```

---

### Task 7: 工程化文件 + 部署

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `README.md`

- [ ] **Step 1: 创建 Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# 暴露端口
EXPOSE 8501

# 启动 Streamlit（后端通过 http 调用本地的 localhost:8000）
# 使用 --server.headless=true 使 Streamlit 可以在容器中运行
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.headless=true"]
```

- [ ] **Step 2: 创建 docker-compose.yml**

```yaml
version: "3.8"

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000

  frontend:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    environment:
      - BACKEND_URL=http://backend:8000
    command: streamlit run frontend/app.py --server.address=0.0.0.0 --server.headless=true
    depends_on:
      - backend
```

- [ ] **Step 3: 创建 README.md**

```markdown
# 🎓 张雪峰 AI 志愿填报顾问

> 「选择比努力更重要，但'有得选'的前提是你足够努力。」

基于 [zhangxuefeng-skill](https://github.com/alchaincyf/zhangxuefeng-skill) 思维框架构建的 AI 志愿填报顾问系统。

## 功能特点

- **数据驱动**：内置专业就业率、院校分数线、行业就业分布等 JSON 知识库
- **思维模型**：5大核心心智模型 + 8条决策启发式 + 完整表达DNA
- **动态 Prompt**：根据问题类型自动注入相关数据和策略
- **多轮对话**：支持上下文记忆，session 隔离
- **模型切换**：支持任何 OpenAI 兼容 API（OpenAI/DeepSeek/通义千问等）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 3. 启动后端

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 启动前端

```bash
streamlit run frontend/app.py
```

### 5. 访问

打开浏览器访问 http://localhost:8501

## Docker 部署

### 单容器模式（开发/演示）

```bash
docker build -t zhangxuefeng-ai .
docker run -p 8501:8501 --env-file .env zhangxuefeng-ai
```

### 分离容器模式（生产）

```bash
docker-compose up -d
```

## 模型切换

编辑 `.env` 文件：

```bash
# OpenAI
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# DeepSeek
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat

# 通义千问
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-plus
```

修改后重启服务即可生效。

## 项目结构

```
├── backend/              # FastAPI 后端
│   ├── main.py           # 应用入口 + API 路由
│   ├── models/           # 数据模型
│   ├── services/         # 业务逻辑（知识库/Prompt/LLM）
│   ├── data/             # JSON 知识库数据
│   └── prompts/          # Prompt 模板模块
├── frontend/             # Streamlit 前端
│   └── app.py            # 多轮对话界面
├── tests/                # 测试
├── requirements.txt      # Python 依赖
├── .env.example          # 环境变量模板
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 运行测试

```bash
pytest tests/ -v --tb=short
```

## 知识库扩展

新增专业数据：

```python
from backend.services.data_updater import add_or_update_data, validate_major

new_major = {
    "人工智能": {
        "employment_rate": 0.93,
        "avg_salary": 15000,
        "top_directions": ["算法工程师", "数据科学家"],
        "resource_threshold": "low",
        "description": "新兴热门专业，薪资高但门槛也高"
    }
}

errors = validate_major(new_major["人工智能"])
if not errors:
    add_or_update_data("majors.json", new_major)
```

## License

MIT
```

- [ ] **Step 4: 验证所有文件创建完成**

Run: `python -c "
from pathlib import Path
files = [
    'requirements.txt', '.env.example', '.gitignore',
    'backend/__init__.py', 'backend/main.py',
    'backend/models/__init__.py', 'backend/models/config.py', 'backend/models/message.py',
    'backend/services/__init__.py', 'backend/services/knowledge_base.py',
    'backend/services/prompt_builder.py', 'backend/services/llm_chain.py',
    'backend/services/data_updater.py',
    'backend/data/majors.json', 'backend/data/universities.json',
    'backend/data/industries.json', 'backend/data/decision_rules.json',
    'backend/prompts/mental_models.txt', 'backend/prompts/decision_heuristics.txt',
    'backend/prompts/expression_dna.txt',
    'frontend/app.py',
    'tests/__init__.py', 'tests/test_knowledge_base.py',
    'tests/test_prompt_builder.py', 'tests/test_llm_chain.py', 'tests/test_api.py',
    'Dockerfile', 'docker-compose.yml', 'README.md',
]
missing = [f for f in files if not Path(f).exists()]
if missing:
    print('MISSING:', missing)
else:
    print('ALL FILES PRESENT')
"
Expected: "ALL FILES PRESENT"

---

### Task 8: 全量测试 + 最终验证

- [ ] **Step 1: 运行全部测试**

Run: `pytest tests/ -v --tb=short`
Expected: ALL TESTS PASS (20+ tests)

- [ ] **Step 2: 验证后端可启动**

Run: `python -c "from backend.main import app; print('FastAPI app created successfully')"`
Expected: "FastAPI app created successfully"

- [ ] **Step 3: 验证前端可导入**

Run: `python -c "import ast; ast.parse(open('frontend/app.py').read()); print('Streamlit app syntax OK')"`
Expected: "Streamlit app syntax OK"

- [ ] **Step 4: 验证依赖可解析**

Run: `pip check` (if dependencies installed) or `python -c "import fastapi, streamlit, langchain_openai, pydantic_settings; print('All imports OK')"`
Expected: 无报错
