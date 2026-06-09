# Vector Knowledge Base 升级架构设计文档

> 中国高校志愿填报咨询系统 · 向量知识库升级方案
> 版本: v4.0 | 日期: 2026-06-08

---

## 目录

1. [现有系统分析](#1-现有系统分析)
2. [新系统架构设计](#2-新系统架构设计)
3. [关键模块代码](#3-关键模块代码)
4. [数据结构升级](#4-数据结构升级)
5. [检索流程说明](#5-检索流程说明)
6. [可解释性设计](#6-可解释性设计)
7. [性能优化方案](#7-性能优化方案)
8. [新旧版本对比](#8-新旧版本对比)
9. [实施路线图](#9-实施路线图)

---

## 1. 现有系统分析

### 1.1 当前架构

```
┌─────────────────────────────────────────────────────┐
│                   用户请求 (Frontend)                │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│              Orchestrator Agent                      │
│  (协调器: 意图解析 → 数据检索 → 规划 → 精炼)        │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│              DataRetrieverV3                         │
│  · SHA-256 伪向量 (384维)                            │
│  · 语义匹配 (60%) + 规则打分 (40%)                   │
│  · 硬编码阈值过滤                                     │
└────────────────────────┬────────────────────────────┘
               ┌─────────┴──────────┐
               ▼                    ▼
┌─────────────────────┐  ┌─────────────────────┐
│   KnowledgeBase     │  │  EmbeddingService   │
│  · JSON 加载         │  │  · SHA-256 hash     │
│  · TTL 内存缓存      │  │  · 无真实语义理解    │
│  · 模糊字符串匹配    │  │  · 无缓存机制        │
└─────────────────────┘  └─────────────────────┘
```

### 1.2 现有核心代码分析

| 文件 | 核心类 | 问题点 |
|------|--------|--------|
| `embedding_service.py` | `EmbeddingService` | SHA-256 哈希生成伪向量，无真实语义能力；每次查询重复计算 |
| `knowledge_base.py` | `KnowledgeBase` | 简单 JSON 加载 + TTL 缓存；查询仅支持字符串包含匹配 |
| `data_retriever.py` | `DataRetrieverV3` | 两阶段评分(语义60%+规则40%)；权重硬编码；硬编码行业-专业映射 |
| `vector_knowledge_base.py` | `VectorKnowledgeBase` | 已有向量文档模型，但未与 FAISS 集成；缓存仅 JSON 文件 |
| `majors.json` | — | 仅含 `employment_rate`, `avg_salary`, `top_directions`, `resource_threshold`, `description` |

### 1.3 痛点总结

1. **Embedding 能力弱** — SHA-256 哈希向量无法捕捉真实语义相似性
2. **检索精度低** — 字符串包含匹配无法处理同义词、近义词
3. **不可扩展** — 硬编码权重和映射，无法配置
4. **无可解释性** — 结果仅返回 match_score，无推荐理由
5. **性能瓶颈** — 每次查询重新计算所有文档 embedding
6. **数据贫乏** — majors.json 缺乏课程、技能、性格适配等维度

---

## 2. 新系统架构设计

### 2.1 总体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          用户请求                                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DataRetrieverV4                                 │
│  ┌─────────────┐  ┌─────────────────────┐  ┌───────────────────┐   │
│  │  Query      │→ │   Multi-Path        │→ │   Re-Ranking      │   │
│  │  Under-     │  │   Recall            │  │   & Explain       │   │
│  │  standing   │  │   (3 paths)         │  │   (可解释)         │   │
│  └─────────────┘  └─────────────────────┘  └───────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐
│ Embedding    │  │   Vector Index  │  │  KnowledgeBase   │
│ Service      │  │   (FAISS/NumPy) │  │  (增强数据模型)   │
│              │  │                 │  │                  │
│ ·Strategy   │  │ ·Lazy Loading   │  │ ·majors.json v2  │
│  Pattern     │  │ ·TopK Query     │  │ ·动态更新         │
│ ·Embedding   │  │ ·Dynamic Update │  │ ·关联图谱         │
│  Cache(LRU)  │  │ ·Persistence    │  │                  │
└──────────────┘  └─────────────────┘  └──────────────────┘
```

### 2.2 模块职责

| 模块 | 职责 | 关键能力 |
|------|------|----------|
| **EmbeddingService** | 文本向量化的统一接口 | 策略模式、多后端、缓存、批量处理 |
| **VectorIndex** | 向量存储与相似度检索 | FAISS/NumPy 双引擎、懒加载、TopK |
| **KnowledgeBase** | 业务数据管理 | 增强数据模型、关联查询、TTL缓存 |
| **DataRetrieverV4** | 多阶段检索编排 | Query理解、多路召回、重排序、可解释 |

### 2.3 目录结构

```
backend/
├── services/
│   ├── embedding_service.py          # [重构] Embedding 策略服务
│   ├── vector_index.py               # [新增] 向量索引管理
│   ├── knowledge_base.py             # [增强] 知识库 (v2 数据模型)
│   └── vector_knowledge_base.py      # [增强] 向量知识库
├── agents/
│   └── data_retriever.py             # [重构] DataRetrieverV4
├── retriever/                        # [新增] 检索子模块
│   ├── __init__.py
│   ├── query_understanding.py        # 查询理解
│   ├── recall/                       # 召回策略
│   │   ├── __init__.py
│   │   ├── semantic_recall.py        # 语义召回
│   │   ├── rule_recall.py            # 规则召回
│   │   └── keyword_recall.py         # 关键词召回
│   ├── reranker.py                   # 重排序器
│   └── explainability.py             # 可解释性引擎
├── cache/
│   ├── embedding_cache.py            # [新增] LRU Embedding 缓存
│   └── query_cache.py                # [新增] 查询 TTL 缓存
├── models/
│   ├── agent_output.py               # [增强] 输出模型
│   └── retrieval_models.py           # [新增] 检索数据模型
└── config/
    └── retrieval_config.yaml         # [新增] 检索配置
```

---

## 3. 关键模块代码

### 3.1 EmbeddingService (策略模式)

```python
"""EmbeddingService: 策略模式实现多后端向量生成。"""

import os
import time
import hashlib
import math
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from collections import OrderedDict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Strategy 抽象基类
# ──────────────────────────────────────────────

class EmbeddingStrategy(ABC):
    """向量生成策略抽象基类。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度。"""
        ...

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """将单条文本编码为向量。"""
        ...

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本。默认逐条调用 encode，子类可覆写优化。"""
        return [self.encode(t) for t in texts]

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ──────────────────────────────────────────────
# 策略 1: HashEmbedding (降级方案)
# ──────────────────────────────────────────────

class HashEmbeddingStrategy(EmbeddingStrategy):
    """基于 SHA-256 的确定性伪向量，作为降级兜底方案。"""

    DIMENSION = 384

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def encode(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.DIMENSION

        vector = [0.0] * self.DIMENSION
        for seed in range(4):
            seed_bytes = seed.to_bytes(4, "big")
            raw = hashlib.sha256(seed_bytes + text.encode("utf-8")).digest()
            for i in range(min(len(raw) * 8, self.DIMENSION)):
                byte_idx = i // 8
                bit_idx = i % 8
                bit = (raw[byte_idx] >> bit_idx) & 1
                value = (bit * 2 - 1) * (1.0 - (i % 13) * 0.05)
                vector[i] += value

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in in vector]
        return vector


# ──────────────────────────────────────────────
# 策略 2: LocalEmbedding (bge-small-zh)
# ──────────────────────────────────────────────

class LocalEmbeddingStrategy(EmbeddingStrategy):
    """使用本地 Sentence-Transformer 模型 (BAAI/bge-small-zh)。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, text: str) -> List[float]:
        embedding = self._model.encode([text], normalize_embeddings=True)[0]
        return embedding.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32,
        )
        return embeddings.tolist()


# ──────────────────────────────────────────────
# 策略 3: APIEmbedding (OpenAI / DeepSeek)
# ──────────────────────────────────────────────

class APIEmbeddingStrategy(EmbeddingStrategy):
    """使用 OpenAI 兼容 API 生成向量 (支持 DeepSeek / OpenAI)。"""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
    ):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._model = model
        self._dim = None  # lazy discover

    @property
    def dimension(self) -> int:
        if self._dim is None:
            result = self._client.embeddings.create(
                model=self._model, input="dim_probe"
            )
            self._dim = len(result.data[0].embedding)
        return self._dim

    def encode(self, text: str) -> List[float]:
        result = self._client.embeddings.create(
            model=self._model, input=[text], encoding_format="float",
        )
        return result.data[0].embedding

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = self._client.embeddings.create(
            model=self._model, input=texts, encoding_format="float",
        )
        return [d.embedding for d in result.data]


# ──────────────────────────────────────────────
# LRU Embedding Cache
# ──────────────────────────────────────────────

class EmbeddingCache:
    """LRU 内存缓存 + 持久化到 JSON。"""

    def __init__(self, max_size: int = 10000, persist_path: Optional[str] = None):
        self._cache: OrderedDict[str, list] = OrderedDict()
        self._max_size = max_size
        self._persist_path = persist_path
        self._load_persist()

    def _persist_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_persist(self):
        import json
        from pathlib import Path
        if self._persist_path and Path(self._persist_path).exists():
            try:
                with open(self._persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key, value in list(data.items())[-self._max_size:]:
                    self._cache[key] = value
            except (json.JSONDecodeError, IOError):
                pass

    def get(self, text: str) -> Optional[List[float]]:
        key = self._persist_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, text: str, embedding: List[float]):
        key = self._persist_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = embedding
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = embedding

    def save(self):
        import json
        from pathlib import Path
        if self._persist_path:
            Path(self._persist_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(dict(self._cache), f, ensure_ascii=False)

    @property
    def size(self) -> int:
        return len(self._cache)


# ──────────────────────────────────────────────
# EmbeddingService 门面类
# ──────────────────────────────────────────────

class EmbeddingService:
    """向量生成服务门面，管理策略选择和缓存。"""

    _instance: Optional["EmbeddingService"] = None

    def __init__(
        self,
        strategy: Optional[EmbeddingStrategy] = None,
        cache_max_size: int = 10000,
        cache_persist_path: Optional[str] = None,
    ):
        self._strategy = strategy or self._auto_select_strategy()
        self._cache = EmbeddingCache(
            max_size=cache_max_size,
            persist_path=cache_persist_path,
        )
        logger.info(
            "EmbeddingService initialized with strategy=%s (dim=%d)",
            self._strategy.name,
            self._strategy.dimension,
        )

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        """单例访问。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _auto_select_strategy() -> EmbeddingStrategy:
        """根据环境变量自动选择策略。"""
        strategy_type = os.getenv("EMBEDDING_STRATEGY", "hash").lower()

        if strategy_type == "api":
            api_key = os.getenv("EMBEDDING_API_KEY", "")
            if not api_key:
                logger.warning("EMBEDDING_API_KEY not set, falling back to hash")
                return HashEmbeddingStrategy()
            return APIEmbeddingStrategy(
                api_key=api_key,
                api_base=os.getenv("EMBEDDING_API_BASE", "https://api.openai.com/v1"),
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            )

        elif strategy_type == "local":
            try:
                return LocalEmbeddingStrategy(
                    model_name=os.getenv("EMBEDDING_LOCAL_MODEL", "BAAI/bge-small-zh"),
                )
            except ImportError:
                logger.warning("sentence_transformers not installed, falling back to hash")
                return HashEmbeddingStrategy()

        # default: hash
        return HashEmbeddingStrategy()

    # ── 公共 API ──

    def get_embedding(self, text: str) -> List[float]:
        """获取单条文本的向量，优先使用缓存。"""
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        embedding = self._strategy.encode(text)
        self._cache.put(text, embedding)
        return embedding

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取向量。"""
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                results.append(None)  # placeholder

        if uncached_texts:
            embeddings = self._strategy.encode_batch(uncached_texts)
            for idx, embedding in zip(uncached_indices, embeddings):
                results[idx] = embedding
                self._cache.put(uncached_texts[uncached_indices.index(text := texts[idx])], embedding)

        return results

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """余弦相似度。"""
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def dimension(self) -> int:
        return self._strategy.dimension

    @property
    def strategy_name(self) -> str:
        return self._strategy.name

    def save_cache(self):
        self._cache.save()
```

### 3.2 VectorIndex (FAISS / NumPy 引擎)

```python
"""VectorIndex: 向量索引管理，支持 FAISS 和 NumPy 双引擎。"""

import os
import json
import logging
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorIndex:
    """向量相似度索引，支持 FAISS (优先) 和 NumPy 回退。

    Args:
        dimension: 向量维度
        engine: 'faiss' 或 'numpy'，None 时自动检测
        persist_path: 持久化路径
    """

    def __init__(
        self,
        dimension: int,
        engine: Optional[str] = None,
        persist_path: Optional[str] = None,
    ):
        self._dimension = dimension
        self._persist_path = persist_path
        self._metadata: List[dict] = []  # 与向量一一对应的元数据
        self._dirty = False

        # 引擎选择
        if engine is None:
            engine = "faiss" if self._check_faiss() else "numpy"

        self._engine = engine
        self._index = None
        self._vectors: Optional[np.ndarray] = None  # numpy 引擎使用
        self._loaded = False

    @staticmethod
    def _check_faiss() -> bool:
        try:
            import faiss
            return True
        except ImportError:
            return False

    # ── 懒加载 ──

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._build_empty_index()
        if self._persist_path and Path(self._persist_path).exists():
            self.load(self._persist_path)
        self._loaded = True

    def _build_empty_index(self):
        if self._engine == "faiss":
            import faiss
            self._index = faiss.IndexFlatIP(self._dimension)  # 内积 (归一化后 = 余弦)
        else:
            self._vectors = np.empty((0, self._dimension), dtype=np.float32)

    # ── 增删改 ──

    def add(self, vector: List[float], metadata: dict = None):
        self.add_batch([vector], [metadata] if metadata else [{}])

    def add_batch(self, vectors: List[List[float]], metadatas: Optional[List[dict]] = None):
        self._ensure_loaded()
        arr = np.array(vectors, dtype=np.float32)

        if self._engine == "faiss":
            self._index.add(arr)
        else:
            self._vectors = np.vstack([self._vectors, arr]) if len(self._vectors) > 0 else arr

        if metadatas:
            self._metadata.extend(metadatas)
        else:
            self._metadata.extend([{}] * len(vectors))
        self._dirty = True

    def update(self, idx: int, vector: List[float], metadata: dict = None):
        """更新指定位置的向量。"""
        self._ensure_loaded()
        arr = np.array([vector], dtype=np.float32)

        if self._engine == "faiss":
            # FAISS IndexFlat 不支持原地更新，需重建
            all_vecs = self.get_all_vectors()
            all_vecs[idx] = arr[0]
            import faiss
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(all_vecs)
        else:
            self._vectors[idx] = arr[0]

        if metadata:
            self._metadata[idx] = metadata
        self._dirty = True

    # ── 查询 ──

    def search(self, query: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        """TopK 相似度检索，返回 [(索引, 相似度)]。"""
        self._ensure_loaded()
        if self._count() == 0:
            return []

        query_arr = np.array([query], dtype=np.float32)

        if self._engine == "faiss":
            k = min(top_k, self._count())
            scores, indices = self._index.search(query_arr, k)
            return [(int(indices[0][i]), float(scores[0][i])) for i in range(k)]
        else:
            # NumPy: 计算余弦相似度
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = self._vectors / norms
            query_norm = np.linalg.norm(query_arr)
            if query_norm > 0:
                query_normalized = query_arr / query_norm
            else:
                return []
            scores = normalized @ query_normalized.T
            scores = scores.flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(int(i), float(scores[i])) for i in top_indices]

    # ── 持久化 ──

    def save(self, path: Optional[str] = None):
        """保存索引和元数据。"""
        self._ensure_loaded()
        save_path = Path(path or self._persist_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self._engine == "faiss":
            import faiss
            faiss.write_index(self._index, str(save_path.with_suffix(".index")))
        else:
            np.save(save_path.with_suffix(".npy"), self._vectors)

        # 保存元数据
        meta_path = save_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        self._dirty = False

    def load(self, path: Optional[str] = None):
        """加载索引和元数据。"""
        load_path = Path(path or self._persist_path)

        if self._engine == "faiss":
            import faiss
            index_file = load_path.with_suffix(".index")
            if index_file.exists():
                self._index = faiss.read_index(str(index_file))
        else:
            npy_file = load_path.with_suffix(".npy")
            if npy_file.exists():
                self._vectors = np.load(npy_file)

        meta_file = load_path.with_suffix(".meta.json")
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

    # ── 工具方法 ──

    def get_all_vectors(self) -> np.ndarray:
        """获取所有向量。"""
        self._ensure_loaded()
        if self._engine == "faiss":
            n = self._count()
            return self._index.reconstruct_n(0, n)
        return self._vectors.copy()

    def get_metadata(self, idx: int) -> dict:
        return self._metadata[idx] if 0 <= idx < len(self._metadata) else {}

    def _count(self) -> int:
        if self._engine == "faiss" and self._index:
            return self._index.ntotal
        elif self._vectors is not None:
            return len(self._vectors)
        return 0

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def engine(self) -> str:
        return self._engine

    @property
    def dimension(self) -> int:
        return self._dimension
```

### 3.3 DataRetrieverV4 (多阶段检索)

```python
"""DataRetrieverV4: 多阶段检索 Agent。

检索流程:
  1. Query Understanding  → 提取意图、兴趣、约束
  2. Multi-Path Recall     → 语义召回 + 规则召回 + 关键词召回
  3. Score Fusion          → 加权融合多路结果
  4. Re-Ranking            → 根据用户画像重排序
  5. Explainability        → 生成推荐理由和匹配依据
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from backend.models.agent_output import UserProfile, DataRetrievalResult
from backend.services.knowledge_base import KnowledgeBase
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_index import VectorIndex

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class RecallItem:
    """单条召回结果。"""
    major_name: str
    score: float
    source: str          # "semantic" | "rule" | "keyword"
    reasons: List[str] = field(default_factory=list)
    data_support: List[str] = field(default_factory=list)


@dataclass
class ExplainInfo:
    """可解释性信息。"""
    match_reason: str          # 为什么匹配
    data_support: List[str]    # 数据依据
    recommend_reason: str      # 推荐理由
    risk_warnings: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# 召回策略
# ──────────────────────────────────────────────

class SemanticRecall:
    """语义召回: 使用向量相似度检索。"""

    def __init__(self, embedding: EmbeddingService, index: VectorIndex):
        self.embedding = embedding
        self.index = index

    def recall(self, query: str, top_k: int = 20) -> List[RecallItem]:
        query_emb = self.embedding.get_embedding(query)
        results = self.index.search(query_emb, top_k=top_k)
        items = []
        for idx, sim_score in results:
            meta = self.index.get_metadata(idx)
            items.append(RecallItem(
                major_name=meta.get("name", ""),
                score=sim_score,
                source="semantic",
                reasons=[f"语义相似度: {sim_score:.3f}"],
                data_support=[meta.get("description", "")],
            ))
        return items


class RuleRecall:
    """规则召回: 基于分数、就业率、资源适配等业务规则。"""

    def recall(self, profile: UserProfile, all_majors: dict) -> List[RecallItem]:
        items = []
        for name, data in all_majors.items():
            score = self._score(name, data, profile)
            if score > 0:
                reasons = self._explain(name, data, profile)
                items.append(RecallItem(
                    major_name=name,
                    score=score,
                    source="rule",
                    reasons=reasons,
                    data_support=[
                        f"就业率: {data.get('employment_rate', 0):.1%}",
                        f"均薪: {data.get('avg_salary', 0)}",
                    ],
                ))
        return sorted(items, key=lambda x: x.score, reverse=True)

    def _score(self, name: str, data: dict, profile: UserProfile) -> float:
        score = 0.0

        # 兴趣匹配 (0-40)
        for interest in profile.interests:
            keywords = data.get("keywords", [])
            if any(interest.lower() in kw.lower() for kw in keywords):
                score += 0.15
            if interest.lower() in name.lower():
                score += 0.25

        # 就业率贡献 (0-25)
        score += data.get("employment_rate", 0.5) * 0.25

        # 薪资贡献 (0-15)
        salary_norm = min(data.get("avg_salary", 0) / 15000.0, 1.0)
        score += salary_norm * 0.15

        # 资源适配 (0-10)
        threshold = data.get("resource_threshold", "medium")
        family = profile.family_resources or "普通"
        score += self._resource_compat(threshold, family) * 0.10

        # 性格适配 (0-10)
        if profile.personality:
            personality_fit = data.get("personality_fit", [])
            if any(p.lower() in profile.personality.lower() for p in personality_fit):
                score += 0.10

        return min(score, 1.0)

    def _resource_compat(self, threshold: str, family: str) -> float:
        levels = {"low": 1, "medium": 2, "high": 3}
        family_map = {
            "充裕": 3, "充足": 3, "高": 3,
            "普通": 2, "一般": 2, "中等": 2,
            "不足": 1, "低": 1, "困难": 1,
        }
        t_val = levels.get(threshold, 2)
        f_val = family_map.get(family, 2)
        return 1.0 if f_val >= t_val else 0.3

    def _explain(self, name: str, data: dict, profile: UserProfile) -> List[str]:
        reasons = []
        for interest in profile.interests:
            if interest.lower() in name.lower():
                reasons.append(f"匹配兴趣「{interest}」")
        if data.get("employment_rate", 0) >= 0.95:
            reasons.append("就业率优秀(≥95%)")
        if data.get("avg_salary", 0) >= 10000:
            reasons.append("薪资水平较高(≥1万)")
        return reasons


class KeywordRecall:
    """关键词召回: 基于行业-专业映射和关键词匹配。"""

    INDUSTRY_MAJOR_MAP = {
        "互联网": ["计算机", "软件", "人工智能", "数据", "物联网", "信息安全"],
        "人工智能": ["人工智能", "数据科学", "计算机", "数学", "算法"],
        "半导体/芯片": ["微电子", "集成电路", "电子", "半导体", "光电"],
        "新能源": ["新能源", "电气工程", "能源", "动力", "材料"],
        "医疗": ["临床", "口腔", "护理", "药学", "医学"],
        "金融": ["金融", "会计", "经济", "财务"],
        "制造业": ["机械", "自动化", "材料", "工业"],
        "汽车": ["车辆", "机械", "自动化", "电子"],
        "通信": ["通信", "电子", "信息", "网络"],
        "网络安全": ["信息安全", "网络", "计算机", "软件"],
        "教育": ["师范", "教育"],
        "公务员/体制内": ["公共管理", "法学", "汉语言", "师范"],
        "航空航天": ["航空航天", "机械", "电子", "自动化"],
        "房地产": ["土木", "建筑", "工程管理"],
    }

    def recall(self, profile: UserProfile, all_majors: dict) -> List[RecallItem]:
        items = []
        matched_majors = set()

        # 从兴趣推导行业
        for interest in profile.interests:
            for industry, keywords in self.INDUSTRY_MAJOR_MAP.items():
                for kw in keywords:
                    if kw.lower() in interest.lower():
                        # 找到关联专业
                        for major_name in all_majors:
                            if any(k.lower() in major_name.lower() for k in keywords):
                                if major_name not in matched_majors:
                                    matched_majors.add(major_name)
                                    items.append(RecallItem(
                                        major_name=major_name,
                                        score=0.5,
                                        source="keyword",
                                        reasons=[f"行业「{industry}」关联专业"],
                                        data_support=[f"兴趣关键词: {interest}"],
                                    ))
                        break

        return items


# ──────────────────────────────────────────────
# 重排序器
# ──────────────────────────────────────────────

class ReRanker:
    """多路召回融合 + 重排序。"""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 10,
    ):
        # 召回源权重，默认: 语义 40%, 规则 35%, 关键词 25%
        self._weights = weights or {
            "semantic": 0.40,
            "rule": 0.35,
            "keyword": 0.25,
        }
        self._top_k = top_k

    def rerank(self, all_items: List[RecallItem]) -> List[RecallItem]:
        """融合多路召回结果并重排序。"""
        # 按专业名聚合
        merged: Dict[str, RecallItem] = {}
        for item in all_items:
            if item.major_name not in merged:
                merged[item.major_name] = RecallItem(
                    major_name=item.major_name,
                    score=0.0,
                    source="fused",
                    reasons=[],
                    data_support=[],
                )
            weight = self._weights.get(item.source, 0.25)
            merged[item.major_name].score += item.score * weight
            merged[item.major_name].reasons.extend(item.reasons)
            merged[item.major_name].data_support.extend(item.data_support)

        # 去重原因
        for item in merged.values():
            item.reasons = list(dict.fromkeys(item.reasons))
            item.data_support = list(dict.fromkeys(item.data_support))

        # 排序取 TopK
        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return ranked[:self._top_k]


# ──────────────────────────────────────────────
# 可解释性引擎
# ──────────────────────────────────────────────

class ExplainEngine:
    """为每个结果生成可解释性信息。"""

    def explain(self, item: RecallItem, major_data: dict, profile: UserProfile) -> ExplainInfo:
        match_reason = self._build_match_reason(item, profile)
        data_support = item.data_support or self._default_support(major_data)
        recommend_reason = self._build_recommend_reason(item, major_data, profile)
        risk_warnings = self._build_risk_warnings(major_data)

        return ExplainInfo(
            match_reason=match_reason,
            data_support=data_support,
            recommend_reason=recommend_reason,
            risk_warnings=risk_warnings,
        )

    def _build_match_reason(self, item: RecallItem, profile: UserProfile) -> str:
        reasons = item.reasons
        if not reasons:
            return "综合评估匹配"
        if len(reasons) == 1:
            return reasons[0]
        return f"{'；'.join(reasons[:3])}"

    def _default_support(self, data: dict) -> List[str]:
        support = []
        if data.get("employment_rate"):
            support.append(f"就业率: {data['employment_rate']:.1%}")
        if data.get("avg_salary"):
            support.append(f"平均薪资: {data['avg_salary']}元/月")
        if data.get("courses"):
            support.append(f"核心课程: {', '.join(data['courses'][:5])}")
        return support

    def _build_recommend_reason(self, item: RecallItem, data: dict, profile: UserProfile) -> str:
        parts = []
        if data.get("description"):
            parts.append(data["description"])
        career_paths = data.get("career_paths", [])
        if career_paths:
            parts.append(f"职业方向: {', '.join(career_paths[:3])}")
        return "。".join(parts) if parts else "综合推荐"

    def _build_risk_warnings(self, data: dict) -> List[str]:
        warnings = []
        if data.get("employment_rate", 1.0) < 0.7:
            warnings.append("就业率偏低，需谨慎考虑")
        if data.get("resource_threshold") == "high":
            warnings.append("该专业对家庭资源要求较高")
        return warnings


# ──────────────────────────────────────────────
# DataRetrieverV4
# ──────────────────────────────────────────────

class DataRetrieverV4:
    """多阶段检索 Agent v4。

    流程: Query理解 → 多路召回 → 融合排序 → 可解释输出
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        embedding: Optional[EmbeddingService] = None,
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 10,
    ):
        self.kb = kb
        self.embedding = embedding or EmbeddingService.get_instance()
        self.index = VectorIndex(
            dimension=self.embedding.dimension,
            persist_path=str(
                Path(__file__).resolve().parent.parent / "cache" / "major_vector_index"
            ),
        )
        self._index_built = False

        # 子模块
        self._semantic_recall = SemanticRecall(self.embedding, self.index)
        self._rule_recall = RuleRecall()
        self._keyword_recall = KeywordRecall()
        self._reranker = ReRanker(weights=weights, top_k=top_k)
        self._explain = ExplainEngine()

    def _ensure_index(self):
        """懒加载: 构建专业向量索引。"""
        if self._index_built:
            return

        all_majors = self.kb.all_majors
        texts = []
        metadatas = []

        for name, data in all_majors.items():
            # 构建搜索文本: 名称 + 描述 + 关键词 + 课程
            parts = [name, data.get("description", "")]
            parts.extend(data.get("keywords", []))
            parts.extend(data.get("courses", []))
            parts.extend(data.get("skills_required", []))
            texts.append(" ".join(parts))
            metadatas.append({"name": name, **data})

        if texts:
            vectors = self.embedding.get_embeddings(texts)
            self.index.add_batch(vectors, metadatas)
            self.index.save()

        self._index_built = True
        logger.info(f"Vector index built: {len(texts)} majors indexed")

    async def retrieve(self, profile: UserProfile) -> DataRetrievalResult:
        """多阶段检索主流程。"""
        self._ensure_index()
        all_majors = self.kb.all_majors

        # Stage 1: Query Understanding → 构建查询文本
        query_text = self._build_query_text(profile)

        # Stage 2: Multi-Path Recall
        semantic_items = self._semantic_recall.recall(query_text, top_k=20)
        rule_items = self._rule_recall.recall(profile, all_majors)
        keyword_items = self._keyword_recall.recall(profile, all_majors)

        all_items = semantic_items + rule_items + keyword_items

        # Stage 3: Score Fusion + Re-Ranking
        ranked = self._reranker.rerank(all_items)

        # Stage 4: Build results with explainability
        majors_result = []
        for item in ranked:
            major_data = all_majors.get(item.major_name, {})
            explain = self._explain.explain(item, major_data, profile)

            majors_result.append({
                "name": item.major_name,
                "employment_rate": major_data.get("employment_rate", 0.0),
                "avg_salary": major_data.get("avg_salary", 0),
                "description": major_data.get("description", ""),
                "top_directions": major_data.get("top_directions", []),
                "resource_threshold": major_data.get("resource_threshold", "medium"),
                "match_score": round(item.score, 3),
                # 可解释性字段
                "match_reason": explain.match_reason,
                "data_support": explain.data_support,
                "recommend_reason": explain.recommend_reason,
                "risk_warnings": explain.risk_warnings,
                "career_paths": major_data.get("career_paths", []),
                "industries": major_data.get("industries", []),
            })

        # Related industries
        industries_result = self._find_related_industries(
            [m["name"] for m in majors_result], profile
        )

        filter_reason = self._build_filter_reason(profile, len(majors_result))

        return DataRetrievalResult(
            majors=majors_result,
            industries=industries_result,
            filter_reason=filter_reason,
        )

    def _build_query_text(self, profile: UserProfile) -> str:
        """根据用户画像构建检索查询文本。"""
        parts = []
        if profile.interests:
            parts.append(f"兴趣方向: {', '.join(profile.interests)}")
        if profile.personality:
            parts.append(f"性格特点: {profile.personality}")
        if profile.target_majors:
            parts.append(f"目标专业: {', '.join(profile.target_majors)}")
        parts.append(f"高考分数: {profile.score}分")
        if profile.province:
            parts.append(f"省份: {profile.province}")
        return " ".join(parts)

    def _find_related_industries(self, major_names: List[str], profile: UserProfile) -> List[dict]:
        all_industries = self.kb.all_industries
        matched = {}

        for major_name in major_names:
            for industry_name, keywords in KeywordRecall.INDUSTRY_MAJOR_MAP.items():
                for kw in keywords:
                    if kw.lower() in major_name.lower():
                        if industry_name in all_industries:
                            data = all_industries[industry_name]
                            matched[industry_name] = {
                                "name": industry_name,
                                "entry_barrier": data.get("entry_barrier", "medium"),
                                "salary_range": data.get("salary_range", {}),
                                "description": data.get("description", ""),
                                "top_employers": data.get("top_employers", []),
                            }
                        break
        return list(matched.values())

    def _build_filter_reason(self, profile: UserProfile, count: int) -> str:
        parts = []
        if profile.interests:
            parts.append(f"兴趣匹配: {', '.join(profile.interests)}")
        parts.append(f"分数段: {profile.score}分")
        parts.append(f"检索策略: 语义+规则+关键词 (加权融合)")
        parts.append(f"检索到 {count} 个候选专业")
        return "。".join(parts)
```

### 3.4 DataRetrievalResult 模型增强

```python
# 在 agent_output.py 中增强 DataRetrievalResult

class DataRetrievalResult(BaseModel):
    """数据检索结果 (v4 增强版)。"""

    majors: list[dict] = Field(
        default_factory=list,
        description="专业列表，每项包含: name, employment_rate, avg_salary, description, "
                    "match_score, match_reason, data_support, recommend_reason, risk_warnings",
    )
    industries: list[dict] = Field(
        default_factory=list,
        description="行业列表",
    )
    filter_reason: str = ""
    retrieval_meta: dict = Field(
        default_factory=dict,
        description="检索元信息: strategy, weights, recall_sources",
    )
```

---

## 4. 数据结构升级

### 4.1 majors.json v2

```json
{
  "人工智能": {
    "name": "人工智能",
    "description": "2025就业率榜首，应届生月薪1.2-2.5万，头部院校年薪30万+，但学历门槛高",
    "employment_rate": 0.982,
    "avg_salary": 13800,
    "top_directions": ["算法工程师", "AI产品工程师", "大模型工程师"],
    "resource_threshold": "low",
    "courses": [
      "高等数学", "线性代数", "概率论与数理统计",
      "Python程序设计", "机器学习", "深度学习",
      "自然语言处理", "计算机视觉", "数据结构与算法"
    ],
    "skills_required": [
      "编程能力", "数学建模", "算法设计",
      "数据分析", "论文阅读", "工程实践"
    ],
    "personality_fit": [
      "研究型", "逻辑型", "创新型", "专注型"
    ],
    "career_paths": [
      "AI算法工程师 → 算法专家 → AI技术总监",
      "AI产品经理 → 产品总监",
      "AI研究员 → 首席科学家"
    ],
    "industries": [
      "人工智能", "互联网", "智能制造",
      "金融科技", "自动驾驶", "医疗健康"
    ],
    "keywords": [
      "人工智能", "AI", "机器学习", "深度学习",
      "算法", "大模型", "NLP", "计算机视觉",
      "数据分析", "智能", "神经网络"
    ]
  },

  "计算机科学与技术": {
    "name": "计算机科学与技术",
    "description": "就业率93%，名校+实习的毕业生仍薪资顶尖，但行业分化加剧",
    "employment_rate": 0.933,
    "avg_salary": 11500,
    "top_directions": ["后端开发", "前端开发", "算法工程师"],
    "resource_threshold": "low",
    "courses": [
      "C/C++程序设计", "数据结构", "操作系统",
      "计算机网络", "数据库原理", "编译原理",
      "软件工程", "离散数学", "计算机组成原理"
    ],
    "skills_required": [
      "编程能力", "系统设计", "数据库",
      "网络协议", "团队协作", "持续学习"
    ],
    "personality_fit": [
      "逻辑型", "实践型", "研究型", "独立型"
    ],
    "career_paths": [
      "软件开发工程师 → 技术专家/架构师",
      "技术管理 → 技术总监/CTO",
      "创业 → 技术创始人"
    ],
    "industries": [
      "互联网", "金融", "通信",
      "智能制造", "电子商务", "游戏"
    ],
    "keywords": [
      "计算机", "软件", "编程", "开发",
      "后端", "前端", "全栈", "系统",
      "架构", "数据结构", "算法"
    ]
  },

  "临床医学": {
    "name": "临床医学",
    "description": "学制长(5+3+规培)，但就业稳定，社会地位高，越老越吃香",
    "employment_rate": 0.850,
    "avg_salary": 9500,
    "top_directions": ["三甲医院医师", "基层医院医师", "医学研究"],
    "resource_threshold": "low",
    "courses": [
      "人体解剖学", "生理学", "病理学",
      "药理学", "内科学", "外科学",
      "诊断学", "妇产科学", "儿科学"
    ],
    "skills_required": [
      "记忆力", "动手能力", "沟通能力",
      "抗压能力", "责任心", "持续学习"
    ],
    "personality_fit": [
      "社会型", "研究型", "责任心型", "耐心型"
    ],
    "career_paths": [
      "住院医师 → 主治医师 → 副主任医师 → 主任医师",
      "医学研究 → 医学教授",
      "私立医院/诊所 → 自主创业"
    ],
    "industries": [
      "医疗", "医药", "健康管理",
      "医学教育", "医疗器械"
    ],
    "keywords": [
      "临床", "医学", "医生", "医院",
      "内科", "外科", "诊断", "治疗",
      "规培", "执业医师"
    ]
  }
}
```

### 4.2 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 专业名称（主键） |
| `description` | string | 专业描述 |
| `employment_rate` | float | 就业率 (0-1) |
| `avg_salary` | int | 平均月薪（元） |
| `top_directions` | list[str] | 热门就业方向 |
| `resource_threshold` | string | 资源门槛 (low/medium/high) |
| `courses` | list[str] | 核心课程列表 |
| `skills_required` | list[str] | 所需能力/技能 |
| `personality_fit` | list[str] | 适配性格类型 |
| `career_paths` | list[str] | 职业发展路径 |
| `industries` | list[str] | 相关行业 |
| `keywords` | list[str] | 检索关键词（含同义词） |

---

## 5. 检索流程说明

### 5.1 完整流程图

```
用户画像 (UserProfile)
       │
       ▼
┌──────────────────────────────────────┐
│  Stage 1: Query Understanding         │
│  ──────────────────────────────────  │
│  · 兴趣 → 关键词提取                  │
│  · 分数 → 分数段分类                  │
│  · 性格 → 适配标签                    │
│  · 构建检索查询文本                   │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  Stage 2: Multi-Path Recall           │
│  ──────────────────────────────────  │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │语义召回 │ │规则召回 │ │关键词  │ │
│  │FAISS    │ │业务规则 │ │行业映射│ │
│  │Top20    │ │全量打分 │ │Top15   │ │
│  └────┬────┘ └────┬────┘ └───┬────┘ │
│       └──────┬─────┘         │       │
│              ▼               │       │
│      合并候选集 (~30项)       │       │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────────────┐
│  Stage 3: Score Fusion                │
│  ──────────────────────────────────  │
│  · 按专业名聚合                       │
│  · 加权融合: 语义40% + 规则35% + 关键词25% │
│  · 去重、去偏                         │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  Stage 4: Re-Ranking & Explain        │
│  ──────────────────────────────────  │
│  · 取 Top10                          │
│  · 生成 match_reason                 │
│  · 生成 data_support                 │
│  · 生成 recommend_reason             │
│  · 生成 risk_warnings                │
└──────────────┬───────────────────────┘
               ▼
        DataRetrievalResult
```

### 5.2 各阶段详细说明

**Stage 1: Query Understanding**
- 将 UserProfile 转换为结构化查询文本
- 示例: `"兴趣方向: 人工智能, 编程 性格特点: 研究型 高考分数: 620分 省份: 广东"`

**Stage 2: Multi-Path Recall**

| 召回路径 | 输入 | 输出 | 方法 |
|----------|------|------|------|
| 语义召回 | 查询文本 → embedding | Top20 专业 | FAISS 向量相似度 |
| 规则召回 | UserProfile + 全量专业 | 全量打分排序 | 业务规则加权评分 |
| 关键词召回 | 兴趣关键词 + 行业映射 | Top15 专业 | 字符串匹配 + 行业映射 |

**Stage 3: Score Fusion**
```
final_score = semantic_score * 0.40 + rule_score * 0.35 + keyword_score * 0.25
```
- 权重可通过配置调整
- 仅出现在单路的结果不会被惩罚

**Stage 4: Re-Ranking & Explain**
- 融合后的结果按分数降序取 Top10
- 为每个结果生成可解释性信息

### 5.3 配置示例

```yaml
# config/retrieval_config.yaml
retrieval:
  embedding:
    strategy: "local"          # hash | local | api
    local_model: "BAAI/bge-small-zh"
    api_base: "https://api.openai.com/v1"
    api_model: "text-embedding-3-small"
    cache_max_size: 10000

  vector_index:
    engine: "auto"             # faiss | numpy | auto
    persist_path: "cache/major_vector_index"

  recall:
    semantic_top_k: 20
    keyword_top_k: 15

  fusion:
    weights:
      semantic: 0.40
      rule: 0.35
      keyword: 0.25

  output:
    top_k: 10
    include_explainability: true
```

---

## 6. 可解释性设计

### 6.1 输出结构

每个推荐专业包含以下可解释性字段:

```json
{
  "name": "人工智能",
  "match_score": 0.892,
  "match_reason": "语义相似度: 0.891；匹配兴趣「人工智能」；就业率优秀(≥95%)",
  "data_support": [
    "就业率: 98.2%",
    "平均薪资: 13800元/月",
    "核心课程: 高等数学, 线性代数, 概率论与数理统计"
  ],
  "recommend_reason": "2025就业率榜首，应届生月薪1.2-2.5万。职业方向: AI算法工程师 → 算法专家 → AI技术总监",
  "risk_warnings": ["学历门槛高，建议读研"]
}
```

### 6.2 可解释性来源

| 字段 | 生成逻辑 |
|------|----------|
| `match_reason` | 合并多路召回的匹配原因 |
| `data_support` | 就业率、薪资、课程等客观数据 |
| `recommend_reason` | 专业描述 + 职业路径 |
| `risk_warnings` | 就业率 < 70%、高资源门槛等风险提示 |

---

## 7. 性能优化方案

### 7.1 LRU Embedding Cache

```
┌──────────────────────────────────────────┐
│           EmbeddingCache (LRU)           │
│  ──────────────────────────────────────  │
│  · max_size: 10,000                      │
│  · 内存: OrderedDict                     │
│  · 持久化: JSON 文件                     │
│  · 淘汰策略: LRU (Least Recently Used)   │
│  · 命中时: move_to_end()                 │
└──────────────────────────────────────────┘
```

- 首次请求: 调用 embedding 策略 → 写入缓存 → 返回
- 缓存命中: 直接从内存返回，零延迟
- 缓存满: 淘汰最久未使用的条目
- 服务关闭: 持久化到 JSON 文件

### 7.2 Query TTL Cache

```python
class QueryCache:
    """查询结果 TTL 缓存。"""

    def __init__(self, ttl: int = 300, max_size: int = 1000):
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._ttl = ttl
        self._max_size = max_size

    def _cache_key(self, profile: UserProfile) -> str:
        raw = json.dumps(profile.model_dump(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, profile: UserProfile) -> Optional[DataRetrievalResult]:
        key = self._cache_key(profile)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                self._cache.move_to_end(key)
                return result
            del self._cache[key]
        return None

    def put(self, profile: UserProfile, result: DataRetrievalResult):
        key = self._cache_key(profile)
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (result, time.time())
```

### 7.3 性能指标预估

| 指标 | V3 | V4 | 改进 |
|------|-----|-----|------|
| Embedding 计算 | 每次查询全量重算 | LRU 缓存，命中率 > 80% | **~10x** |
| 相似度搜索 | O(n) 逐条计算 | FAISS O(1) 索引查找 | **~100x** (n>100) |
| 首次检索延迟 | ~200ms | ~500ms (含模型加载) | - |
| 缓存命中延迟 | ~50ms | ~5ms | **~10x** |
| 内存占用 | ~5MB | ~50MB (含 FAISS 索引) | - |

---

## 8. 新旧版本对比

### 8.1 架构对比

| 维度 | V3 (当前) | V4 (升级后) |
|------|-----------|-------------|
| **Embedding** | SHA-256 伪向量，无真实语义 | 策略模式: Hash/Local/API 三选一 |
| **向量索引** | 无索引，逐条计算相似度 | FAISS / NumPy 双引擎，TopK 检索 |
| **检索策略** | 语义(60%) + 规则(40%)，硬编码权重 | 3路召回，可配置权重融合 |
| **召回路径** | 2条 (语义 + 规则) | 3条 (语义 + 规则 + 关键词) |
| **数据模型** | 5字段 (就业率/薪资/方向/门槛/描述) | 12字段 (+课程/技能/性格/职业/行业/关键词) |
| **可解释性** | 仅 filter_reason | match_reason + data_support + recommend_reason + risk_warnings |
| **缓存机制** | TTL 内存缓存 (KB 级别) | LRU Embedding 缓存 + Query TTL 缓存 + 持久化 |
| **配置管理** | 硬编码 | YAML 配置 + 环境变量 |
| **可扩展性** | 低 (硬编码) | 高 (策略模式 + 插件化召回) |

### 8.2 检索精度对比

| 场景 | V3 表现 | V4 预期 |
|------|---------|---------|
| "我想学编程" | 仅匹配含"编程"字符串的专业 | 语义理解 "编程" → 计算机/软件/人工智能 |
| "我性格内向适合什么" | 无法处理 | 匹配 personality_fit 字段 |
| "未来想进互联网行业" | 硬编码映射，覆盖有限 | 关键词召回 + 语义召回双路覆盖 |
| "620分广东考生" | 分数过滤后按兴趣排序 | 分数段适配 + 多路召回融合 |

### 8.3 代码对比

```
                    V3                          V4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
embedding_service   105 lines, 1 class          200+ lines, 5 classes
                    1 策略                       3 策略 + 策略模式
                    无缓存                      LRU 缓存 + 持久化

data_retriever      304 lines, 1 class          300+ lines, 7 classes
                    2 阶段                       4 阶段
                    硬编码权重                   可配置权重
                    无解释                       完整可解释性

knowledge_base      192 lines                   增强版 (兼容旧 API)
                    5 字段                      12 字段
                    字符串匹配                   语义 + 关键词双匹配
```

---

## 9. 实施路线图

### Phase 1: 基础设施 (1-2 天)

- [ ] 实现 EmbeddingService 策略模式
  - [ ] EmbeddingStrategy 抽象基类
  - [ ] HashEmbeddingStrategy (保留现有逻辑)
  - [ ] 本地模型 / API 策略骨架
- [ ] 实现 EmbeddingCache (LRU + 持久化)
- [ ] 实现 VectorIndex (NumPy 先行，FAISS 后续)
- [ ] 更新 config.py 添加 embedding 配置项

### Phase 2: 数据升级 (1 天)

- [ ] 设计 majors.json v2 数据结构
- [ ] 编写数据迁移脚本
- [ ] 更新 KnowledgeBase 兼容新旧数据格式
- [ ] 补充 personality_fit, courses, skills_required 等字段

### Phase 3: 检索引擎 (2-3 天)

- [ ] 实现 SemanticRecall
- [ ] 实现 RuleRecall (重构现有打分逻辑)
- [ ] 实现 KeywordRecall
- [ ] 实现 ReRanker (加权融合)
- [ ] 实现 ExplainEngine
- [ ] 组装 DataRetrieverV4

### Phase 4: 集成与测试 (2 天)

- [ ] 更新 Orchestrator 使用 DataRetrieverV4
- [ ] 编写集成测试
- [ ] 性能基准测试
- [ ] 灰度发布 (A/B 对比)

### Phase 5: 优化与 FAISS (1 天)

- [ ] 集成 FAISS 引擎
- [ ] 添加 QueryCache
- [ ] 监控与调优

---

## 附录 A: 环境变量配置

```bash
# Embedding 策略选择: hash | local | api
EMBEDDING_STRATEGY=local

# 本地模型配置 (strategy=local)
EMBEDDING_LOCAL_MODEL=BAAI/bge-small-zh

# API 配置 (strategy=api)
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_API_BASE=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

# 向量索引引擎: faiss | numpy | auto
VECTOR_INDEX_ENGINE=auto

# 缓存配置
EMBEDDING_CACHE_MAX_SIZE=10000
EMBEDDING_CACHE_PATH=backend/cache/embeddings_cache.json

# 检索权重
RETRIEVAL_WEIGHT_SEMANTIC=0.40
RETRIEVAL_WEIGHT_RULE=0.35
RETRIEVAL_WEIGHT_KEYWORD=0.25
```

## 附录 B: 依赖清单

```txt
# 核心依赖 (已有)
pydantic>=2.0
pydantic-settings>=2.0

# Embedding 策略 (新增，按需安装)
sentence-transformers>=2.2.0    # LocalEmbeddingStrategy
openai>=1.0                     # APIEmbeddingStrategy
faiss-cpu>=1.7.0                # FAISS 向量索引
numpy>=1.24.0                   # NumPy 回退引擎

# 可选
PyYAML>=6.0                     # 配置文件解析
```

## 附录 C: 兼容性说明

1. **向后兼容**: DataRetrieverV4 完全兼容 DataRetrieverV3 的接口签名
2. **降级策略**: EmbeddingService 默认使用 HashEmbedding，无需安装额外依赖
3. **数据迁移**: majors.json v2 向后兼容 v1 字段，旧字段全部保留
4. **渐进迁移**: 可通过环境变量切换 V3/V4，支持灰度发布
