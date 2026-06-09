"""EmbeddingService: 策略模式实现多后端向量生成。"""

import os
import math
import time
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
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
            vector = [v / norm for v in vector]
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
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = embedding

    def save(self):
        if self._persist_path:
            Path(self._persist_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(dict(self._cache), f, ensure_ascii=False)

    @property
    def size(self) -> int:
        return len(self._cache)


# ──────────────────────────────────────────────
# EmbeddingService 门面类
# ─────────────────────────────────────────────

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

    @classmethod
    def reset_instance(cls):
        """重置单例（用于测试）。"""
        cls._instance = None

    @staticmethod
    def _auto_select_strategy() -> EmbeddingStrategy:
        """根据环境变量自动选择策略。"""
        strategy_type = os.getenv("EMBEDDING_STRATEGY", "hash").lower()

        if strategy_type == "api":
            api_key = os.getenv("EMBEDDING_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning("EMBEDDING_API_KEY not set, falling back to hash")
                return HashEmbeddingStrategy()
            return APIEmbeddingStrategy(
                api_key=api_key,
                api_base=os.getenv("EMBEDDING_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")),
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
        """批量获取向量，自动处理缓存命中和未命中。"""
        if not texts:
            return []

        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            embeddings = self._strategy.encode_batch(uncached_texts)
            for i, embedding in zip(uncached_indices, embeddings):
                results[i] = embedding
                self._cache.put(texts[i], embedding)

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
