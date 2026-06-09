"""Phase 1: EmbeddingService 策略模式 + Cache + VectorIndex 测试。"""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.embedding_service import (
    EmbeddingService,
    EmbeddingCache,
    HashEmbeddingStrategy,
)
from backend.services.vector_index import VectorIndex


# ──────────────────────────────────────────────
# HashEmbeddingStrategy 测试
# ────────────────────────────────────────────

class TestHashEmbeddingStrategy:
    """Hash 向量策略测试。"""

    def setup_method(self):
        self.strategy = HashEmbeddingStrategy()

    def test_dimension(self):
        assert self.strategy.dimension == 384

    def test_empty_text(self):
        result = self.strategy.encode("")
        assert len(result) == 384
        assert all(v == 0.0 for v in result)

    def test_deterministic(self):
        """相同文本应生成相同向量。"""
        text = "计算机科学与技术"
        v1 = self.strategy.encode(text)
        v2 = self.strategy.encode(text)
        assert v1 == v2

    def test_different_texts_produce_different_vectors(self):
        v1 = self.strategy.encode("计算机")
        v2 = self.strategy.encode("金融学")
        assert v1 != v2

    def test_normalized(self):
        """向量应归一化。"""
        text = "人工智能"
        v = self.strategy.encode(text)
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-5

    def test_batch_encode(self):
        texts = ["计算机", "金融学", "医学"]
        results = self.strategy.encode_batch(texts)
        assert len(results) == 3
        assert all(len(v) == 384 for v in results)


# ──────────────────────────────────────────────
# EmbeddingService 测试
# ──────────────────────────────────────────────

class TestEmbeddingService:
    """EmbeddingService 门面测试。"""

    def setup_method(self):
        # 使用 Hash 策略进行单元测试（无需外部依赖）
        self.service = EmbeddingService(
            strategy=HashEmbeddingStrategy(),
            cache_max_size=100,
        )

    def test_get_embedding(self):
        v = self.service.get_embedding("计算机")
        assert len(v) == 384

    def test_get_embeddings(self):
        texts = ["计算机", "金融学", "医学"]
        results = self.service.get_embeddings(texts)
        assert len(results) == 3
        assert all(len(v) == 384 for v in results)

    def test_cache_hit(self):
        """测试缓存命中。"""
        v1 = self.service.get_embedding("计算机")
        v2 = self.service.get_embedding("计算机")
        assert v1 == v2

    def test_cosine_similarity_same(self):
        """相同向量余弦相似度应为 1。"""
        v = self.service.get_embedding("测试")
        sim = self.service.cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_different(self):
        v1 = self.service.get_embedding("计算机")
        v2 = self.service.get_embedding("金融学")
        sim = self.service.cosine_similarity(v1, v2)
        assert -1.0 <= sim <= 1.0

    def test_cosine_similarity_empty(self):
        assert self.service.cosine_similarity([], []) == 0.0

    def test_dimension_property(self):
        assert self.service.dimension == 384

    def test_strategy_name(self):
        assert self.service.strategy_name == "HashEmbeddingStrategy"

    def test_batch_partial_cache(self):
        """测试部分缓存命中的批量编码。"""
        # 先编码一个
        self.service.get_embedding("计算机")
        # 批量编码（含已缓存和未缓存）
        results = self.service.get_embeddings(["计算机", "金融学"])
        assert len(results) == 2
        assert all(len(v) == 384 for v in results)


# ──────────────────────────────────────────────
# EmbeddingCache 测试
# ──────────────────────────────────────────────

class TestEmbeddingCache:
    """LRU 缓存测试。"""

    def test_basic_put_get(self):
        cache = EmbeddingCache(max_size=10)
        cache.put("hello", [0.1, 0.2, 0.3])
        result = cache.get("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_cache_miss(self):
        cache = EmbeddingCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = EmbeddingCache(max_size=2)
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        cache.put("c", [3.0])  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == [2.0]
        assert cache.get("c") == [3.0]

    def test_lru_access_refreshes(self):
        cache = EmbeddingCache(max_size=2)
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        cache.get("a")  # refresh "a"
        cache.put("c", [3.0])  # should evict "b"
        assert cache.get("a") == [1.0]
        assert cache.get("b") is None
        assert cache.get("c") == [3.0]

    def test_update_existing(self):
        cache = EmbeddingCache(max_size=10)
        cache.put("key", [1.0])
        cache.put("key", [2.0])
        assert cache.get("key") == [2.0]
        assert cache.size == 1


# ──────────────────────────────────────────────
# VectorIndex 测试 (NumPy engine)
# ──────────────────────────────────────────────

class TestVectorIndex:
    """向量索引测试。"""

    def setup_method(self):
        self.index = VectorIndex(dimension=4, engine="numpy")

    def test_add_and_search(self):
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        metadatas = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        self.index.add_batch(vectors, metadatas)

        results = self.index.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # First should be index 0 (exact match)
        assert results[0][0] == 0

    def test_empty_index_search(self):
        results = self.index.search([1.0, 0.0], top_k=5)
        assert results == []

    def test_add_single(self):
        self.index.add([1.0, 0.0, 0.0, 0.0], {"name": "test"})
        assert self.index.count == 1

    def test_get_metadata(self):
        self.index.add([1.0, 0.0, 0.0, 0.0], {"name": "test"})
        meta = self.index.get_metadata(0)
        assert meta["name"] == "test"

    def test_update(self):
        self.index.add([1.0, 0.0, 0.0, 0.0], {"name": "old"})
        self.index.update(0, [0.0, 1.0, 0.0, 0.0], {"name": "new"})
        meta = self.index.get_metadata(0)
        assert meta["name"] == "new"

    def test_out_of_range_metadata(self):
        assert self.index.get_metadata(999) == {}

    def test_persist_and_load(self, tmp_path):
        persist_path = str(tmp_path / "test_index")
        idx = VectorIndex(dimension=4, engine="numpy", persist_path=persist_path)
        idx.add([1.0, 0.0, 0.0, 0.0], {"name": "saved"})
        idx.save()

        # Load into new instance
        idx2 = VectorIndex(dimension=4, engine="numpy", persist_path=persist_path)
        idx2._ensure_loaded()
        assert idx2.count == 1
        assert idx2.get_metadata(0)["name"] == "saved"

    def test_engine_property(self):
        assert self.index.engine == "numpy"

    def test_dimension_property(self):
        assert self.index.dimension == 4
