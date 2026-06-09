# 向量知识库 + 智能推荐系统 Implementation Plan - Phase 1

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基础向量知识库模块，支持结构化数据的文档化、向量化和语义搜索。

**Architecture:** 新增 VectorKnowledgeBase 类，复用现有 EmbeddingService，为三类新数据源（高校名单、高考分数线、学科评估）创建 JSON 数据文件并实现文档化转换。

**Tech Stack:** Python, Pydantic, 现有 EmbeddingService (hash-based 384-dim), JSON, pytest

---

## File Structure

```
backend/services/
├── vector_knowledge_base.py    # 新增: VectorKnowledgeBase 核心类
├── embedding_service.py        # 已有，复用
└── knowledge_base.py           # 已有，不变

backend/data/
├── universities_list.json      # 新增: 完整高校名单（示例数据）
├── gaokao_scores.json          # 新增: 高考分数线（示例数据）
└── subject_review.json         # 新增: 学科评估（示例数据）

backend/cache/                   # 新增: 缓存目录
└── embeddings_cache.json       # Embedding 持久化缓存

tests/
└── test_vector_knowledge_base.py  # 新增: 单元测试
```

---

### Task 1: 创建数据模型 (VectorDocument + Pydantic Models)

**Files:**
- Create: `backend/services/vector_knowledge_base.py`

- [ ] **Step 1: 创建数据模型和 VectorKnowledgeBase 骨架**

```python
"""向量知识库模块：支持语义搜索的文档存储和检索。"""

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


# 向量知识库的类别枚举
CATEGORY_UNIVERSITY_BASIC = "university_basic"
CATEGORY_SCORE_BATCH = "score_batch"
CATEGORY_SCORE_SCHOOL = "score_school"
CATEGORY_SCORE_MAJOR = "score_major"
CATEGORY_SUBJECT_EVAL = "subject_eval"


@dataclass
class VectorDocument:
    """向量文档模型"""
    id: str                           # 唯一标识，格式: "category:base_id:variant_type"
    category: str                     # 数据类别
    base_id: str                      # 原始数据ID
    variant_type: str                 # 变体类型: basic/location_based/score_based/major_based/comparison
    text: str                         # 文档化文本
    embedding: list[float] = field(default_factory=list)  # 向量表示
    metadata: dict = field(default_factory=dict)          # 原始结构化数据

    def to_dict(self) -> dict:
        """序列化"""
        return {
            "id": self.id,
            "category": self.category,
            "base_id": self.base_id,
            "variant_type": self.variant_type,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VectorDocument":
        """反序列化"""
        return cls(
            id=data["id"],
            category=data["category"],
            base_id=data["base_id"],
            variant_type=data["variant_type"],
            text=data["text"],
            embedding=data.get("embedding", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SearchResult:
    """搜索结果"""
    document: VectorDocument
    score: float                    # 相似度得分
    rank: int = 0                   # 排序名次


@dataclass
class RiskClassificationResult:
    """冲稳保分类结果"""
    charge: list                    # 冲刺院校列表
    stable: list                    # 稳妥院校列表
    safe: list                      # 保底院校列表
```

- [ ] **Step 2: 创建 EmbeddingCache 类**

在 `backend/services/vector_knowledge_base.py` 末尾追加：

```python
class EmbeddingCache:
    """Embedding 持久化缓存"""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "embeddings_cache.json"
        self._cache: dict[str, list[float]] = {}
        self._load()

    def _load(self):
        """从文件加载缓存"""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save(self):
        """保存缓存到文件"""
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)
        except IOError:
            pass

    def _hash_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        """获取缓存的 embedding"""
        key = self._hash_key(text)
        return self._cache.get(key)

    def set(self, text: str, embedding: list[float]):
        """缓存 embedding"""
        key = self._hash_key(text)
        self._cache[key] = embedding

    def save(self):
        """持久化缓存"""
        self._save()

    def size(self) -> int:
        """返回缓存条目数"""
        return len(self._cache)
```

- [ ] **Step 3: 创建 cache 目录占位文件**

Create: `backend/cache/.gitkeep`

```
# This directory stores embedding caches
```

---

### Task 2: 实现 VectorKnowledgeBase 核心方法

**Files:**
- Modify: `backend/services/vector_knowledge_base.py` (追加到文件末尾)
- Test: `tests/test_vector_knowledge_base.py`

- [ ] **Step 1: 写测试 - 文档增删查**

Create: `tests/test_vector_knowledge_base.py`

```python
"""向量知识库模块测试。"""

import pytest
from pathlib import Path
import sys
import tempfile
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.vector_knowledge_base import (
    VectorDocument,
    SearchResult,
    EmbeddingCache,
)


class TestVectorDocument:
    """VectorDocument 模型测试"""

    def test_create_document(self):
        doc = VectorDocument(
            id="university_basic:清华:basic",
            category="university_basic",
            base_id="清华",
            variant_type="basic",
            text="清华大学位于北京，是一所985高校。",
            metadata={"name": "清华大学", "province": "北京"},
        )
        assert doc.id == "university_basic:清华:basic"
        assert doc.category == "university_basic"
        assert "北京" in doc.text

    def test_to_dict_and_from_dict(self):
        doc = VectorDocument(
            id="test:001:basic",
            category="test",
            base_id="001",
            variant_type="basic",
            text="测试文本",
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
        )
        d = doc.to_dict()
        restored = VectorDocument.from_dict(d)
        assert restored.id == doc.id
        assert restored.text == doc.text
        assert restored.embedding == doc.embedding
        assert restored.metadata == doc.metadata
```

- [ ] **Step 2: 运行测试验证通过**

Run: `pytest tests/test_vector_knowledge_base.py::TestVectorDocument -v`

Expected: All tests PASS

- [ ] **Step 3: 写测试 - EmbeddingCache**

在 `tests/test_vector_knowledge_base.py` 末尾追加：

```python
class TestEmbeddingCache:
    """EmbeddingCache 测试"""

    def test_cache_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))
            cache.set("hello world", [0.1, 0.2, 0.3])
            result = cache.get("hello world")
            assert result == [0.1, 0.2, 0.3]

    def test_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))
            result = cache.get("nonexistent")
            assert result is None

    def test_cache_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))
            cache.set("persist_test", [1.0, 2.0])
            cache.save()

            # 新实例应能加载缓存
            cache2 = EmbeddingCache(cache_dir=Path(tmpdir))
            result = cache2.get("persist_test")
            assert result == [1.0, 2.0]

    def test_cache_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))
            assert cache.size() == 0
            cache.set("a", [1.0])
            cache.set("b", [2.0])
            assert cache.size() == 2
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_vector_knowledge_base.py::TestEmbeddingCache -v`

Expected: All tests PASS

- [ ] **Step 5: 写 VectorKnowledgeBase 核心类**

在 `backend/services/vector_knowledge_base.py` 末尾追加：

```python
class VectorKnowledgeBase:
    """向量知识库，支持语义搜索"""

    def __init__(self, data_dir: Path = DATA_DIR, cache_dir: Path = CACHE_DIR):
        self._data_dir = data_dir
        self._cache = EmbeddingCache(cache_dir=cache_dir)
        self._documents: list[VectorDocument] = []
        self._embedding_service = None  # lazy init

    def _get_embedding_service(self):
        """延迟初始化 EmbeddingService"""
        if self._embedding_service is None:
            from backend.services.embedding_service import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    # -- 文档管理 --

    def add_document(self, doc: VectorDocument) -> None:
        """添加单个文档"""
        self._documents.append(doc)

    def add_documents(self, docs: list[VectorDocument]) -> None:
        """批量添加文档"""
        self._documents.extend(docs)

    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """根据 ID 获取文档"""
        for doc in self._documents:
            if doc.id == doc_id:
                return doc
        return None

    def get_documents_by_category(self, category: str) -> list[VectorDocument]:
        """获取指定类别的所有文档"""
        return [d for d in self._documents if d.category == category]

    @property
    def document_count(self) -> int:
        """文档总数"""
        return len(self._documents)

    @property
    def categories(self) -> list[str]:
        """所有类别"""
        return list(set(d.category for d in self._documents))

    # -- 向量化 --

    def embed_document(self, doc: VectorDocument) -> VectorDocument:
        """为单个文档计算 embedding"""
        embedding = self._get_embedding_service().get_embedding(doc.text)
        doc.embedding = embedding
        # 缓存
        self._cache.set(doc.text, embedding)
        return doc

    def embed_all(self) -> None:
        """为所有未向量化的文档计算 embedding"""
        for doc in self._documents:
            if not doc.embedding:
                # 先查缓存
                cached = self._cache.get(doc.text)
                if cached:
                    doc.embedding = cached
                else:
                    self.embed_document(doc)
        self._cache.save()

    # -- 搜索 --

    def semantic_search(
        self,
        query: str,
        category: str = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        语义搜索

        Args:
            query: 查询文本
            category: 可选，限定数据类别
            top_k: 返回结果数量

        Returns:
            按相似度降序的搜索结果列表
        """
        if not query or not self._documents:
            return []

        # 过滤类别
        candidates = self._documents
        if category:
            candidates = [d for d in candidates if d.category == category]

        if not candidates:
            return []

        # 计算查询向量
        cached = self._cache.get(query)
        if cached:
            query_embedding = cached
        else:
            query_embedding = self._get_embedding_service().get_embedding(query)
            self._cache.set(query, query_embedding)

        # 计算相似度
        embedding_svc = self._get_embedding_service()
        scored = []
        for doc in candidates:
            if doc.embedding:
                score = embedding_svc.cosine_similarity(query_embedding, doc.embedding)
                scored.append(SearchResult(document=doc, score=score))

        # 排序
        scored.sort(key=lambda x: x.score, reverse=True)
        scored = scored[:top_k]

        # 设置排名
        for i, result in enumerate(scored):
            result.rank = i + 1

        return scored

    # -- 持久化 --

    def save_index(self, filepath: str) -> None:
        """保存知识库索引到文件"""
        data = {
            "documents": [doc.to_dict() for doc in self._documents],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_index(self, filepath: str) -> None:
        """从文件加载知识库索引"""
        path = Path(filepath)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._documents = [
            VectorDocument.from_dict(d) for d in data.get("documents", [])
        ]

    def clear(self) -> None:
        """清空知识库"""
        self._documents.clear()
```

- [ ] **Step 6: 写 VectorKnowledgeBase 测试**

在 `tests/test_vector_knowledge_base.py` 末尾追加：

```python
class TestVectorKnowledgeBase:
    """VectorKnowledgeBase 核心功能测试"""

    def _create_test_kb(self):
        """创建带测试数据的 VectorKnowledgeBase"""
        from backend.services.vector_knowledge_base import VectorKnowledgeBase

        kb = VectorKnowledgeBase()
        kb.add_documents([
            VectorDocument(
                id="test:tsinghua:basic",
                category="university_basic",
                base_id="tsinghua",
                variant_type="basic",
                text="清华大学位于北京，是一所985高校，工科强势。",
                metadata={"name": "清华大学", "province": "北京", "tier": "985"},
            ),
            VectorDocument(
                id="test:pku:basic",
                category="university_basic",
                base_id="pku",
                variant_type="basic",
                text="北京大学位于北京，是一所985高校，文科理科都很强。",
                metadata={"name": "北京大学", "province": "北京", "tier": "985"},
            ),
            VectorDocument(
                id="test:zju:basic",
                category="university_basic",
                base_id="zju",
                variant_type="basic",
                text="浙江大学位于浙江杭州，是一所985高校，工科和计算机强。",
                metadata={"name": "浙江大学", "province": "浙江", "tier": "985"},
            ),
        ])
        kb.embed_all()
        return kb

    def test_add_and_count(self):
        kb = self._create_test_kb()
        assert kb.document_count == 3

    def test_get_document(self):
        kb = self._create_test_kb()
        doc = kb.get_document("test:tsinghua:basic")
        assert doc is not None
        assert doc.metadata["name"] == "清华大学"

    def test_get_document_not_found(self):
        kb = self._create_test_kb()
        doc = kb.get_document("nonexistent")
        assert doc is None

    def test_get_documents_by_category(self):
        kb = self._create_test_kb()
        docs = kb.get_documents_by_category("university_basic")
        assert len(docs) == 3

        docs = kb.get_documents_by_category("nonexistent")
        assert len(docs) == 0

    def test_categories(self):
        kb = self._create_test_kb()
        assert "university_basic" in kb.categories

    def test_embed_all(self):
        kb = VectorKnowledgeBase()
        kb.add_document(VectorDocument(
            id="test:1:basic",
            category="test",
            base_id="1",
            variant_type="basic",
            text="测试embedding",
        ))
        assert kb._documents[0].embedding == []
        kb.embed_all()
        assert len(kb._documents[0].embedding) == 384  # EmbeddingService.DIMENSION

    def test_semantic_search_basic(self):
        kb = self._create_test_kb()
        results = kb.semantic_search("北京的大学", top_k=2)
        assert len(results) == 2
        assert results[0].rank == 1
        # 清华和北大都在北京，应该排在前面
        assert "北京" in results[0].document.metadata.get("province", "")

    def test_semantic_search_with_category(self):
        kb = self._create_test_kb()
        results = kb.semantic_search("大学", category="university_basic", top_k=5)
        assert len(results) <= 3  # 最多3条

        results = kb.semantic_search("大学", category="nonexistent", top_k=5)
        assert len(results) == 0

    def test_semantic_search_empty_query(self):
        kb = self._create_test_kb()
        results = kb.semantic_search("", top_k=5)
        assert len(results) == 0

    def test_save_and_load_index(self):
        import tempfile
        kb = self._create_test_kb()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            kb.save_index(f.name)
            kb2 = VectorKnowledgeBase()
            kb2.load_index(f.name)
            assert kb2.document_count == kb.document_count
            Path(f.name).unlink()

    def test_clear(self):
        kb = self._create_test_kb()
        assert kb.document_count == 3
        kb.clear()
        assert kb.document_count == 0
```

- [ ] **Step 7: 运行全部测试验证通过**

Run: `pytest tests/test_vector_knowledge_base.py -v`

Expected: All tests PASS

---

### Task 3: 创建示例数据文件

**Files:**
- Create: `backend/data/universities_list.json`
- Create: `backend/data/gaokao_scores.json`
- Create: `backend/data/subject_review.json`

- [ ] **Step 1: 创建高校名单数据**

Create: `backend/data/universities_list.json`

```json
{
  "4111010003": {
    "school_name": "清华大学",
    "province": "北京",
    "competent_department": "教育部",
    "location": "北京",
    "level": "本科",
    "is_private": false,
    "tier": "985"
  },
  "411101001": {
    "school_name": "北京大学",
    "province": "北京",
    "competent_department": "教育部",
    "location": "北京",
    "level": "本科",
    "is_private": false,
    "tier": "985"
  },
  "4133010335": {
    "school_name": "浙江大学",
    "province": "浙江",
    "competent_department": "教育部",
    "location": "杭州",
    "level": "本科",
    "is_private": false,
    "tier": "985"
  },
  "4131010248": {
    "school_name": "上海交通大学",
    "province": "上海",
    "competent_department": "教育部",
    "location": "上海",
    "level": "本科",
    "is_private": false,
    "tier": "985"
  },
  "4134010357": {
    "school_name": "合肥工业大学",
    "province": "安徽",
    "competent_department": "教育部",
    "location": "合肥",
    "level": "本科",
    "is_private": false,
    "tier": "211"
  },
  "4134010358": {
    "school_name": "中国科学技术大学",
    "province": "安徽",
    "competent_department": "中国科学院",
    "location": "合肥",
    "level": "本科",
    "is_private": false,
    "tier": "985"
  },
  "4134010359": {
    "school_name": "安徽大学",
    "province": "安徽",
    "competent_department": "安徽省",
    "location": "合肥",
    "level": "本科",
    "is_private": false,
    "tier": "211"
  },
  "4134010360": {
    "school_name": "安徽工业大学",
    "province": "安徽",
    "competent_department": "安徽省",
    "location": "马鞍山",
    "level": "本科",
    "is_private": false,
    "tier": ""
  }
}
```

- [ ] **Step 2: 创建高考分数线数据**

Create: `backend/data/gaokao_scores.json`

```json
{
  "batch_scores": {
    "安徽_2025_本科_一批_理科": {
      "province": "安徽",
      "year": 2025,
      "category": "本科",
      "batch": "一批",
      "subject_type": "理科",
      "score_line": 515
    },
    "安徽_2025_本科_二批_理科": {
      "province": "安徽",
      "year": 2025,
      "category": "本科",
      "batch": "二批",
      "subject_type": "理科",
      "score_line": 450
    }
  },
  "school_scores": {
    "安徽_2025_理科_清华大学": {
      "province": "安徽",
      "year": 2025,
      "subject_type": "理科",
      "school_name": "清华大学",
      "min_score": 680,
      "avg_score": 690,
      "min_rank": 50,
      "admission_count": 15,
      "provincial_line": 515,
      "line_diff": 165
    },
    "安徽_2025_理科_合肥工业大学": {
      "province": "安徽",
      "year": 2025,
      "subject_type": "理科",
      "school_name": "合肥工业大学",
      "min_score": 572,
      "avg_score": 578,
      "min_rank": 18000,
      "admission_count": 800,
      "provincial_line": 515,
      "line_diff": 57
    },
    "安徽_2025_理科_安徽大学": {
      "province": "安徽",
      "year": 2025,
      "subject_type": "理科",
      "school_name": "安徽大学",
      "min_score": 555,
      "avg_score": 560,
      "min_rank": 30000,
      "admission_count": 1200,
      "provincial_line": 515,
      "line_diff": 40
    },
    "安徽_2025_理科_安徽工业大学": {
      "province": "安徽",
      "year": 2025,
      "subject_type": "理科",
      "school_name": "安徽工业大学",
      "min_score": 535,
      "avg_score": 540,
      "min_rank": 45000,
      "admission_count": 1500,
      "provincial_line": 515,
      "line_diff": 20
    }
  },
  "major_scores": {}
}
```

- [ ] **Step 3: 创建学科评估数据**

Create: `backend/data/subject_review.json`

```json
{
  "4111010003_0812_4": {
    "round": 4,
    "year": 2016,
    "category_name": "工学",
    "subject_code": "0812",
    "subject_name": "计算机科学与技术",
    "school_code": "10003",
    "school_name": "清华大学",
    "rank": 1,
    "grade": "A+"
  },
  "411101001_0812_4": {
    "round": 4,
    "year": 2016,
    "category_name": "工学",
    "subject_code": "0812",
    "subject_name": "计算机科学与技术",
    "school_code": "10001",
    "school_name": "北京大学",
    "rank": 2,
    "grade": "A+"
  },
  "4133010335_0812_4": {
    "round": 4,
    "year": 2016,
    "category_name": "工学",
    "subject_code": "0812",
    "subject_name": "计算机科学与技术",
    "school_code": "10335",
    "school_name": "浙江大学",
    "rank": 3,
    "grade": "A+"
  },
  "4134010357_0812_4": {
    "round": 4,
    "year": 2016,
    "category_name": "工学",
    "subject_code": "0812",
    "subject_name": "计算机科学与技术",
    "school_code": "10359",
    "school_name": "合肥工业大学",
    "rank": 30,
    "grade": "B+"
  },
  "4134010359_0812_4": {
    "round": 4,
    "year": 2016,
    "category_name": "工学",
    "subject_code": "0812",
    "subject_name": "计算机科学与技术",
    "school_code": "10357",
    "school_name": "安徽大学",
    "rank": 45,
    "grade": "B"
  }
}
```

---

### Task 4: 实现数据转换脚本 (Document Enhancement)

**Files:**
- Create: `backend/scripts/build_vector_kb.py`
- Test: `tests/test_vector_knowledge_base.py` (追加转换测试)

- [ ] **Step 1: 创建数据转换函数**

Create: `backend/scripts/build_vector_kb.py`

```python
"""数据转换和向量化脚本：将 JSON 数据转换为向量知识库文档。"""

import sys
from pathlib import Path

# 确保能导入 backend
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.services.vector_knowledge_base import (
    VectorDocument,
    VectorKnowledgeBase,
    CATEGORY_UNIVERSITY_BASIC,
    CATEGORY_SCORE_BATCH,
    CATEGORY_SCORE_SCHOOL,
    CATEGORY_SUBJECT_EVAL,
)


def build_university_docs(universities: dict) -> list[VectorDocument]:
    """为高校数据生成多个语义变体文档"""
    docs = []

    for school_id, data in universities.items():
        name = data.get("school_name", "")
        province = data.get("province", "")
        location = data.get("location", "")
        tier = data.get("tier", "")
        department = data.get("competent_department", "")
        level = data.get("level", "")
        is_private = data.get("is_private", False)

        private_note = "民办" if is_private else "公办"
        tier_note = f"{tier}院校" if tier else ""

        # 变体1: 基础描述
        docs.append(VectorDocument(
            id=f"{CATEGORY_UNIVERSITY_BASIC}:{school_id}:basic",
            category=CATEGORY_UNIVERSITY_BASIC,
            base_id=school_id,
            variant_type="basic",
            text=f"{name}位于{location}，主管部门为{department}，办学层次为{level}，是一所{private_note}普通高等学校。{tier_note}",
            metadata=data,
        ))

        # 变体2: 地域导向
        docs.append(VectorDocument(
            id=f"{CATEGORY_UNIVERSITY_BASIC}:{school_id}:location_based",
            category=CATEGORY_UNIVERSITY_BASIC,
            base_id=school_id,
            variant_type="location_based",
            text=f"{province}{location}的大学推荐:{name}，是一所{private_note}{tier_note}。主管部门{department}。",
            metadata=data,
        ))

        # 变体3: 学校类型导向
        if tier:
            docs.append(VectorDocument(
                id=f"{CATEGORY_UNIVERSITY_BASIC}:{school_id}:tier_based",
                category=CATEGORY_UNIVERSITY_BASIC,
                base_id=school_id,
                variant_type="tier_based",
                text=f"如果想报考{tier}院校，{name}是不错的选择。该校位于{location}，{private_note}。",
                metadata=data,
            ))

    return docs


def build_batch_score_docs(batch_scores: dict) -> list[VectorDocument]:
    """为批次分数线数据生成文档"""
    docs = []

    for score_id, data in batch_scores.items():
        province = data.get("province", "")
        year = data.get("year", "")
        category = data.get("category", "")
        batch = data.get("batch", "")
        subject = data.get("subject_type", "")
        score = data.get("score_line", 0)

        docs.append(VectorDocument(
            id=f"{CATEGORY_SCORE_BATCH}:{score_id}:basic",
            category=CATEGORY_SCORE_BATCH,
            base_id=score_id,
            variant_type="basic",
            text=f"{year}年{province}{category}{batch}{subject}录取分数线为{score}分。",
            metadata=data,
        ))

    return docs


def build_school_score_docs(school_scores: dict) -> list[VectorDocument]:
    """为学校录取分数线数据生成文档"""
    docs = []

    for score_id, data in school_scores.items():
        province = data.get("province", "")
        year = data.get("year", "")
        subject = data.get("subject_type", "")
        school = data.get("school_name", "")
        min_score = data.get("min_score", 0)
        avg_score = data.get("avg_score", 0)
        line_diff = data.get("line_diff", 0)

        # 变体1: 分数导向
        docs.append(VectorDocument(
            id=f"{CATEGORY_SCORE_SCHOOL}:{score_id}:score_based",
            category=CATEGORY_SCORE_SCHOOL,
            base_id=score_id,
            variant_type="score_based",
            text=f"{year}年{school}在{province}{subject}录取最低分{min_score}分，平均分{avg_score}分，高出省控线{line_diff}分。",
            metadata=data,
        ))

        # 变体2: 学校导向
        docs.append(VectorDocument(
            id=f"{CATEGORY_SCORE_SCHOOL}:{score_id}:school_based",
            category=CATEGORY_SCORE_SCHOOL,
            base_id=score_id,
            variant_type="school_based",
            text=f"{school}{year}年在{province}的{subject}录取情况:最低分{min_score}，平均分{avg_score}。",
            metadata=data,
        ))

    return docs


def build_subject_eval_docs(subject_data: dict) -> list[VectorDocument]:
    """为学科评估数据生成文档"""
    docs = []

    for eval_id, data in subject_data.items():
        school = data.get("school_name", "")
        subject = data.get("subject_name", "")
        category = data.get("category_name", "")
        grade = data.get("grade", "")
        rank = data.get("rank", 0)
        year = data.get("year", 0)
        round_num = data.get("round", 0)

        # 变体1: 学科评估导向
        docs.append(VectorDocument(
            id=f"{CATEGORY_SUBJECT_EVAL}:{eval_id}:eval_based",
            category=CATEGORY_SUBJECT_EVAL,
            base_id=eval_id,
            variant_type="eval_based",
            text=f"{school}的{subject}（{category}）学科，在第{round_num}轮学科评估中整体水平排名第{rank}，评估结果为{grade}。评估年份:{year}。",
            metadata=data,
        ))

        # 变体2: 专业推荐导向
        grade_desc = "全国顶尖" if grade in ["A+", "A", "A-"] else "较强" if grade in ["B+", "B"] else ""
        if grade_desc:
            docs.append(VectorDocument(
                id=f"{CATEGORY_SUBJECT_EVAL}:{eval_id}:recommend_based",
                category=CATEGORY_SUBJECT_EVAL,
                base_id=eval_id,
                variant_type="recommend_based",
                text=f"如果想学{subject}专业，{school}是不错的选择，该学科评估{grade}，{grade_desc}，全国排名第{rank}。",
                metadata=data,
            ))

    return docs


def build_all(data_dir: Path = None) -> VectorKnowledgeBase:
    """构建完整的向量知识库"""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"

    kb = VectorKnowledgeBase(data_dir=data_dir)

    # 1. 加载高校名单
    universities_file = data_dir / "universities_list.json"
    if universities_file.exists():
        import json
        with open(universities_file, "r", encoding="utf-8") as f:
            universities = json.load(f)
        kb.add_documents(build_university_docs(universities))

    # 2. 加载高考分数线
    gaokao_file = data_dir / "gaokao_scores.json"
    if gaokao_file.exists():
        import json
        with open(gaokao_file, "r", encoding="utf-8") as f:
            gaokao_data = json.load(f)
        kb.add_documents(build_batch_score_docs(gaokao_data.get("batch_scores", {})))
        kb.add_documents(build_school_score_docs(gaokao_data.get("school_scores", {})))

    # 3. 加载学科评估
    subject_file = data_dir / "subject_review.json"
    if subject_file.exists():
        import json
        with open(subject_file, "r", encoding="utf-8") as f:
            subject_data = json.load(f)
        kb.add_documents(build_subject_eval_docs(subject_data))

    # 4. 向量化
    kb.embed_all()

    return kb


if __name__ == "__main__":
    kb = build_all()
    print(f"知识库构建完成:")
    print(f"  文档总数: {kb.document_count}")
    print(f"  类别: {kb.categories}")
    print(f"  缓存大小: {kb._cache.size()}")

    # 测试搜索
    print("\n测试搜索 '北京的大学':")
    results = kb.semantic_search("北京的大学", category="university_basic", top_k=3)
    for r in results:
        print(f"  [{r.rank}] {r.document.metadata.get('school_name', '')} (score: {r.score:.3f})")

    print("\n测试搜索 '计算机专业强校':")
    results = kb.semantic_search("计算机专业强校", category="subject_eval", top_k=3)
    for r in results:
        print(f"  [{r.rank}] {r.document.metadata.get('school_name', '')} - {r.document.metadata.get('subject_name', '')} (score: {r.score:.3f})")
```

- [ ] **Step 2: 写数据转换测试**

在 `tests/test_vector_knowledge_base.py` 末尾追加：

```python
class TestDataTransformation:
    """数据转换脚本测试"""

    def test_build_university_docs(self):
        from backend.scripts.build_vector_kb import build_university_docs

        test_data = {
            "001": {
                "school_name": "清华大学",
                "province": "北京",
                "competent_department": "教育部",
                "location": "北京",
                "level": "本科",
                "is_private": False,
                "tier": "985",
            }
        }
        docs = build_university_docs(test_data)
        # 每条数据应生成至少3个变体
        assert len(docs) >= 3
        # 检查变体类型
        variant_types = [d.variant_type for d in docs]
        assert "basic" in variant_types
        assert "location_based" in variant_types
        assert "tier_based" in variant_types

    def test_build_batch_score_docs(self):
        from backend.scripts.build_vector_kb import build_batch_score_docs

        test_data = {
            "test_batch_001": {
                "province": "安徽",
                "year": 2025,
                "category": "本科",
                "batch": "一批",
                "subject_type": "理科",
                "score_line": 515,
            }
        }
        docs = build_batch_score_docs(test_data)
        assert len(docs) == 1
        assert "515" in docs[0].text
        assert "安徽" in docs[0].text

    def test_build_school_score_docs(self):
        from backend.scripts.build_vector_kb import build_school_score_docs

        test_data = {
            "test_school_001": {
                "province": "安徽",
                "year": 2025,
                "subject_type": "理科",
                "school_name": "合肥工业大学",
                "min_score": 572,
                "avg_score": 578,
                "line_diff": 57,
            }
        }
        docs = build_school_score_docs(test_data)
        assert len(docs) >= 2
        # 检查包含关键信息
        combined_text = " ".join(d.text for d in docs)
        assert "合肥工业大学" in combined_text
        assert "572" in combined_text

    def test_build_subject_eval_docs(self):
        from backend.scripts.build_vector_kb import build_subject_eval_docs

        test_data = {
            "test_eval_001": {
                "round": 4,
                "year": 2016,
                "category_name": "工学",
                "subject_name": "计算机科学与技术",
                "school_name": "清华大学",
                "rank": 1,
                "grade": "A+",
            }
        }
        docs = build_subject_eval_docs(test_data)
        assert len(docs) >= 1
        # A+ 应生成推荐变体
        variant_types = [d.variant_type for d in docs]
        assert "eval_based" in variant_types
        assert "recommend_based" in variant_types

    def test_build_all_integration(self):
        """集成测试: 构建完整知识库并搜索"""
        from backend.scripts.build_vector_kb import build_all
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # 复制测试数据
            import shutil
            src_data = Path(__file__).resolve().parent.parent / "backend" / "data"
            for f in ["universities_list.json", "gaokao_scores.json", "subject_review.json"]:
                src = src_data / f
                if src.exists():
                    shutil.copy(src, data_dir / f)

            kb = build_all(data_dir=data_dir)
            assert kb.document_count > 0
            # 所有文档都应有 embedding
            for doc in kb._documents:
                assert len(doc.embedding) == 384

            # 测试搜索
            results = kb.semantic_search("北京的大学", category="university_basic", top_k=3)
            assert len(results) > 0
```

- [ ] **Step 3: 运行测试验证通过**

Run: `pytest tests/test_vector_knowledge_base.py::TestDataTransformation -v`

Expected: All tests PASS

- [ ] **Step 4: 运行脚本验证输出**

Run: `python backend/scripts/build_vector_kb.py`

Expected: 输出类似:
```
知识库构建完成:
  文档总数: XX
  类别: ['university_basic', 'score_batch', 'score_school', 'subject_eval']
  缓存大小: XX

测试搜索 '北京的大学':
  [1] 清华大学 (score: 0.XXX)
  ...
```

---

### Task 5: 运行全部测试验证

- [ ] **Step 1: 运行所有测试**

Run: `pytest tests/test_vector_knowledge_base.py -v --tb=short`

Expected: All tests PASS

- [ ] **Step 2: 运行全部现有测试确保无破坏**

Run: `pytest tests/ -v --tb=short`

Expected: All existing tests + new tests PASS

---

### Task 6: Commit

- [ ] **Step 1: 提交代码**

```bash
git add backend/services/vector_knowledge_base.py
git add backend/scripts/build_vector_kb.py
git add backend/data/universities_list.json
git add backend/data/gaokao_scores.json
git add backend/data/subject_review.json
git add backend/cache/.gitkeep
git add tests/test_vector_knowledge_base.py
git commit -m "feat: 向量知识库基础模块 - 支持文档化、向量化和语义搜索

- 新增 VectorKnowledgeBase 核心类
- 新增 EmbeddingCache 持久化缓存
- 新增三类示例数据(高校名单/高考分数线/学科评估)
- 新增数据转换脚本 build_vector_kb.py
- 每条数据生成多语义变体(3-5个)提升召回率
- 完整单元测试覆盖"
```

---

## Phase 2+ 后续计划（不在本 Plan 范围内）

Phase 1 完成后，系统已具备：
- 向量知识库核心能力
- 数据文档化和多语义变体
- Embedding 缓存

后续 Phase 将依次实现：

**Phase 2: 用户画像 + 混合检索**
- 升级 UserProfiler 解析更多字段
- 实现 StructuredFilter 结构化过滤
- 实现多路召回融合

**Phase 3: Rerank + 冲稳保**
- 实现多因子 Reranker
- 实现 RiskClassifier
- 实现推荐理由生成

**Phase 4: Agent 集成**
- 工具注册表
- Orchestrator 升级
- 端到端流程

**Phase 5: 性能优化**
- FAISS 索引支持
- 增量更新
- 异步支持
