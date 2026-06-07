"""向量知识库模块：支持语义搜索的文档存储和检索。"""

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

# Category constants
CATEGORY_UNIVERSITY_BASIC = "university_basic"
CATEGORY_SCORE_BATCH = "score_batch"
CATEGORY_SCORE_SCHOOL = "score_school"
CATEGORY_SCORE_MAJOR = "score_major"
CATEGORY_SUBJECT_EVAL = "subject_eval"


@dataclass
class VectorDocument:
    """向量文档模型"""
    id: str
    category: str
    base_id: str
    variant_type: str
    text: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
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
    score: float
    rank: int = 0


@dataclass
class RiskClassificationResult:
    """冲稳保分类结果"""
    charge: list
    stable: list
    safe: list


class EmbeddingCache:
    """Embedding 持久化缓存"""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "embeddings_cache.json"
        self._cache: dict[str, list[float]] = {}
        self._load()

    def _load(self):
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save(self):
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)
        except IOError:
            pass

    def _hash_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        key = self._hash_key(text)
        return self._cache.get(key)

    def set(self, text: str, embedding: list[float]):
        key = self._hash_key(text)
        self._cache[key] = embedding

    def save(self):
        self._save()

    def size(self) -> int:
        return len(self._cache)


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
        self._cache.set(doc.text, embedding)
        return doc

    def embed_all(self) -> None:
        """为所有未向量化的文档计算 embedding"""
        for doc in self._documents:
            if not doc.embedding:
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

        candidates = self._documents
        if category:
            candidates = [d for d in candidates if d.category == category]

        if not candidates:
            return []

        cached = self._cache.get(query)
        if cached:
            query_embedding = cached
        else:
            query_embedding = self._get_embedding_service().get_embedding(query)
            self._cache.set(query, query_embedding)

        embedding_svc = self._get_embedding_service()
        scored = []
        for doc in candidates:
            if doc.embedding:
                score = embedding_svc.cosine_similarity(query_embedding, doc.embedding)
                scored.append(SearchResult(document=doc, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        scored = scored[:top_k]

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
