"""VectorIndex: 向量索引管理，支持 FAISS 和 NumPy 双引擎。

V5 升级内容:
1. 多字段语义增强 (build_major_document)
2. 向量索引持久化 (save/load/exists)
3. 增量更新机制 (add_major/remove_major_by_name)
4. 索引版本管理
"""

import json
import logging
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# V5: 多字段文档构建
def build_major_document(major: dict, name: str = "") -> str:
    """为专业构建融合文本，用于 embedding 计算。

    整合专业名称、描述、课程、技能、性格适配、职业路径、关键词等字段，
    生成语义丰富的文本用于向量检索。

    Args:
        major: 专业数据字典
        name: 专业名称（如果 major 中不包含）

    Returns:
        融合后的文本字符串
    """
    parts = []

    # 专业名称
    major_name = name or major.get("name", "")
    if major_name:
        parts.append(f"专业名称：{major_name}")

    # 专业描述
    desc = major.get("description", "")
    if desc:
        parts.append(f"专业描述：{desc}")

    # 核心课程
    courses = major.get("courses", [])
    if courses:
        parts.append(f"核心课程：{'、'.join(courses)}")

    # 技能要求
    skills = major.get("skills_required", [])
    if skills:
        parts.append(f"技能要求：{'、'.join(skills)}")

    # 适合人群
    personality = major.get("personality_fit", [])
    if personality:
        parts.append(f"适合人群：{'、'.join(personality)}")

    # 就业方向
    career_paths = major.get("career_paths", [])
    if career_paths:
        parts.append(f"就业方向：{'；'.join(career_paths)}")

    # 关键词
    keywords = major.get("keywords", [])
    if keywords:
        parts.append(f"关键词：{'、'.join(keywords)}")

    # 所属行业
    industries = major.get("industries", [])
    if industries:
        parts.append(f"所属行业：{'、'.join(industries)}")

    return " ".join(parts)


def build_major_document_hash(major: dict, name: str = "") -> str:
    """计算专业文档的哈希值，用于增量更新检测。
    
    Args:
        major: 专业数据字典
        name: 专业名称
        
    Returns:
        MD5 哈希字符串
    """
    doc = build_major_document(major, name)
    return hashlib.md5(doc.encode("utf-8")).hexdigest()


class VectorIndex:
    """向量相似度索引，支持 FAISS (优先) 和 NumPy 回退。

    V5 特性:
    - 持久化存储（FAISS index / .npy + .meta.json）
    - 增量更新（add_by_name / remove_by_name）
    - 文档哈希检测（避免重复插入过期数据）

    Args:
        dimension: 向量维度
        engine: 'faiss' 或 'numpy'，None 时自动检测
        persist_path: 持久化路径（不含扩展名）
    """

    # 索引版本号，用于检测数据结构变更
    INDEX_VERSION = "v5.0"

    def __init__(
        self,
        dimension: int,
        engine: Optional[str] = None,
        persist_path: Optional[str] = None,
    ):
        self._dimension = dimension
        self._persist_path = persist_path
        self._metadata: List[dict] = []
        self._doc_hashes: Dict[str, str] = {}  # name -> doc_hash
        self._dirty = False

        # 自动检测 faiss 可用性
        faiss_available = self._check_faiss()

        if engine is None or engine == "auto":
            engine = "faiss" if faiss_available else "numpy"
        elif engine == "faiss" and not faiss_available:
            logger.warning("FAISS 引擎指定但未安装，回退到 NumPy 引擎")
            engine = "numpy"

        self._engine = engine
        self._index = None
        self._vectors: Optional[np.ndarray] = None
        self._loaded = False

    @staticmethod
    def _check_faiss() -> bool:
        try:
            import faiss  # noqa: F401
            return True
        except ImportError:
            return False

    # ── 持久化检查 ──

    @staticmethod
    def exists(path: str) -> bool:
        """检查持久化索引文件是否存在。
        
        Args:
            path: 持久化路径（不含扩展名）
            
        Returns:
            True 如果索引文件和元数据文件都存在
        """
        p = Path(path)
        index_exists = p.with_suffix(".index").exists() or p.with_suffix(".npy").exists()
        meta_exists = p.with_suffix(".meta.json").exists()
        return index_exists and meta_exists

    # ── 懒加载 ──

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._build_empty_index()
        if self._persist_path and self.exists(self._persist_path):
            self.load(self._persist_path)
        self._loaded = True

    def _build_empty_index(self):
        if self._engine == "faiss":
            import faiss
            self._index = faiss.IndexFlatIP(self._dimension)
        else:
            self._vectors = np.empty((0, self._dimension), dtype=np.float32)

    # ── 增删改 ──

    def add(self, vector: List[float], metadata: Optional[dict] = None):
        self.add_batch([vector], [metadata] if metadata else [{}])

    def add_batch(self, vectors: List[List[float]], metadatas: Optional[List[dict]] = None):
        self._ensure_loaded()
        arr = np.array(vectors, dtype=np.float32)

        if self._engine == "faiss":
            self._index.add(arr)
        else:
            if len(self._vectors) > 0:
                self._vectors = np.vstack([self._vectors, arr])
            else:
                self._vectors = arr

        if metadatas:
            self._metadata.extend(metadatas)
        else:
            self._metadata.extend([{}] * len(vectors))
        self._dirty = True

    def update(self, idx: int, vector: List[float], metadata: Optional[dict] = None):
        """更新指定位置的向量。"""
        self._ensure_loaded()
        arr = np.array([vector], dtype=np.float32)

        if self._engine == "faiss":
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

    # ── V5 增量更新 ──

    def add_by_name(
        self, name: str, vector: List[float], major_data: dict,
        doc_hash: Optional[str] = None,
    ) -> bool:
        """按名称增量添加专业，支持重复检测。

        Args:
            name: 专业名称
            vector: 向量
            major_data: 专业完整数据
            doc_hash: 文档哈希，如果未提供则自动计算

        Returns:
            True 表示新增成功，False 表示已存在且未变更
        """
        self._ensure_loaded()

        if doc_hash is None:
            doc_hash = build_major_document_hash(major_data, name)

        # 重复检测
        if name in self._doc_hashes:
            if self._doc_hashes[name] == doc_hash:
                logger.debug("Major '%s' already exists with same content, skipping", name)
                return False
            # 内容变更，需要更新
            logger.info("Major '%s' content changed, updating", name)
            self._update_by_name(name, vector, major_data, doc_hash)
            return True

        # 新增
        metadata = {"name": name, **major_data}
        self.add(vector, metadata)
        self._doc_hashes[name] = doc_hash
        logger.info("Added major '%s' to vector index", name)
        return True

    def _update_by_name(
        self, name: str, vector: List[float], major_data: dict, doc_hash: str,
    ):
        """更新已有专业的向量。"""
        idx = self._find_index_by_name(name)
        if idx is None:
            logger.warning("Major '%s' not found for update, adding instead", name)
            self.add_by_name(name, vector, major_data, doc_hash)
            return

        metadata = {"name": name, **major_data}
        self.update(idx, vector, metadata)
        self._doc_hashes[name] = doc_hash

    def remove_by_name(self, name: str) -> bool:
        """按名称移除专业。

        Args:
            name: 专业名称

        Returns:
            True 表示成功移除，False 表示未找到
        """
        self._ensure_loaded()
        idx = self._find_index_by_name(name)
        if idx is None:
            return False

        if self._engine == "faiss":
            # FAISS 不支持直接删除，需要重建索引
            self._remove_by_rebuild(name)
        else:
            # NumPy 可以直接删除
            self._vectors = np.delete(self._vectors, idx, axis=0)
            del self._metadata[idx]

        self._doc_hashes.pop(name, None)
        self._dirty = True
        logger.info("Removed major '%s' from vector index", name)
        return True

    def _remove_by_rebuild(self, name: str):
        """通过重建索引移除专业（FAISS 引擎）。"""
        all_vecs = self.get_all_vectors()
        idx = self._find_index_by_name(name)
        if idx is None:
            return

        # 过滤掉要删除的向量
        keep_indices = [i for i in range(len(all_vecs)) if i != idx]
        kept_vecs = all_vecs[keep_indices] if keep_indices else np.empty((0, self._dimension), dtype=np.float32)
        kept_metadata = [self._metadata[i] for i in keep_indices]

        import faiss
        self._index = faiss.IndexFlatIP(self._dimension)
        if len(kept_vecs) > 0:
            self._index.add(kept_vecs)
        self._metadata = kept_metadata

    def _find_index_by_name(self, name: str) -> Optional[int]:
        """查找专业在索引中的位置。"""
        for i, meta in enumerate(self._metadata):
            if meta.get("name") == name:
                return i
        return None

    def rebuild_from_scratch(
        self, majors: Dict[str, dict], embedding_fn,
    ):
        """从给定专业数据全量重建索引。

        Args:
            majors: {name: major_data} 字典
            embedding_fn: 批量 embedding 函数，接收 List[str] 返回 List[List[float]]
        """
        self._ensure_loaded()

        # 构建文档文本和哈希
        texts = []
        names = []
        metadatas = []
        new_hashes = {}

        for name, data in majors.items():
            doc = build_major_document(data, name)
            texts.append(doc)
            names.append(name)
            metadatas.append({"name": name, **data})
            new_hashes[name] = hashlib.md5(doc.encode("utf-8")).hexdigest()

        # 批量计算 embedding
        vectors = embedding_fn(texts)

        # 重建索引
        arr = np.array(vectors, dtype=np.float32)
        if self._engine == "faiss":
            import faiss
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(arr)
        else:
            self._vectors = arr

        self._metadata = metadatas
        self._doc_hashes = new_hashes
        self._dirty = True
        logger.info("Rebuilt vector index: %d majors", len(names))

    def needs_rebuild(self, majors: Dict[str, dict]) -> bool:
        """检查是否需要重建索引（对比专业集合是否一致）。

        Args:
            majors: {name: major_data} 字典

        Returns:
            True 如果专业集合有变化（增删改）
        """
        if len(self._doc_hashes) != len(majors):
            return True

        for name, data in majors.items():
            expected_hash = build_major_document_hash(data, name)
            if self._doc_hashes.get(name) != expected_hash:
                return True

        return False

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

        # 保存元数据 + 文档哈希 + 版本信息
        meta_data = {
            "version": self.INDEX_VERSION,
            "dimension": self._dimension,
            "engine": self._engine,
            "count": self._count(),
            "doc_hashes": self._doc_hashes,
            "metadata": self._metadata,
        }
        meta_path = save_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        self._dirty = False
        logger.info("Vector index saved to %s (engine=%s, count=%d)", save_path, self._engine, self._count())

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
                meta_data = json.load(f)
            self._metadata = meta_data.get("metadata", [])
            self._doc_hashes = meta_data.get("doc_hashes", {})

        logger.info("Vector index loaded from %s (engine=%s, count=%d)", load_path, self._engine, self._count())

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

    def get_name_to_index_map(self) -> Dict[str, int]:
        """返回专业名称到索引位置的映射。"""
        result = {}
        for i, meta in enumerate(self._metadata):
            name = meta.get("name")
            if name:
                result[name] = i
        return result

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

    @property
    def count(self) -> int:
        return self._count()
