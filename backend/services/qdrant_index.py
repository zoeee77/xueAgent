"""QdrantIndex: Qdrant 向量数据库封装。

提供与 VectorIndex 类似的接口，但底层使用 Qdrant Cloud 存储。
支持语义检索、增量更新、payload 过滤查询。
"""

import logging
from typing import List, Optional, Tuple, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.models.config import settings

logger = logging.getLogger(__name__)


class QdrantIndex:
    """Qdrant 向量索引封装。

    Args:
        dimension: 向量维度
        collection_name: Qdrant Collection 名称
        url: Qdrant 服务地址
        api_key: Qdrant API Key
    """

    def __init__(
        self,
        dimension: int,
        collection_name: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self._dimension = dimension
        self._collection_name = collection_name or settings.qdrant_collection
        self._url = url or settings.qdrant_url
        self._api_key = api_key or settings.qdrant_api_key
        self._client: Optional[QdrantClient] = None
        self._ready = False

        self._init_client()

    # -- 初始化 --

    def _init_client(self) -> None:
        """初始化 Qdrant 客户端并检查连接。"""
        if not self._url or not self._api_key:
            logger.warning("Qdrant URL 或 API Key 未配置，Qdrant 引擎不可用")
            return

        try:
            self._client = QdrantClient(url=self._url, api_key=self._api_key)
            info = self._client.info()
            logger.info("Qdrant 连接成功: version=%s", info.version)
            self._ready = True
        except Exception as e:
            logger.error("Qdrant 连接失败: %s", e)
            self._ready = False

    @property
    def is_ready(self) -> bool:
        """Qdrant 是否可用。"""
        return self._ready and self._client is not None

    @property
    def engine(self) -> str:
        return "qdrant"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def count(self) -> int:
        """获取 Collection 中的向量总数。"""
        if not self.is_ready:
            return 0
        try:
            return self._client.count(collection_name=self._collection_name).count
        except Exception:
            return 0

    @property
    def is_dirty(self) -> bool:
        return False  # Qdrant 云端存储，无需 dirty 标记

    # -- Collection 管理 --

    def ensure_collection(self, distance: str = "COSINE") -> None:
        """确保 Collection 存在，不存在则创建。"""
        if not self.is_ready:
            raise RuntimeError("Qdrant 客户端未就绪")

        try:
            collections = self._client.get_collections().collections
            names = [c.name for c in collections]
            if self._collection_name in names:
                logger.info("Collection '%s' 已存在", self._collection_name)
                return

            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._dimension,
                    distance=models.Distance[distance.upper()],
                ),
            )
            logger.info("Collection '%s' 创建成功", self._collection_name)
        except UnexpectedResponse as e:
            logger.error("创建 Collection 失败: %s", e)
            raise

    def needs_rebuild(self, all_majors: Dict[str, dict]) -> bool:
        """检查是否需要重建（对比向量数量）。"""
        if not self.is_ready:
            return True
        current_count = self.count
        return current_count != len(all_majors)

    # -- 批量写入 --

    def add_batch(
        self,
        vectors: List[List[float]],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """批量添加向量到 Collection。

        Args:
            vectors: 向量列表
            metadatas: 元数据列表，与 vectors 一一对应
        """
        if not self.is_ready:
            raise RuntimeError("Qdrant 客户端未就绪")

        self.ensure_collection()

        points = []
        for i, vec in enumerate(vectors):
            payload = {}
            if metadatas and i < len(metadatas):
                payload = dict(metadatas[i])

            points.append(
                models.PointStruct(
                    id=i,
                    vector=vec,
                    payload=payload,
                )
            )

        # 分批上传（Qdrant Cloud 单次限制）
        batch_size = 100
        total_uploaded = 0
        for start in range(0, len(points), batch_size):
            batch = points[start: start + batch_size]
            self._client.upsert(
                collection_name=self._collection_name,
                points=batch,
            )
            total_uploaded += len(batch)
            logger.info("已上传 %d/%d 个向量", total_uploaded, len(points))

        logger.info("全部 %d 个向量上传完成", total_uploaded)

    def upload_batch(
        self,
        vectors: List[List[float]],
        metadatas: Optional[List[dict]] = None,
    ) -> int:
        """兼容 VectorIndex 接口的别名。"""
        self.add_batch(vectors, metadatas)
        return len(vectors)

    # -- 增量更新 --

    def add_by_name(
        self, name: str, vector: List[float], metadata: Optional[dict] = None
    ) -> bool:
        """增量添加/更新单个专业。

        使用专业名称的哈希作为 ID，确保同一名专业可被覆盖更新。
        """
        if not self.is_ready:
            return False

        import hashlib
        point_id = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % (10 ** 9)

        payload = metadata or {"name": name}
        if "name" not in payload:
            payload["name"] = name

        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        return True

    def remove_by_name(self, name: str) -> bool:
        """按专业名称删除向量点。"""
        if not self.is_ready:
            return False

        import hashlib
        point_id = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % (10 ** 9)

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[point_id]),
        )
        return True

    # -- 检索 --

    def search(
        self,
        query: List[float],
        top_k: int = 5,
        with_payload: bool = False,
    ) -> List[Tuple[int, float]]:
        """语义检索。

        Args:
            query: 查询向量
            top_k: 返回结果数量
            with_payload: 是否同时返回 payload（Qdrant 特性，避免 N+1 查询）

        Returns:
            with_payload=False: [(point_id, score), ...]
            with_payload=True:  [(point_id, score, payload), ...]
        """
        if not self.is_ready:
            raise RuntimeError("Qdrant 客户端未就绪")

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query,
            limit=top_k,
            with_payload=with_payload,
            with_vectors=False,
        )

        if with_payload:
            return [(int(r.id), float(r.score), r.payload or {}) for r in response.points]
        return [(int(r.id), float(r.score)) for r in response.points]

    def search_with_payload(
        self,
        query: List[float],
        top_k: int = 5,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """语义检索 + payload 过滤（Qdrant 特有功能）。

        Args:
            query: 查询向量
            top_k: 返回结果数量
            filter_payload: payload 过滤条件，如 {"province": "河南"}

        Returns:
            [(point_id, score, payload), ...]
        """
        if not self.is_ready:
            raise RuntimeError("Qdrant 客户端未就绪")

        qdrant_filter = None
        if filter_payload:
            conditions = []
            for key, value in filter_payload.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
            if conditions:
                qdrant_filter = models.Filter(must=conditions)

        # Use query_points API (qdrant-client 1.x+)
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
        )
        return [(int(r.id), float(r.score), r.payload or {}) for r in response.points]

    # -- 持久化 --
    # Qdrant 云端存储，save/load 为 noop，但保持接口兼容

    def save(self, path: str) -> None:
        logger.info("Qdrant 索引已持久化到云端 (Collection=%s)", self._collection_name)

    @classmethod
    def exists(cls, path: str) -> bool:
        return True  # 云端存储，始终存在

    def load(self, path: str) -> None:
        logger.info("Qdrant 索引从云端加载 (Collection=%s)", self._collection_name)

    # -- 工具方法 --

    def get_metadata(self, point_id: int) -> dict:
        """获取指定点的元数据。"""
        if not self.is_ready:
            return {}
        try:
            points = self._client.retrieve(
                collection_name=self._collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            if points:
                return points[0].payload or {}
        except Exception:
            pass
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """获取 Collection 统计信息。"""
        if not self.is_ready:
            return {"status": "not_ready"}

        try:
            info = self._client.get_collection(self._collection_name)
            return {
                "status": "ready",
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "config": {
                    "dimension": self._dimension,
                },
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
