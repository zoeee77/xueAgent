"""混合检索模块：结构化过滤 + 向量召回 + 多查询扩展。"""

from dataclasses import dataclass, field
from typing import Any, Optional

from backend.services.vector_knowledge_base import VectorDocument


@dataclass
class FilterCondition:
    """过滤条件"""
    field: str
    operator: str  # "=", "!=", "in", "range"
    value: Any


class StructuredFilter:
    """结构化过滤器"""

    def __init__(self, conditions: list[FilterCondition] = None):
        self._conditions: list[FilterCondition] = conditions or []

    def add_condition(self, condition: FilterCondition) -> "StructuredFilter":
        """添加过滤条件"""
        self._conditions.append(condition)
        return self

    def apply(self, documents: list[VectorDocument]) -> list[VectorDocument]:
        """应用过滤条件，返回符合条件的文档"""
        result = documents
        for condition in self._conditions:
            result = self._apply_single(result, condition)
        return result

    def _apply_single(
        self, documents: list[VectorDocument], condition: FilterCondition
    ) -> list[VectorDocument]:
        """应用单个过滤条件"""
        filtered = []
        for doc in documents:
            field_value = doc.metadata.get(condition.field)
            if condition.operator == "=":
                if field_value == condition.value:
                    filtered.append(doc)
            elif condition.operator == "!=":
                if field_value != condition.value:
                    filtered.append(doc)
            elif condition.operator == "in":
                if field_value in condition.value:
                    filtered.append(doc)
            elif condition.operator == "range":
                if isinstance(condition.value, (list, tuple)) and len(condition.value) == 2:
                    low, high = condition.value
                    if isinstance(field_value, (int, float)):
                        if low <= field_value <= high:
                            filtered.append(doc)
        return filtered

    @property
    def conditions(self) -> list[FilterCondition]:
        return self._conditions

    def clear(self) -> None:
        self._conditions.clear()


class QueryExpander:
    """多查询扩展器：将用户画像扩展为多个查询文本"""

    def expand(self, profile: dict) -> list[str]:
        """
        将用户画像字典扩展为多个查询

        Args:
            profile: 用户画像字典，包含:
                - target_majors: list[str] 目标专业列表
                - city_preference: list[str] 城市偏好
                - score: int 高考分数
                - school_types: list[str] 学校类型
                - province: str 生源省份

        Returns:
            查询文本列表
        """
        queries = []

        # 专业查询
        target_majors = profile.get("target_majors", [])
        if target_majors:
            for major in target_majors[:2]:  # 最多取前2个专业
                queries.append(f"{major}专业强校")
                queries.append(f"{major}学科建设好的大学")

        # 省份查询（提前判断，用于城市去重）
        province = profile.get("province", "")

        # 地域查询
        city_pref = profile.get("city_preference", [])
        if city_pref:
            for city in city_pref[:2]:
                # 如果城市与省份相同，跳过"XX的大学"（由省份查询处理或完全跳过）
                if city == province:
                    continue
                queries.append(f"{city}的大学")
                queries.append(f"{city}本地高校")

        # 省份查询
        if province and province not in city_pref:
            queries.append(f"{province}的大学")

        # 分数段查询
        score = profile.get("score")
        if score:
            queries.append(f"录取分数{score}左右的大学")

        # 学校类型查询
        school_types = profile.get("school_types", [])
        if school_types:
            for stype in school_types:
                queries.append(f"{stype}院校")

        # 去重
        return list(dict.fromkeys(queries))


@dataclass
class HybridSearchResult:
    """混合检索结果"""
    document: VectorDocument
    combined_score: float       # 综合得分
    filter_score: float = 0.0   # 结构化过滤得分
    vector_score: float = 0.0   # 向量相似度得分
    rank: int = 0


class HybridSearch:
    """混合检索：结构化过滤 + 向量召回融合"""

    def __init__(self, vector_kb):
        """
        Args:
            vector_kb: VectorKnowledgeBase 实例
        """
        self._kb = vector_kb

    def search(
        self,
        query: str,
        filters: list[FilterCondition] = None,
        category: str = None,
        top_k: int = 20,
        weights: dict = None,
    ) -> list[HybridSearchResult]:
        """
        混合检索

        Args:
            query: 查询文本
            filters: 结构化过滤条件列表
            category: 可选，限定数据类别
            top_k: 返回结果数量
            weights: 融合权重 {"filter": 1.0, "vector": 0.3}

        Returns:
            按综合得分降序的混合检索结果列表
        """
        if weights is None:
            weights = {"filter": 1.0, "vector": 0.3}

        # 1. 获取候选文档
        if category:
            candidates = self._kb.get_documents_by_category(category)
        else:
            candidates = self._kb._documents

        if not candidates:
            return []

        # 2. 结构化过滤（如果有过滤条件）
        filter_results = candidates
        if filters:
            sf = StructuredFilter(filters)
            filter_results = sf.apply(candidates)

        # 3. 向量召回
        vector_results = self._kb.semantic_search(query, category=category, top_k=top_k)

        # 4. 融合
        return self._merge_results(
            filter_results=filter_results,
            vector_results=vector_results,
            weights=weights,
            has_filters=filters is not None and len(filters) > 0,
        )

    def _merge_results(
        self,
        filter_results: list[VectorDocument],
        vector_results: list,  # list[SearchResult]
        weights: dict,
        has_filters: bool = False,
    ) -> list[HybridSearchResult]:
        """
        融合多路召回结果
        策略: 取并集，去重，加权排序
        当有过滤器时，向量召回结果需通过过滤才能加入
        """
        all_docs: dict[str, HybridSearchResult] = {}

        filter_w = weights.get("filter", 1.0)
        vector_w = weights.get("vector", 0.3)

        # 过滤结果集合（用于快速判断）
        filter_ids = {doc.id for doc in filter_results} if has_filters else set()

        # 结构化过滤结果（权重最高，因为必须满足）
        for doc in filter_results:
            all_docs[doc.id] = HybridSearchResult(
                document=doc,
                combined_score=filter_w,
                filter_score=filter_w,
                vector_score=0.0,
            )

        # 向量召回结果
        for result in vector_results:
            doc = result.document
            # 如果有过滤器，向量召回结果也必须通过过滤
            if has_filters and doc.id not in filter_ids:
                continue
            v_score = result.score * vector_w
            if doc.id in all_docs:
                all_docs[doc.id].combined_score += v_score
                all_docs[doc.id].vector_score = result.score
            else:
                all_docs[doc.id] = HybridSearchResult(
                    document=doc,
                    combined_score=v_score,
                    filter_score=0.0,
                    vector_score=result.score,
                )

        # 按综合得分排序
        sorted_results = sorted(
            all_docs.values(), key=lambda x: x.combined_score, reverse=True
        )

        # 设置排名
        for i, r in enumerate(sorted_results):
            r.rank = i + 1

        return sorted_results
