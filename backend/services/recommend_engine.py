"""推荐引擎：整合向量知识库、混合检索、重排序和推荐报告生成。"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend.scripts.build_vector_kb import build_all
from backend.services.hybrid_search import FilterCondition, HybridSearch
from backend.services.recommender import (
    UserProfile,
    Reranker,
    RecommendationReport,
    build_recommendation_report,
)
from backend.services.vector_knowledge_base import VectorKnowledgeBase

logger = logging.getLogger(__name__)


class RecommendEngine:
    """推荐引擎：端到端的志愿填报推荐流程"""

    def __init__(self, data_dir: Path = None):
        self._kb: Optional[VectorKnowledgeBase] = None
        self._data_dir = data_dir

    def load(self) -> None:
        """加载知识库（如果尚未加载）"""
        if self._kb is None:
            logger.info("Loading knowledge base...")
            self._kb = build_all(data_dir=self._data_dir)
            logger.info(f"Knowledge base loaded: {self._kb.document_count} documents")

    def recommend(
        self,
        user_profile: UserProfile,
        top_k: int = 20,
    ) -> RecommendationReport:
        """
        完整推荐流程

        Args:
            user_profile: 用户画像
            top_k: 召回数量

        Returns:
            推荐报告
        """
        self.load()

        # 1. 构建结构化过滤条件
        filters = self._build_filters(user_profile)

        # 2. 混合检索（学校基本信息）
        province = user_profile.province or ""
        subject_type = user_profile.subject_type or ""
        query = f"{province} {subject_type} 大学".strip()
        if not query or query.isspace():
            query = "大学"

        hs = HybridSearch(self._kb)
        search_results = hs.search(
            query=query,
            filters=filters,
            category="university_basic",
            top_k=top_k,
        )

        if not search_results:
            # 无过滤条件重试
            search_results = hs.search(
                query="大学",
                category="university_basic",
                top_k=top_k,
            )

        # 3. 转换为 VectorDocument 列表
        docs = [r.document for r in search_results]

        # 4. 重排序
        reranker = Reranker()
        rerank_results = reranker.rerank(docs, user_profile)

        # 5. 构建推荐报告
        report = build_recommendation_report(user_profile, rerank_results)

        return report

    def _build_filters(self, profile: UserProfile) -> list[FilterCondition]:
        """根据用户画像构建结构化过滤条件"""
        filters = []

        if profile.province:
            filters.append(FilterCondition("province", "=", profile.province))

        if profile.school_types:
            filters.append(FilterCondition("tier", "in", profile.school_types))

        if profile.degree_level:
            filters.append(FilterCondition("level", "=", profile.degree_level))

        # 分数范围过滤（根据用户分数推断）
        if profile.score:
            # 录取线在 user_score-30 到 user_score+30 之间
            filters.append(
                FilterCondition("min_score", "range", [profile.score - 30, profile.score + 30])
            )

        return filters


def convert_profile_to_recommender(
    agent_profile,  # backend.models.agent_output.UserProfile
) -> UserProfile:
    """将现有的 Pydantic UserProfile 转换为 recommender 的 UserProfile"""
    from backend.services.recommender import UserProfile as RecommenderProfile

    # city_preference: agent 中是 Optional[str]，recommender 中是 list
    city_pref = []
    if agent_profile.city_preference:
        city_pref = [agent_profile.city_preference]

    return RecommenderProfile(
        score=agent_profile.score if agent_profile.score else None,
        province=agent_profile.province if agent_profile.province else None,
        subject_type=agent_profile.subject_type,
        target_majors=agent_profile.target_majors or [],
        risk_preference=agent_profile.risk_preference,
        city_preference=city_pref,
        school_types=agent_profile.school_types or [],
        degree_level=agent_profile.degree_level or "本科",
        constraints=agent_profile.constraints or [],
    )
