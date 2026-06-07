"""推荐工具：Agent可调用的志愿填报推荐工具。"""

import logging
from typing import Optional

from backend.services.recommend_engine import RecommendEngine, convert_profile_to_recommender
from backend.services.recommender import UserProfile as RecommenderProfile
from backend.models.agent_output import UserProfile as AgentProfile

logger = logging.getLogger(__name__)

_engine: Optional[RecommendEngine] = None


def get_engine() -> RecommendEngine:
    """获取或创建推荐引擎（单例）"""
    global _engine
    if _engine is None:
        _engine = RecommendEngine()
    return _engine


def recommend_by_profile(
    agent_profile: AgentProfile,
    top_k: int = 20,
):
    """
    使用 Agent UserProfile 进行推荐

    Args:
        agent_profile: Agent 系统的 UserProfile
        top_k: 召回数量

    Returns:
        RecommendationReport
    """
    profile = convert_profile_to_recommender(agent_profile)
    engine = get_engine()
    return engine.recommend(profile, top_k=top_k)


def recommend_by_text(
    user_text: str,
    score: int = 0,
    province: str = "",
    subject_type: str = "",
    top_k: int = 20,
):
    """
    使用自然语言 + 结构化参数进行推荐

    Args:
        user_text: 用户自然语言输入
        score: 高考分数
        province: 省份
        subject_type: 科类
        top_k: 召回数量

    Returns:
        RecommendationReport
    """
    profile = RecommenderProfile(
        score=score if score else None,
        province=province if province else None,
        subject_type=subject_type if subject_type else None,
    )
    engine = get_engine()
    return engine.recommend(profile, top_k=top_k)
