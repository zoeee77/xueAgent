"""推荐模块：多因子重排序 + 冲稳保分类 + 推荐理由生成。"""

from dataclasses import dataclass, field
from typing import Optional

from backend.services.vector_knowledge_base import VectorDocument


@dataclass
class UserProfile:
    """用户画像"""
    score: Optional[int] = None
    province: Optional[str] = None
    subject_type: Optional[str] = None
    target_majors: list = field(default_factory=list)
    risk_preference: str = "均衡"  # 冲/稳/保/均衡
    city_preference: list = field(default_factory=list)
    school_types: list = field(default_factory=list)
    degree_level: str = "本科"
    career_goals: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    confidence: dict = field(default_factory=dict)


@dataclass
class RerankResult:
    """重排序结果"""
    document: VectorDocument
    final_score: float
    breakdown: dict = field(default_factory=dict)  # 各因子得分明细


@dataclass
class RerankConfig:
    """排序配置"""
    embedding_weight: float = 0.2
    score_match_weight: float = 0.35
    subject_eval_weight: float = 0.25
    preference_weight: float = 0.2

    def adjust_for_conservative_user(self):
        """保守用户：提高分数匹配权重"""
        self.score_match_weight = 0.5
        self.embedding_weight = 0.1
        self.subject_eval_weight = 0.2
        self.preference_weight = 0.2

    def adjust_for_aggressive_user(self):
        """激进用户：降低分数匹配权重"""
        self.score_match_weight = 0.2
        self.embedding_weight = 0.3
        self.subject_eval_weight = 0.25
        self.preference_weight = 0.25


@dataclass
class RiskInfo:
    """冲稳保风险信息"""
    document: VectorDocument
    level: str          # 冲/稳/保
    score_diff: int     # 分数差
    min_score: int
    avg_score: int
    description: str = ""


@dataclass
class SchoolRecommend:
    """单所学校推荐"""
    school_name: str
    risk_level: str          # 冲/稳/保
    score_diff: int
    min_score: int
    avg_score: int
    reasons: list = field(default_factory=list)
    major_advantage: str = ""
    location_advantage: str = ""
    risk_warning: str = ""
    raw_data: dict = field(default_factory=dict)


@dataclass
class RecommendationReport:
    """推荐报告"""
    user_profile: UserProfile
    batch_info: dict = field(default_factory=dict)
    charge_schools: list = field(default_factory=list)  # list[SchoolRecommend]
    stable_schools: list = field(default_factory=list)
    safe_schools: list = field(default_factory=list)
    overall_advice: str = ""
    risk_warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Reranker:
    """重排序器：多因子综合评分"""

    def __init__(self, config: RerankConfig = None):
        self._config = config or RerankConfig()

    @property
    def config(self) -> RerankConfig:
        return self._config

    def rerank(
        self,
        documents: list[VectorDocument],
        user_profile: UserProfile,
    ) -> list[RerankResult]:
        """
        计算综合得分并排序

        Args:
            documents: 候选文档列表
            user_profile: 用户画像

        Returns:
            按综合得分降序的重排序结果
        """
        # 根据用户偏好调整权重
        if user_profile.risk_preference == "保":
            self._config.adjust_for_conservative_user()
        elif user_profile.risk_preference == "冲":
            self._config.adjust_for_aggressive_user()

        results = []

        for doc in documents:
            embedding_score = self._calc_embedding_score(doc)
            score_match = self._calc_score_match(doc, user_profile.score)
            subject_eval = self._calc_subject_eval(doc, user_profile.target_majors)
            pref_match = self._calc_preference_match(doc, user_profile)

            c = self._config
            final_score = (
                c.embedding_weight * embedding_score +
                c.score_match_weight * score_match +
                c.subject_eval_weight * subject_eval +
                c.preference_weight * pref_match
            )

            results.append(RerankResult(
                document=doc,
                final_score=final_score,
                breakdown={
                    "embedding": embedding_score,
                    "score_match": score_match,
                    "subject_eval": subject_eval,
                    "preference": pref_match,
                }
            ))

        results.sort(key=lambda x: x.final_score, reverse=True)
        return results

    def _calc_embedding_score(self, doc: VectorDocument) -> float:
        """语义匹配得分 (0-1)，从embedding搜索的score中获取"""
        # 对于已搜索结果，文档本身不存储score，给中值
        # 实际使用时由 HybridSearch 传入 vector_score
        return 0.5

    def _calc_score_match(self, doc: VectorDocument, user_score: Optional[int]) -> float:
        """
        计算分数匹配度
        理想情况: 用户分数略高于录取线 (1-20分)
        """
        min_score = doc.metadata.get("min_score", 0)
        avg_score = doc.metadata.get("avg_score", 0)

        if not user_score or not min_score:
            return 0.5

        ref_score = avg_score if avg_score else min_score
        diff = user_score - ref_score

        if diff < 0:
            return 0.0
        elif diff <= 5:
            return 1.0
        elif diff <= 20:
            return 0.9
        elif diff <= 50:
            return 0.7
        else:
            return 0.4

    def _calc_subject_eval(self, doc: VectorDocument, target_majors: list[str]) -> float:
        """学科评估得分 (0-1)"""
        subject_eval = doc.metadata.get("subject_eval")
        if not subject_eval:
            return 0.5

        if not target_majors:
            # 无目标专业，取最高评估
            best_grade = max(subject_eval.values()) if subject_eval else ""
            return self._grade_to_score(best_grade)

        # 有目标专业，计算匹配专业的平均评估
        matched_grades = []
        for major in target_majors:
            for key, grade in subject_eval.items():
                if major in key:
                    matched_grades.append(self._grade_to_score(grade))

        if matched_grades:
            return sum(matched_grades) / len(matched_grades)
        return 0.5

    def _grade_to_score(self, grade: str) -> float:
        """学科评估等级转换为得分"""
        grade_map = {
            "A+": 1.0,
            "A": 0.95,
            "A-": 0.9,
            "B+": 0.8,
            "B": 0.7,
            "B-": 0.6,
            "C+": 0.5,
            "C": 0.4,
            "C-": 0.3,
        }
        return grade_map.get(grade, 0.5)

    def _calc_preference_match(self, doc: VectorDocument, user_profile: UserProfile) -> float:
        """用户偏好匹配得分 (0-1)"""
        score = 0.0
        count = 0

        # 地域匹配
        if user_profile.city_preference:
            count += 1
            for city in user_profile.city_preference:
                doc_loc = doc.metadata.get("location", "") + doc.metadata.get("province", "")
                if city in doc_loc:
                    score += 1.0
                    break

        # 省份匹配
        if user_profile.province:
            count += 1
            if user_profile.province == doc.metadata.get("province", ""):
                score += 1.0

        # 学校类型匹配
        if user_profile.school_types:
            count += 1
            doc_tier = doc.metadata.get("tier", "")
            if doc_tier in user_profile.school_types:
                score += 1.0

        # 学历层次匹配
        if user_profile.degree_level:
            count += 1
            if user_profile.degree_level == doc.metadata.get("level", "本科"):
                score += 1.0

        if count == 0:
            return 0.5
        return score / count


class RiskClassifier:
    """冲稳保分类器"""

    CHARGE_THRESHOLD: int = -10
    SAFE_THRESHOLD: int = 10

    def classify(
        self,
        user_score: int,
        rerank_results: list[RerankResult],
    ) -> tuple[list[RiskInfo], list[RiskInfo], list[RiskInfo]]:
        """
        将候选学校分为冲/稳/保三档

        Returns:
            (charge, stable, safe) 三个列表
        """
        charge = []
        stable = []
        safe = []

        for result in rerank_results:
            min_score = result.document.metadata.get("min_score", 0)
            avg_score = result.document.metadata.get("avg_score", min_score)

            ref_score = avg_score if avg_score else min_score
            diff = user_score - ref_score

            if diff < self.CHARGE_THRESHOLD:
                level = "冲"
                desc = f"您的分数比往年录取线低{abs(diff)}分，有机会但需冲刺"
            elif diff <= self.SAFE_THRESHOLD:
                level = "稳"
                desc = f"您的分数与往年录取线相当(差{diff}分)，录取概率较高"
            else:
                level = "保"
                desc = f"您的分数比往年录取线高{diff}分，录取把握很大"

            risk_info = RiskInfo(
                document=result.document,
                level=level,
                score_diff=diff,
                min_score=min_score,
                avg_score=avg_score,
                description=desc,
            )

            if level == "冲":
                charge.append(risk_info)
            elif level == "稳":
                stable.append(risk_info)
            else:
                safe.append(risk_info)

        return charge[:8], stable[:10], safe[:8]


def generate_recommendation_reasons(
    doc: VectorDocument,
    user_profile: UserProfile,
) -> list[str]:
    """生成推荐理由"""
    reasons = []

    # 1. 分数匹配理由
    if user_profile.score:
        avg_score = doc.metadata.get("avg_score", 0)
        min_score = doc.metadata.get("min_score", 0)
        if avg_score:
            score_diff = user_profile.score - avg_score
            if -5 <= score_diff <= 15:
                direction = "高出" if score_diff > 0 else "略低"
                reasons.append(f"分数匹配度高（{direction}{abs(score_diff)}分）")

    # 2. 学科评估理由
    subject_eval = doc.metadata.get("subject_eval")
    if subject_eval:
        for major, grade in subject_eval.items():
            if grade in ["A+", "A", "A-"]:
                reasons.append(f"{major}专业全国顶尖（评估{grade}）")
            elif grade in ["B+", "B"]:
                reasons.append(f"{major}专业较强（评估{grade}）")

    # 3. 地域匹配理由
    if user_profile.city_preference:
        for city in user_profile.city_preference:
            doc_loc = doc.metadata.get("location", "") + doc.metadata.get("province", "")
            if city in doc_loc:
                location = doc.metadata.get("location", doc.metadata.get("province", ""))
                reasons.append(f"位于{location}，符合地域偏好")
                break

    # 4. 学校类型理由
    if user_profile.school_types:
        doc_tier = doc.metadata.get("tier", "")
        if doc_tier in user_profile.school_types:
            reasons.append(f"{doc_tier}院校，符合学校类型要求")

    # 5. 省份匹配理由
    if user_profile.province and user_profile.province == doc.metadata.get("province", ""):
        reasons.append(f"本地院校，认可度高")

    return reasons[:4]


def build_recommendation_report(
    user_profile: UserProfile,
    rerank_results: list[RerankResult],
    batch_info: dict = None,
) -> RecommendationReport:
    """
    构建完整推荐报告

    Args:
        user_profile: 用户画像
        rerank_results: 重排序结果
        batch_info: 批次线信息

    Returns:
        完整推荐报告
    """
    classifier = RiskClassifier()
    charge, stable, safe = classifier.classify(user_profile.score or 0, rerank_results)

    def build_school_recommend(risk_info: RiskInfo) -> SchoolRecommend:
        doc = risk_info.document
        reasons = generate_recommendation_reasons(doc, user_profile)

        # 专业优势
        subject_eval = doc.metadata.get("subject_eval", {})
        major_advantage = ""
        if subject_eval:
            best = max(subject_eval.items(), key=lambda x: {"A+": 5, "A": 4, "A-": 3, "B+": 2, "B": 1}.get(x[1], 0))
            major_advantage = f"{best[0]}评估{best[1]}"

        # 地域优势
        location = doc.metadata.get("location", doc.metadata.get("province", ""))
        location_advantage = f"位于{location}"

        # 风险提示
        risk_warning = risk_info.description

        return SchoolRecommend(
            school_name=doc.metadata.get("school_name", doc.metadata.get("name", "")),
            risk_level=risk_info.level,
            score_diff=risk_info.score_diff,
            min_score=risk_info.min_score,
            avg_score=risk_info.avg_score,
            reasons=reasons,
            major_advantage=major_advantage,
            location_advantage=location_advantage,
            risk_warning=risk_warning,
            raw_data=doc.metadata,
        )

    # 生成总体建议
    advice_parts = []
    if user_profile.score:
        provincial_line = batch_info.get("score_line", 0) if batch_info else 0
        if provincial_line:
            diff = user_profile.score - provincial_line
            advice_parts.append(f"您的分数高出省控线{diff}分")

    if user_profile.target_majors:
        advice_parts.append(f"建议关注{', '.join(user_profile.target_majors)}相关专业")

    if user_profile.city_preference:
        advice_parts.append(f"地域偏好: {', '.join(user_profile.city_preference)}")

    advice_parts.append("建议志愿顺序: 冲刺2-3所 → 稳妥4-5所 → 保底2-3所")

    overall_advice = "；".join(advice_parts) + "。" if advice_parts else ""

    # 风险提示
    warnings = []
    if charge and len(charge) > 5:
        warnings.append("冲刺院校较多，建议适当减少")
    if not stable:
        warnings.append("稳妥院校为空，请扩大分数范围")
    if not safe:
        warnings.append("保底院校为空，存在滑档风险")

    return RecommendationReport(
        user_profile=user_profile,
        batch_info=batch_info or {},
        charge_schools=[build_school_recommend(r) for r in charge],
        stable_schools=[build_school_recommend(r) for r in stable],
        safe_schools=[build_school_recommend(r) for r in safe],
        overall_advice=overall_advice,
        risk_warnings=warnings,
        metadata={"version": "1.0"},
    )
