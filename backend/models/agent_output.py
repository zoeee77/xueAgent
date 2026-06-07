"""智能体结构化输出模型。"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """用户画像分析结果。"""

    score: int
    province: str
    interests: list[str]
    personality: Optional[str] = None
    family_resources: Optional[str] = None
    risk_preference: Literal["冲", "稳", "保"]
    constraints: list[str] = Field(default_factory=list)
    subject_type: Optional[str] = None
    target_majors: list[str] = Field(default_factory=list)
    city_preference: Optional[str] = None
    school_types: list[str] = Field(default_factory=list)
    degree_level: str = "本科"


class DataRetrievalResult(BaseModel):
    """数据检索结果。"""

    majors: list[dict] = Field(
        default_factory=list,
        description="专业列表，每项包含 name, employment_rate, avg_salary, description",
    )
    industries: list[dict] = Field(
        default_factory=list,
        description="行业列表，每项包含 name, entry_barrier, salary_range, description",
    )
    filter_reason: str = ""


class RoleOpinion(BaseModel):
    """单个角色的意见。"""

    role_name: str
    recommendation: str
    reasoning: str
    score: int = Field(ge=0, le=100)


class MultiRoleResult(BaseModel):
    """多角色推理结果。"""

    opinions: list[RoleOpinion] = Field(default_factory=list)
    consensus: str = ""
    conflicts: list[str] = Field(default_factory=list)


class PlanOption(BaseModel):
    """单个志愿方案选项。"""

    risk_level: Literal["冲", "稳", "保"]
    major: str
    universities: list[str]
    reason: str
    expected_score: int


class PlanResult(BaseModel):
    """规划器输出结果。"""

    options: list[PlanOption] = Field(default_factory=list)


class RankItem(BaseModel):
    """单个排序项。"""

    option: PlanOption
    total_score: float = Field(ge=0.0, le=100.0)
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="维度评分：employment, match, risk, salary, growth",
    )
    rank: int


class RankResult(BaseModel):
    """排序器输出结果。"""

    ranked_list: list[RankItem] = Field(default_factory=list)


class DevilAdvocateResult(BaseModel):
    """反对者分析结果。"""

    objections: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    alternative_suggestions: list[str] = Field(default_factory=list)


class ExplanationResult(BaseModel):
    """解释器输出结果。"""

    why_recommended: str
    why_first: str
    why_not_others: str
    risk_warnings: list[str] = Field(default_factory=list)


class IntentParseResult(BaseModel):
    """意图解析结果。"""

    filter_criteria: dict = Field(
        default_factory=dict,
        description="筛选条件，如 {\"exclude_majors\": [], \"prefer_provinces\": [], \"prefer_risk\": \"\"}",
    )
    is_feedback: bool = False
    feedback_type: Optional[Literal["exclude", "prefer", "change_risk"]] = None


class RefineResult(BaseModel):
    """精炼器输出结果。"""

    updated_plan: PlanResult
    changes_made: list[str] = Field(default_factory=list)
