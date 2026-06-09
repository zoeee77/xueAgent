"""方案生成智能体：基于多角色分析和用户画像生成冲/稳/保三套志愿方案。"""

import logging

from backend.agents.structured_output import parse_structured
from backend.models.agent_output import (
    UserProfile,
    DataRetrievalResult,
    MultiRoleResult,
    PlanOption,
    PlanResult,
)
from backend.services.knowledge_base import KnowledgeBase
from backend.fallback.fallback_handler import FallbackHandler
from backend.services.llm_chain import chat_sync

logger = logging.getLogger(__name__)

# 冲/稳/保三档的分数偏移范围
_TIER_OFFSETS: dict[str, tuple[int, int]] = {
    "冲": (0, 20),       # 冲：profile score + 0~20
    "稳": (-15, 5),      # 稳：profile score - 15 ~ +5
    "保": (-50, -20),    # 保：profile score - 50 ~ -20
}


class Planner:
    """方案生成智能体。

    接收多角色分析结果、数据检索结果和用户画像，生成 3 套志愿方案：
    冲（激进）、稳（平衡）、保（保守）。
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.fallback = FallbackHandler()

    async def generate(
        self,
        role_result: MultiRoleResult,
        data_result: DataRetrievalResult,
        profile: UserProfile,
    ) -> PlanResult:
        """生成冲/稳/保三套志愿方案。

        Args:
            role_result: 多角色分析结果
            data_result: 数据检索结果
            profile: 用户画像

        Returns:
            PlanResult: 包含 3 个 PlanOption 的方案结果
        """
        try:
            prompt = self._build_planner_prompt(role_result, data_result, profile)
            system_prompt = (
                "你是一位资深的高考志愿填报规划专家。"
                "请根据提供的用户画像、数据检索结果和多角色分析意见，"
                "生成冲/稳/保三套志愿方案。"
                "每套方案必须包含风险等级、推荐专业、院校列表、推荐理由和预期分数线。"
                "请只输出合法的 JSON，不要包含任何额外解释。"
            )

            response = chat_sync(question=prompt, system_prompt=system_prompt)
            plan_result = parse_structured(response, PlanResult)

            # 校验必须返回 3 个选项
            if len(plan_result.options) == 3:
                return plan_result

            logger.warning(
                "LLM returned %d options instead of 3, falling back to rule-based",
                len(plan_result.options),
            )
        except Exception as e:
            logger.warning("LLM plan generation failed: %s", e)

        return self._rule_based_generate(role_result, data_result, profile)

    def _rule_based_generate(
        self,
        role_result: MultiRoleResult,
        data_result: DataRetrievalResult,
        profile: UserProfile,
    ) -> PlanResult:
        """基于规则降级生成方案（LLM 失败时的兜底逻辑）。"""
        options = []
        for tier, offset in _TIER_OFFSETS.items():
            universities = self._match_universities(profile.score, offset, self.kb)
            major = self._select_major_for_tier(tier, data_result)
            expected_score = profile.score + (offset[0] + offset[1]) // 2
            reason = self._build_reason(tier, major, profile, role_result)

            options.append(
                PlanOption(
                    risk_level=tier,
                    major=major,
                    universities=universities,
                    reason=reason,
                    expected_score=expected_score,
                )
            )

        return PlanResult(options=options)

    def _match_universities(
        self,
        score: int,
        range_offset: tuple[int, int],
        kb: KnowledgeBase,
    ) -> list[str]:
        """根据分数和偏移量在知识库中查找匹配的院校。

        Args:
            score: 用户高考分数
            range_offset: 分数偏移范围 (low, high)，相对于 score 的偏移
            kb: 知识库实例

        Returns:
            匹配的院校名称列表（最多 5 所）
        """
        low = score + range_offset[0]
        high = score + range_offset[1]
        universities = kb.all_universities

        matched = []
        for name, info in universities.items():
            min_score = info.get("min_score_2025", 0)
            if low <= min_score <= high:
                matched.append((name, min_score))

        # 按最低录取分排序
        matched.sort(key=lambda x: x[1])

        # 取最多 5 所
        return [name for name, _ in matched[:5]]

    def _build_planner_prompt(
        self,
        role_result: MultiRoleResult,
        data_result: DataRetrievalResult,
        profile: UserProfile,
    ) -> str:
        """构建用于 LLM 方案生成的提示词。

        Args:
            role_result: 多角色分析结果
            data_result: 数据检索结果
            profile: 用户画像

        Returns:
            完整的提示词字符串
        """
        profile_str = self._format_profile(profile)
        data_str = self._format_data_result(data_result)
        role_str = self._format_role_result(role_result)

        return (
            f"请基于以下信息生成冲/稳/保三套志愿方案。\n\n"
            f"## 用户画像\n"
            f"{profile_str}\n\n"
            f"## 数据检索结果\n"
            f"{data_str}\n\n"
            f"## 多角色分析意见\n"
            f"{role_str}\n\n"
            f"## 方案生成要求\n"
            f"请生成 3 套方案，分别为「冲」「稳」「保」：\n"
            f"1. **冲**：推荐录取分数略高于用户分数的院校（{profile.score} ~ {profile.score + 20} 分范围）\n"
            f"2. **稳**：推荐录取分数与用户分数匹配的院校（{profile.score - 15} ~ {profile.score + 5} 分范围）\n"
            f"3. **保**：推荐录取分数低于用户分数的院校（{profile.score - 50} ~ {profile.score - 20} 分范围）\n\n"
            f"每套方案需结合多角色分析意见选择专业，并从知识库中匹配对应分数段的院校。\n\n"
            f"请以 JSON 格式输出，必须符合以下结构：\n"
            f"{{\n"
            f'  "options": [\n'
            f"    {{\n"
            f'      "risk_level": "冲",\n'
            f'      "major": "推荐专业名称",\n'
            f'      "universities": ["院校1", "院校2", "院校3"],\n'
            f'      "reason": "推荐理由",\n'
            f'      "expected_score": 预期录取分数\n'
            f"    }},\n"
            f"    {{ ... }}\n"
            f"  ]\n"
            f"}}\n\n"
            f"注意：\n"
            f"1. risk_level 必须是「冲」「稳」「保」之一\n"
            f"2. universities 至少包含 1 所院校，最多 5 所\n"
            f"3. expected_score 必须是整数\n"
            f"4. 请只输出合法 JSON，不要包含其他内容"
        )

    def _format_profile(self, profile: UserProfile) -> str:
        """格式化用户画像。"""
        lines = [
            f"高考分数: {profile.score}",
            f"所在省份: {profile.province}",
            f"兴趣方向: {', '.join(profile.interests) if profile.interests else '无'}",
        ]
        if profile.personality:
            lines.append(f"性格特点: {profile.personality}")
        if profile.family_resources:
            lines.append(f"家庭资源: {profile.family_resources}")
        lines.append(f"风险偏好: {profile.risk_preference}")
        if profile.constraints:
            lines.append(f"约束条件: {', '.join(profile.constraints)}")
        return "\n".join(lines)

    def _format_data_result(self, data_result: DataRetrievalResult) -> str:
        """格式化数据检索结果。"""
        parts = []
        if data_result.majors:
            parts.append("## 推荐专业")
            for i, m in enumerate(data_result.majors, 1):
                parts.append(
                    f"{i}. {m.get('name', '未知')} - "
                    f"就业率: {m.get('employment_rate', '未知')}, "
                    f"平均薪资: {m.get('avg_salary', '未知')}元"
                )
                if m.get("description"):
                    parts.append(f"   描述: {m['description']}")

        if data_result.industries:
            parts.append("\n## 相关行业")
            for i, ind in enumerate(data_result.industries, 1):
                parts.append(
                    f"{i}. {ind.get('name', '未知')} - "
                    f"门槛: {ind.get('entry_barrier', '未知')}, "
                    f"薪资范围: {ind.get('salary_range', '未知')}"
                )

        if data_result.filter_reason:
            parts.append(f"\n筛选理由: {data_result.filter_reason}")

        return "\n".join(parts) if parts else "暂无数据检索结果"

    def _format_role_result(self, role_result: MultiRoleResult) -> str:
        """格式化多角色分析结果。"""
        parts = []
        for opinion in role_result.opinions:
            parts.append(
                f"### {opinion.role_name}\n"
                f"推荐: {opinion.recommendation}\n"
                f"理由: {opinion.reasoning}\n"
                f"打分: {opinion.score}/100"
            )

        if role_result.consensus:
            parts.append(f"\n### 共识\n{role_result.consensus}")

        if role_result.conflicts:
            parts.append("\n### 分歧")
            for conflict in role_result.conflicts:
                parts.append(f"- {conflict}")

        return "\n".join(parts) if parts else "暂无多角色分析结果"

    def _select_major_for_tier(
        self, tier: str, data_result: DataRetrievalResult
    ) -> str:
        """为指定档位选择一个专业（规则降级时使用）。

        Args:
            tier: 档位（冲/稳/保）
            data_result: 数据检索结果

        Returns:
            专业名称
        """
        majors = data_result.majors
        if not majors:
            return "综合评估"

        if tier == "冲":
            # 冲档选排名靠前的专业（更有挑战）
            return majors[0].get("name", "未知")
        elif tier == "稳":
            # 稳档选中间位置的专业
            idx = min(len(majors) // 2, len(majors) - 1)
            return majors[idx].get("name", "未知")
        else:
            # 保档选排名靠后但稳妥的专业
            return majors[-1].get("name", "未知")

    def _build_reason(
        self,
        tier: str,
        major: str,
        profile: UserProfile,
        role_result: MultiRoleResult,
    ) -> str:
        """生成推荐理由（规则降级时使用）。

        Args:
            tier: 档位
            major: 专业名称
            profile: 用户画像
            role_result: 多角色分析结果

        Returns:
            推荐理由字符串
        """
        tier_desc = {
            "冲": "冲刺型方案，适合在分数基础上适当拔高，挑战更高水平院校",
            "稳": "稳妥型方案，专业与分数匹配度高，录取概率较大",
            "保": "保底型方案，确保有学可上，降低落榜风险",
        }.get(tier, "")

        # 从角色意见中提取相关观点
        role_mentions = []
        for opinion in role_result.opinions[:2]:  # 取前两个角色的意见
            if major in opinion.recommendation:
                role_mentions.append(f"{opinion.role_name}认可该专业")

        reason_parts = [tier_desc]
        if role_mentions:
            reason_parts.append("、".join(role_mentions))
        reason_parts.append(f"结合您的兴趣方向（{', '.join(profile.interests) if profile.interests else '综合评估'}）")

        return "；".join(reason_parts)
