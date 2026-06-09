"""精炼器智能体：根据用户反馈意图调整志愿方案。"""

import logging
from typing import Optional

from backend.agents.structured_output import parse_structured
from backend.models.agent_output import (
    UserProfile,
    IntentParseResult,
    PlanResult,
    PlanOption,
    RefineResult,
)
from backend.services.knowledge_base import KnowledgeBase
from backend.services.llm_chain import chat_sync

logger = logging.getLogger(__name__)

# 冲/稳/保三档的分数偏移范围
_TIER_OFFSETS: dict[str, tuple[int, int]] = {
    "冲": (0, 20),
    "稳": (-15, 5),
    "保": (-50, -20),
}

_RISK_MAP = {
    "冲": "冲",
    "稳": "稳",
    "保": "保",
}

_REFINER_SYSTEM = """你是一个高考志愿填报方案的精炼专家。你的任务是根据用户的反馈意图，调整已有的志愿方案。

调整规则：
1. 如果用户要求排除某些专业（exclude_majors），从方案中移除或替换这些专业
2. 如果用户偏好某些地区（prefer_provinces），优先选择该地区的院校
3. 如果用户调整风险偏好（prefer_risk），重新调整冲/稳/保方案
4. 如果用户偏好某些专业（prefer_majors），优先选择这些专业

请返回调整后的方案，并记录所做的变更。只输出合法的 JSON，不要包含任何额外解释。"""


class Refiner:
    """精炼器：根据用户反馈意图调整志愿方案。"""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    async def refine(
        self,
        plan_result: PlanResult,
        intent_result: IntentParseResult,
        profile: UserProfile,
        kb: Optional[KnowledgeBase] = None,
    ) -> RefineResult:
        """根据用户反馈意图精炼志愿方案。

        Args:
            plan_result: 当前的志愿方案
            intent_result: 意图解析结果
            profile: 用户画像
            kb: 知识库（可选，优先使用构造时传入的）

        Returns:
            RefineResult: 包含更新后的方案和变更记录
        """
        active_kb = kb or self.kb
        changes_made: list[str] = []

        # 首先尝试 LLM 精炼
        try:
            llm_result = await self._llm_based_refine(
                plan_result, intent_result, profile, active_kb
            )
            if llm_result is not None and llm_result.updated_plan.options:
                return llm_result
        except Exception as e:
            logger.warning("LLM-based refinement failed, falling back to rule-based: %s", e)

        # 规则精炼
        updated_options = list(plan_result.options)

        filter_criteria = intent_result.filter_criteria
        exclude_majors = filter_criteria.get("exclude_majors", [])
        prefer_provinces = filter_criteria.get("prefer_provinces", [])
        prefer_majors = filter_criteria.get("prefer_majors", [])
        prefer_risk = filter_criteria.get("prefer_risk")

        # 1. 排除专业
        if exclude_majors:
            updated_options, change = self._apply_exclude_majors(
                updated_options, exclude_majors, profile, active_kb
            )
            if change:
                changes_made.append(change)

        # 2. 偏好专业
        if prefer_majors:
            updated_options, change = self._apply_prefer_majors(
                updated_options, prefer_majors, profile, active_kb
            )
            if change:
                changes_made.append(change)

        # 3. 偏好省份
        if prefer_provinces:
            updated_options, change = self._apply_prefer_provinces(
                updated_options, prefer_provinces, profile, active_kb
            )
            if change:
                changes_made.append(change)

        # 4. 调整风险偏好
        if prefer_risk:
            updated_options, change = self._apply_risk_adjustment(
                updated_options, prefer_risk, profile, active_kb
            )
            if change:
                changes_made.append(change)

        updated_plan = PlanResult(options=updated_options)

        return RefineResult(
            updated_plan=updated_plan,
            changes_made=changes_made,
        )

    async def _llm_based_refine(
        self,
        plan_result: PlanResult,
        intent_result: IntentParseResult,
        profile: UserProfile,
        kb: KnowledgeBase,
    ) -> Optional[RefineResult]:
        """使用 LLM 精炼方案。

        Args:
            plan_result: 当前方案
            intent_result: 意图解析结果
            profile: 用户画像
            kb: 知识库

        Returns:
            RefineResult 或 None
        """
        prompt = self._build_refiner_prompt(plan_result, intent_result, profile, kb)

        try:
            response = chat_sync(
                question=prompt,
                system_prompt=_REFINER_SYSTEM,
            )
            result = parse_structured(response, RefineResult)
            return result
        except Exception as e:
            logger.warning("Failed to parse LLM response into RefineResult: %s", e)
            return None

    def _build_refiner_prompt(
        self,
        plan_result: PlanResult,
        intent_result: IntentParseResult,
        profile: UserProfile,
        kb: KnowledgeBase,
    ) -> str:
        """构建精炼器的 LLM 提示词。

        Args:
            plan_result: 当前方案
            intent_result: 意图解析结果
            profile: 用户画像
            kb: 知识库

        Returns:
            提示词字符串
        """
        plan_str = self._format_plan(plan_result)
        intent_str = self._format_intent(intent_result)
        profile_str = self._format_profile(profile)

        return (
            f"请根据用户反馈精炼以下志愿方案：\n\n"
            f"## 用户画像\n"
            f"{profile_str}\n\n"
            f"## 当前方案\n"
            f"{plan_str}\n\n"
            f"## 用户反馈意图\n"
            f"{intent_str}\n\n"
            f"请返回调整后的方案 JSON，必须符合以下结构：\n"
            f"{{\n"
            f'  "updated_plan": {{\n'
            f'    "options": [\n'
            f"      {{\n"
            f'        "risk_level": "冲/稳/保",\n'
            f'        "major": "专业名称",\n'
            f'        "universities": ["院校1", "院校2"],\n'
            f'        "reason": "推荐理由",\n'
            f'        "expected_score": 预期分数\n'
            f"      }}\n"
            f"    ]\n"
            f"  }},\n"
            f'  "changes_made": ["变更1", "变更2"]\n'
            f"}}\n\n"
            f"注意：\n"
            f"1. 必须保留冲/稳/保三套方案\n"
            f"2. 根据用户反馈调整专业和院校选择\n"
            f"3. changes_made 描述具体做了哪些调整"
        )

    def _format_plan(self, plan_result: PlanResult) -> str:
        """格式化当前方案。"""
        parts = []
        for i, opt in enumerate(plan_result.options, 1):
            parts.append(
                f"{i}. [{opt.risk_level}] 专业: {opt.major}, "
                f"院校: {', '.join(opt.universities)}, "
                f"预期分数: {opt.expected_score}, "
                f"理由: {opt.reason}"
            )
        return "\n".join(parts) if parts else "暂无方案"

    def _format_intent(self, intent_result: IntentParseResult) -> str:
        """格式化意图解析结果。"""
        parts = []
        if intent_result.is_feedback:
            parts.append(f"反馈类型: {intent_result.feedback_type}")
        criteria = intent_result.filter_criteria
        if criteria.get("exclude_majors"):
            parts.append(f"排除专业: {', '.join(criteria['exclude_majors'])}")
        if criteria.get("prefer_majors"):
            parts.append(f"偏好专业: {', '.join(criteria['prefer_majors'])}")
        if criteria.get("prefer_provinces"):
            parts.append(f"偏好地区: {', '.join(criteria['prefer_provinces'])}")
        if criteria.get("prefer_risk"):
            parts.append(f"风险偏好: {criteria['prefer_risk']}")
        return "\n".join(parts) if parts else "无明确筛选条件"

    def _format_profile(self, profile: UserProfile) -> str:
        """格式化用户画像。"""
        lines = [
            f"分数: {profile.score}",
            f"省份: {profile.province}",
            f"兴趣: {', '.join(profile.interests) if profile.interests else '无'}",
            f"风险偏好: {profile.risk_preference}",
        ]
        return "\n".join(lines)

    def _apply_exclude_majors(
        self,
        options: list[PlanOption],
        exclude_majors: list[str],
        profile: UserProfile,
        kb: KnowledgeBase,
    ) -> tuple[list[PlanOption], Optional[str]]:
        """应用排除专业条件。

        Args:
            options: 当前方案选项
            exclude_majors: 要排除的专业列表
            profile: 用户画像
            kb: 知识库

        Returns:
            (更新后的选项列表, 变更记录)
        """
        changes = []
        new_options = []

        for option in options:
            # 检查当前专业是否被排除
            should_exclude = False
            for excluded in exclude_majors:
                if excluded in option.major or option.major in excluded:
                    should_exclude = True
                    break

            if should_exclude:
                # 查找替代专业
                replacement = self._find_alternative_major(
                    exclude_majors, option.risk_level, kb
                )
                if replacement:
                    reason = (
                        f"原专业 {option.major} 被用户排除，"
                        f"替换为 {replacement}（{option.risk_level}档推荐）"
                    )
                    new_option = PlanOption(
                        risk_level=option.risk_level,
                        major=replacement,
                        universities=option.universities,
                        reason=reason,
                        expected_score=option.expected_score,
                    )
                    new_options.append(new_option)
                    changes.append(f"排除 {option.major}，替换为 {replacement}")
                else:
                    changes.append(f"排除 {option.major}，暂无合适替代")
            else:
                new_options.append(option)

        change_record = f"排除专业: {', '.join(exclude_majors)}" if changes else None
        return new_options, change_record

    def _apply_prefer_majors(
        self,
        options: list[PlanOption],
        prefer_majors: list[str],
        profile: UserProfile,
        kb: KnowledgeBase,
    ) -> tuple[list[PlanOption], Optional[str]]:
        """应用偏好专业条件。

        Args:
            options: 当前方案选项
            prefer_majors: 偏好的专业列表
            profile: 用户画像
            kb: 知识库

        Returns:
            (更新后的选项列表, 变更记录)
        """
        changes = []
        new_options = []

        for option in options:
            # 检查是否已有偏好专业
            already_preferred = any(
                pref in option.major or option.major in pref
                for pref in prefer_majors
            )
            if already_preferred:
                # 更新推荐理由
                new_reason = f"{option.reason}；符合用户偏好的专业方向"
                new_option = PlanOption(
                    risk_level=option.risk_level,
                    major=option.major,
                    universities=option.universities,
                    reason=new_reason,
                    expected_score=option.expected_score,
                )
                new_options.append(new_option)
            else:
                # 尝试替换为偏好专业
                replacement = prefer_majors[0]  # 使用第一个偏好专业
                major_info = kb.query_major(replacement)
                if major_info:
                    reason = (
                        f"根据用户偏好选择 {replacement}，"
                        f"{major_info.get('description', '')}"
                    )
                else:
                    reason = f"根据用户偏好选择 {replacement}"

                new_option = PlanOption(
                    risk_level=option.risk_level,
                    major=replacement,
                    universities=option.universities,
                    reason=reason,
                    expected_score=option.expected_score,
                )
                new_options.append(new_option)
                changes.append(f"调整为偏好专业 {replacement}")

        change_record = f"偏好专业: {', '.join(prefer_majors)}" if changes else None
        return new_options, change_record

    def _apply_prefer_provinces(
        self,
        options: list[PlanOption],
        prefer_provinces: list[str],
        profile: UserProfile,
        kb: KnowledgeBase,
    ) -> tuple[list[PlanOption], Optional[str]]:
        """应用偏好省份条件。

        Args:
            options: 当前方案选项
            prefer_provinces: 偏好的省份列表
            profile: 用户画像
            kb: 知识库

        Returns:
            (更新后的选项列表, 变更记录)
        """
        changes = []
        new_options = []

        for option in options:
            # 查找偏好省份的替代院校
            new_universities = self._find_universities_in_provinces(
                prefer_provinces,
                profile.score,
                _TIER_OFFSETS.get(option.risk_level, (-10, 10)),
                kb,
            )

            if new_universities:
                reason = (
                    f"优先选择用户偏好地区（{', '.join(prefer_provinces)}）的院校"
                )
                new_option = PlanOption(
                    risk_level=option.risk_level,
                    major=option.major,
                    universities=new_universities,
                    reason=reason,
                    expected_score=option.expected_score,
                )
                new_options.append(new_option)
                changes.append(
                    f"调整院校至偏好地区: {', '.join(prefer_provinces)}"
                )
            else:
                new_options.append(option)

        change_record = (
            f"偏好地区: {', '.join(prefer_provinces)}" if changes else None
        )
        return new_options, change_record

    def _apply_risk_adjustment(
        self,
        options: list[PlanOption],
        prefer_risk: str,
        profile: UserProfile,
        kb: KnowledgeBase,
    ) -> tuple[list[PlanOption], Optional[str]]:
        """应用风险偏好调整。

        Args:
            options: 当前方案选项
            prefer_risk: 偏好的风险偏好（冲/稳/保）
            profile: 用户画像
            kb: 知识库

        Returns:
            (更新后的选项列表, 变更记录)
        """
        changes = []

        # 如果用户偏好某一档位，需要调整方案重心
        # 例如偏好"保"，则三档都往更保守的方向偏移
        if prefer_risk == "保":
            # 冲 -> 稳, 稳 -> 保, 保 -> 更保
            new_options = []
            for option in options:
                if option.risk_level == "冲":
                    new_tier = "稳"
                elif option.risk_level == "稳":
                    new_tier = "保"
                else:
                    new_tier = "保"

                offset = _TIER_OFFSETS[new_tier]
                universities = self._match_universities(
                    profile.score, offset, kb
                )
                expected_score = profile.score + (offset[0] + offset[1]) // 2
                reason = f"根据用户风险偏好调整，原 {option.risk_level} 档改为 {new_tier} 档"

                new_options.append(
                    PlanOption(
                        risk_level=new_tier,
                        major=option.major,
                        universities=universities,
                        reason=reason,
                        expected_score=expected_score,
                    )
                )
            changes.append(f"风险偏好调整为更保守（整体向 {prefer_risk} 倾斜）")
            return new_options, changes[0]

        elif prefer_risk == "冲":
            # 保 -> 稳, 稳 -> 冲, 冲 -> 更冲
            new_options = []
            for option in options:
                if option.risk_level == "冲":
                    new_tier = "冲"
                elif option.risk_level == "稳":
                    new_tier = "冲"
                else:
                    new_tier = "稳"

                offset = _TIER_OFFSETS[new_tier]
                universities = self._match_universities(
                    profile.score, offset, kb
                )
                expected_score = profile.score + (offset[0] + offset[1]) // 2
                reason = f"根据用户风险偏好调整，原 {option.risk_level} 档改为 {new_tier} 档"

                new_options.append(
                    PlanOption(
                        risk_level=new_tier,
                        major=option.major,
                        universities=universities,
                        reason=reason,
                        expected_score=expected_score,
                    )
                )
            changes.append(f"风险偏好调整为更激进（整体向 {prefer_risk} 倾斜）")
            return new_options, changes[0]

        elif prefer_risk == "稳":
            # 全部调整为稳
            new_options = []
            offset = _TIER_OFFSETS["稳"]
            for option in options:
                universities = self._match_universities(
                    profile.score, offset, kb
                )
                expected_score = profile.score + (offset[0] + offset[1]) // 2
                reason = f"根据用户风险偏好调整为稳妥方案"

                new_options.append(
                    PlanOption(
                        risk_level="稳",
                        major=option.major,
                        universities=universities,
                        reason=reason,
                        expected_score=expected_score,
                    )
                )
            changes.append(f"风险偏好调整为稳妥（全部方案统一为 稳 档）")
            return new_options, changes[0]

        return options, None

    def _find_alternative_major(
        self,
        exclude_majors: list[str],
        tier: str,
        kb: KnowledgeBase,
    ) -> Optional[str]:
        """查找替代专业。

        Args:
            exclude_majors: 要排除的专业
            tier: 当前档位
            kb: 知识库

        Returns:
            替代专业名称或 None
        """
        all_majors = kb.all_majors
        exclude_set = set(exclude_majors)

        # 按就业率排序，找到不在排除列表中的专业
        candidates = []
        for name, info in all_majors.items():
            if not any(exc in name or name in exc for exc in exclude_set):
                candidates.append((
                    name,
                    info.get("employment_rate", 0),
                    info.get("avg_salary", 0),
                ))

        if not candidates:
            return None

        # 按就业率和薪资综合排序
        candidates.sort(key=lambda x: x[1] * 0.6 + x[2] / 10000 * 0.4, reverse=True)

        # 根据档位选择
        if tier == "冲":
            return candidates[0][0] if candidates else None
        elif tier == "稳":
            idx = min(len(candidates) // 2, len(candidates) - 1)
            return candidates[idx][0] if candidates else None
        else:
            # 保档选择就业率高的
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0] if candidates else None

    def _find_universities_in_provinces(
        self,
        provinces: list[str],
        score: int,
        range_offset: tuple[int, int],
        kb: KnowledgeBase,
    ) -> list[str]:
        """在指定省份中查找匹配的院校。

        Args:
            provinces: 省份列表
            score: 用户分数
            range_offset: 分数偏移
            kb: 知识库

        Returns:
            匹配的院校列表
        """
        low = score + range_offset[0]
        high = score + range_offset[1]
        all_universities = kb.all_universities

        matched = []
        for name, info in all_universities.items():
            uni_province = info.get("province", "")
            min_score = info.get("min_score_2025", 0)
            if uni_province in provinces and low <= min_score <= high:
                matched.append((name, min_score))

        matched.sort(key=lambda x: x[1])
        return [name for name, _ in matched[:5]]

    def _match_universities(
        self,
        score: int,
        range_offset: tuple[int, int],
        kb: KnowledgeBase,
    ) -> list[str]:
        """根据分数和偏移量查找匹配的院校。

        Args:
            score: 用户分数
            range_offset: 分数偏移
            kb: 知识库

        Returns:
            匹配的院校列表
        """
        low = score + range_offset[0]
        high = score + range_offset[1]
        all_universities = kb.all_universities

        matched = []
        for name, info in all_universities.items():
            min_score = info.get("min_score_2025", 0)
            if low <= min_score <= high:
                matched.append((name, min_score))

        matched.sort(key=lambda x: x[1])
        return [name for name, _ in matched[:5]]
