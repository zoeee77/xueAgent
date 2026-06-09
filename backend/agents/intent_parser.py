"""意图解析智能体：判断用户消息是新咨询还是反馈，并提取筛选条件。"""

import json
import logging
import re
from typing import Optional

from backend.agents.structured_output import parse_structured
from backend.models.agent_output import UserProfile, IntentParseResult
from backend.services.llm_chain import chat_sync

logger = logging.getLogger(__name__)

# 一级城市列表
_FIRST_TIER_CITIES = ["北京", "上海", "广州", "深圳"]

# 风险关键词到偏好值的映射
_RISK_KEYWORDS = {
    "稳": ["稳定", "稳妥", "保稳", "求稳", "保险", "安定"],
    "冲": ["冲刺", "冲击", "冒险", "挑战", "拼一拼", "搏一搏"],
    "保": ["保底", "保稳", "保守", "保险", "求稳"],
}

# 排除专业关键词映射（用户表达的负面意向 → 专业类别）
_EXCLUDE_MAJOR_KEYWORDS = {
    "计算机": ["计算机类", "计算机科学与技术", "软件工程", "信息安全"],
    "编程": ["计算机类", "软件工程"],
    "代码": ["计算机类", "软件工程"],
    "医学": ["医学类", "临床医学", "口腔医学"],
    "师范": ["师范类", "教育学"],
    "土木": ["土木类", "土木工程", "建筑学"],
    "金融": ["金融学", "金融类", "经济学"],
    "法律": ["法学", "法律"],
    "外语": ["外语类", "英语", "翻译"],
    "艺术": ["艺术类", "设计类"],
}

# 偏好省份关键词映射
_PREFER_PROVINCE_KEYWORDS = {
    "一线": _FIRST_TIER_CITIES,
    "一线城市": _FIRST_TIER_CITIES,
    "大城市": _FIRST_TIER_CITIES,
    "北京": ["北京"],
    "上海": ["上海"],
    "广州": ["广州"],
    "深圳": ["深圳"],
    "江浙沪": ["江苏", "浙江", "上海"],
    "长三角": ["江苏", "浙江", "上海"],
    "珠三角": ["广东", "广州", "深圳"],
    "沿海": ["北京", "上海", "广州", "深圳", "江苏", "浙江", "福建", "山东"],
}

# 意图解析的系统提示词
_INTENT_PARSER_SYSTEM = """你是一个高考志愿填报助手中的意图解析模块。你的任务是分析用户的后续消息，判断用户意图并提取筛选条件。

分析规则：
1. 判断消息类型：
   - 如果是全新的咨询请求（与之前的志愿规划无关），is_feedback = false
   - 如果是对已有规划结果的反馈或细化要求，is_feedback = true
2. 提取筛选条件（filter_criteria）：
   - 排除专业：如"不喜欢计算机" → {"exclude_majors": ["计算机类"]}
   - 偏好地区：如"想去一线城市" → {"prefer_provinces": ["北京", "上海", "广州", "深圳"]}
   - 风险偏好：如"想稳定" → {"prefer_risk": "保"}
   - 偏好专业：如"想学电子" → {"prefer_majors": ["电子类"]}
3. 反馈类型（feedback_type）：
   - "exclude"：用户想排除某些选项
   - "prefer"：用户想偏好某些选项
   - "change_risk"：用户想调整风险偏好
   - null：新咨询或无明确反馈类型

请只输出合法的 JSON，不要包含任何额外解释。"""


class IntentParser:
    """解析用户意图，判断消息类型并提取筛选条件。"""

    async def parse(
        self,
        user_message: str,
        current_profile: Optional[UserProfile] = None,
    ) -> IntentParseResult:
        """解析用户消息的意图。

        Args:
            user_message: 用户的输入消息
            current_profile: 当前用户画像（如果有，说明已有上下文）

        Returns:
            IntentParseResult: 包含筛选条件、是否反馈、反馈类型的结果
        """
        # 首先尝试 LLM 解析
        try:
            result = await self._llm_based_parse(user_message, current_profile)
            if result is not None:
                return result
        except Exception as e:
            logger.warning("LLM-based intent parsing failed, falling back to keyword rules: %s", e)

        # LLM 失败时回退到关键词规则
        return self._keyword_based_parse(user_message, current_profile)

    async def _llm_based_parse(
        self,
        user_message: str,
        current_profile: Optional[UserProfile],
    ) -> Optional[IntentParseResult]:
        """使用 LLM 解析用户意图。

        Args:
            user_message: 用户消息
            current_profile: 当前用户画像

        Returns:
            IntentParseResult 或 None（如果解析失败）
        """
        prompt = self._build_llm_prompt(user_message, current_profile)

        try:
            response = chat_sync(
                question=prompt,
                system_prompt=_INTENT_PARSER_SYSTEM,
            )
            result = parse_structured(response, IntentParseResult)
            return result
        except Exception as e:
            logger.warning("Failed to parse LLM response into IntentParseResult: %s", e)
            return None

    def _build_llm_prompt(
        self,
        user_message: str,
        current_profile: Optional[UserProfile],
    ) -> str:
        """构建发送给 LLM 的提示词。

        Args:
            user_message: 用户消息
            current_profile: 当前用户画像

        Returns:
            提示词字符串
        """
        context_info = "当前没有用户画像，这可能是第一次对话。"
        if current_profile is not None:
            context_info = (
                f"当前用户画像：\n"
                f"  - 分数: {current_profile.score}\n"
                f"  - 省份: {current_profile.province}\n"
                f"  - 兴趣: {', '.join(current_profile.interests) if current_profile.interests else '无'}\n"
                f"  - 风险偏好: {current_profile.risk_preference}"
            )

        return (
            f"请分析以下用户消息的意图：\n\n"
            f"## 上下文\n"
            f"{context_info}\n\n"
            f"## 用户消息\n"
            f'"{user_message}"\n\n'
            f"请返回如下字段的 JSON：\n"
            f"- filter_criteria: dict，包含可能的 exclude_majors, prefer_provinces, prefer_risk, prefer_majors\n"
            f"- is_feedback: bool，是否为对已有结果的反馈\n"
            f"- feedback_type: str 或 null，可选值为 'exclude', 'prefer', 'change_risk', null"
        )

    def _keyword_based_parse(
        self,
        user_message: str,
        current_profile: Optional[UserProfile],
    ) -> IntentParseResult:
        """使用关键词规则解析用户意图（兜底逻辑）。

        Args:
            user_message: 用户消息
            current_profile: 当前用户画像

        Returns:
            IntentParseResult: 解析结果
        """
        filter_criteria: dict = {
            "exclude_majors": [],
            "prefer_provinces": [],
            "prefer_majors": [],
        }
        prefer_risk: Optional[str] = None
        is_feedback = current_profile is not None
        feedback_type: Optional[str] = None

        # 提取排除专业
        exclude_majors = self._extract_exclude_majors(user_message)
        if exclude_majors:
            filter_criteria["exclude_majors"] = exclude_majors
            is_feedback = True
            feedback_type = "exclude"

        # 提取偏好省份
        prefer_provinces = self._extract_prefer_provinces(user_message)
        if prefer_provinces:
            filter_criteria["prefer_provinces"] = prefer_provinces
            is_feedback = True
            if feedback_type is None:
                feedback_type = "prefer"

        # 提取偏好专业
        prefer_majors = self._extract_prefer_majors(user_message)
        if prefer_majors:
            filter_criteria["prefer_majors"] = prefer_majors
            is_feedback = True
            if feedback_type is None:
                feedback_type = "prefer"

        # 提取风险偏好
        risk = self._extract_risk_preference(user_message)
        if risk:
            filter_criteria["prefer_risk"] = risk
            is_feedback = True
            feedback_type = "change_risk"

        return IntentParseResult(
            filter_criteria=filter_criteria,
            is_feedback=is_feedback,
            feedback_type=feedback_type,
        )

    def _extract_exclude_majors(self, message: str) -> list[str]:
        """从消息中提取要排除的专业。

        Args:
            message: 用户消息

        Returns:
            要排除的专业类别列表
        """
        exclude_majors = []

        # 检查否定关键词
        negative_patterns = [
            r"不[喜欢想想要爱]",
            r"排除",
            r"避开",
            r"不要",
            r"别选",
            r"拒绝",
            r"反感",
            r"讨厌",
        ]

        has_negative = any(re.search(pattern, message) for pattern in negative_patterns)
        if not has_negative:
            return exclude_majors

        # 匹配专业关键词
        for keyword, majors in _EXCLUDE_MAJOR_KEYWORDS.items():
            if keyword in message:
                exclude_majors.extend(majors)

        # 去重
        return list(dict.fromkeys(exclude_majors))

    def _extract_prefer_provinces(self, message: str) -> list[str]:
        """从消息中提取偏好省份。

        Args:
            message: 用户消息

        Returns:
            偏好的省份列表
        """
        prefer_provinces = []

        # 偏好地区的触发词
        prefer_triggers = [
            r"想去?",
            r"希望.*在",
            r"倾向.*在",
            r"偏好",
            r"优先",
            r"最好.*在",
        ]

        has_prefer = any(re.search(pattern, message) for pattern in prefer_triggers)

        for keyword, provinces in _PREFER_PROVINCE_KEYWORDS.items():
            if keyword in message:
                prefer_provinces.extend(provinces)

        # 如果有明确触发词但没有匹配到关键词，尝试直接匹配省份名
        if not prefer_provinces and has_prefer:
            all_provinces = set()
            for provinces in _PREFER_PROVINCE_KEYWORDS.values():
                all_provinces.update(provinces)
            for province in all_provinces:
                if province in message:
                    prefer_provinces.append(province)

        return list(dict.fromkeys(prefer_provinces))

    def _extract_prefer_majors(self, message: str) -> list[str]:
        """从消息中提取偏好专业。

        Args:
            message: 用户消息

        Returns:
            偏好的专业列表
        """
        prefer_majors = []

        # 偏好触发词
        prefer_triggers = [
            r"想学",
            r"喜欢",
            r"感兴趣",
            r"倾向",
            r"偏好",
            r"优先考虑",
            r"希望.*专业",
        ]

        has_prefer = any(re.search(pattern, message) for pattern in prefer_triggers)
        if not has_prefer:
            return prefer_majors

        # 常见专业关键词映射
        major_keywords = {
            "计算机": ["计算机类"],
            "电子": ["电子信息类", "电子科学与技术"],
            "自动化": ["自动化类"],
            "机械": ["机械类"],
            "医学": ["医学类"],
            "金融": ["金融学", "金融类"],
            "法律": ["法学"],
            "师范": ["师范类"],
            "建筑": ["建筑类"],
            "通信": ["通信工程", "电子信息类"],
            "电气": ["电气工程", "电气类"],
        }

        for keyword, majors in major_keywords.items():
            if keyword in message:
                prefer_majors.extend(majors)

        return list(dict.fromkeys(prefer_majors))

    def _extract_risk_preference(self, message: str) -> Optional[str]:
        """从消息中提取风险偏好。

        Args:
            message: 用户消息

        Returns:
            "冲"、"稳"、"保" 或 None
        """
        # 检查风险关键词
        for risk_level, keywords in _RISK_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message:
                    return risk_level

        # 特殊表达
        if any(w in message for w in ["求稳", "稳妥", "稳定", "安稳"]):
            return "保"
        if any(w in message for w in ["冲一冲", "拼一拼", "搏一搏", "冒险"]):
            return "冲"

        return None
