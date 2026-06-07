"""用户画像分析智能体：根据用户输入提取约束、偏好和风险倾向。"""

import json
import logging
from typing import Optional

from backend.agents.structured_output import parse_structured
from backend.models.agent_output import UserProfile
from backend.services.llm_chain import chat_sync

logger = logging.getLogger(__name__)

_USER_PROFILER_SYSTEM = """你是一个高考志愿填报助手中的用户画像分析模块。你的任务是根据用户提供的信息，提取用户画像并返回结构化的 JSON。

分析规则：
1. 根据高考分数（score）判断风险偏好（risk_preference）：
   - score >= 600：风险偏好为"冲"（可以冲击更好的学校）
   - score 在 550-599 之间：风险偏好为"稳"（稳妥选择为主）
   - score < 550：风险偏好为"保"（保底学校为主）
2. 从兴趣（interests）、性格（personality）中提取约束条件（constraints）
3. 家庭资源（family_resources）可能影响专业选择倾向

请只输出合法的 JSON，不要包含任何额外解释。"""


class UserProfiler:
    """分析用户输入并构建用户画像。"""

    async def analyze(self, user_input: dict) -> UserProfile:
        """分析用户输入，返回用户画像。

        Args:
            user_input: 包含 score, province, interests, personality, family_resources 的字典

        Returns:
            UserProfile: 用户画像分析结果
        """
        # 首先尝试 LLM 分析
        try:
            profile = await self._llm_based_profile(user_input)
            if profile is not None:
                return profile
        except Exception as e:
            logger.warning("LLM-based profiling failed, falling back to rule-based: %s", e)

        # LLM 失败时回退到规则分析
        return self._rule_based_profile(user_input)

    async def _llm_based_profile(self, user_input: dict) -> Optional[UserProfile]:
        """使用 LLM 分析用户画像。

        Args:
            user_input: 用户输入字典

        Returns:
            UserProfile 或 None（如果解析失败）
        """
        prompt = self._build_llm_prompt(user_input)

        try:
            response = chat_sync(
                question=prompt,
                system_prompt=_USER_PROFILER_SYSTEM,
            )
            profile = parse_structured(response, UserProfile)
            return profile
        except Exception as e:
            logger.warning("Failed to parse LLM response into UserProfile: %s", e)
            return None

    def _build_llm_prompt(self, user_input: dict) -> str:
        """构建发送给 LLM 的提示词。

        Args:
            user_input: 用户输入字典

        Returns:
            提示词字符串
        """
        return (
            f"请分析以下用户信息并输出用户画像 JSON：\n\n"
            f"分数: {user_input.get('score', '未知')}\n"
            f"省份: {user_input.get('province', '未知')}\n"
            f"科类: {user_input.get('subject_type', '未知')}\n"
            f"兴趣: {user_input.get('interests', [])}\n"
            f"性格: {user_input.get('personality', '未知')}\n"
            f"家庭资源: {user_input.get('family_resources', '未知')}\n"
            f"目标城市: {user_input.get('city_preference', '未知')}\n"
            f"学校类型偏好: {user_input.get('school_types', [])}\n\n"
            f"请返回如下字段的 JSON：score, province, interests, personality, "
            f"family_resources, risk_preference（冲/稳/保）, constraints（约束条件列表）, "
            f"subject_type（科类：理科/文科/物理类/历史类）, target_majors（目标专业列表）, "
            f"city_preference（目标城市）, school_types（学校类型列表：985/211/公办/民办）, "
            f"degree_level（学历层次，默认本科）"
        )

    def _rule_based_profile(self, user_input: dict) -> UserProfile:
        """使用硬编码规则创建用户画像。

        Args:
            user_input: 用户输入字典

        Returns:
            UserProfile: 基于规则的用户画像
        """
        score = int(user_input.get("score", 0))
        province = str(user_input.get("province", ""))
        interests = user_input.get("interests", [])
        personality = user_input.get("personality")
        family_resources = user_input.get("family_resources")

        # 根据分数确定风险偏好
        risk_preference = self._determine_risk_preference(score)

        # 提取科类
        subject_type = self._extract_subject_type(user_input)

        # 提取目标专业
        target_majors = self._extract_target_majors(interests, user_input)

        # 提取城市偏好
        city_preference = self._extract_city_preference(user_input)

        # 提取学校类型
        school_types = self._extract_school_types(user_input)

        # 从兴趣和性格中提取约束
        constraints = self._extract_constraints(interests, personality, family_resources)

        return UserProfile(
            score=score,
            province=province,
            interests=interests if interests else [],
            personality=personality,
            family_resources=family_resources,
            risk_preference=risk_preference,
            constraints=constraints,
            subject_type=subject_type,
            target_majors=target_majors,
            city_preference=city_preference,
            school_types=school_types,
            degree_level="本科",
        )

    def _extract_subject_type(self, user_input: dict) -> Optional[str]:
        """从用户输入中提取科类。

        Args:
            user_input: 用户输入字典

        Returns:
            科类字符串（理科/文科/物理类/历史类）或 None
        """
        # 直接从 subject_type 字段获取
        subject_type = user_input.get("subject_type")
        if subject_type:
            return subject_type

        # 从 interests 中查找
        interests = user_input.get("interests", [])
        keywords = ["理科", "文科", "物理类", "历史类"]
        for interest in interests:
            for kw in keywords:
                if kw in interest:
                    return kw

        return None

    def _extract_target_majors(self, interests: list[str], user_input: dict) -> list[str]:
        """从兴趣和输入中提取目标专业。

        Args:
            interests: 兴趣列表
            user_input: 用户输入字典

        Returns:
            目标专业列表
        """
        # 排除科类关键词
        subject_keywords = {"理科", "文科", "物理类", "历史类"}
        majors = []
        for interest in interests:
            if interest not in subject_keywords and interest:
                majors.append(interest)

        # 从 user_input 的 target_majors 字段获取
        if user_input.get("target_majors"):
            for major in user_input["target_majors"]:
                if major not in majors:
                    majors.append(major)

        return majors

    def _extract_city_preference(self, user_input: dict) -> Optional[str]:
        """从用户输入中提取城市偏好。

        Args:
            user_input: 用户输入字典

        Returns:
            城市字符串或 None
        """
        city = user_input.get("city_preference")
        if city:
            return city

        # 从 interests 中查找城市名（简单匹配常见城市）
        interests = user_input.get("interests", [])
        common_cities = [
            "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆",
            "武汉", "西安", "天津", "苏州", "长沙", "郑州", "青岛", "大连",
            "厦门", "宁波", "无锡", "合肥", "济南", "福州", "昆明", "哈尔滨",
            "长春", "沈阳", "南昌", "贵阳", "南宁", "兰州", "海口",
        ]
        for interest in interests:
            for city in common_cities:
                if city in interest:
                    return city

        return None

    def _extract_school_types(self, user_input: dict) -> list[str]:
        """从用户输入中提取学校类型偏好。

        Args:
            user_input: 用户输入字典

        Returns:
            学校类型列表（985/211/公办/民办）
        """
        school_types = user_input.get("school_types", [])
        if school_types:
            return school_types

        # 从 interests 中查找学校类型关键词
        interests = user_input.get("interests", [])
        keywords = ["985", "211", "公办", "民办"]
        result = []
        for interest in interests:
            for kw in keywords:
                if kw in interest and kw not in result:
                    result.append(kw)

        return result

    def _determine_risk_preference(self, score: int) -> str:
        """根据分数确定风险偏好。

        Args:
            score: 高考分数

        Returns:
            "冲", "稳", 或 "保"
        """
        if score >= 600:
            return "冲"
        elif score >= 550:
            return "稳"
        else:
            return "保"

    def _extract_constraints(
        self,
        interests: list[str],
        personality: Optional[str],
        family_resources: Optional[str],
    ) -> list[str]:
        """从用户信息中提取约束条件。

        Args:
            interests: 兴趣列表
            personality: 性格描述
            family_resources: 家庭资源描述

        Returns:
            约束条件列表
        """
        constraints = []

        if interests:
            constraints.append(f"兴趣方向: {', '.join(interests)}")

        if personality:
            constraints.append(f"性格特点: {personality}")

        if family_resources:
            constraints.append(f"家庭资源: {family_resources}")

        return constraints
