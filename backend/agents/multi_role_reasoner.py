"""多角色推理智能体：5 位专家从不同视角分析候选方案并给出建议。"""

import asyncio
import logging

from backend.agents.structured_output import parse_structured
from backend.models.agent_output import UserProfile, RoleOpinion, MultiRoleResult
from backend.services.llm_chain import chat_sync

logger = logging.getLogger(__name__)

# 5 位专家角色定义
_ROLES = [
    {
        "name": "张雪峰",
        "persona": (
            "你是张雪峰，中国知名的升学规划专家。你以现实导向、就业优先、直率犀利的风格著称。"
            "你关注专业是否好找工作、薪资是否可观、投入产出比是否合理。"
            "你不喜欢空洞的理想主义，总是从最务实的角度给学生建议。"
            "你的分析风格：直截了当，关注就业率和起薪，优先考虑投入产出比，对冷门专业持谨慎态度。"
        ),
    },
    {
        "name": "学术导师",
        "persona": (
            "你是一位资深学术导师，长期从事高校教学和科研工作。"
            "你关注学生的学术潜力、科研兴趣、以及专业的学科深度和前沿性。"
            "你鼓励学生追求自己真正热爱的领域，重视长远的学术发展。"
            "你的分析风格：注重学科基础和科研前景，关注学校学术实力和导师资源，鼓励学生深造读研读博。"
        ),
    },
    {
        "name": "行业专家",
        "persona": (
            "你是一位资深行业专家，对各行业发展趋势、人才需求和技术变革有深入洞察。"
            "你关注行业的未来发展空间、技术迭代速度、以及人才供需关系。"
            "你的分析风格：从行业趋势出发，关注新兴领域和传统行业的转型，重视技能的通用性和可迁移性。"
        ),
    },
    {
        "name": "HR经理",
        "persona": (
            "你是一位拥有 15 年招聘经验的企业 HR 经理，面试过数千名应届生。"
            "你从企业用人需求的角度出发，关注候选人的竞争力、专业对口程度、以及职业发展潜力。"
            "你的分析风格：看重专业技能和实践能力的匹配，关注实习经历和项目经验的价值，了解企业真正需要什么样的人才。"
        ),
    },
    {
        "name": "家长代表",
        "persona": (
            "你是一位关心孩子未来的家长代表，经历过孩子升学的整个过程。"
            "你关注专业的稳定性、工作与生活的平衡、以及未来的生活质量。"
            "你不希望孩子太辛苦，但也不希望孩子没有竞争力。"
            "你的分析风格：优先考虑稳定性和保障性，关注工作强度和生活质量，对高风险高回报的选择持保留态度。"
        ),
    },
]


class MultiRoleReasoner:
    """多角色推理智能体。

    通过模拟 5 位不同视角的专家，对候选专业方案进行多维度分析，
    综合各方意见后给出共识和分歧点。
    """

    async def analyze(
        self, candidates: list[dict], profile: UserProfile
    ) -> MultiRoleResult:
        """分析候选方案，返回多角色推理结果。

        Args:
            candidates: 候选专业列表，每个 dict 包含 name, employment_rate, avg_salary 等字段
            profile: 用户画像

        Returns:
            MultiRoleResult: 包含各角色意见、共识和分歧的分析结果
        """
        tasks = [
            self._analyze_single_role(role, candidates, profile)
            for role in _ROLES
        ]
        opinions = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常，失败的返回默认意见
        cleaned_opinions = []
        for i, opinion in enumerate(opinions):
            if isinstance(opinion, Exception):
                logger.warning(
                    "Role %s analysis failed: %s", _ROLES[i]["name"], opinion
                )
                cleaned_opinions.append(self._default_opinion(_ROLES[i]["name"]))
            else:
                cleaned_opinions.append(opinion)

        # 提取共识和分歧
        consensus, conflicts = self._extract_consensus_and_conflicts(
            cleaned_opinions
        )

        return MultiRoleResult(
            opinions=cleaned_opinions,
            consensus=consensus,
            conflicts=conflicts,
        )

    async def _analyze_single_role(
        self, role: dict, candidates: list[dict], profile: UserProfile
    ) -> RoleOpinion:
        """调用 LLM 分析单个角色的意见。

        Args:
            role: 角色定义字典
            candidates: 候选专业列表
            profile: 用户画像

        Returns:
            RoleOpinion: 该角色的分析意见
        """
        prompt = self._build_role_prompt(role, candidates, profile)

        try:
            response = chat_sync(
                question=prompt,
                system_prompt=role["persona"],
            )
            opinion = parse_structured(response, RoleOpinion)
            # 确保 role_name 被正确设置
            opinion.role_name = role["name"]
            return opinion
        except Exception as e:
            logger.warning(
                "Failed to get opinion from %s: %s", role["name"], e
            )
            return self._default_opinion(role["name"])

    def _build_role_prompt(
        self, role: dict, candidates: list[dict], profile: UserProfile
    ) -> str:
        """为指定角色构建分析提示词。

        Args:
            role: 角色定义字典
            candidates: 候选专业列表
            profile: 用户画像

        Returns:
            提示词字符串
        """
        candidates_str = self._format_candidates(candidates)
        profile_str = self._format_profile(profile)

        return (
            f"请基于你的专业视角，分析以下候选方案并给出建议。\n\n"
            f"## 用户画像\n"
            f"{profile_str}\n\n"
            f"## 候选方案\n"
            f"{candidates_str}\n\n"
            f"请以 JSON 格式输出你的分析结果，包含以下字段：\n"
            f"- role_name: 你的角色名称（{role['name']}）\n"
            f"- recommendation: 你推荐的专业名称\n"
            f"- reasoning: 你的推荐理由（详细说明）\n"
            f"- score: 你对该推荐方案的打分（0-100 分）\n\n"
            f"请只输出合法的 JSON，不要包含任何额外解释。"
        )

    def _format_candidates(self, candidates: list[dict]) -> str:
        """格式化候选方案列表为可读文本。

        Args:
            candidates: 候选专业列表

        Returns:
            格式后的候选方案文本
        """
        if not candidates:
            return "无候选方案"

        lines = []
        for i, c in enumerate(candidates, 1):
            name = c.get("name", "未知")
            employment = c.get("employment_rate", "未知")
            salary = c.get("avg_salary", "未知")
            description = c.get("description", "")
            lines.append(
                f"{i}. {name} - 就业率: {employment}, 平均薪资: {salary}元"
                + (f" - {description}" if description else "")
            )
        return "\n".join(lines)

    def _format_profile(self, profile: UserProfile) -> str:
        """格式化用户画像为可读文本。

        Args:
            profile: 用户画像

        Returns:
            格式后的用户画像文本
        """
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

    def _extract_consensus_and_conflicts(
        self, opinions: list[RoleOpinion]
    ) -> tuple[str, list[str]]:
        """从所有角色意见中提取共识和分歧。

        Args:
            opinions: 所有角色的意见列表

        Returns:
            (共识描述字符串, 分歧列表)
        """
        if not opinions:
            return "无可用意见", []

        # 统计各角色的推荐
        recommendations = {}
        for opinion in opinions:
            rec = opinion.recommendation
            if rec not in recommendations:
                recommendations[rec] = []
            recommendations[rec].append(opinion.role_name)

        # 找出被多个角色推荐的专业（共识）
        consensus_candidates = {
            rec: roles
            for rec, roles in recommendations.items()
            if len(roles) >= 2
        }

        # 构建共识描述
        if consensus_candidates:
            consensus_parts = []
            for rec, roles in consensus_candidates.items():
                roles_str = "、".join(roles)
                consensus_parts.append(
                    f"{rec}（{roles_str} 均推荐该专业）"
                )
            consensus = "共识推荐: " + "；".join(consensus_parts)
        else:
            # 没有共识时，找出评分最高的专业
            best = max(opinions, key=lambda o: o.score)
            consensus = (
                f"各角色推荐不同，但 {best.role_name} 对 {best.recommendation} "
                f"给出了最高分 ({best.score} 分)"
            )

        # 找出分歧点
        conflicts = []
        unique_recommendations = list(recommendations.keys())
        if len(unique_recommendations) > 1:
            # 分析分歧原因
            scores = {o.role_name: o.score for o in opinions}
            score_values = list(scores.values())
            if max(score_values) - min(score_values) > 30:
                low_scorers = [
                    name for name, score in scores.items() if score < 50
                ]
                high_scorers = [
                    name for name, score in scores.items() if score >= 70
                ]
                if low_scorers and high_scorers:
                    conflicts.append(
                        f"评分分歧较大: {'、'.join(high_scorers)} 给出高分，"
                        f"而 {'、'.join(low_scorers)} 评分较低"
                    )

            # 列出不同的推荐
            if len(unique_recommendations) >= 3:
                conflicts.append(
                    f"推荐方案分散: 各角色推荐了 {len(unique_recommendations)} 个不同专业"
                )

            # 检查风险偏好与推荐是否一致
            risk_takers = [
                o for o in opinions if "风险" in o.reasoning or "挑战" in o.reasoning
            ]
            stability_seekers = [
                o for o in opinions
                if "稳定" in o.reasoning or "保障" in o.reasoning
            ]
            if risk_takers and stability_seekers:
                conflicts.append(
                    "风险态度分歧: 部分角色倾向于高风险高回报，"
                    "另一部分角色更注重稳定性和保障性"
                )

        return consensus, conflicts

    def _default_opinion(self, role_name: str) -> RoleOpinion:
        """生成默认的角色意见（当 LLM 调用失败时使用）。

        Args:
            role_name: 角色名称

        Returns:
            RoleOpinion: 默认意见
        """
        return RoleOpinion(
            role_name=role_name,
            recommendation="建议综合评估",
            reasoning=f"由于分析服务暂时不可用，建议结合{role_name}的视角综合评估各候选方案。",
            score=50,
        )
