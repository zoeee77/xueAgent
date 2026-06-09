"""Orchestrator: 总协调器，串联所有子智能体完成志愿填报流程。"""

import asyncio
import logging
import time
from typing import Any, Optional

from backend.agents.data_retriever import DataRetrieverV4
from backend.agents.intent_parser import IntentParser
from backend.agents.multi_role_reasoner import MultiRoleReasoner
from backend.agents.planner import Planner
from backend.agents.refiner import Refiner
from backend.agents.tool_agent import ToolAgent
from backend.agents.user_profiler import UserProfiler
from backend.agents.structured_output import parse_structured
from backend.fallback.fallback_handler import FallbackHandler
from backend.logging_config import trace_id_ctx
from backend.models.agent_output import (
    DevilAdvocateResult,
    ExplanationResult,
    PlanOption,
    PlanResult,
    RankItem,
    RankResult,
    UserProfile,
)
from backend.services.knowledge_base import KnowledgeBase
from backend.services.llm_chain import chat_sync
from backend.state.agent_state import AgentState, AgentStatus, StepName
from backend.state.state_manager import StateManager

logger = logging.getLogger(__name__)

# 单 Agent 超时时间（秒）
_AGENT_TIMEOUT = 30
_AGENT_TIMEOUT_MULTI_ROLE = 120  # 多角色并行 LLM 调用需要更长时间


# ---------------------------------------------------------------------------
# 缺失子智能体的轻量实现
# ---------------------------------------------------------------------------

class Ranker:
    """排序器：对方案结果进行多维度打分排序。"""

    async def rank(
        self,
        plan_result: PlanResult,
        profile: UserProfile,
    ) -> RankResult:
        """对方案进行排序。

        Args:
            plan_result: 方案结果
            profile: 用户画像

        Returns:
            RankResult: 排序后的结果
        """
        try:
            prompt = self._build_rank_prompt(plan_result, profile)
            system_prompt = (
                "你是一位志愿填报排序专家。请对给定的志愿方案进行多维度打分排序。"
                "考虑就业率、专业匹配度、风险适配、薪资前景、发展空间等维度。"
                "请只输出合法的 JSON，不要包含任何额外解释。"
            )
            response = chat_sync(question=prompt, system_prompt=system_prompt)
            result = parse_structured(response, RankResult)
            return result
        except Exception as e:
            logger.warning("LLM ranking failed, falling back to rule-based: %s", e)
            return self._rule_based_rank(plan_result, profile)

    def _rule_based_rank(
        self,
        plan_result: PlanResult,
        profile: UserProfile,
    ) -> RankResult:
        """基于规则的排序兜底逻辑。"""
        ranked = []
        for i, option in enumerate(plan_result.options):
            total = 50.0  # 基础分
            breakdown: dict[str, float] = {}

            # 风险适配
            risk_match = 1.0 if option.risk_level == profile.risk_preference else 0.6
            breakdown["risk"] = risk_match * 25.0
            total += breakdown["risk"]

            # 匹配度（兴趣关键词）
            match_score = 0.0
            if profile.interests:
                for interest in profile.interests:
                    if interest.lower() in option.major.lower():
                        match_score += 0.5
                match_score = min(match_score, 1.0)
            else:
                match_score = 0.5  # 无兴趣偏好时给中等分
            breakdown["match"] = match_score * 25.0
            total += breakdown["match"]

            # 就业 & 薪资（占位）
            breakdown["employment"] = 15.0
            breakdown["salary"] = 10.0
            breakdown["growth"] = 10.0
            total += breakdown["employment"] + breakdown["salary"] + breakdown["growth"]

            total = min(max(total, 0.0), 100.0)

            ranked.append(
                RankItem(
                    option=option,
                    total_score=round(total, 2),
                    breakdown=breakdown,
                    rank=i + 1,
                )
            )

        ranked.sort(key=lambda x: x.total_score, reverse=True)
        for i, item in enumerate(ranked):
            item.rank = i + 1

        return RankResult(ranked_list=ranked)

    def _build_rank_prompt(
        self, plan_result: PlanResult, profile: UserProfile
    ) -> str:
        options_str = "\n".join(
            f"{i + 1}. [{o.risk_level}] {o.major} - {o.reason}"
            for i, o in enumerate(plan_result.options)
        )
        return (
            f"请对以下志愿方案进行排序打分：\n\n"
            f"## 用户画像\n"
            f"分数: {profile.score}, 省份: {profile.province}, "
            f"兴趣: {', '.join(profile.interests) if profile.interests else '无'}, "
            f"风险偏好: {profile.risk_preference}\n\n"
            f"## 待排序方案\n{options_str}\n\n"
            f"请以 JSON 格式输出，结构如下：\n"
            f'{{"ranked_list": [{{"option": {{...}}, "total_score": 85.0, '
            f'"breakdown": {{"employment": 15.0, "match": 20.0, "risk": 20.0, '
            f'"salary": 15.0, "growth": 15.0}}, "rank": 1}}]}}'
        )


class DevilAdvocate:
    """反对者：对方案提出质疑和风险提醒。"""

    async def challenge(
        self,
        plan_result: PlanResult,
        profile: UserProfile,
    ) -> DevilAdvocateResult:
        """对方案进行挑战性分析。

        Args:
            plan_result: 方案结果
            profile: 用户画像

        Returns:
            DevilAdvocateResult: 质疑和风险提醒
        """
        try:
            prompt = self._build_challenge_prompt(plan_result, profile)
            system_prompt = (
                "你是一位反对者（Devil's Advocate）。你的职责是对给定的志愿方案提出质疑，"
                "找出潜在风险，并提供替代建议。请从最谨慎的角度分析问题。"
                "请只输出合法的 JSON，不要包含任何额外解释。"
            )
            response = chat_sync(question=prompt, system_prompt=system_prompt)
            result = parse_structured(response, DevilAdvocateResult)
            return result
        except Exception as e:
            logger.warning("LLM devil advocate failed, using rule-based: %s", e)
            return self._rule_based_challenge(plan_result, profile)

    def _rule_based_challenge(
        self, plan_result: PlanResult, profile: UserProfile
    ) -> DevilAdvocateResult:
        """基于规则的兜底挑战分析。"""
        objections = []
        risks = []
        suggestions = []

        for option in plan_result.options:
            if option.risk_level == "冲":
                objections.append(
                    f"冲档方案「{option.major}」录取分数较高，"
                    f"存在落榜风险（预期分数 {option.expected_score}，您的分数 {profile.score}）。"
                )
                risks.append("冲刺方案可能无法达到录取线")
            elif option.risk_level == "保":
                suggestions.append(
                    f"保档方案「{option.major}」虽安全，但可能过于保守，"
                    f"建议适当考虑中间档位的选择。"
                )

        if not profile.interests:
            risks.append("未明确兴趣方向，专业选择可能不够精准")

        if not objections:
            objections.append("当前方案整体合理，未见明显异议")
        if not risks:
            risks.append("风险提示：高考志愿填报需结合当年实际招生计划调整")
        if not suggestions:
            suggestions.append("建议关注目标院校的最新招生政策和专业调整信息")

        return DevilAdvocateResult(
            objections=objections,
            risks=risks,
            alternative_suggestions=suggestions,
        )

    def _build_challenge_prompt(
        self, plan_result: PlanResult, profile: UserProfile
    ) -> str:
        options_str = "\n".join(
            f"- [{o.risk_level}] {o.major}: {o.reason} (预期分数: {o.expected_score})"
            for o in plan_result.options
        )
        return (
            f"请对以下志愿方案提出质疑和风险提醒：\n\n"
            f"## 用户画像\n"
            f"分数: {profile.score}, 风险偏好: {profile.risk_preference}\n\n"
            f"## 待分析方案\n{options_str}\n\n"
            f"请以 JSON 格式输出：\n"
            f'{{"objections": ["质疑1", "质疑2"], "risks": ["风险1"], '
            f'"alternative_suggestions": ["建议1"]}}'
        )


class Explainer:
    """解释器：对排序结果和反对者意见生成可读解释。"""

    async def explain(
        self,
        rank_result: RankResult,
        devil_result: DevilAdvocateResult,
        profile: UserProfile,
    ) -> ExplanationResult:
        """生成方案解释。

        Args:
            rank_result: 排序结果
            devil_result: 反对者分析结果
            profile: 用户画像

        Returns:
            ExplanationResult: 解释说明
        """
        try:
            prompt = self._build_explain_prompt(rank_result, devil_result, profile)
            system_prompt = (
                "你是一位志愿填报解释专家。请用通俗易懂的语言向用户解释"
                "为什么推荐这些方案、为什么首选排在第一位、为什么不选其他方案，"
                "以及需要注意哪些风险。"
                "请只输出合法的 JSON，不要包含任何额外解释。"
            )
            response = chat_sync(question=prompt, system_prompt=system_prompt)
            result = parse_structured(response, ExplanationResult)
            return result
        except Exception as e:
            logger.warning("LLM explanation failed, using rule-based: %s", e)
            return self._rule_based_explain(rank_result, devil_result, profile)

    def _rule_based_explain(
        self,
        rank_result: RankResult,
        devil_result: DevilAdvocateResult,
        profile: UserProfile,
    ) -> ExplanationResult:
        """基于规则的兜底解释。"""
        ranked = rank_result.ranked_list
        first = ranked[0] if ranked else None

        why_recommended = (
            f"基于您的分数（{profile.score}分）、兴趣方向（"
            f"{', '.join(profile.interests) if profile.interests else '综合评估'}）"
            f"以及风险偏好（{profile.risk_preference}），系统为您生成了冲/稳/保三套方案，"
            f"并从就业率、专业匹配度、风险适配等多个维度进行了综合排序。"
        )

        if first:
            why_first = (
                f"排在首位的是「{first.option.risk_level}」档的 {first.option.major}，"
                f"综合得分 {first.total_score} 分。"
                f"{first.option.reason}"
            )
        else:
            why_first = "暂无首选方案。"

        why_not_others = (
            "其他方案在不同维度上各有优劣，系统已将其纳入备选。"
            "反对者提出的质疑已记录在风险提示中。"
        )

        risk_warnings = devil_result.risks if devil_result else []

        return ExplanationResult(
            why_recommended=why_recommended,
            why_first=why_first,
            why_not_others=why_not_others,
            risk_warnings=risk_warnings,
        )

    def _build_explain_prompt(
        self,
        rank_result: RankResult,
        devil_result: DevilAdvocateResult,
        profile: UserProfile,
    ) -> str:
        ranked_str = "\n".join(
            f"{i + 1}. [{r.option.risk_level}] {r.option.major} (得分: {r.total_score})"
            for i, r in enumerate(rank_result.ranked_list)
        )
        risks_str = "\n".join(f"- {r}" for r in devil_result.risks) if devil_result.risks else "无明显风险"
        return (
            f"请解释以下排序结果和风险提醒：\n\n"
            f"## 用户画像\n"
            f"分数: {profile.score}, 风险偏好: {profile.risk_preference}\n\n"
            f"## 排序结果\n{ranked_str}\n\n"
            f"## 风险提醒\n{risks_str}\n\n"
            f"请以 JSON 格式输出：\n"
            f'{{"why_recommended": "推荐理由", "why_first": "首选理由", '
            f'"why_not_others": "不选其他的原因", "risk_warnings": ["风险1"]}}'
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """总协调器：串联所有子智能体，完成志愿填报的完整流程。"""

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.state_manager = StateManager()

        # 初始化所有子智能体
        self.user_profiler = UserProfiler()
        self.data_retriever = DataRetrieverV4(kb=kb)
        self.multi_role_reasoner = MultiRoleReasoner()
        self.planner = Planner(kb=kb)
        self.ranker = Ranker()
        self.devil_advocate = DevilAdvocate()
        self.explainer = Explainer()
        self.refiner = Refiner(kb=kb)
        self.intent_parser = IntentParser()
        self.tool_agent = ToolAgent()
        self.fallback_handler = FallbackHandler()

    # -- public API ----------------------------------------------------------

    async def execute(
        self, user_input: dict, trace_id: Optional[str] = None
    ) -> dict:
        """执行完整的志愿填报流程。

        Args:
            user_input: 用户输入，包含 score, province, interests 等
            trace_id: 追踪 ID（用于日志链路追踪）

        Returns:
            包含所有子智能体输出的结构化结果字典
        """
        # 1. 设置 trace_id 上下文
        trace_id = trace_id or f"trace-{int(time.time())}"
        trace_id_ctx.set(trace_id)
        logger.info("[%s] Orchestrator started", trace_id)

        profile: Optional[UserProfile] = None
        data_result = None
        role_result = None
        plan_result = None
        rank_result = None
        devil_result = None
        explain_result = None

        try:
            # 2. UserProfiler.analyze
            profile = await self._run_with_timeout(
                self.user_profiler.analyze(user_input),
                step=StepName.USER_PROFILE,
                trace_id=trace_id,
            )
            logger.info("[%s] UserProfiler completed", trace_id)

            # 3. DataRetrieverV4.retrieve
            data_result = await self._run_with_timeout(
                self.data_retriever.retrieve(profile),
                step=StepName.DATA_RETRIEVE,
                trace_id=trace_id,
            )
            logger.info("[%s] DataRetriever completed", trace_id)

            # 4. MultiRoleReasoner.analyze
            role_result = await self._run_with_timeout(
                self.multi_role_reasoner.analyze(data_result.majors, profile),
                step=StepName.MULTI_ROLE_REASON,
                trace_id=trace_id,
                timeout=_AGENT_TIMEOUT_MULTI_ROLE,
            )
            logger.info("[%s] MultiRoleReasoner completed", trace_id)

            # 5. Planner.generate
            plan_result = await self._run_with_timeout(
                self.planner.generate(role_result, data_result, profile),
                step=StepName.PLAN,
                trace_id=trace_id,
            )
            logger.info("[%s] Planner completed", trace_id)

            # 6. Ranker.rank
            rank_result = await self._run_with_timeout(
                self.ranker.rank(plan_result, profile),
                step=StepName.RANK,
                trace_id=trace_id,
            )
            logger.info("[%s] Ranker completed", trace_id)

            # 7. DevilAdvocate.challenge
            devil_result = await self._run_with_timeout(
                self.devil_advocate.challenge(plan_result, profile),
                step=StepName.EXPLAIN,  # reuse EXPLAIN step for devil advocate tracking
                trace_id=trace_id,
            )
            logger.info("[%s] DevilAdvocate completed", trace_id)

            # 8. Explainer.explain
            explain_result = await self._run_with_timeout(
                self.explainer.explain(rank_result, devil_result, profile),
                step=StepName.EXPLAIN,
                trace_id=trace_id,
            )
            logger.info("[%s] Explainer completed", trace_id)

            # 9. 返回结构化结果
            result = self._build_execute_result(
                profile=profile,
                data_result=data_result,
                role_result=role_result,
                plan_result=plan_result,
                rank_result=rank_result,
                devil_result=devil_result,
                explain_result=explain_result,
                trace_id=trace_id,
            )
            logger.info("[%s] Orchestrator execute completed successfully", trace_id)
            return result

        except Exception as e:
            # 10. 失败处理
            logger.error("[%s] Orchestrator execute failed: %s", trace_id, e)
            return await self._handle_failure(
                user_input=user_input,
                profile=profile,
                error=str(e),
                trace_id=trace_id,
            )

    async def refine(
        self, user_input: dict, intent_result, trace_id: Optional[str] = None
    ) -> dict:
        """执行完整流程后，根据用户反馈意图精炼方案。

        Args:
            user_input: 用户输入
            intent_result: IntentParseResult，用户意图解析结果
            trace_id: 追踪 ID

        Returns:
            包含精炼后方案的结果字典
        """
        # 1. 先执行正常 pipeline
        execute_result = await self.execute(user_input, trace_id=trace_id)

        # 如果执行已经是 fallback 模式，直接返回
        if execute_result.get("is_fallback"):
            return execute_result

        trace_id = execute_result.get("trace_id", trace_id)
        profile = execute_result.get("profile")

        if profile is None:
            logger.warning("[%s] No profile available for refine, returning execute result", trace_id)
            return execute_result

        try:
            # 从 execute_result 中重建 PlanResult
            plan_result = self._rebuild_plan_result(execute_result)

            # 2. 运行 Refiner.refine
            refine_result = await self._run_with_timeout(
                self.refiner.refine(plan_result, intent_result, profile, self.kb),
                step=StepName.REFINE,
                trace_id=trace_id,
            )
            logger.info("[%s] Refiner completed", trace_id)

            # 3. 返回精炼结果
            result = self._build_refine_result(
                execute_result=execute_result,
                refine_result=refine_result,
                trace_id=trace_id,
            )
            logger.info("[%s] Orchestrator refine completed successfully", trace_id)
            return result

        except Exception as e:
            logger.error("[%s] Orchestrator refine failed: %s", trace_id, e)
            return execute_result  # 精炼失败时返回原始执行结果

    # -- internal helpers ----------------------------------------------------

    async def _run_with_timeout(
        self,
        coro,
        step: StepName,
        trace_id: str,
        timeout: Optional[int] = None,
    ) -> Any:
        """带超时和状态追踪的执行器。

        Args:
            coro: 要执行的协程
            step: 当前步骤名称
            trace_id: 追踪 ID
            timeout: 超时时间（秒），默认使用全局配置

        Returns:
            协程的执行结果

        Raises:
            TimeoutError: 执行超时
            Exception: 执行失败
        """
        # 创建初始状态
        await self.state_manager.create_state(trace_id, step)
        await self.state_manager.update_state(
            trace_id,
            status=AgentStatus.RUNNING,
            started_at=time.time(),
        )

        actual_timeout = timeout if timeout is not None else _AGENT_TIMEOUT
        try:
            result = await asyncio.wait_for(coro, timeout=actual_timeout)

            await self.state_manager.update_state(
                trace_id,
                status=AgentStatus.SUCCESS,
                completed_at=time.time(),
            )
            return result

        except asyncio.TimeoutError:
            logger.error("[%s] Agent %s timed out after %ds", trace_id, step.value, _AGENT_TIMEOUT)
            await self.state_manager.update_state(
                trace_id,
                status=AgentStatus.TIMEOUT,
                completed_at=time.time(),
                error_message=f"Timeout after {_AGENT_TIMEOUT}s",
            )
            raise

        except Exception as e:
            await self.state_manager.update_state(
                trace_id,
                status=AgentStatus.FAILED,
                completed_at=time.time(),
                error_message=str(e),
            )
            raise

    async def _handle_failure(
        self,
        user_input: dict,
        profile: Optional[UserProfile],
        error: str,
        trace_id: str,
    ) -> dict:
        """处理管道执行失败。

        Args:
            user_input: 原始用户输入
            profile: 已获取的用户画像（可能为 None）
            error: 错误信息
            trace_id: 追踪 ID

        Returns:
            兜底结果字典
        """
        await self.state_manager.create_state(trace_id, StepName.FALLBACK)
        await self.state_manager.update_state(
            trace_id, status=AgentStatus.RUNNING, started_at=time.time()
        )

        user_profile_for_fallback = {}
        if profile:
            user_profile_for_fallback = {
                "score": profile.score,
                "province": profile.province,
                "interests": profile.interests,
            }
        else:
            user_profile_for_fallback = {
                "score": user_input.get("score", 0),
                "province": user_input.get("province", ""),
                "interests": user_input.get("interests", []),
            }

        fallback_result = await self.fallback_handler.handle_failure(
            user_profile=user_profile_for_fallback,
            knowledge_base=self.kb,
        )

        await self.state_manager.update_state(
            trace_id,
            status=AgentStatus.SUCCESS if fallback_result.get("success") else AgentStatus.FAILED,
            completed_at=time.time(),
        )

        return {
            "trace_id": trace_id,
            "is_fallback": True,
            "error": error,
            "fallback_result": fallback_result,
        }

    def _build_execute_result(
        self,
        profile: UserProfile,
        data_result,
        role_result,
        plan_result: PlanResult,
        rank_result: RankResult,
        devil_result: DevilAdvocateResult,
        explain_result: ExplanationResult,
        trace_id: str,
    ) -> dict:
        """构建 execute 方法的返回字典。"""
        return {
            "trace_id": trace_id,
            "success": True,
            "is_fallback": False,
            "profile": {
                "score": profile.score,
                "province": profile.province,
                "interests": profile.interests,
                "personality": profile.personality,
                "family_resources": profile.family_resources,
                "risk_preference": profile.risk_preference,
                "constraints": profile.constraints,
            },
            "data_retrieval": {
                "majors": data_result.majors,
                "industries": data_result.industries,
                "filter_reason": data_result.filter_reason,
            },
            "multi_role_analysis": {
                "opinions": [
                    {
                        "role_name": o.role_name,
                        "recommendation": o.recommendation,
                        "reasoning": o.reasoning,
                        "score": o.score,
                    }
                    for o in role_result.opinions
                ],
                "consensus": role_result.consensus,
                "conflicts": role_result.conflicts,
            },
            "plan": {
                "options": [
                    {
                        "risk_level": o.risk_level,
                        "major": o.major,
                        "universities": o.universities,
                        "reason": o.reason,
                        "expected_score": o.expected_score,
                    }
                    for o in plan_result.options
                ]
            },
            "rank": {
                "ranked_list": [
                    {
                        "rank": r.rank,
                        "total_score": r.total_score,
                        "breakdown": r.breakdown,
                        "option": {
                            "risk_level": r.option.risk_level,
                            "major": r.option.major,
                            "universities": r.option.universities,
                            "reason": r.option.reason,
                            "expected_score": r.option.expected_score,
                        },
                    }
                    for r in rank_result.ranked_list
                ],
            },
            "devil_advocate": {
                "objections": devil_result.objections,
                "risks": devil_result.risks,
                "alternative_suggestions": devil_result.alternative_suggestions,
            },
            "explanation": {
                "why_recommended": explain_result.why_recommended,
                "why_first": explain_result.why_first,
                "why_not_others": explain_result.why_not_others,
                "risk_warnings": explain_result.risk_warnings,
            },
        }

    def _rebuild_plan_result(self, execute_result: dict) -> PlanResult:
        """从 execute_result 中重建 PlanResult 对象。"""
        options = []
        plan_data = execute_result.get("plan", {})
        for opt in plan_data.get("options", []):
            options.append(
                PlanOption(
                    risk_level=opt["risk_level"],
                    major=opt["major"],
                    universities=opt["universities"],
                    reason=opt["reason"],
                    expected_score=opt["expected_score"],
                )
            )
        return PlanResult(options=options)

    def _build_refine_result(
        self,
        execute_result: dict,
        refine_result,
        trace_id: str,
    ) -> dict:
        """构建 refine 方法的返回字典。"""
        refined_plan = refine_result.updated_plan

        result = dict(execute_result)
        result["trace_id"] = trace_id
        result["refined"] = True
        result["refined_plan"] = {
            "options": [
                {
                    "risk_level": o.risk_level,
                    "major": o.major,
                    "universities": o.universities,
                    "reason": o.reason,
                    "expected_score": o.expected_score,
                }
                for o in refined_plan.options
            ]
        }
        result["changes_made"] = refine_result.changes_made
        return result
