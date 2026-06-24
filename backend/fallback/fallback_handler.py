"""Fallback handler: uses KnowledgeBase (PostgreSQL) for data access."""

from backend.services.knowledge_base import KnowledgeBase


class FallbackHandler:
    """Handles LLM failures with rule-based recommendations."""

    def __init__(self):
        self._kb: KnowledgeBase | None = None

    def _get_kb(self) -> KnowledgeBase:
        """Lazy-load KnowledgeBase."""
        if self._kb is None:
            self._kb = KnowledgeBase()
        return self._kb

    # -- public API ----------------------------------------------------------

    async def handle_failure(
        self, user_profile: dict, knowledge_base=None
    ) -> dict:
        """
        Generate fallback recommendations based on user profile
        (score, province, interests) using rule-based logic.
        """
        score = user_profile.get("score", 0)
        province = user_profile.get("province", "")
        interests = user_profile.get("interests", [])

        try:
            plans = self.get_fallback_plan(score, province, interests, knowledge_base)
            return {
                "success": True,
                "is_fallback": True,
                "plans": plans,
                "explanation": (
                    f"LLM 调用失败，已基于规则引擎为您生成分档志愿推荐。"
                    f"您的分数为 {score}，省份为 {province}。"
                ),
                "warning": (
                    "以下为规则引擎推荐，仅供参考。"
                    "建议结合实际情况和专业老师意见进行最终决策。"
                ),
            }
        except Exception as exc:
            return {
                "success": False,
                "is_fallback": True,
                "plans": [],
                "explanation": f"规则引擎也未能生成推荐: {exc}",
                "warning": "系统异常，请稍后重试。",
            }

    def get_fallback_plan(
        self,
        score: int,
        province: str,
        interests: list[str],
        kb=None,
    ) -> list[dict]:
        """
        Return 3 plans (冲 / 稳 / 保) based on simple score matching.
        """
        # Use provided kb or lazy-load from PostgreSQL
        knowledge_base = kb or self._get_kb()
        majors = knowledge_base.all_majors
        universities = knowledge_base.all_universities

        # --- select high-employment majors (> 0.9) filtered by interests ----
        good_majors = {
            name: info
            for name, info in majors.items()
            if info.get("employment_rate", 0) > 0.9
        }

        if interests:
            # keep majors whose name or top_directions mention any interest keyword
            matched = {}
            for name, info in good_majors.items():
                directions = " ".join(info.get("top_directions", []))
                text = name + directions
                if any(kw in text for kw in interests):
                    matched[name] = info
            if matched:
                good_majors = matched

        # sort by employment_rate desc, then avg_salary desc
        ranked_majors = sorted(
            good_majors.items(),
            key=lambda x: (
                x[1].get("employment_rate", 0),
                x[1].get("avg_salary", 0),
            ),
            reverse=True,
        )

        # --- pick universities by score tier --------------------------------
        score_ranges: dict[str, tuple[int, int]] = {
            "冲": (score - 15, score),
            "稳": (score - 35, score - 16),
            "保": (score - 60, score - 36),
        }

        plans = []
        for tier, (low, high) in score_ranges.items():
            unis = self._filter_unis_by_score(universities, province, low, high)
            majors_for_tier = self._pick_majors_for_tier(
                tier, ranked_majors, score
            )
            plans.append(
                {
                    "tier": tier,
                    "score_range": f"{low}-{high}",
                    "universities": unis,
                    "majors": majors_for_tier,
                }
            )

        return plans

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _filter_unis_by_score(
        universities: dict, province: str, low: int, high: int
    ) -> list[dict]:
        results = []
        for name, info in universities.items():
            min_score = info.get("min_score_2025", 0)
            if low <= min_score <= high:
                results.append(
                    {
                        "name": name,
                        "province": info.get("province", ""),
                        "tier": info.get("tier", ""),
                        "min_score_2025": min_score,
                        "avg_score_2025": info.get("avg_score_2025", 0),
                        "description": info.get("description", ""),
                    }
                )
        # if province specified, prioritise local unis
        if province:
            results.sort(
                key=lambda u: (0 if u["province"] == province else 1, u["min_score_2025"]),
                reverse=False,
            )
        else:
            results.sort(key=lambda u: u["min_score_2025"])
        return results[:5]

    @staticmethod
    def _pick_majors_for_tier(
        tier: str, ranked_majors: list[tuple[str, dict]], score: int
    ) -> list[dict]:
        # "冲" picks top-3, "稳" picks next 3-5, "保" picks the rest (up to 5)
        count = {"冲": 3, "稳": 4, "保": 5}.get(tier, 3)
        start = {"冲": 0, "稳": 3, "保": 7}.get(tier, 0)
        slice_ = ranked_majors[start : start + count]
        return [
            {
                "name": name,
                "employment_rate": info.get("employment_rate", 0),
                "avg_salary": info.get("avg_salary", 0),
                "top_directions": info.get("top_directions", []),
                "description": info.get("description", ""),
            }
            for name, info in slice_
        ]
