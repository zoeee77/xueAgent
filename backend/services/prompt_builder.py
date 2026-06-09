"""Prompt构建器：加载prompt模块，根据问题类型动态注入数据。"""

from pathlib import Path
import pathlib as pl
from typing import Optional

from .knowledge_base import KnowledgeBase

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
ROLES_DIR = PROMPTS_DIR / "roles"


class PromptBuilder:
    """构建System Prompt，支持模块化加载和动态数据注入。"""

    def __init__(self, knowledge_base: KnowledgeBase, prompts_dir: Path = PROMPTS_DIR):
        self._kb = knowledge_base
        self._prompts_dir = prompts_dir
        self._mental_models = self._load_prompt("mental_models.txt")
        self._decision_heuristics = self._load_prompt("decision_heuristics.txt")
        self._expression_dna = self._load_prompt("expression_dna.txt")

    def _load_prompt(self, filename: str) -> str:
        filepath = self._prompts_dir / filename
        if not filepath.exists():
            return ""
        return filepath.read_text(encoding="utf-8")

    @property
    def mental_models(self) -> str:
        return self._mental_models

    @property
    def decision_heuristics(self) -> str:
        return self._decision_heuristics

    @property
    def expression_dna(self) -> str:
        return self._expression_dna

    # ------------------------------------------------------------------
    # Role-scoped prompt loading (for MultiRoleReasoner)
    # ------------------------------------------------------------------

    # Role name (Chinese display) -> directory name (English)
    ROLE_DIR_MAP = {
        "张雪峰": "zhangxuefeng",
        "学术导师": "academic_mentor",
        "行业专家": "industry_expert",
        "HR经理": "hr_manager",
        "家长代表": "parent_representative",
    }

    def load_role_prompts(self, role_name: str) -> dict[str, str]:
        """Load the 3 constraint files for a specific role.

        Args:
            role_name: e.g. "张雪峰", "学术导师", "行业专家", "HR经理", "家长代表"

        Returns:
            dict with keys: mental_models, decision_heuristics, expression_dna
        """
        dir_name = self.ROLE_DIR_MAP.get(role_name, role_name)
        role_dir = ROLES_DIR / dir_name
        return {
            "mental_models": self._read_role_file(role_dir, "mental_models.txt"),
            "decision_heuristics": self._read_role_file(role_dir, "decision_heuristics.txt"),
            "expression_dna": self._read_role_file(role_dir, "expression_dna.txt"),
        }

    def _read_role_file(self, role_dir: Path, filename: str) -> str:
        filepath = role_dir / filename
        if not filepath.exists():
            return ""
        return filepath.read_text(encoding="utf-8")

    @staticmethod
    def list_available_roles() -> list[str]:
        """List all available role display names (Chinese)."""
        if not ROLES_DIR.exists():
            return []
        return list(PromptBuilder.ROLE_DIR_MAP.keys())

    # ------------------------------------------------------------------
    # Question classification & system prompt building (chat endpoint)
    # ------------------------------------------------------------------

    def classify_question(self, question: str) -> str:
        """根据问题内容分类。
        Returns:
            "major" | "university" | "industry" | "general"
        """
        q = question.lower()
        university_keywords = ["学校", "大学", "院校", "报哪个学校", "分数线"]

        # 优先检查具体行业名
        specific_industries = ["互联网", "制造", "能源", "农业"]
        if any(i in q for i in specific_industries):
            return "industry"

        has_industry_keyword = any(kw in q for kw in ["行业", "工资", "去哪", "做什么"])
        if has_industry_keyword:
            return "industry"

        specific_majors = ["金融", "计算机", "法学", "医学", "新闻", "电气", "会计", "土木"]
        has_specific_major = any(m in q for m in specific_majors)
        if has_specific_major:
            return "major"

        if any(kw in q for kw in university_keywords):
            return "university"

        if any(kw in q for kw in ["专业", "学什么", "选专业", "就业前景"]):
            return "major"

        return "general"

    def _extract_keywords(self, question: str) -> dict:
        """从问题中提取关键信息。"""
        import re
        info = {
            "score": None,
            "province": None,
            "major": None,
            "university": None,
            "industry": None,
        }
        score_match = re.search(r"(\d{3})\s*分", question)
        if score_match:
            info["score"] = int(score_match.group(1))
        provinces = ["河南", "河北", "山东", "江苏", "浙江", "广东", "四川", "湖北", "湖南",
                     "安徽", "江西", "福建", "陕西", "山西", "辽宁", "吉林", "黑龙江",
                     "云南", "贵州", "广西", "甘肃", "新疆", "宁夏", "青海", "西藏",
                     "内蒙古", "海南", "北京", "上海", "天津", "重庆"]
        for p in provinces:
            if p in question:
                info["province"] = p
                break
        return info

    def build_system_prompt(self, question: str, history: Optional[list] = None) -> str:
        """构建完整的 System Prompt。"""
        question_type = self.classify_question(question)
        keywords = self._extract_keywords(question)
        parts = []

        parts.append("## 表达风格\n")
        parts.append(self._expression_dna)
        parts.append("")

        parts.append("## 思维模型\n")
        parts.append(self._mental_models)
        parts.append("")

        parts.append("## 决策启发式\n")
        parts.append(self._decision_heuristics)
        parts.append("")

        parts.append("## 相关数据参考\n")
        data_injected = False

        if question_type == "major" or keywords.get("major"):
            major_keywords = ["金融", "计算机", "法学", "医学", "新闻", "电气", "会计", "土木"]
            for m in major_keywords:
                if m in question:
                    data = self._kb.query_major(m)
                    if data:
                        parts.append(f"### {m}专业数据\n")
                        parts.append(f"- 就业率: {data.get('employment_rate', 'N/A')}")
                        parts.append(f"- 平均薪资: {data.get('avg_salary', 'N/A')}")
                        parts.append(f"- 主要去向: {', '.join(data.get('top_directions', []))}")
                        parts.append(f"- 资源门槛: {data.get('resource_threshold', 'N/A')}")
                        parts.append(f"- 说明: {data.get('description', '')}")
                        parts.append("")
                        data_injected = True

        if question_type == "university" or keywords.get("province"):
            strategy = self._kb.get_score_strategy(keywords["score"]) if keywords.get("score") else None
            if strategy:
                parts.append(f"### 分数段策略({keywords['score']}分)\n")
                parts.append(strategy)
                parts.append("")
                data_injected = True

        if question_type == "industry":
            industry_keywords = ["金融", "互联网", "医疗", "制造"]
            for ind in industry_keywords:
                if ind in question:
                    data = self._kb.query_industry(ind)
                    if data:
                        parts.append(f"### {ind}行业数据\n")
                        parts.append(f"- 进入门槛: {data.get('entry_barrier', 'N/A')}")
                        parts.append(f"- 家庭资源依赖: {'是' if data.get('family_resource_dependent') else '否'}")
                        parts.append(f"- 主要雇主: {', '.join(data.get('top_employers', []))}")
                        dist = data.get("graduate_distribution", {})
                        if dist:
                            parts.append(f"- 毕业去向: 顶尖{dist.get('top_tier',0)*100:.0f}% 中层{dist.get('mid_tier',0)*100:.0f}% 基层{dist.get('grassroots',0)*100:.0f}%")
                        parts.append(f"- 说明: {data.get('description', '')}")
                        parts.append("")
                        data_injected = True

        if not data_injected:
            parts.append("## 相关数据参考\n")
            parts.append("（用户问题未匹配到具体数据，请基于思维模型和启发式回答）\n")

        is_first_turn = not history or len(history) <= 1
        if is_first_turn:
            parts.append("## 重要\n")
            parts.append("首次回复时请说：「我以张雪峰视角和你聊，基于公开言论推断，非本人观点。」")
            parts.append("后续对话不再重复此声明。")

        return "\n".join(parts)

    def get_all_modules(self) -> dict[str, str]:
        """返回所有 prompt 模块。"""
        return {
            "mental_models": self._mental_models,
            "decision_heuristics": self._decision_heuristics,
            "expression_dna": self._expression_dna,
        }