"""Prompt 构建器测试。含完整性、动态注入、回归、边界、模块独立性测试。"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.prompt_builder import PromptBuilder
from backend.services.knowledge_base import KnowledgeBase


@pytest.fixture
def builder():
    """创建 PromptBuilder 实例（使用测试数据）。"""
    kb = KnowledgeBase()
    return PromptBuilder(knowledge_base=kb)


class TestPromptModules:
    """模块独立性测试。"""

    def test_mental_models_loaded(self, builder):
        modules = builder.get_all_modules()
        assert "mental_models" in modules
        assert len(modules["mental_models"]) > 0

    def test_decision_heuristics_loaded(self, builder):
        modules = builder.get_all_modules()
        assert "decision_heuristics" in modules
        assert len(modules["decision_heuristics"]) > 0

    def test_expression_dna_loaded(self, builder):
        modules = builder.get_all_modules()
        assert "expression_dna" in modules
        assert len(modules["expression_dna"]) > 0


class TestQuestionClassification:
    """问题分类测试。"""

    def test_classify_major(self, builder):
        assert builder.classify_question("我想学计算机专业") == "major"
        assert builder.classify_question("金融学就业前景怎么样") == "major"

    def test_classify_university(self, builder):
        assert builder.classify_question("560分报哪个学校") == "university"
        assert builder.classify_question("郑州大学分数线") == "university"

    def test_classify_industry(self, builder):
        assert builder.classify_question("金融行业工资高吗") == "industry"
        assert builder.classify_question("互联网就业前景") == "industry"

    def test_classify_general(self, builder):
        assert builder.classify_question("我该怎么选") == "general"


class TestPromptCompleteness:
    """完整性测试：验证 System Prompt 包含全部三个模块。"""

    def test_system_prompt_contains_mental_models(self, builder):
        prompt = builder.build_system_prompt("选什么专业好")
        assert "现实优先" in prompt
        assert "阶层流动" in prompt
        assert "就业导向" in prompt

    def test_system_prompt_contains_decision_heuristics(self, builder):
        prompt = builder.build_system_prompt("选什么专业好")
        assert "先选城市" in prompt
        assert "技术类优先" in prompt

    def test_system_prompt_contains_expression_dna(self, builder):
        prompt = builder.build_system_prompt("选什么专业好")
        assert "东北大哥" in prompt
        assert "接地气" in prompt


class TestDynamicInjection:
    """动态注入测试：验证不同问题类型注入正确数据。"""

    def test_major_injection(self, builder):
        prompt = builder.build_system_prompt("金融学好不好就业")
        # "金融学" 在KB中不存在，但模糊匹配到"金融学"，注入的关键词是"金融"
        assert "金融" in prompt
        assert "就业率" in prompt
        assert "平均薪资" in prompt

    def test_university_score_injection(self, builder):
        prompt = builder.build_system_prompt("河南560分选什么专业")
        assert "560" in prompt
        assert "优先选城市" in prompt

    def test_industry_injection(self, builder):
        prompt = builder.build_system_prompt("金融行业工资怎么样")
        assert "金融" in prompt
        assert "进入门槛" in prompt
        assert "家庭资源依赖" in prompt


class TestRegressionAnchors:
    """模板回归测试：验证关键锚点字符串存在。"""

    def test_anchor_reality_first(self, builder):
        prompt = builder.build_system_prompt("随便问")
        assert "现实优先原则" in prompt

    def test_anchor_employment_rate(self, builder):
        prompt = builder.build_system_prompt("金融学")
        assert "就业率" in prompt

    def test_anchor_data_driven(self, builder):
        prompt = builder.build_system_prompt("随便问")
        assert "数据" in prompt


class TestBoundaryCases:
    """边界测试。"""

    def test_empty_question(self, builder):
        prompt = builder.build_system_prompt("")
        assert len(prompt) > 0
        assert "现实优先" in prompt

    def test_no_match_data(self, builder):
        prompt = builder.build_system_prompt("我想学一个不存在的专业叫量子按摩学")
        assert len(prompt) > 0
        assert "未匹配到具体数据" in prompt

    def test_history_suppresses_disclaimer(self, builder):
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好啊"},
            {"role": "user", "content": "金融学怎么样"},
        ]
        prompt = builder.build_system_prompt("金融学怎么样", history=history)
        assert "首次回复时请说" not in prompt

    def test_first_turn_shows_disclaimer(self, builder):
        prompt = builder.build_system_prompt("金融学怎么样")
        assert "首次回复时请说" in prompt