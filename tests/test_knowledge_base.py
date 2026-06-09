"""知识库模块测试。"""

import pytest
from pathlib import Path
import tempfile
import json
import sys

# 确保能导入 backend 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.knowledge_base import KnowledgeBase


@pytest.fixture
def temp_kb():
    """创建带测试数据的临时知识库。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # 写入测试数据
        (data_dir / "majors.json").write_text(
            json.dumps({
                "金融学": {
                    "employment_rate": 0.72,
                    "avg_salary": 8500,
                    "top_directions": ["银行柜员"],
                    "resource_threshold": "high",
                    "description": "测试专业"
                },
                "计算机科学与技术": {
                    "employment_rate": 0.91,
                    "avg_salary": 12000,
                    "top_directions": ["后端开发"],
                    "resource_threshold": "low",
                    "description": "测试专业2"
                }
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (data_dir / "universities.json").write_text(
            json.dumps({
                "郑州大学": {
                    "province": "河南",
                    "tier": "211",
                    "min_score_2024": 580,
                    "avg_score_2024": 610,
                    "description": "测试院校"
                }
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (data_dir / "industries.json").write_text(
            json.dumps({
                "金融": {
                    "entry_barrier": "high",
                    "family_resource_dependent": True,
                    "top_employers": ["银行"],
                    "graduate_distribution": {"top_tier": 0.15, "mid_tier": 0.35, "grassroots": 0.50},
                    "description": "测试行业"
                }
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (data_dir / "decision_rules.json").write_text(
            json.dumps({
                "score_range_strategies": {
                    "650+": "冲刺985",
                    "600-650": "兼顾学校和专业",
                    "550-600": "优先选城市",
                    "500-550": "专业>城市>学校",
                    "500以下": "优先实用技能"
                },
                "priority_rules": {
                    "high_resource_family": "学校 > 专业 > 城市",
                    "low_resource_family": "专业 > 城市 > 学校"
                }
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        yield KnowledgeBase(data_dir=data_dir, cache_ttl=60)


class TestKnowledgeBaseQueryMajor:
    """专业查询测试。"""

    def test_exact_match(self, temp_kb):
        result = temp_kb.query_major("金融学")
        assert result is not None
        assert result["employment_rate"] == 0.72
        assert result["resource_threshold"] == "high"

    def test_fuzzy_match(self, temp_kb):
        result = temp_kb.query_major("金融")
        assert result is not None
        assert "employment_rate" in result

    def test_no_match(self, temp_kb):
        result = temp_kb.query_major("不存在的专业")
        assert result is None

    def test_empty_name(self, temp_kb):
        result = temp_kb.query_major("")
        assert result is None


class TestKnowledgeBaseQueryUniversity:
    """院校查询测试。"""

    def test_exact_match(self, temp_kb):
        result = temp_kb.query_university("郑州大学")
        assert result is not None
        assert result["province"] == "河南"
        assert result["tier"] == "211"

    def test_with_province(self, temp_kb):
        result = temp_kb.query_university("郑州", province="河南")
        assert result is not None

    def test_no_match(self, temp_kb):
        result = temp_kb.query_university("不存在的学校")
        assert result is None


class TestKnowledgeBaseQueryIndustry:
    """行业查询测试。"""

    def test_exact_match(self, temp_kb):
        result = temp_kb.query_industry("金融")
        assert result is not None
        assert result["entry_barrier"] == "high"
        assert result["family_resource_dependent"] is True

    def test_fuzzy_match(self, temp_kb):
        result = temp_kb.query_industry("金")
        assert result is not None

    def test_no_match(self, temp_kb):
        result = temp_kb.query_industry("不存在的行业")
        assert result is None


class TestKnowledgeBaseDecisionRules:
    """决策规则测试。"""

    def test_score_strategy_high(self, temp_kb):
        result = temp_kb.get_score_strategy(660)
        assert result is not None
        assert "985" in result

    def test_score_strategy_mid(self, temp_kb):
        result = temp_kb.get_score_strategy(620)
        assert result is not None
        assert "兼顾" in result

    def test_score_strategy_low(self, temp_kb):
        result = temp_kb.get_score_strategy(520)
        assert result is not None
        assert "专业" in result

    def test_score_strategy_very_low(self, temp_kb):
        result = temp_kb.get_score_strategy(480)
        assert result is not None

    def test_priority_rule_low_resource(self, temp_kb):
        result = temp_kb.get_priority_rule("low")
        assert result is not None
        assert "专业" in result

    def test_priority_rule_high_resource(self, temp_kb):
        result = temp_kb.get_priority_rule("high")
        assert result is not None
        assert "学校" in result


class TestKnowledgeBaseCache:
    """缓存测试。"""

    def test_cache_returns_same_result(self, temp_kb):
        result1 = temp_kb.query_major("金融学")
        result2 = temp_kb.query_major("金融学")
        assert result1 == result2

    def test_cache_handles_no_match(self, temp_kb):
        result1 = temp_kb.query_major("不存在")
        result2 = temp_kb.query_major("不存在")
        assert result1 is None
        assert result2 is None
