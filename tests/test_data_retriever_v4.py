"""Phase 3: DataRetrieverV4 多阶段检索测试。"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agents.data_retriever import DataRetrieverV4, QueryContext, RecallItem
from backend.models.agent_output import UserProfile
from backend.services.knowledge_base import KnowledgeBase


@pytest.fixture
def v2_kb():
    """使用 v2 数据结构的临时知识库。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        (data_dir / "majors.json").write_text(
            json.dumps({
                "计算机科学与技术": {
                    "employment_rate": 0.933,
                    "avg_salary": 11500,
                    "top_directions": ["后端开发"],
                    "resource_threshold": "low",
                    "description": "万金油专业",
                    "courses": ["数据结构", "操作系统"],
                    "skills_required": ["编程"],
                    "personality_fit": ["逻辑型"],
                    "career_paths": ["工程师 -> 架构师"],
                    "industries": ["互联网"],
                    "keywords": ["计算机", "软件", "编程"]
                },
                "人工智能": {
                    "employment_rate": 0.982,
                    "avg_salary": 13800,
                    "top_directions": ["算法"],
                    "resource_threshold": "low",
                    "description": "AI 热门",
                    "courses": ["机器学习", "深度学习"],
                    "skills_required": ["算法", "数学"],
                    "personality_fit": ["研究型"],
                    "career_paths": ["AI 工程师 -> AI 专家"],
                    "industries": ["人工智能", "互联网"],
                    "keywords": ["人工智能", "AI", "机器学习", "大模型"]
                },
                "金融学": {
                    "employment_rate": 0.720,
                    "avg_salary": 10500,
                    "top_directions": ["银行"],
                    "resource_threshold": "high",
                    "description": "资源密集型",
                    "courses": ["金融学"],
                    "skills_required": ["分析"],
                    "personality_fit": ["社交型"],
                    "career_paths": ["银行 -> 行长"],
                    "industries": ["金融"],
                    "keywords": ["金融", "银行", "投资"]
                },
                "临床医学": {
                    "employment_rate": 0.850,
                    "avg_salary": 9500,
                    "top_directions": ["医师"],
                    "resource_threshold": "low",
                    "description": "稳定职业",
                    "courses": ["解剖学"],
                    "skills_required": ["记忆力"],
                    "personality_fit": ["责任型"],
                    "career_paths": ["住院医 -> 主任"],
                    "industries": ["医疗"],
                    "keywords": ["临床", "医学", "医生"]
                },
                "电子信息工程": {
                    "employment_rate": 0.965,
                    "avg_salary": 9800,
                    "top_directions": ["硬件"],
                    "resource_threshold": "low",
                    "description": "软硬兼备",
                    "courses": ["电路"],
                    "skills_required": ["电路设计"],
                    "personality_fit": ["逻辑型"],
                    "career_paths": ["工程师 -> 总监"],
                    "industries": ["通信", "电子"],
                    "keywords": ["电子", "信息", "硬件", "嵌入式"]
                }
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (data_dir / "universities.json").write_text("{}")
        (data_dir / "industries.json").write_text(
            json.dumps({
                "互联网": {
                    "entry_barrier": "medium",
                    "salary_range": {"low": 8000, "avg": 15000, "high": 35000},
                    "description": "互联网行业",
                    "top_employers": ["阿里", "腾讯"]
                },
                "人工智能": {
                    "entry_barrier": "high",
                    "salary_range": {"low": 12000, "avg": 20000, "high": 50000},
                    "description": "AI 行业",
                    "top_employers": ["百度", "商汤"]
                },
                "通信": {
                    "entry_barrier": "medium",
                    "salary_range": {"low": 8000, "avg": 12000, "high": 25000},
                    "description": "通信行业",
                    "top_employers": ["华为", "中兴"]
                }
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (data_dir / "decision_rules.json").write_text("{}")
        yield KnowledgeBase(data_dir=data_dir, cache_ttl=60)


@pytest.fixture
def retriever(v2_kb):
    return DataRetrieverV4(kb=v2_kb)


# ──────────────────────────────────────────────
# Stage 1: Query Understanding 测试
# ─────────────────────────────────────────────

class TestQueryUnderstand:
    """第一阶段: Query 理解测试。"""

    def test_basic_profile(self, retriever, v2_kb):
        profile = UserProfile(
            score=580, province="河南", interests=["计算机"],
            personality="逻辑型", family_resources="普通",
            risk_preference="稳",
        )
        ctx = retriever._query_understand(profile)
        assert ctx.interest_keywords == ["计算机"]
        assert ctx.score == 580
        assert ctx.province == "河南"
        assert ctx.personality == "逻辑型"
        assert "计算机" in ctx.query_text
        assert "逻辑型" in ctx.query_text

    def test_empty_interests(self, retriever, v2_kb):
        profile = UserProfile(
            score=580, province="河南", interests=[],
            personality=None, family_resources=None,
            risk_preference="稳",
        )
        ctx = retriever._query_understand(profile)
        assert ctx.interest_keywords == []
        assert ctx.query_text == ""


# ──────────────────────────────────────────────
# Stage 2: Multi-Path Recall 测试
# ──────────────────────────────────────────────

class TestMultiPathRecall:
    """第二阶段: 多路召回测试。"""

    def test_multi_path_deduplication(self, retriever, v2_kb):
        """多路召回应该去重。"""
        ctx = QueryContext(
            interest_keywords=["计算机"], personality="逻辑型",
            query_text="计算机 逻辑型", family_resources="普通"
        )
        items = retriever._multi_path_recall(ctx)
        # 每个专业只应出现一次
        names = list(items.keys())
        assert len(names) == len(set(names))
        # 计算机相关专业应该被召回
        assert "计算机科学与技术" in names

    def test_recall_includes_computer(self, retriever, v2_kb):
        """召回应该包含计算机相关专业。"""
        ctx = QueryContext(
            interest_keywords=["计算机"], personality="逻辑型",
            query_text="计算机 逻辑型", family_resources="普通"
        )
        items = retriever._multi_path_recall(ctx)
        assert "计算机科学与技术" in items
        # Should also find AI via keyword/industry mapping
        assert "人工智能" in items

    def test_rule_recall_filters_high_resource(self, retriever, v2_kb):
        """规则召回应对高资源需求专业进行过滤。"""
        ctx = QueryContext(family_resources="不足", score=580)
        items = retriever._multi_path_recall(ctx)
        # 金融学资源要求 high, 家庭资源不足时不应通过规则召回
        if "金融学" in items:
            assert "rule" not in items["金融学"].scores, "金融学不应通过规则召回"

    def test_recall_scores_exist(self, retriever, v2_kb):
        """召回项应有至少一个来源的分数。"""
        ctx = QueryContext(
            interest_keywords=["计算机"], personality="逻辑型",
            query_text="计算机 逻辑型", family_resources="普通"
        )
        items = retriever._multi_path_recall(ctx)
        for name, item in items.items():
            assert len(item.scores) > 0, f"{name} 没有任何召回分数"


# ──────────────────────────────────────────────
# Stage 3: Score Fusion 测试
# ──────────────────────────────────────────────

class TestScoreFusion:
    """第三阶段: 分数融合测试。"""

    def test_weighted_fusion(self, retriever, v2_kb):
        ctx = QueryContext(interest_keywords=["计算机"], family_resources="普通")
        items = retriever._multi_path_recall(ctx)

        fused = retriever._score_fusion(items, ctx)
        assert len(fused) > 0
        # 最高分的应该是计算机相关
        top_name = fused[0].name
        assert top_name in ["计算机科学与技术", "人工智能"]

    def test_fusion_scores_normalized(self, retriever, v2_kb):
        ctx = QueryContext(interest_keywords=["计算机"], family_resources="普通")
        items = retriever._multi_path_recall(ctx)
        fused = retriever._score_fusion(items, ctx)

        for item in fused:
            final = item.scores.get("final", 0)
            assert 0.0 <= final <= 1.0, f"{item.name} final score {final} out of range"

    def test_fusion_sorting(self, retriever, v2_kb):
        ctx = QueryContext(interest_keywords=["计算机"], family_resources="普通")
        items = retriever._multi_path_recall(ctx)
        fused = retriever._score_fusion(items, ctx)

        scores = [item.scores.get("final", 0) for item in fused]
        assert scores == sorted(scores, reverse=True)


# ──────────────────────────────────────────────
# Stage 4: Re-Ranking + Explain 测试
# ──────────────────────────────────────────────

class TestReRankAndExplain:
    """第四阶段: 重排序 + 可解释性测试。"""

    def test_explainable_output(self, retriever, v2_kb):
        ctx = QueryContext(interest_keywords=["计算机"], family_resources="普通")
        items = retriever._multi_path_recall(ctx)
        fused = retriever._score_fusion(items, ctx)
        results = retriever._re_rank_and_explain(fused, ctx)

        assert len(results) > 0
        # 每个结果都应该包含可解释性字段
        for r in results:
            assert "match_reason" in r
            assert "data_support" in r
            assert "recommend_reason" in r
            assert "risk_warnings" in r
            assert "courses" in r
            assert "career_paths" in r

    def test_match_reason_not_empty(self, retriever, v2_kb):
        ctx = QueryContext(interest_keywords=["计算机"], family_resources="普通")
        items = retriever._multi_path_recall(ctx)
        fused = retriever._score_fusion(items, ctx)
        results = retriever._re_rank_and_explain(fused, ctx)

        for r in results:
            assert r["match_reason"], f"{r['name']} 的 match_reason 为空"

    def test_data_support_format(self, retriever, v2_kb):
        ctx = QueryContext(interest_keywords=["计算机"], family_resources="普通")
        items = retriever._multi_path_recall(ctx)
        fused = retriever._score_fusion(items, ctx)
        results = retriever._re_rank_and_explain(fused, ctx)

        for r in results:
            supports = r["data_support"]
            assert any("就业率" in s for s in supports), f"{r['name']} 缺少就业率"
            assert any("薪资" in s for s in supports), f"{r['name']} 缺少薪资"

    def test_risk_warnings_for_high_resource(self, retriever, v2_kb):
        ctx = QueryContext(interest_keywords=["金融"], family_resources="不足")
        items = retriever._multi_path_recall(ctx)
        fused = retriever._score_fusion(items, ctx)
        results = retriever._re_rank_and_explain(fused, ctx)

        finance_result = [r for r in results if "金融" in r["name"]]
        if finance_result:
            warnings = finance_result[0]["risk_warnings"]
            assert any("资源" in w for w in warnings), "应提示资源不足风险"


# ─────────────────────────────────────────────
# 完整流程测试
# ──────────────────────────────────────────────

class TestFullRetrievalPipeline:
    """完整检索流程端到端测试。"""

    def test_full_pipeline_computer_interest(self, retriever, v2_kb):
        profile = UserProfile(
            score=580, province="河南", interests=["计算机"],
            personality="逻辑型", family_resources="普通",
            risk_preference="稳",
        )
        result = asyncio.get_event_loop().run_until_complete(
            retriever.retrieve(profile)
        )

        assert result.majors is not None
        assert len(result.majors) > 0
        # Top result should be computer-related
        top_major = result.majors[0]["name"]
        assert "计算机" in top_major or "人工智能" in top_major
        # Should have industries
        assert len(result.industries) > 0
        # Filter reason should be present
        assert result.filter_reason
        # Should have retrieval metadata
        assert "strategy" in result.retrieval_meta
        assert result.retrieval_meta["strategy"] == "v4"

    def test_full_pipeline_no_interests(self, retriever, v2_kb):
        """无兴趣时仍能返回结果（基于规则）。"""
        profile = UserProfile(
            score=580, province="河南", interests=[],
            personality=None, family_resources="普通",
            risk_preference="稳",
        )
        result = asyncio.get_event_loop().run_until_complete(
            retriever.retrieve(profile)
        )

        # 无兴趣时应有基于规则的推荐
        assert len(result.majors) > 0

    def test_high_score_all_majors(self, retriever, v2_kb):
        """高分用户能看到所有专业。"""
        profile = UserProfile(
            score=680, province="北京", interests=["计算机", "金融"],
            personality="研究型", family_resources="充裕",
            risk_preference="冲",
        )
        result = asyncio.get_event_loop().run_until_complete(
            retriever.retrieve(profile)
        )

        # 高分 + 充裕资源应能看到更多专业
        assert len(result.majors) >= 2
