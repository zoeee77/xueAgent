"""混合检索模块测试。"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.vector_knowledge_base import VectorDocument
from backend.services.hybrid_search import (
    FilterCondition,
    StructuredFilter,
    QueryExpander,
    HybridSearchResult,
    HybridSearch,
)


class TestFilterCondition:
    def test_create_condition(self):
        c = FilterCondition(field="province", operator="=", value="安徽")
        assert c.field == "province"
        assert c.operator == "="
        assert c.value == "安徽"


class TestStructuredFilter:
    def test_equal_filter(self):
        docs = [
            VectorDocument(id="1", category="t", base_id="1", variant_type="v",
                          text="test1", metadata={"province": "安徽"}),
            VectorDocument(id="2", category="t", base_id="2", variant_type="v",
                          text="test2", metadata={"province": "北京"}),
        ]
        sf = StructuredFilter([FilterCondition("province", "=", "安徽")])
        result = sf.apply(docs)
        assert len(result) == 1
        assert result[0].metadata["province"] == "安徽"

    def test_in_filter(self):
        docs = [
            VectorDocument(id="1", category="t", base_id="1", variant_type="v",
                          text="test1", metadata={"tier": "985"}),
            VectorDocument(id="2", category="t", base_id="2", variant_type="v",
                          text="test2", metadata={"tier": "211"}),
            VectorDocument(id="3", category="t", base_id="3", variant_type="v",
                          text="test3", metadata={"tier": ""}),
        ]
        sf = StructuredFilter([FilterCondition("tier", "in", ["985", "211"])])
        result = sf.apply(docs)
        assert len(result) == 2

    def test_range_filter(self):
        docs = [
            VectorDocument(id="1", category="t", base_id="1", variant_type="v",
                          text="test1", metadata={"min_score": 550}),
            VectorDocument(id="2", category="t", base_id="2", variant_type="v",
                          text="test2", metadata={"min_score": 580}),
            VectorDocument(id="3", category="t", base_id="3", variant_type="v",
                          text="test3", metadata={"min_score": 620}),
        ]
        sf = StructuredFilter([FilterCondition("min_score", "range", [540, 590])])
        result = sf.apply(docs)
        assert len(result) == 2

    def test_multiple_filters(self):
        docs = [
            VectorDocument(id="1", category="t", base_id="1", variant_type="v",
                          text="test1", metadata={"province": "安徽", "tier": "211"}),
            VectorDocument(id="2", category="t", base_id="2", variant_type="v",
                          text="test2", metadata={"province": "安徽", "tier": "985"}),
            VectorDocument(id="3", category="t", base_id="3", variant_type="v",
                          text="test3", metadata={"province": "北京", "tier": "985"}),
        ]
        sf = StructuredFilter([
            FilterCondition("province", "=", "安徽"),
            FilterCondition("tier", "=", "211"),
        ])
        result = sf.apply(docs)
        assert len(result) == 1
        assert result[0].id == "1"

    def test_empty_filters(self):
        docs = [
            VectorDocument(id="1", category="t", base_id="1", variant_type="v", text="test1"),
        ]
        sf = StructuredFilter()
        result = sf.apply(docs)
        assert len(result) == 1

    def test_clear_filters(self):
        sf = StructuredFilter([FilterCondition("a", "=", "b")])
        sf.clear()
        assert len(sf.conditions) == 0

    def test_add_condition_chaining(self):
        sf = StructuredFilter()
        result = sf.add_condition(FilterCondition("a", "=", "b"))
        assert result is sf  # returns self for chaining
        assert len(sf.conditions) == 1

    def test_not_equal_filter(self):
        docs = [
            VectorDocument(id="1", category="t", base_id="1", variant_type="v",
                          text="test1", metadata={"is_private": True}),
            VectorDocument(id="2", category="t", base_id="2", variant_type="v",
                          text="test2", metadata={"is_private": False}),
        ]
        sf = StructuredFilter([FilterCondition("is_private", "!=", True)])
        result = sf.apply(docs)
        assert len(result) == 1
        assert result[0].metadata["is_private"] is False


class TestQueryExpander:
    def test_expand_with_majors(self):
        profile = {"target_majors": ["计算机", "软件工程"]}
        expander = QueryExpander()
        queries = expander.expand(profile)
        assert "计算机专业强校" in queries
        assert "计算机学科建设好的大学" in queries
        assert "软件工程专业强校" in queries

    def test_expand_with_city(self):
        profile = {"city_preference": ["北京", "上海"]}
        expander = QueryExpander()
        queries = expander.expand(profile)
        assert "北京的大学" in queries
        assert "北京本地高校" in queries
        assert "上海的大学" in queries

    def test_expand_with_score(self):
        profile = {"score": 580}
        expander = QueryExpander()
        queries = expander.expand(profile)
        assert "录取分数580左右的大学" in queries

    def test_expand_with_school_types(self):
        profile = {"school_types": ["985", "211"]}
        expander = QueryExpander()
        queries = expander.expand(profile)
        assert "985院校" in queries
        assert "211院校" in queries

    def test_expand_with_province(self):
        profile = {"province": "安徽", "city_preference": []}
        expander = QueryExpander()
        queries = expander.expand(profile)
        assert "安徽的大学" in queries

    def test_expand_province_not_dup_with_city(self):
        """如果省份已在city_preference中，不再添加省份查询"""
        profile = {"province": "安徽", "city_preference": ["安徽"]}
        expander = QueryExpander()
        queries = expander.expand(profile)
        assert "安徽的大学" not in queries

    def test_expand_empty_profile(self):
        expander = QueryExpander()
        queries = expander.expand({})
        assert queries == []

    def test_expand_combined_profile(self):
        profile = {
            "score": 580,
            "province": "安徽",
            "target_majors": ["计算机"],
            "city_preference": ["合肥"],
            "school_types": ["211"],
        }
        expander = QueryExpander()
        queries = expander.expand(profile)
        # Should have queries from multiple dimensions
        assert len(queries) >= 5


class TestHybridSearch:
    """HybridSearch 集成测试"""

    def _create_kb(self):
        from backend.services.vector_knowledge_base import VectorKnowledgeBase
        kb = VectorKnowledgeBase()
        kb.add_documents([
            VectorDocument(
                id="test:hfut:basic", category="university_basic", base_id="hfut",
                variant_type="basic",
                text="合肥工业大学位于安徽合肥，是一所211公办院校。",
                metadata={"name": "合肥工业大学", "province": "安徽", "tier": "211", "min_score": 572},
            ),
            VectorDocument(
                id="test:ahu:basic", category="university_basic", base_id="ahu",
                variant_type="basic",
                text="安徽大学位于安徽合肥，是一所211公办院校。",
                metadata={"name": "安徽大学", "province": "安徽", "tier": "211", "min_score": 555},
            ),
            VectorDocument(
                id="test:tsinghua:basic", category="university_basic", base_id="tsinghua",
                variant_type="basic",
                text="清华大学位于北京，是一所985公办院校。",
                metadata={"name": "清华大学", "province": "北京", "tier": "985", "min_score": 680},
            ),
        ])
        kb.embed_all()
        return kb

    def test_hybrid_search_no_filters(self):
        kb = self._create_kb()
        hs = HybridSearch(kb)
        results = hs.search("安徽的大学", category="university_basic", top_k=5)
        assert len(results) > 0
        # 安徽的学校应该在前面
        assert results[0].document.metadata["province"] == "安徽"

    def test_hybrid_search_with_filter(self):
        kb = self._create_kb()
        hs = HybridSearch(kb)
        results = hs.search(
            "大学",
            filters=[FilterCondition("province", "=", "安徽")],
            category="university_basic",
            top_k=5,
        )
        # 过滤后只有安徽的学校
        for r in results:
            assert r.document.metadata["province"] == "安徽"

    def test_hybrid_search_with_range_filter(self):
        kb = self._create_kb()
        hs = HybridSearch(kb)
        results = hs.search(
            "大学",
            filters=[FilterCondition("min_score", "range", [540, 580])],
            category="university_basic",
            top_k=5,
        )
        # 应该排除清华(680分)
        for r in results:
            assert r.document.metadata["name"] != "清华大学"

    def test_hybrid_search_empty_result(self):
        kb = self._create_kb()
        hs = HybridSearch(kb)
        results = hs.search(
            "大学",
            filters=[FilterCondition("province", "=", "不存在的省份")],
            category="university_basic",
            top_k=5,
        )
        assert len(results) == 0

    def test_hybrid_search_scores(self):
        """测试融合得分：过滤结果得分 > 向量结果得分"""
        kb = self._create_kb()
        hs = HybridSearch(kb)
        results = hs.search(
            "北京",
            filters=[FilterCondition("province", "=", "安徽")],
            category="university_basic",
            top_k=5,
        )
        # 安徽的学校通过过滤+向量双路召回，北京的学校仅向量召回
        # 所以安徽的学校combined_score应该更高
        if results and len(results) > 1:
            assert results[0].document.metadata["province"] == "安徽"

    def test_hybrid_result_ranking(self):
        kb = self._create_kb()
        hs = HybridSearch(kb)
        results = hs.search("大学", category="university_basic", top_k=5)
        assert results[0].rank == 1
        for i, r in enumerate(results):
            assert r.rank == i + 1
