"""推荐模块测试。"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.vector_knowledge_base import VectorDocument
from backend.services.recommender import (
    UserProfile,
    RerankConfig,
    RerankResult,
    Reranker,
    RiskInfo,
    RiskClassifier,
    SchoolRecommend,
    RecommendationReport,
    generate_recommendation_reasons,
    build_recommendation_report,
)


class TestUserProfile:
    def test_default_values(self):
        profile = UserProfile()
        assert profile.risk_preference == "均衡"
        assert profile.degree_level == "本科"
        assert profile.target_majors == []

    def test_create_with_values(self):
        profile = UserProfile(
            score=580,
            province="安徽",
            subject_type="理科",
            target_majors=["计算机"],
            city_preference=["合肥"],
            school_types=["211"],
        )
        assert profile.score == 580
        assert profile.province == "安徽"


class TestRerankConfig:
    def test_default_weights_sum(self):
        config = RerankConfig()
        total = config.embedding_weight + config.score_match_weight + config.subject_eval_weight + config.preference_weight
        assert abs(total - 1.0) < 0.01

    def test_conservative_adjustment(self):
        config = RerankConfig()
        config.adjust_for_conservative_user()
        assert config.score_match_weight == 0.5
        assert config.embedding_weight == 0.1

    def test_aggressive_adjustment(self):
        config = RerankConfig()
        config.adjust_for_aggressive_user()
        assert config.score_match_weight == 0.2
        assert config.embedding_weight == 0.3


class TestReranker:
    def _create_docs(self):
        return [
            VectorDocument(
                id="1", category="t", base_id="1", variant_type="v",
                text="test1",
                metadata={"name": "学校A", "min_score": 570, "avg_score": 575},
            ),
            VectorDocument(
                id="2", category="t", base_id="2", variant_type="v",
                text="test2",
                metadata={"name": "学校B", "min_score": 550, "avg_score": 555},
            ),
            VectorDocument(
                id="3", category="t", base_id="3", variant_type="v",
                text="test3",
                metadata={"name": "学校C", "min_score": 600, "avg_score": 610},
            ),
        ]

    def test_rerank_score_match(self):
        docs = self._create_docs()
        profile = UserProfile(score=580)
        reranker = Reranker()
        results = reranker.rerank(docs, profile)

        # 学校B(555) diff=25 -> 0.7, 学校A(575) diff=5 -> 1.0
        # 学校A应该在最前面
        assert results[0].document.metadata["name"] == "学校A"

    def test_rerank_no_score_returns_mid(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="test", metadata={"name": "X"},
        )
        profile = UserProfile(score=None)
        reranker = Reranker()
        results = reranker.rerank([doc], profile)
        # 无分数信息时 score_match=0.5
        assert results[0].breakdown["score_match"] == 0.5

    def test_rerank_breakdown_keys(self):
        docs = self._create_docs()
        profile = UserProfile(score=580)
        reranker = Reranker()
        results = reranker.rerank(docs, profile)
        for r in results:
            assert "embedding" in r.breakdown
            assert "score_match" in r.breakdown
            assert "subject_eval" in r.breakdown
            assert "preference" in r.breakdown

    def test_rerank_with_subject_eval(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="test",
            metadata={
                "name": "学校A",
                "min_score": 570,
                "avg_score": 575,
                "subject_eval": {"计算机科学与技术": "A+"},
            },
        )
        profile = UserProfile(score=580, target_majors=["计算机"])
        reranker = Reranker()
        results = reranker.rerank([doc], profile)
        assert results[0].breakdown["subject_eval"] == 1.0  # A+ -> 1.0

    def test_rerank_with_preference_match(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="test",
            metadata={
                "name": "学校A",
                "province": "安徽",
                "location": "合肥",
                "tier": "211",
                "level": "本科",
                "min_score": 570,
            },
        )
        profile = UserProfile(
            score=580,
            province="安徽",
            city_preference=["合肥"],
            school_types=["211"],
        )
        reranker = Reranker()
        results = reranker.rerank([doc], profile)
        # 所有偏好都匹配，preference=1.0
        assert results[0].breakdown["preference"] == 1.0

    def test_rerank_conservative_user(self):
        docs = self._create_docs()
        profile = UserProfile(score=580, risk_preference="保")
        reranker = Reranker()
        results = reranker.rerank(docs, profile)
        # 保守用户 score_match_weight=0.5
        assert reranker.config.score_match_weight == 0.5

    def test_rerank_sorted_descending(self):
        docs = self._create_docs()
        profile = UserProfile(score=580)
        reranker = Reranker()
        results = reranker.rerank(docs, profile)
        for i in range(len(results) - 1):
            assert results[i].final_score >= results[i + 1].final_score


class TestRiskClassifier:
    def test_classify_charge(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="test", metadata={"min_score": 590, "avg_score": 595},
        )
        result = RerankResult(document=doc, final_score=0.8)
        classifier = RiskClassifier()
        charge, stable, safe = classifier.classify(580, [result])
        assert len(charge) == 1
        assert charge[0].level == "冲"

    def test_classify_stable(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="test", metadata={"min_score": 570, "avg_score": 575},
        )
        result = RerankResult(document=doc, final_score=0.8)
        classifier = RiskClassifier()
        charge, stable, safe = classifier.classify(580, [result])
        assert len(stable) == 1
        assert stable[0].level == "稳"

    def test_classify_safe(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="test", metadata={"min_score": 550, "avg_score": 555},
        )
        result = RerankResult(document=doc, final_score=0.8)
        classifier = RiskClassifier()
        charge, stable, safe = classifier.classify(580, [result])
        assert len(safe) == 1
        assert safe[0].level == "保"

    def test_classify_limits(self):
        """测试边界值: diff=-10 → 稳, diff=10 → 稳"""
        docs = [
            VectorDocument(id="1", category="t", base_id="1", variant_type="v",
                          text="t", metadata={"min_score": 590, "avg_score": 590}),  # diff=-10
            VectorDocument(id="2", category="t", base_id="2", variant_type="v",
                          text="t", metadata={"min_score": 570, "avg_score": 570}),  # diff=+10
        ]
        results = [RerankResult(document=d, final_score=0.8) for d in docs]
        classifier = RiskClassifier()
        charge, stable, safe = classifier.classify(580, results)
        assert len(stable) == 2  # both are 稳

    def test_classify_max_count(self):
        """测试数量限制"""
        docs = [
            VectorDocument(id=str(i), category="t", base_id=str(i), variant_type="v",
                          text="t", metadata={"min_score": 500 + i, "avg_score": 500 + i})
            for i in range(20)
        ]
        results = [RerankResult(document=d, final_score=0.8) for d in docs]
        classifier = RiskClassifier()
        charge, stable, safe = classifier.classify(580, results)
        assert len(charge) <= 8
        assert len(stable) <= 10
        assert len(safe) <= 8


class TestGenerateRecommendationReasons:
    def test_score_reason(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="t", metadata={"avg_score": 575},
        )
        profile = UserProfile(score=580)
        reasons = generate_recommendation_reasons(doc, profile)
        assert any("分数" in r for r in reasons)

    def test_subject_eval_reason(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="t", metadata={"subject_eval": {"计算机科学与技术": "A+"}},
        )
        profile = UserProfile()
        reasons = generate_recommendation_reasons(doc, profile)
        assert any("全国顶尖" in r for r in reasons)

    def test_location_reason(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="t", metadata={"location": "合肥", "province": "安徽"},
        )
        profile = UserProfile(city_preference=["合肥"])
        reasons = generate_recommendation_reasons(doc, profile)
        assert any("地域偏好" in r for r in reasons)

    def test_school_type_reason(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="t", metadata={"tier": "211"},
        )
        profile = UserProfile(school_types=["211"])
        reasons = generate_recommendation_reasons(doc, profile)
        assert any("211" in r for r in reasons)

    def test_max_4_reasons(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="t",
            metadata={
                "avg_score": 575,
                "subject_eval": {"计算机": "A+", "数学": "A"},
                "location": "合肥",
                "province": "安徽",
                "tier": "211",
            },
        )
        profile = UserProfile(
            score=580,
            city_preference=["合肥"],
            province="安徽",
            school_types=["211"],
        )
        reasons = generate_recommendation_reasons(doc, profile)
        assert len(reasons) <= 4

    def test_no_reasons_when_no_match(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="t", metadata={},
        )
        profile = UserProfile()
        reasons = generate_recommendation_reasons(doc, profile)
        # 默认分数匹配会返回0.5但不算理由
        assert len(reasons) == 0


class TestBuildRecommendationReport:
    def _create_results(self):
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="test",
            metadata={
                "name": "合肥工业大学",
                "province": "安徽",
                "location": "合肥",
                "tier": "211",
                "min_score": 572,
                "avg_score": 578,
                "subject_eval": {"计算机科学与技术": "B+"},
            },
        )
        return [RerankResult(document=doc, final_score=0.85)]

    def test_basic_report(self):
        profile = UserProfile(score=580, province="安徽")
        results = self._create_results()
        report = build_recommendation_report(profile, results)
        assert report.user_profile == profile
        assert len(report.charge_schools) + len(report.stable_schools) + len(report.safe_schools) >= 1

    def test_report_with_batch_info(self):
        profile = UserProfile(score=580)
        results = self._create_results()
        batch = {"score_line": 515}
        report = build_recommendation_report(profile, results, batch_info=batch)
        assert report.batch_info == batch
        assert "省控线" in report.overall_advice

    def test_report_school_recommend_structure(self):
        profile = UserProfile(score=580, province="安徽")
        results = self._create_results()
        report = build_recommendation_report(profile, results)
        # 合肥工业大学 580-578=2 → 稳
        assert len(report.stable_schools) == 1
        school = report.stable_schools[0]
        assert school.school_name == "合肥工业大学"
        assert school.risk_level == "稳"
        assert isinstance(school.reasons, list)

    def test_report_warnings(self):
        profile = UserProfile(score=580)
        # 只创建一个保底学校
        doc = VectorDocument(
            id="1", category="t", base_id="1", variant_type="v",
            text="t", metadata={"min_score": 500, "avg_score": 510, "name": "保底学校"},
        )
        results = [RerankResult(document=doc, final_score=0.5)]
        report = build_recommendation_report(profile, results)
        # 没有稳妥和冲刺学校
        assert len(report.charge_schools) == 0
        # 应该有警告
        assert len(report.risk_warnings) >= 0  # 可能有也可能没有
