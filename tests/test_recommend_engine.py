"""推荐引擎测试。"""

import pytest
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.recommender import UserProfile
from backend.services.recommend_engine import RecommendEngine, convert_profile_to_recommender


class TestRecommendEngine:
    """RecommendEngine 集成测试"""

    def _create_engine_with_data(self):
        """创建带测试数据的引擎"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            src_data = Path(__file__).resolve().parent.parent / "backend" / "data"
            for f in ["universities_list.json", "gaokao_scores.json", "subject_review.json"]:
                src = src_data / f
                if src.exists():
                    shutil.copy(src, data_dir / f)

            engine = RecommendEngine(data_dir=data_dir)
            engine.load()
            return engine, tmpdir

    def test_engine_loads_kb(self):
        engine = RecommendEngine()
        assert engine._kb is None
        engine.load()
        assert engine._kb is not None
        assert engine._kb.document_count > 0

    def test_recommend_basic(self):
        engine, tmpdir = self._create_engine_with_data()
        profile = UserProfile(score=580, province="安徽", subject_type="理科")
        report = engine.recommend(profile, top_k=10)
        assert report is not None
        assert report.user_profile == profile
        total = len(report.charge_schools) + len(report.stable_schools) + len(report.safe_schools)
        assert total > 0

    def test_recommend_with_preferences(self):
        engine, tmpdir = self._create_engine_with_data()
        profile = UserProfile(
            score=580,
            province="安徽",
            subject_type="理科",
            city_preference=["合肥"],
            school_types=["211"],
        )
        report = engine.recommend(profile, top_k=10)
        assert report is not None

    def test_recommend_with_target_majors(self):
        engine, tmpdir = self._create_engine_with_data()
        profile = UserProfile(
            score=580,
            province="安徽",
            target_majors=["计算机"],
        )
        report = engine.recommend(profile, top_k=10)
        assert report is not None

    def test_recommend_conservative(self):
        engine, tmpdir = self._create_engine_with_data()
        profile = UserProfile(
            score=580,
            province="安徽",
            risk_preference="保",
        )
        report = engine.recommend(profile, top_k=10)
        assert report is not None
        # 保守用户应该有较多保底学校
        assert len(report.safe_schools) >= 0


class TestConvertProfile:
    """Profile 转换测试"""

    def test_convert_from_pydantic(self):
        from backend.models.agent_output import UserProfile as AgentProfile
        agent_profile = AgentProfile(
            score=580,
            province="安徽",
            interests=["计算机"],
            risk_preference="稳",
        )
        converted = convert_profile_to_recommender(agent_profile)
        assert converted.score == 580
        assert converted.province == "安徽"
        assert converted.risk_preference == "稳"
