"""推荐工具测试。"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models.agent_output import UserProfile as AgentProfile
from backend.tools.recommend_tool import recommend_by_text


class TestRecommendTool:
    def test_recommend_by_text(self):
        report = recommend_by_text(
            user_text="我是安徽的，理科580分",
            score=580,
            province="安徽",
            subject_type="理科",
            top_k=10,
        )
        assert report is not None
        total = len(report.charge_schools) + len(report.stable_schools) + len(report.safe_schools)
        assert total > 0
