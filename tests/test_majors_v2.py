"""Phase 2: 知识库数据结构 v2 测试。"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.knowledge_base import KnowledgeBase

# 指向真实数据文件
DATA_DIR = Path(__file__).resolve().parent.parent / "backend" / "data"


@pytest.fixture
def real_kb():
    """使用真实数据文件的知识库。"""
    return KnowledgeBase(data_dir=DATA_DIR, cache_ttl=60)


@pytest.fixture
def temp_kb_v2():
    """创建带 v2 数据结构的临时知识库。"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # 写入 v2 测试数据
        (data_dir / "majors.json").write_text(
            json.dumps({
                "计算机科学与技术": {
                    "employment_rate": 0.933,
                    "avg_salary": 11500,
                    "top_directions": ["后端开发", "前端开发"],
                    "resource_threshold": "low",
                    "description": "测试专业",
                    "courses": ["数据结构", "操作系统", "计算机网络"],
                    "skills_required": ["编程能力", "系统设计", "算法"],
                    "personality_fit": ["逻辑型", "实践型"],
                    "career_paths": ["工程师 -> 架构师 -> CTO"],
                    "industries": ["互联网", "金融"],
                    "keywords": ["计算机", "软件", "编程", "开发"]
                },
                "人工智能": {
                    "employment_rate": 0.982,
                    "avg_salary": 13800,
                    "top_directions": ["算法工程师"],
                    "resource_threshold": "low",
                    "description": "AI 专业",
                    "courses": ["机器学习", "深度学习"],
                    "skills_required": ["算法设计", "数学建模"],
                    "personality_fit": ["研究型", "逻辑型"],
                    "career_paths": ["AI 工程师 -> AI 专家"],
                    "industries": ["人工智能", "互联网"],
                    "keywords": ["人工智能", "AI", "机器学习", "大模型"]
                }
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (data_dir / "universities.json").write_text("{}")
        (data_dir / "industries.json").write_text("{}")
        (data_dir / "decision_rules.json").write_text("{}")
        yield KnowledgeBase(data_dir=data_dir, cache_ttl=60)


class TestMajorV2Fields:
    """v2 字段测试。"""

    def test_courses_field(self, temp_kb_v2):
        major = temp_kb_v2.query_major("计算机科学与技术")
        assert "courses" in major
        assert isinstance(major["courses"], list)
        assert len(major["courses"]) > 0
        assert "数据结构" in major["courses"]

    def test_skills_required_field(self, temp_kb_v2):
        major = temp_kb_v2.query_major("计算机科学与技术")
        assert "skills_required" in major
        assert "编程能力" in major["skills_required"]

    def test_personality_fit_field(self, temp_kb_v2):
        major = temp_kb_v2.query_major("人工智能")
        assert "personality_fit" in major
        assert "研究型" in major["personality_fit"]

    def test_career_paths_field(self, temp_kb_v2):
        major = temp_kb_v2.query_major("计算机科学与技术")
        assert "career_paths" in major
        assert isinstance(major["career_paths"], list)
        assert "->" in major["career_paths"][0]

    def test_industries_field(self, temp_kb_v2):
        major = temp_kb_v2.query_major("人工智能")
        assert "industries" in major
        assert "人工智能" in major["industries"]

    def test_keywords_field(self, temp_kb_v2):
        major = temp_kb_v2.query_major("计算机科学与技术")
        assert "keywords" in major
        assert "计算机" in major["keywords"]
        assert "编程" in major["keywords"]


class TestRealDataV2Fields:
    """真实数据 v2 字段完整性测试。"""

    REQUIRED_V2_FIELDS = [
        "courses", "skills_required", "personality_fit",
        "career_paths", "industries", "keywords",
    ]
    REQUIRED_V1_FIELDS = [
        "employment_rate", "avg_salary", "top_directions",
        "resource_threshold", "description",
    ]

    def test_all_majors_have_v2_fields(self, real_kb):
        """所有专业都应包含 v2 扩展字段。"""
        for name, data in real_kb.all_majors.items():
            for field in self.REQUIRED_V2_FIELDS:
                assert field in data, f"{name} 缺少字段 {field}"

    def test_all_majors_have_v1_fields(self, real_kb):
        """所有专业都应包含 v1 基础字段（向后兼容）。"""
        for name, data in real_kb.all_majors.items():
            for field in self.REQUIRED_V1_FIELDS:
                assert field in data, f"{name} 缺少字段 {field}"

    def test_keywords_not_empty(self, real_kb):
        """所有专业的关键词列表不应为空。"""
        for name, data in real_kb.all_majors.items():
            assert len(data["keywords"]) > 0, f"{name} 的 keywords 为空"

    def test_courses_not_empty(self, real_kb):
        """所有专业的课程列表不应为空。"""
        for name, data in real_kb.all_majors.items():
            assert len(data["courses"]) > 0, f"{name} 的 courses 为空"

    def test_career_paths_not_empty(self, real_kb):
        """所有专业的职业路径列表不应为空。"""
        for name, data in real_kb.all_majors.items():
            assert len(data["career_paths"]) > 0, f"{name} 的 career_paths 为空"

    def test_majors_count(self, real_kb):
        """验证专业数量。"""
        assert len(real_kb.all_majors) >= 30


class TestKeywordSearch:
    """关键词搜索测试。"""

    def test_search_by_keyword(self, real_kb):
        """通过关键词搜索能找到匹配的专业。"""
        all_majors = real_kb.all_majors
        matched = [
            name for name, data in all_majors.items()
            if "编程" in data.get("keywords", [])
        ]
        assert len(matched) > 0

    def test_search_by_personality(self, real_kb):
        """通过性格类型搜索能找到匹配的专业。"""
        all_majors = real_kb.all_majors
        matched = [
            name for name, data in all_majors.items()
            if "研究型" in data.get("personality_fit", [])
        ]
        assert len(matched) > 0

    def test_search_by_industry(self, real_kb):
        """通过行业搜索能找到匹配的专业。"""
        all_majors = real_kb.all_majors
        matched = [
            name for name, data in all_majors.items()
            if "互联网" in data.get("industries", [])
        ]
        assert len(matched) > 0
