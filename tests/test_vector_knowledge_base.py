"""向量知识库模块测试。"""

import pytest
from pathlib import Path
import sys
import tempfile
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.vector_knowledge_base import (
    VectorDocument,
    SearchResult,
    EmbeddingCache,
    VectorKnowledgeBase,
)


class TestVectorDocument:
    """VectorDocument 模型测试"""

    def test_create_document(self):
        doc = VectorDocument(
            id="university_basic:清华:basic",
            category="university_basic",
            base_id="清华",
            variant_type="basic",
            text="清华大学位于北京，是一所985高校。",
            metadata={"name": "清华大学", "province": "北京"},
        )
        assert doc.id == "university_basic:清华:basic"
        assert doc.category == "university_basic"
        assert "北京" in doc.text

    def test_to_dict_and_from_dict(self):
        doc = VectorDocument(
            id="test:001:basic",
            category="test",
            base_id="001",
            variant_type="basic",
            text="测试文本",
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
        )
        d = doc.to_dict()
        restored = VectorDocument.from_dict(d)
        assert restored.id == doc.id
        assert restored.text == doc.text
        assert restored.embedding == doc.embedding
        assert restored.metadata == doc.metadata


class TestEmbeddingCache:
    """EmbeddingCache 测试"""

    def test_cache_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))
            cache.set("hello world", [0.1, 0.2, 0.3])
            result = cache.get("hello world")
            assert result == [0.1, 0.2, 0.3]

    def test_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))
            result = cache.get("nonexistent")
            assert result is None

    def test_cache_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))
            cache.set("persist_test", [1.0, 2.0])
            cache.save()

            cache2 = EmbeddingCache(cache_dir=Path(tmpdir))
            result = cache2.get("persist_test")
            assert result == [1.0, 2.0]

    def test_cache_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))
            assert cache.size() == 0
            cache.set("a", [1.0])
            cache.set("b", [2.0])
            assert cache.size() == 2


class TestVectorKnowledgeBase:
    """VectorKnowledgeBase 核心功能测试"""

    def _create_test_kb(self):
        kb = VectorKnowledgeBase()
        kb.add_documents([
            VectorDocument(
                id="test:tsinghua:basic",
                category="university_basic",
                base_id="tsinghua",
                variant_type="basic",
                text="清华大学位于北京，是一所985高校，工科强势。",
                metadata={"name": "清华大学", "province": "北京", "tier": "985"},
            ),
            VectorDocument(
                id="test:pku:basic",
                category="university_basic",
                base_id="pku",
                variant_type="basic",
                text="北京大学位于北京，是一所985高校，文科理科都很强。",
                metadata={"name": "北京大学", "province": "北京", "tier": "985"},
            ),
            VectorDocument(
                id="test:zju:basic",
                category="university_basic",
                base_id="zju",
                variant_type="basic",
                text="浙江大学位于浙江杭州，是一所985高校，工科和计算机强。",
                metadata={"name": "浙江大学", "province": "浙江", "tier": "985"},
            ),
        ])
        kb.embed_all()
        return kb

    def test_add_and_count(self):
        kb = self._create_test_kb()
        assert kb.document_count == 3

    def test_get_document(self):
        kb = self._create_test_kb()
        doc = kb.get_document("test:tsinghua:basic")
        assert doc is not None
        assert doc.metadata["name"] == "清华大学"

    def test_get_document_not_found(self):
        kb = self._create_test_kb()
        doc = kb.get_document("nonexistent")
        assert doc is None

    def test_get_documents_by_category(self):
        kb = self._create_test_kb()
        docs = kb.get_documents_by_category("university_basic")
        assert len(docs) == 3

        docs = kb.get_documents_by_category("nonexistent")
        assert len(docs) == 0

    def test_categories(self):
        kb = self._create_test_kb()
        assert "university_basic" in kb.categories

    def test_embed_all(self):
        from backend.services.vector_knowledge_base import VectorKnowledgeBase

        kb = VectorKnowledgeBase()
        kb.add_document(VectorDocument(
            id="test:1:basic",
            category="test",
            base_id="1",
            variant_type="basic",
            text="测试embedding",
        ))
        assert kb._documents[0].embedding == []
        kb.embed_all()
        assert len(kb._documents[0].embedding) == 384

    def test_semantic_search_basic(self):
        kb = self._create_test_kb()
        results = kb.semantic_search("北京的大学", top_k=2)
        assert len(results) == 2
        assert results[0].rank == 1
        assert "北京" in results[0].document.metadata.get("province", "")

    def test_semantic_search_with_category(self):
        kb = self._create_test_kb()
        results = kb.semantic_search("大学", category="university_basic", top_k=5)
        assert len(results) <= 3

        results = kb.semantic_search("大学", category="nonexistent", top_k=5)
        assert len(results) == 0

    def test_semantic_search_empty_query(self):
        kb = self._create_test_kb()
        results = kb.semantic_search("", top_k=5)
        assert len(results) == 0

    def test_save_and_load_index(self):
        kb = self._create_test_kb()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        kb.save_index(filepath)
        kb2 = VectorKnowledgeBase()
        kb2.load_index(filepath)
        assert kb2.document_count == kb.document_count
        Path(filepath).unlink()

    def test_clear(self):
        kb = self._create_test_kb()
        assert kb.document_count == 3
        kb.clear()
        assert kb.document_count == 0


class TestDataTransformation:
    """数据转换脚本测试"""

    def test_build_university_docs(self):
        from backend.scripts.build_vector_kb import build_university_docs

        test_data = {
            "001": {
                "school_name": "清华大学",
                "province": "北京",
                "competent_department": "教育部",
                "location": "北京",
                "level": "本科",
                "is_private": False,
                "tier": "985",
            }
        }
        docs = build_university_docs(test_data)
        assert len(docs) >= 3
        variant_types = [d.variant_type for d in docs]
        assert "basic" in variant_types
        assert "location_based" in variant_types
        assert "tier_based" in variant_types

    def test_build_batch_score_docs(self):
        from backend.scripts.build_vector_kb import build_batch_score_docs

        test_data = {
            "test_batch_001": {
                "province": "安徽",
                "year": 2025,
                "category": "本科",
                "batch": "一批",
                "subject_type": "理科",
                "score_line": 515,
            }
        }
        docs = build_batch_score_docs(test_data)
        assert len(docs) == 1
        assert "515" in docs[0].text
        assert "安徽" in docs[0].text

    def test_build_school_score_docs(self):
        from backend.scripts.build_vector_kb import build_school_score_docs

        test_data = {
            "test_school_001": {
                "province": "安徽",
                "year": 2025,
                "subject_type": "理科",
                "school_name": "合肥工业大学",
                "min_score": 572,
                "avg_score": 578,
                "line_diff": 57,
            }
        }
        docs = build_school_score_docs(test_data)
        assert len(docs) >= 2
        combined_text = " ".join(d.text for d in docs)
        assert "合肥工业大学" in combined_text
        assert "572" in combined_text

    def test_build_subject_eval_docs(self):
        from backend.scripts.build_vector_kb import build_subject_eval_docs

        test_data = {
            "test_eval_001": {
                "round": 4,
                "year": 2016,
                "category_name": "工学",
                "subject_name": "计算机科学与技术",
                "school_name": "清华大学",
                "rank": 1,
                "grade": "A+",
            }
        }
        docs = build_subject_eval_docs(test_data)
        assert len(docs) >= 1
        variant_types = [d.variant_type for d in docs]
        assert "eval_based" in variant_types
        assert "recommend_based" in variant_types

    def test_build_all_integration(self):
        """集成测试: 构建完整知识库并搜索"""
        from backend.scripts.build_vector_kb import build_all
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            src_data = Path(__file__).resolve().parent.parent / "backend" / "data"
            for f in ["universities_list.json", "gaokao_scores.json", "subject_review.json"]:
                src = src_data / f
                if src.exists():
                    shutil.copy(src, data_dir / f)

            kb = build_all(data_dir=data_dir)
            assert kb.document_count > 0
            for doc in kb._documents:
                assert len(doc.embedding) == 384

            results = kb.semantic_search("北京的大学", category="university_basic", top_k=3)
            assert len(results) > 0
