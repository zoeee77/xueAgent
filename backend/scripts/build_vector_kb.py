"""数据转换和向量化脚本：将 JSON 数据转换为向量知识库文档。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.services.vector_knowledge_base import (
    VectorDocument,
    VectorKnowledgeBase,
    CATEGORY_UNIVERSITY_BASIC,
    CATEGORY_SCORE_BATCH,
    CATEGORY_SCORE_SCHOOL,
    CATEGORY_SUBJECT_EVAL,
)


def build_university_docs(universities: dict) -> list[VectorDocument]:
    """为高校数据生成多个语义变体文档"""
    docs = []

    for school_id, data in universities.items():
        name = data.get("school_name", "")
        province = data.get("province", "")
        location = data.get("location", "")
        tier = data.get("tier", "")
        department = data.get("competent_department", "")
        level = data.get("level", "")
        is_private = data.get("is_private", False)

        private_note = "民办" if is_private else "公办"
        tier_note = f"{tier}院校" if tier else ""

        docs.append(VectorDocument(
            id=f"{CATEGORY_UNIVERSITY_BASIC}:{school_id}:basic",
            category=CATEGORY_UNIVERSITY_BASIC,
            base_id=school_id,
            variant_type="basic",
            text=f"{name}位于{location}，主管部门为{department}，办学层次为{level}，是一所{private_note}普通高等学校。{tier_note}",
            metadata=data,
        ))

        docs.append(VectorDocument(
            id=f"{CATEGORY_UNIVERSITY_BASIC}:{school_id}:location_based",
            category=CATEGORY_UNIVERSITY_BASIC,
            base_id=school_id,
            variant_type="location_based",
            text=f"{province}{location}的大学推荐:{name}，是一所{private_note}{tier_note}。主管部门{department}。",
            metadata=data,
        ))

        if tier:
            docs.append(VectorDocument(
                id=f"{CATEGORY_UNIVERSITY_BASIC}:{school_id}:tier_based",
                category=CATEGORY_UNIVERSITY_BASIC,
                base_id=school_id,
                variant_type="tier_based",
                text=f"如果想报考{tier}院校，{name}是不错的选择。该校位于{location}，{private_note}。",
                metadata=data,
            ))

    return docs


def build_batch_score_docs(batch_scores: dict) -> list[VectorDocument]:
    """为批次分数线数据生成文档"""
    docs = []

    for score_id, data in batch_scores.items():
        province = data.get("province", "")
        year = data.get("year", "")
        category = data.get("category", "")
        batch = data.get("batch", "")
        subject = data.get("subject_type", "")
        score = data.get("score_line", 0)

        docs.append(VectorDocument(
            id=f"{CATEGORY_SCORE_BATCH}:{score_id}:basic",
            category=CATEGORY_SCORE_BATCH,
            base_id=score_id,
            variant_type="basic",
            text=f"{year}年{province}{category}{batch}{subject}录取分数线为{score}分。",
            metadata=data,
        ))

    return docs


def build_school_score_docs(school_scores: dict) -> list[VectorDocument]:
    """为学校录取分数线数据生成文档"""
    docs = []

    for score_id, data in school_scores.items():
        province = data.get("province", "")
        year = data.get("year", "")
        subject = data.get("subject_type", "")
        school = data.get("school_name", "")
        min_score = data.get("min_score", 0)
        avg_score = data.get("avg_score", 0)
        line_diff = data.get("line_diff", 0)

        docs.append(VectorDocument(
            id=f"{CATEGORY_SCORE_SCHOOL}:{score_id}:score_based",
            category=CATEGORY_SCORE_SCHOOL,
            base_id=score_id,
            variant_type="score_based",
            text=f"{year}年{school}在{province}{subject}录取最低分{min_score}分，平均分{avg_score}分，高出省控线{line_diff}分。",
            metadata=data,
        ))

        docs.append(VectorDocument(
            id=f"{CATEGORY_SCORE_SCHOOL}:{score_id}:school_based",
            category=CATEGORY_SCORE_SCHOOL,
            base_id=score_id,
            variant_type="school_based",
            text=f"{school}{year}年在{province}的{subject}录取情况:最低分{min_score}，平均分{avg_score}。",
            metadata=data,
        ))

    return docs


def build_subject_eval_docs(subject_data: dict) -> list[VectorDocument]:
    """为学科评估数据生成文档"""
    docs = []

    for eval_id, data in subject_data.items():
        school = data.get("school_name", "")
        subject = data.get("subject_name", "")
        category = data.get("category_name", "")
        grade = data.get("grade", "")
        rank = data.get("rank", 0)
        year = data.get("year", 0)
        round_num = data.get("round", 0)

        docs.append(VectorDocument(
            id=f"{CATEGORY_SUBJECT_EVAL}:{eval_id}:eval_based",
            category=CATEGORY_SUBJECT_EVAL,
            base_id=eval_id,
            variant_type="eval_based",
            text=f"{school}的{subject}（{category}）学科，在第{round_num}轮学科评估中整体水平排名第{rank}，评估结果为{grade}。评估年份:{year}。",
            metadata=data,
        ))

        grade_desc = "全国顶尖" if grade in ["A+", "A", "A-"] else "较强" if grade in ["B+", "B"] else ""
        if grade_desc:
            docs.append(VectorDocument(
                id=f"{CATEGORY_SUBJECT_EVAL}:{eval_id}:recommend_based",
                category=CATEGORY_SUBJECT_EVAL,
                base_id=eval_id,
                variant_type="recommend_based",
                text=f"如果想学{subject}专业，{school}是不错的选择，该学科评估{grade}，{grade_desc}，全国排名第{rank}。",
                metadata=data,
            ))

    return docs


def build_all(data_dir: Path = None) -> VectorKnowledgeBase:
    """构建完整的向量知识库"""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"

    kb = VectorKnowledgeBase(data_dir=data_dir)

    universities_file = data_dir / "universities_list.json"
    if universities_file.exists():
        with open(universities_file, "r", encoding="utf-8") as f:
            universities = json.load(f)
        kb.add_documents(build_university_docs(universities))

    gaokao_file = data_dir / "gaokao_scores.json"
    if gaokao_file.exists():
        with open(gaokao_file, "r", encoding="utf-8") as f:
            gaokao_data = json.load(f)
        kb.add_documents(build_batch_score_docs(gaokao_data.get("batch_scores", {})))
        kb.add_documents(build_school_score_docs(gaokao_data.get("school_scores", {})))

    subject_file = data_dir / "subject_review.json"
    if subject_file.exists():
        with open(subject_file, "r", encoding="utf-8") as f:
            subject_data = json.load(f)
        kb.add_documents(build_subject_eval_docs(subject_data))

    kb.embed_all()

    return kb


if __name__ == "__main__":
    kb = build_all()
    print(f"知识库构建完成:")
    print(f"  文档总数: {kb.document_count}")
    print(f"  类别: {kb.categories}")
    print(f"  缓存大小: {kb._cache.size()}")

    print("\n测试搜索 '北京的大学':")
    results = kb.semantic_search("北京的大学", category="university_basic", top_k=3)
    for r in results:
        print(f"  [{r.rank}] {r.document.metadata.get('school_name', '')} (score: {r.score:.3f})")

    print("\n测试搜索 '计算机专业强校':")
    results = kb.semantic_search("计算机专业强校", category="subject_eval", top_k=3)
    for r in results:
        print(f"  [{r.rank}] {r.document.metadata.get('school_name', '')} - {r.document.metadata.get('subject_name', '')} (score: {r.score:.3f})")
