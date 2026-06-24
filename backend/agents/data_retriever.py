"""数据检索 Agent V5: 多阶段检索 + 持久化向量索引 + 多字段语义增强。"""

import logging
import time
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field

from backend.models.agent_output import UserProfile, DataRetrievalResult
from backend.models.config import settings
from backend.services.knowledge_base import KnowledgeBase
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_index import VectorIndex, build_major_document

# Qdrant 向量数据库（可选，当 vector_index_engine="qdrant" 时使用）
from backend.services.qdrant_index import QdrantIndex

logger = logging.getLogger(__name__)

# 行业-专业关键词映射（用于关键词召回）
_INDUSTRY_MAJOR_MAP = {
    "互联网": ["计算机", "软件", "人工智能", "数据", "物联网", "信息安全"],
    "人工智能": ["人工智能", "数据科学", "计算机", "数学", "算法"],
    "半导体/芯片": ["微电子", "集成电路", "电子", "半导体", "光电"],
    "新能源": ["新能源", "电气工程", "能源", "动力", "材料"],
    "医疗": ["临床", "口腔", "护理", "药学", "医学"],
    "金融": ["金融", "会计", "经济", "财务"],
    "制造业": ["机械", "自动化", "材料", "工业"],
    "汽车": ["车辆", "机械", "自动化", "电子"],
    "通信": ["通信", "电子", "信息", "网络"],
    "网络安全": ["信息安全", "网络", "计算机", "软件"],
    "教育": ["师范", "教育"],
    "公务员/体制内": ["公共管理", "法学", "汉语言", "师范"],
    "航空航天": ["航空航天", "机械", "电子", "自动化"],
    "房地产": ["土木", "建筑", "工程管理"],
}


# ──────────────────────────────────────────────
# 中间数据结构
# ──────────────────────────────────────────────

@dataclass
class QueryContext:
    """第一阶段: Query 理解结果。"""
    interest_keywords: List[str] = field(default_factory=list)
    score: int = 0
    province: str = ""
    risk_preference: str = ""
    personality: str = ""
    family_resources: str = ""
    exclude_majors: List[str] = field(default_factory=list)
    prefer_provinces: List[str] = field(default_factory=list)
    prefer_school_types: List[str] = field(default_factory=list)
    query_text: str = ""

    def __post_init__(self):
        # 构建查询文本（用于语义检索）
        parts = self.interest_keywords.copy()
        if self.personality:
            parts.append(self.personality)
        # 只有在有 interest_keywords 时才加入 risk_preference
        if parts and self.risk_preference:
            parts.append(self.risk_preference)
        self.query_text = " ".join(parts) if parts else ""


@dataclass
class RecallItem:
    """单个召回项。"""
    name: str
    data: dict
    scores: Dict[str, float] = field(default_factory=dict)  # {来源: 分数}

    @property
    def raw_score(self) -> float:
        return sum(self.scores.values())


# ──────────────────────────────────────────────
# QueryCache: 基于查询上下文的 TTL 缓存
# ──────────────────────────────────────────────

class QueryCache:
    """TTL 查询结果缓存，用于提升重复查询的检索性能。
    
    Args:
        ttl_seconds: 缓存有效期（秒）
        max_size: 最大缓存条目数
    """
    
    def __init__(self, ttl_seconds: int = 300, max_size: int = 500):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._cache: OrderedDict[str, tuple] = OrderedDict()  # key -> (result, expire_time)
    
    def _make_key(self, ctx: QueryContext) -> str:
        """根据查询上下文生成缓存键。"""
        key_parts = [
            ",".join(sorted(ctx.interest_keywords)),
            str(ctx.score),
            ctx.family_resources,
            ctx.personality,
        ]
        raw_key = "|".join(key_parts)
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()
    
    def get(self, ctx: QueryContext) -> Optional[DataRetrievalResult]:
        """获取缓存结果，过期自动删除。"""
        key = self._make_key(ctx)
        if key in self._cache:
            result, expire_time = self._cache[key]
            if time.time() < expire_time:
                self._cache.move_to_end(key)
                return result
            else:
                del self._cache[key]
        return None
    
    def put(self, ctx: QueryContext, result: DataRetrievalResult):
        """写入缓存结果。"""
        key = self._make_key(ctx)
        expire_time = time.time() + self._ttl
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = (result, expire_time)
    
    @property
    def size(self) -> int:
        return len(self._cache)
    
    def clear(self):
        self._cache.clear()


# ──────────────────────────────────────────────
# DataRetrieverV4
# ──────────────────────────────────────────────

class DataRetrieverV4:
    """多阶段检索器。

    四阶段流程:
    1. Query Understanding - 解析用户画像为结构化查询
    2. Multi-Path Recall    - 语义/规则/关键词 3路并行召回
    3. Score Fusion        - 可配置权重融合
    4. Re-Ranking          - 重排序 + 可解释输出
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.embedding = EmbeddingService()
        self.vector_index: Optional[VectorIndex] = None
        
        # 索引持久化路径
        self._index_path = str(
            Path(__file__).resolve().parent.parent / "cache" / "major_vector_index_v5"
        )
        
        # 懒加载标志
        self._index_built = False
        
        # 查询缓存（TTL 300秒，最大500条）
        self.query_cache = QueryCache(ttl_seconds=300, max_size=500)

        # 权重（可从配置加载）
        self._w_semantic = settings.retrieval_weight_semantic
        self._w_rule = settings.retrieval_weight_rule
        self._w_keyword = settings.retrieval_weight_keyword
    
    def _ensure_vector_index(self) -> None:
        """懒加载构建向量索引（V5: 支持持久化、多字段 embedding、Qdrant 云端）。"""
        if self._index_built:
            return

        all_majors = self.kb.all_majors
        all_names = list(all_majors.keys())

        # V5: 支持 Qdrant 云端向量数据库
        engine = settings.vector_index_engine.lower()
        if engine == "qdrant":
            self._init_qdrant_index(all_majors, all_names)
        else:
            self._init_local_index(all_majors, all_names)

        self._index_built = True

    def _init_qdrant_index(self, all_majors: Dict[str, dict], all_names: List[str]) -> None:
        """初始化 Qdrant 云端向量索引。"""
        logger.info("Initializing Qdrant vector index (Collection=%s)...", settings.qdrant_collection)

        self.vector_index = QdrantIndex(
            dimension=self.embedding.dimension,
            collection_name=settings.qdrant_collection,
        )

        if not self.vector_index.is_ready:
            logger.warning(
                "Qdrant 不可用，回退到本地向量索引 (engine=%s)",
                settings.vector_index_engine,
            )
            self._init_local_index(all_majors, all_names)
            return

        # 检查是否需要重建
        if self.vector_index.needs_rebuild(all_majors):
            logger.info("Qdrant Collection 数据不一致，重建中...")
            self._rebuild_index(all_majors, all_names)
            self.vector_index.save(settings.qdrant_collection)
        else:
            logger.info(
                "Qdrant 索引已就绪: %d majors",
                self.vector_index.count,
            )

    def _init_local_index(self, all_majors: Dict[str, dict], all_names: List[str]) -> None:
        """初始化本地向量索引 (FAISS/NumPy)。"""
        self.vector_index = VectorIndex(
            dimension=self.embedding.dimension,
            engine=settings.vector_index_engine,
            persist_path=self._index_path,
        )

        # V5: 优先加载持久化索引，不存在则构建新索引
        if VectorIndex.exists(self._index_path):
            logger.info("Loading persisted vector index from %s", self._index_path)
            # 索引已在 VectorIndex._ensure_loaded 中自动加载
            self._index_built = True

            # 检查是否需要重建（对比哈希）
            if self.vector_index.needs_rebuild(all_majors):
                logger.info("Index outdated, rebuilding...")
                self._rebuild_index(all_majors, all_names)
                self.vector_index.save(self._index_path)
            else:
                logger.info(
                    "Vector index loaded: %d majors (engine=%s)",
                    self.vector_index.count, self.vector_index.engine,
                )
        else:
            # 首次构建索引
            logger.info("Building new vector index...")
            self._rebuild_index(all_majors, all_names)
            # 保存索引到磁盘
            self.vector_index.save(self._index_path)
            logger.info("Vector index saved to %s", self._index_path)
        
        self._index_built = True
    
    def _rebuild_index(self, all_majors: Dict[str, dict], all_names: List[str]) -> None:
        """全量重建向量索引（使用多字段融合文本）。
        
        Args:
            all_majors: {专业名: 专业数据} 字典
            all_names: 专业名称列表
        """
        # V5: 使用多字段融合文本构建 embedding
        texts = [build_major_document(data, name) for name, data in all_majors.items()]
        embeddings = self.embedding.get_embeddings(texts)
        metadatas = [{"name": name, **data} for name, data in all_majors.items()]
        
        self.vector_index.add_batch(embeddings, metadatas)
        logger.info(
            "Vector index built: %d majors, engine=%s",
            len(all_names), self.vector_index.engine,
        )
    
    def add_major(self, name: str, major_data: dict) -> bool:
        """增量添加专业到索引。
        
        Args:
            name: 专业名称
            major_data: 专业数据字典
            
        Returns:
            True 表示新增/更新成功，False 表示内容未变更
        """
        if self.vector_index is None or not self._index_built:
            self._ensure_vector_index()
        
        # 使用多字段融合文本计算 embedding
        doc_text = build_major_document(major_data, name)
        embedding = self.embedding.get_embedding(doc_text)
        
        return self.vector_index.add_by_name(name, embedding, major_data)
    
    def remove_major(self, name: str) -> bool:
        """从索引中移除专业。
        
        Args:
            name: 专业名称
            
        Returns:
            True 表示成功移除，False 表示未找到
        """
        if self.vector_index is None or not self._index_built:
            return False
        
        result = self.vector_index.remove_by_name(name)
        if result and self.vector_index.is_dirty:
            self.vector_index.save(self._index_path)
        return result

    # ── 公共 API ──

    async def retrieve(self, profile: UserProfile) -> DataRetrievalResult:
        """执行完整的四阶段检索流程。

        Args:
            profile: 用户画像

        Returns:
            检索结果（含可解释性信息）
        """
        t0 = time.time()

        # Stage 1: Query Understanding
        ctx = self._query_understand(profile)
        logger.info(
            "Stage1 Query Understanding: interests=%s, score=%d, personality=%s",
            ctx.interest_keywords, ctx.score, ctx.personality,
        )

        # 阶段 0: 查询缓存检查
        cached = self.query_cache.get(ctx)
        if cached is not None:
            elapsed = time.time() - t0
            logger.info("Query cache hit (%.1fms)", elapsed * 1000)
            return cached

        # Stage 2: Multi-Path Recall
        recall_items = self._multi_path_recall(ctx)
        logger.info("Stage2 Recall: %d unique items from 3 paths", len(recall_items))

        if not recall_items:
            # 降级：返回空结果
            result = DataRetrievalResult(
                majors=[],
                industries=[],
                filter_reason="未找到匹配的专业",
                retrieval_meta={
                    "strategy": "v4",
                    "weights": {"semantic": self._w_semantic, "rule": self._w_rule, "keyword": self._w_keyword},
                    "recall_sources": {},
                },
            )
            self.query_cache.put(ctx, result)
            return result

        # Stage 3: Score Fusion
        fused = self._score_fusion(recall_items, ctx)
        logger.info("Stage3 Fusion: top score=%.3f", fused[0].raw_score if fused else 0)

        # Stage 4: Re-Ranking + Explain
        majors_result = self._re_rank_and_explain(fused, ctx)
        industries_result = self._find_related_industries(majors_result, ctx)

        elapsed = time.time() - t0
        filter_reason = self._build_filter_reason(ctx, len(majors_result), elapsed)

        result = DataRetrievalResult(
            majors=majors_result,
            industries=industries_result,
            filter_reason=filter_reason,
            retrieval_meta={
                "strategy": "v4",
                "weights": {"semantic": self._w_semantic, "rule": self._w_rule, "keyword": self._w_keyword},
                "recall_sources": {
                    "semantic_count": sum(1 for r in recall_items.values() if r.scores.get("semantic", 0) > 0),
                    "rule_count": sum(1 for r in recall_items.values() if r.scores.get("rule", 0) > 0),
                    "keyword_count": sum(1 for r in recall_items.values() if r.scores.get("keyword", 0) > 0),
                },
                "elapsed_ms": round(elapsed * 1000, 2),
            },
        )
        
        # 写入查询缓存
        self.query_cache.put(ctx, result)
        return result

    # ── Stage 1: Query Understanding ──

    def _query_understand(self, profile: UserProfile) -> QueryContext:
        """解析用户画像为结构化查询上下文。"""
        ctx = QueryContext(
            interest_keywords=list(profile.interests) if profile.interests else [],
            score=profile.score,
            province=profile.province,
            risk_preference=profile.risk_preference,
            personality=profile.personality or "",
            family_resources=profile.family_resources or "普通",
            exclude_majors=list(profile.constraints) if profile.constraints else [],
        )
        return ctx

    # ── Stage 2: Multi-Path Recall ──

    def _multi_path_recall(self, ctx: QueryContext) -> Dict[str, RecallItem]:
        """多路并行召回，返回去重后的候选集合。

        Returns:
            {major_name: RecallItem}
        """
        all_majors = self.kb.all_majors
        items: Dict[str, RecallItem] = {}

        def _ensure(name: str):
            if name not in items and name in all_majors:
                items[name] = RecallItem(name=name, data=all_majors[name])

        # Path 1: 语义召回（embedding 向量检索）
        self._recall_semantic(ctx, items, _ensure)

        # Path 2: 规则召回（结构化过滤）
        self._recall_rule(ctx, items, _ensure)

        # Path 3: 关键词召回（关键词+行业映射）
        self._recall_keyword(ctx, items, _ensure)

        return items

    def _recall_semantic(
        self, ctx: QueryContext, items: Dict[str, RecallItem], _ensure
    ):
        """语义召回：基于 VectorIndex 的 FAISS/Numpy 向量检索。
        
        V5: 使用多字段融合文本构建的专业向量索引，语义表达能力显著增强。
        """
        if not ctx.query_text:
            return
        
        # 确保向量索引已构建
        self._ensure_vector_index()
        
        if self.vector_index is None:
            return
        
        # 获取查询文本的向量
        query_embedding = self.embedding.get_embedding(ctx.query_text)
        
        # 使用 VectorIndex 进行 TopK 检索
        all_names = list(self.kb.all_majors.keys())
        top_k = len(all_names)
        
        # V5 优化：Qdrant 引擎支持 with_payload 参数，避免 N+1 HTTP 请求
        try:
            results = self.vector_index.search(query_embedding, top_k=top_k, with_payload=True)
            use_payload = True
        except TypeError:
            # 本地 VectorIndex 不支持 with_payload 参数，回退到旧接口
            results = self.vector_index.search(query_embedding, top_k=top_k)
            use_payload = False
        
        # 将检索结果填入召回项
        kb_names = set(self.kb.all_majors.keys())
        if use_payload:
            # Qdrant: (idx, score, payload)
            for idx, score, meta in results:
                name = meta.get("name", "")
                if not name and idx < len(all_names):
                    name = all_names[idx]
                if name and name in kb_names and score > 0.0:
                    _ensure(name)
                    items[name].scores["semantic"] = score
        else:
            # 本地 VectorIndex: (idx, score)
            for idx, score in results:
                meta = self.vector_index.get_metadata(idx)
                name = meta.get("name", "")
                if not name and idx < len(all_names):
                    name = all_names[idx]
                if name and name in kb_names and score > 0.0:
                    _ensure(name)
                    items[name].scores["semantic"] = score

    def _recall_rule(
        self, ctx: QueryContext, items: Dict[str, RecallItem], _ensure
    ):
        """规则召回：基于结构化字段过滤。"""
        for name, data in self.kb.all_majors.items():
            score = 0.0

            # 就业率得分
            emp_rate = data.get("employment_rate", 0.5)
            score += emp_rate * 0.4

            # 薪资得分
            salary = data.get("avg_salary", 0)
            score += min(salary / 15000.0, 1.0) * 0.3

            # 资源兼容性得分
            threshold = data.get("resource_threshold", "medium")
            family = ctx.family_resources
            if self._resource_compatible(threshold, family):
                score += 0.3
            else:
                # 资源不兼容时，大幅降低分数（仅保留部分基础分）
                score *= 0.3

            if score > 0.3:  # 阈值过滤
                _ensure(name)
                items[name].scores["rule"] = score

    def _recall_keyword(
        self, ctx: QueryContext, items: Dict[str, RecallItem], _ensure
    ):
        """关键词召回：基于关键词匹配 + 行业映射。"""
        all_majors = self.kb.all_majors

        for name, data in all_majors.items():
            kw_score = 0.0

            # 1. 兴趣关键词 → 专业名匹配
            for kw in ctx.interest_keywords:
                if kw.lower() in name.lower():
                    kw_score += 0.5

            # 2. 兴趣关键词 → 专业关键词列表匹配
            major_keywords = data.get("keywords", [])
            for kw in ctx.interest_keywords:
                for mkw in major_keywords:
                    if kw.lower() in mkw.lower() or mkw.lower() in kw.lower():
                        kw_score += 0.3

            # 3. 兴趣关键词 → 行业 → 专业 间接匹配
            for kw in ctx.interest_keywords:
                for industry_name, industry_kws in _INDUSTRY_MAJOR_MAP.items():
                    if any(kw.lower() in ikw.lower() or ikw.lower() in kw.lower()
                           for ikw in industry_kws):
                        # 检查该专业是否属于该行业
                        major_industries = data.get("industries", [])
                        if industry_name in major_industries:
                            kw_score += 0.2

            if kw_score > 0.1:
                _ensure(name)
                items[name].scores["keyword"] = kw_score

    # ── Stage 3: Score Fusion ──

    def _score_fusion(
        self, items: Dict[str, RecallItem], ctx: QueryContext
    ) -> List[RecallItem]:
        """加权融合多路召回分数，排序返回。

        final_score = w_semantic * semantic + w_rule * rule + w_keyword * keyword
        """
        fused = []

        for name, item in items.items():
            s_semantic = item.scores.get("semantic", 0.0)
            s_rule = item.scores.get("rule", 0.0)
            s_keyword = item.scores.get("keyword", 0.0)

            # 归一化各维度分数
            s_semantic = min(s_semantic, 1.0)
            s_rule = min(s_rule, 1.0)
            s_keyword = min(s_keyword, 1.0)

            final = (
                self._w_semantic * s_semantic +
                self._w_rule * s_rule +
                self._w_keyword * s_keyword
            )

            item.scores["final"] = final
            fused.append(item)

        # 按最终分数排序
        fused.sort(key=lambda x: x.scores.get("final", 0.0), reverse=True)
        return fused

    # ── Stage 4: Re-Ranking + Explain ──

    def _re_rank_and_explain(
        self, fused: List[RecallItem], ctx: QueryContext
    ) -> List[dict]:
        """重排序并生成可解释性输出。

        返回 top 10 专业，每个包含：
        - match_reason: 匹配原因
        - data_support: 数据支撑
        - recommend_reason: 推荐理由
        - risk_warnings: 风险提示
        """
        all_majors = self.kb.all_majors
        top_n = min(10, len(fused))
        results = []

        for i in range(top_n):
            item = fused[i]
            name = item.name
            data = item.data

            # 构建 match_reason
            match_reasons = []
            s_semantic = item.scores.get("semantic", 0.0)
            s_keyword = item.scores.get("keyword", 0.0)
            if s_semantic > 0.1:
                match_reasons.append(f"语义相似度: {s_semantic:.3f}")
            if s_keyword > 0.1:
                keywords_matched = []
                for ikw in ctx.interest_keywords:
                    if ikw.lower() in name.lower():
                        keywords_matched.append(ikw)
                    else:
                        major_kws = data.get("keywords", [])
                        if any(ikw.lower() in mkw.lower() or mkw.lower() in ikw.lower()
                               for mkw in major_kws):
                            keywords_matched.append(ikw)
                if keywords_matched:
                    match_reasons.append(f"匹配兴趣: {', '.join(keywords_matched)}")
            if item.scores.get("rule", 0.0) > 0.3:
                match_reasons.append(f"规则匹配得分: {item.scores['rule']:.3f}")

            # 构建 data_support
            emp_rate = data.get("employment_rate", 0.0)
            avg_salary = data.get("avg_salary", 0)
            data_support = [
                f"就业率: {emp_rate * 100:.1f}%",
                f"平均薪资: {avg_salary}元/月",
            ]
            career_paths = data.get("career_paths", [])
            if career_paths:
                data_support.append(f"职业路径: {career_paths[0]}")

            # 构建 recommend_reason
            recommend_reason = data.get("description", "")

            # 构建 risk_warnings
            risk_warnings = []
            resource_threshold = data.get("resource_threshold", "medium")
            if resource_threshold == "high" and not self._resource_compatible(
                resource_threshold, ctx.family_resources
            ):
                risk_warnings.append("该专业需要较多家庭资源支持")
            if emp_rate < 0.75:
                risk_warnings.append("就业率偏低，需谨慎考虑")
            if avg_salary < 6000:
                risk_warnings.append("起薪偏低，需关注后续发展空间")

            results.append({
                "name": name,
                "employment_rate": emp_rate,
                "avg_salary": avg_salary,
                "description": recommend_reason,
                "top_directions": data.get("top_directions", []),
                "resource_threshold": resource_threshold,
                "match_score": round(item.scores.get("final", 0.0), 3),
                "match_reason": "；".join(match_reasons) if match_reasons else "基础推荐",
                "data_support": data_support,
                "recommend_reason": recommend_reason,
                "risk_warnings": risk_warnings,
                "courses": data.get("courses", []),
                "skills_required": data.get("skills_required", []),
                "personality_fit": data.get("personality_fit", []),
                "career_paths": career_paths,
                "industries": data.get("industries", []),
            })

        return results

    # ─ 辅助方法 ──

    def _resource_compatible(self, threshold: str, family_resources: str) -> bool:
        resource_levels = {"low": 1, "medium": 2, "high": 3}
        threshold_val = resource_levels.get(threshold, 2)
        family_map = {
            "充裕": 3, "充足": 3, "高": 3,
            "普通": 2, "一般": 2, "中等": 2,
            "不足": 1, "低": 1, "困难": 1,
        }
        family_val = family_map.get(family_resources, 2)
        return family_val >= threshold_val

    def _find_related_industries(
        self, majors_result: List[dict], ctx: QueryContext
    ) -> List[dict]:
        """基于检索结果查找相关行业。"""
        all_industries = self.kb.all_industries
        matched = {}

        # 从 majors 的 industries 字段直接获取
        for major in majors_result:
            for industry_name in major.get("industries", []):
                if industry_name in all_industries:
                    ind_data = all_industries[industry_name]
                    matched[industry_name] = {
                        "name": industry_name,
                        "entry_barrier": ind_data.get("entry_barrier", "medium"),
                        "salary_range": ind_data.get("salary_range", {}),
                        "description": ind_data.get("description", ""),
                        "top_employers": ind_data.get("top_employers", []),
                    }

        # 也从兴趣关键词查找行业
        for kw in ctx.interest_keywords:
            for industry_name, industry_kws in _INDUSTRY_MAJOR_MAP.items():
                if any(kw.lower() in ikw.lower() or ikw.lower() in kw.lower()
                       for ikw in industry_kws):
                    if industry_name in all_industries and industry_name not in matched:
                        ind_data = all_industries[industry_name]
                        matched[industry_name] = {
                            "name": industry_name,
                            "entry_barrier": ind_data.get("entry_barrier", "medium"),
                            "salary_range": ind_data.get("salary_range", {}),
                            "description": ind_data.get("description", ""),
                            "top_employers": ind_data.get("top_employers", []),
                        }

        return list(matched.values())

    def _build_filter_reason(
        self, ctx: QueryContext, result_count: int, elapsed_ms: float
    ) -> str:
        parts = []
        if ctx.interest_keywords:
            parts.append(f"兴趣匹配: {', '.join(ctx.interest_keywords)}")
        parts.append(f"分数段: {ctx.score}分")
        parts.append(f"检索到 {result_count} 个候选专业")
        parts.append(f"耗时: {elapsed_ms:.0f}ms")
        return "。".join(parts)
