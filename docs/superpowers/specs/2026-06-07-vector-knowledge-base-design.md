# 高考志愿填报智能 Agent 系统设计文档

**日期**: 2026-06-07\
**主题**: 从"数据检索"到"决策推荐"的系统升级\
**版本**: v2.0

***

## 1. 系统愿景

> **目标**: 构建一个像张雪峰一样能"给建议"的AI，而不是只会"查数据"的AI

### 1.1 核心能力升级

| 能力维度 | 当前状态     | 升级目标         |
| ---- | -------- | ------------ |
| 信息获取 | JSON精确查询 | 混合检索（结构化+向量） |
| 用户理解 | 无        | 用户画像解析       |
| 决策能力 | 无        | 冲稳保推荐模型      |
| 推理能力 | 单一LLM调用  | 多步Agent推理    |
| 输出质量 | 原始数据     | 可解释推荐报告      |
| 数据规模 | 千级       | 十万级可扩展       |

***

## 2. 系统架构总览

### 2.1 新架构图（文字描述）

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户交互层                                    │
│                    (Web/App/小程序)                                  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent 调度层 (Orchestrator)                        │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ 意图解析  │───▶│ 用户画像  │───▶│ 任务规划  │───▶│ 工具调度  │      │
│  │ Parser   │    │ Profiler │    │ Planner  │    │ Executor │      │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘      │
│                                                       │             │
│                              ┌────────────────────────┘             │
│                              ▼                                      │
│                    ┌──────────────────┐                             │
│                    │  多角色推理引擎    │                             │
│                    │ MultiRoleReasoner│                             │
│                    └────────┬─────────┘                             │
│                             │                                       │
│                    ┌────────▼─────────┐                             │
│                    │   Refiner + 输出  │                             │
│                    └────────┬─────────┘                             │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       工具层 (Tools)                                 │
│                                                                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │结构化筛选   │  │向量检索     │  │分数匹配     │  │学科评估     │    │
│  │FilterTool  │  │SearchTool  │  │ScoreTool   │  │EvalTool    │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
│                                                                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                    │
│  │冲稳保分类   │  │Rerank排序   │  │推荐解释     │                    │
│  │RiskTool    │  │RerankTool  │  │ExplainTool │                    │
│  └────────────┘  └────────────┘  └────────────┘                    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       数据服务层                                     │
│                                                                     │
│  ┌─────────────────┐          ┌─────────────────┐                   │
│  │ KnowledgeBase   │          │VectorKnowledge  │                   │
│  │ (精确查询)       │          │Base (语义检索)   │                   │
│  │                 │          │                 │                   │
│  │ - universities  │          │ - university_   │                   │
│  │ - majors        │          │   basic          │                   │
│  │ - industries    │          │ - score_batch    │                   │
│  │ - decision_rules│          │ - score_school   │                   │
│  └─────────────────┘          │ - score_major    │                   │
│                               │ - subject_eval   │                   │
│                               └────────┬────────┘                   │
│                                        │                            │
│  ┌─────────────────┐          ┌────────▼────────┐                   │
│  │ EmbeddingService│          │  EmbeddingCache  │                   │
│  │ (向量化)         │◀────────▶│  (持久化)        │                   │
│  └─────────────────┘          └─────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       数据存储层                                     │
│                                                                     │
│  JSON文件 ←→ 向量索引文件 ←→ FAISS/HNSW索引 ←→ 用户记忆DB             │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心数据流

```
用户输入: "安徽理科580分，想学计算机，求推荐"
    │
    ▼
[1] 意图解析 → 识别为"志愿填报推荐"意图
    │
    ▼
[2] 用户画像解析 → {score:580, province:"安徽", subject:"理科", target_major:"计算机"}
    │
    ▼
[3] 任务规划 → 拆解为多个子任务:
    ├─ 查询安徽理科2025年批次线
    ├─ 检索计算机相关院校
    ├─ 匹配分数段(560-600)的学校
    └─ 获取学科评估数据
    │
    ▼
[4] 工具执行 → 混合检索:
    ├─ 结构化过滤: 省份=安徽, 年份=2025, 科类=理科
    ├─ 向量召回: "计算机强校"、"安徽本地院校"
    └─ 分数匹配: 录取线在560-600之间的学校
    │
    ▼
[5] 结果融合 → Rerank排序
    final_score = 0.3*embedding + 0.4*分数匹配 + 0.2*学科评估 + 0.1*偏好匹配
    │
    ▼
[6] 冲稳保分类 → 
    冲: 分数差 < -10 的学校
    稳: -10 ≤ 分数差 ≤ 10 的学校
    保: 分数差 > 10 的学校
    │
    ▼
[7] 推荐理由生成 → 为每所学校生成解释
    "推荐: 合肥工业大学(稳)
     - 分数匹配度高(高出省控线15分)
     - 计算机专业较强(学科评估B+)
     - 位于安徽，地理匹配"
    │
    ▼
[8] 输出推荐报告 → 结构化JSON + 自然语言解释
```

***

## 3. 模块详细设计

### 3.1 用户画像解析模块 (UserProfiler)

#### 3.1.1 功能

将用户自然语言输入解析为结构化画像JSON。

#### 3.1.2 数据模型

```python
@dataclass
class UserProfile:
    """用户画像"""
    score: Optional[int]           # 高考分数
    province: Optional[str]        # 生源省份
    subject_type: Optional[str]    # 科类: 理科/文科/物理类/历史类
    target_majors: List[str]       # 目标专业列表
    risk_preference: str           # 风险偏好: 冲/稳/保/均衡
    city_preference: List[str]     # 城市偏好
    school_types: List[str]        # 学校类型: 985/211/公办/民办
    degree_level: str              # 学历层次: 本科/专科
    career_goals: List[str]        # 职业目标
    constraints: List[str]         # 约束条件(如"不去偏远地区")
    
    # 置信度评分(0-1)，表示每个字段的可靠程度
    confidence: Dict[str, float]
```

#### 3.1.3 解析策略

**三层解析器**:

```
Layer 1: 规则解析 (RuleParser)
  - 正则提取数字(分数)
  - 关键词匹配(省份、科类)
  - 快速、确定性高

Layer 2: LLM解析 (LLMParser)
  - 复杂语义理解
  - 缺失信息推理
  - 输出标准JSON

Layer 3: 画像补全 (ProfileCompleter)
  - 基于分数推断批次
  - 基于省份推断科类名称
  - 基于历史对话补全
```

#### 3.1.4 示例

**输入**: "我是安徽的，今年考了580，理科，想学计算机相关的，最好能在本地"

**输出**:

```json
{
  "score": 580,
  "province": "安徽",
  "subject_type": "理科",
  "target_majors": ["计算机", "软件工程", "人工智能"],
  "risk_preference": "稳",
  "city_preference": ["安徽", "合肥"],
  "school_types": [],
  "degree_level": "本科",
  "career_goals": [],
  "constraints": ["优先安徽本地"],
  "confidence": {
    "score": 0.99,
    "province": 0.99,
    "subject_type": 0.99,
    "target_majors": 0.85,
    "city_preference": 0.80
  }
}
```

***

### 3.2 混合检索系统 (HybridSearch)

#### 3.2.1 核心接口

```python
def hybrid_search(
    query: str,              # 原始查询文本
    filters: Dict,           # 结构化过滤条件
    user_profile: UserProfile, # 用户画像
    top_k: int = 20          # 召回数量
) -> List[SearchResult]
```

#### 3.2.2 检索流程

```
                    用户查询 + filters
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
    ┌─────────────┐       ┌─────────────┐
    │ 结构化过滤   │       │ 向量召回     │
    │ (Hard)      │       │ (Soft)      │
    │             │       │             │
    │ - 省份      │       │ - 语义匹配   │
    │ - 年份      │       │ - 专业匹配   │
    │ - 科类      │       │ - 城市匹配   │
    │ - 批次      │       │ - 偏好匹配   │
    │ - 学校类型   │       │             │
    └──────┬──────┘       └──────┬──────┘
           │                     │
           └──────────┬──────────┘
                      ▼
              ┌─────────────┐
              │ 结果融合     │
              │ (Union)     │
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │ Rerank排序   │
              └──────┬──────┘
                     ▼
              ┌─────────────┐
              │ Top-K返回    │
              └─────────────┘
```

#### 3.2.3 结构化过滤 (Hard Filter)

```python
class FilterCondition:
    """过滤条件"""
    field: str           # 过滤字段
    operator: str        # =, !=, in, range
    value: Any           # 过滤值


class StructuredFilter:
    """结构化过滤器"""
    
    def apply(self, documents: List[VectorDocument]) -> List[VectorDocument]:
        """应用过滤条件，返回符合条件的文档"""
        for condition in self.conditions:
            if condition.operator == "=":
                documents = [d for d in documents if d.metadata.get(condition.field) == condition.value]
            elif condition.operator == "in":
                documents = [d for d in documents if d.metadata.get(condition.field) in condition.value]
            elif condition.operator == "range":
                documents = [d for d in documents 
                            if condition.value[0] <= d.metadata.get(condition.field, 0) <= condition.value[1]]
        return documents
```

**过滤字段映射**:

| 用户画像字段        | 数据字段                  | 操作符   |
| ------------- | --------------------- | ----- |
| province      | 生源地/所在地               | =     |
| subject\_type | 文理分科                  | =     |
| school\_types | tier (985/211)        | in    |
| score         | min\_score/max\_score | range |
| degree\_level | 办学层次                  | =     |

#### 3.2.4 向量召回 (Soft Recall)

**多查询策略**:

```python
def expand_queries(user_profile: UserProfile) -> List[str]:
    """将用户画像扩展为多个查询"""
    queries = []
    
    # 专业查询
    if user_profile.target_majors:
        queries.append(f"{user_profile.target_majors[0]}专业强校")
        queries.append(f"{user_profile.target_majors[0]}学科建设好的大学")
    
    # 地域查询
    if user_profile.city_preference:
        queries.append(f"{user_profile.city_preference[0]}的大学")
        queries.append(f"{user_profile.city_preference[0]}本地高校")
    
    # 分数段查询
    if user_profile.score:
        queries.append(f"录取分数{user_profile.score}左右的大学")
    
    return queries
```

#### 3.2.5 结果融合

```python
def merge_results(
    filter_results: List[VectorDocument],
    vector_results: List[VectorDocument],
    score_results: List[VectorDocument],
    weights: Dict[str, float] = None
) -> List[VectorDocument]:
    """
    融合多路召回结果
    策略: 取并集，去重，加权排序
    """
    all_docs = {}
    
    # 结构化过滤结果(权重最高，因为必须满足)
    for doc in filter_results:
        all_docs[doc.id] = {"doc": doc, "score": 1.0}
    
    # 向量召回结果
    for i, doc in enumerate(vector_results):
        if doc.id in all_docs:
            all_docs[doc.id]["score"] += 0.3
        else:
            all_docs[doc.id] = {"doc": doc, "score": 0.3}
    
    # 分数匹配结果
    for i, doc in enumerate(score_results):
        if doc.id in all_docs:
            all_docs[doc.id]["score"] += 0.5
        else:
            all_docs[doc.id] = {"doc": doc, "score": 0.5}
    
    # 按分数排序
    sorted_results = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
    return [r["doc"] for r in sorted_results]
```

***

### 3.3 文档增强策略 (DocumentEnhancement)

#### 3.3.1 问题

当前每条数据只有一个文本描述 → 语义表达能力弱，召回率低。

#### 3.3.2 解决方案

一条数据生成 3\~5 条不同语义表达的文档变体。

#### 3.3.3 增强模板

**以高校数据为例**:

```python
def enhance_university_doc(university: Dict) -> List[Dict]:
    """为一条高校数据生成多个语义变体"""
    name = university["name"]
    province = university.get("province", "")
    tier = university.get("tier", "")
    desc = university.get("description", "")
    
    variants = []
    
    # 变体1: 基础描述
    variants.append({
        "text": f"{name}位于{province}，是一所{tier}高校。{desc}",
        "variant_type": "basic"
    })
    
    # 变体2: 地域导向
    variants.append({
        "text": f"{province}的{tier}大学推荐:{name}。{desc}",
        "variant_type": "location_based"
    })
    
    # 变体3: 分数导向
    if "min_score_2025" in university:
        variants.append({
            "text": f"{name}2025年最低录取分{university['min_score_2025']}分，{tier}院校。{desc}",
            "variant_type": "score_based"
        })
    
    # 变体4: 专业导向
    if "strong_majors" in university:
        majors = "、".join(university["strong_majors"])
        variants.append({
            "text": f"如果想学{majors}等专业，{name}是不错的选择。{desc}",
            "variant_type": "major_based"
        })
    
    # 变体5: 对比导向
    variants.append({
        "text": f"与同类{tier}院校相比，{name}的特点是{desc}",
        "variant_type": "comparison"
    })
    
    return variants
```

#### 3.3.4 存储结构

```python
class EnhancedDocument:
    """增强文档"""
    base_id: str           # 原始数据ID
    variant_id: str        # 变体ID (base_id:variant_type)
    variant_type: str      # 变体类型
    text: str              # 文本内容
    embedding: List[float] # 向量
    metadata: Dict         # 原始数据(所有变体共享同一份)
```

**效果**:

* 原始: 1条数据 → 1个文档 → 1次向量匹配

* 增强后: 1条数据 → 5个文档 → 5次向量匹配 → 召回率提升3-5倍

***

### 3.4 Rerank 排序模型 (Reranker)

#### 3.4.1 当前问题

仅使用 embedding 相似度排序 → 无法体现分数匹配、学科评估等关键因素。

#### 3.4.2 多因子排序模型

```python
@dataclass
class RerankConfig:
    """排序配置"""
    # 权重配置
    embedding_weight: float = 0.2      # 语义匹配权重
    score_match_weight: float = 0.35   # 分数匹配权重(最高)
    subject_eval_weight: float = 0.25  # 学科评估权重
    preference_weight: float = 0.2     # 用户偏好权重
    
    # 可动态调整
    def adjust_for_conservative_user(self):
        """保守用户：提高分数匹配权重"""
        self.score_match_weight = 0.5
        self.embedding_weight = 0.1
    
    def adjust_for_aggressive_user(self):
        """激进用户：降低分数匹配权重"""
        self.score_match_weight = 0.2
        self.embedding_weight = 0.3
```

#### 3.4.3 评分计算

```python
class Reranker:
    """重排序器"""
    
    def rerank(
        self,
        documents: List[VectorDocument],
        user_profile: UserProfile,
        config: RerankConfig
    ) -> List[RerankResult]:
        """
        计算综合得分并排序
        """
        results = []
        
        for doc in documents:
            # 1. 语义匹配得分 (0-1)
            embedding_score = self._calc_embedding_score(doc)
            
            # 2. 分数匹配得分 (0-1)
            score_match = self._calc_score_match(doc, user_profile.score)
            
            # 3. 学科评估得分 (0-1)
            subject_eval = self._calc_subject_eval(doc, user_profile.target_majors)
            
            # 4. 偏好匹配得分 (0-1)
            pref_match = self._calc_preference_match(doc, user_profile)
            
            # 综合得分
            final_score = (
                config.embedding_weight * embedding_score +
                config.score_match_weight * score_match +
                config.subject_eval_weight * subject_eval +
                config.preference_weight * pref_match
            )
            
            results.append(RerankResult(
                document=doc,
                final_score=final_score,
                breakdown={
                    "embedding": embedding_score,
                    "score_match": score_match,
                    "subject_eval": subject_eval,
                    "preference": pref_match
                }
            ))
        
        # 按综合得分降序
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
    
    def _calc_score_match(self, doc: VectorDocument, user_score: int) -> float:
        """
        计算分数匹配度
        理想情况: 用户分数略高于录取线 (1-20分)
        """
        min_score = doc.metadata.get("min_score", 0)
        avg_score = doc.metadata.get("avg_score", 0)
        
        if not user_score or not min_score:
            return 0.5  # 无分数信息，给中值
        
        diff = user_score - min_score
        
        if diff < 0:
            return 0.0  # 用户分数低于录取线
        elif diff <= 5:
            return 1.0  # 刚好过线，最佳匹配
        elif diff <= 20:
            return 0.9  # 略高，好匹配
        elif diff <= 50:
            return 0.7  # 偏高，可接受
        else:
            return 0.4  # 太高，浪费分数
```

***

### 3.5 冲稳保推荐模型 (RiskClassifier)

#### 3.5.1 核心逻辑

```python
class RiskClassifier:
    """冲稳保分类器"""
    
    # 可配置的阈值
    CHARGE_THRESHOLD: int = -10   # 冲: 用户分数比录取线低10分以上
    SAFE_THRESHOLD: int = 10      # 稳: 用户分数比录取线高10分以内
    # 保: 用户分数比录取线高10分以上
    
    def classify(
        self,
        user_score: int,
        documents: List[RerankResult]
    ) -> RiskClassificationResult:
        """
        将候选学校分为冲/稳/保三档
        """
        charge = []   # 冲刺
        stable = []   # 稳妥
        safe = []     # 保底
        
        for result in documents:
            min_score = result.document.metadata.get("min_score", 0)
            avg_score = result.document.metadata.get("avg_score", min_score)
            
            # 使用平均分作为参考
            ref_score = avg_score if avg_score else min_score
            diff = user_score - ref_score
            
            risk_info = RiskInfo(
                result=result,
                score_diff=diff,
                min_score=min_score,
                avg_score=avg_score
            )
            
            if diff < self.CHARGE_THRESHOLD:
                risk_info.level = "冲"
                risk_info.description = f"您的分数比往年录取线低{abs(diff)}分，有机会但需冲刺"
                charge.append(risk_info)
            elif diff <= self.SAFE_THRESHOLD:
                risk_info.level = "稳"
                risk_info.description = f"您的分数与往年录取线相当(差{diff}分)，录取概率较高"
                stable.append(risk_info)
            else:
                risk_info.level = "保"
                risk_info.description = f"您的分数比往年录取线高{diff}分，录取把握很大"
                safe.append(risk_info)
        
        return RiskClassificationResult(
            charge=charge[:8],   # 冲: 最多8所
            stable=stable[:10],  # 稳: 最多10所
            safe=safe[:8]        # 保: 最多8所
        )
```

#### 3.5.2 输出结构

```json
{
  "冲": [
    {
      "school": "浙江大学",
      "score_diff": -5,
      "min_score": 585,
      "avg_score": 595,
      "reason": "分数略低，但该校在安徽有断档可能，可冲刺",
      "risk_level": "高"
    }
  ],
  "稳": [
    {
      "school": "合肥工业大学",
      "score_diff": 8,
      "min_score": 572,
      "avg_score": 578,
      "reason": "分数匹配度高，计算机专业评估B+，录取概率大",
      "risk_level": "中"
    }
  ],
  "保": [
    {
      "school": "安徽大学",
      "score_diff": 25,
      "min_score": 555,
      "avg_score": 560,
      "reason": "分数优势明显，本地211，可作为保底",
      "risk_level": "低"
    }
  ]
}
```

***

### 3.6 Agent 调度系统 (AgentOrchestrator)

#### 3.6.1 工具注册表

```python
class ToolRegistry:
    """工具注册表"""
    
    tools = {
        "user_parser": UserParserTool,           # 用户画像解析
        "batch_score_query": BatchScoreQueryTool, # 批次线查询
        "school_filter": SchoolFilterTool,        # 学校筛选
        "vector_search": VectorSearchTool,        # 向量检索
        "score_match": ScoreMatchTool,            # 分数匹配
        "subject_eval": SubjectEvalTool,          # 学科评估查询
        "rerank": RerankTool,                     # 重排序
        "risk_classify": RiskClassifyTool,        # 冲稳保分类
        "explain": ExplainTool,                   # 推荐理由生成
    }
```

#### 3.6.2 Agent 执行流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent 执行流程                                     │
└─────────────────────────────────────────────────────────────────────┘

Step 1: 意图识别 (IntentParser)
  Input: "安徽理科580分，想学计算机，求推荐"
  Output: intent="recommend", confidence=0.95
  
Step 2: 用户画像解析 (UserProfiler)
  Input: 用户文本 + intent
  Output: UserProfile(score=580, province="安徽", ...)
  
Step 3: 任务规划 (Planner)
  Input: UserProfile
  Output: TaskList[
    Task1: query_batch_score(安徽, 理科, 2025),
    Task2: search_schools(filters={province:"安徽", majors:["计算机"]}),
    Task3: match_score_range(580, range=[-20, +30]),
    Task4: query_subject_eval(majors=["计算机"])
  ]
  
Step 4: 工具执行 (ToolExecutor)
  并行执行各工具:
  - batch_score_query → 批次线数据
  - school_filter → 符合条件的学校列表
  - vector_search → 语义相关学校
  - subject_eval → 计算机学科评估结果
  
Step 5: 结果融合 (DataRetriever)
  合并多路召回结果，去重
  
Step 6: Rerank排序 (MultiRoleReasoner)
  计算综合得分，重新排序
  
Step 7: 冲稳保分类 (MultiRoleReasoner)
  按分数差分类
  
Step 8: 推荐理由生成 (Refiner)
  为每所学校生成解释
  
Step 9: 输出构建 (StructuredOutput)
  组装最终推荐报告
```

#### 3.6.3 与现有Agent系统集成

**现有系统已有的Agent组件**:

| 组件                | 文件                       | 当前职责   | 升级后职责        |
| ----------------- | ------------------------ | ------ | ------------ |
| IntentParser      | `intent_parser.py`       | 意图识别   | + 识别推荐意图     |
| UserProfiler      | `user_profiler.py`       | (空/简单) | 新增: 结构化画像解析  |
| Planner           | `planner.py`             | 任务规划   | + 生成检索任务列表   |
| DataRetriever     | `data_retriever.py`      | 数据检索   | + 混合检索+融合    |
| MultiRoleReasoner | `multi_role_reasoner.py` | 推理     | + Rerank+冲稳保 |
| Refiner           | `refiner.py`             | 输出优化   | + 推荐理由生成     |
| Orchestrator      | `orchestrator.py`        | 流程调度   | + 工具调度编排     |

**兼容方案**:

* 保留现有接口，不破坏已有功能

* 新增能力通过工具注册表扩展

* Orchestrator根据意图选择执行路径

***

### 3.7 向量库优化 (VectorKB Optimization)

#### 3.7.1 细粒度分类

```python
CATEGORIES = {
    "university_basic": "高校基本信息",
    "score_batch": "批次录取分数线",
    "score_school": "学校录取分数线",
    "score_major": "专业录取分数线",
    "subject_eval": "学科评估结果",
    "major_info": "专业信息",
    "industry_info": "行业信息",
}
```

#### 3.7.2 Embedding缓存

```python
class EmbeddingCache:
    """Embedding持久化缓存"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, List[float]] = {}
    
    def get(self, text: str) -> Optional[List[float]]:
        """获取缓存的embedding"""
        key = self._hash_key(text)
        if key in self.memory_cache:
            return self.memory_cache[key]
        # 从文件加载
        return self._load_from_file(key)
    
    def set(self, text: str, embedding: List[float]):
        """缓存embedding"""
        key = self._hash_key(text)
        self.memory_cache[key] = embedding
        self._save_to_file(key, embedding)
    
    def _hash_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
```

#### 3.7.3 FAISS/HNSW索引支持

```python
class VectorIndex:
    """向量索引抽象"""
    
    def __init__(self, backend: str = "brute_force"):
        """
        backend: 
          - "brute_force": 暴力搜索(< 1万条)
          - "faiss": FAISS索引(1万-10万条)
          - "hnsw": HNSW图索引(> 10万条)
        """
        self.backend = backend
        self._index = self._build_index(backend)
    
    def add(self, embedding: List[float], doc_id: str):
        """添加向量到索引"""
        ...
    
    def search(self, query_embedding: List[float], top_k: int) -> List[str]:
        """搜索相似向量"""
        ...
```

#### 3.7.4 性能预估

| 数据量      | 索引方式          | 查询延迟    | 内存占用    |
| -------- | ------------- | ------- | ------- |
| < 1万     | 暴力搜索          | < 50ms  | \~20MB  |
| 1万-10万   | FAISS IVF     | < 100ms | \~200MB |
| 10万-100万 | HNSW          | < 200ms | \~2GB   |
| > 100万   | Milvus/Qdrant | < 100ms | 分布式     |

***

### 3.8 输出能力优化 (OutputGenerator)

#### 3.8.1 推荐报告结构

```python
@dataclass
class RecommendationReport:
    """推荐报告"""
    user_profile: UserProfile          # 用户画像
    batch_info: Dict                    # 批次线信息
    charge_schools: List[SchoolRecommend]  # 冲刺院校
    stable_schools: List[SchoolRecommend]  # 稳妥院校
    safe_schools: List[SchoolRecommend]    # 保底院校
    overall_advice: str                  # 总体建议
    risk_warnings: List[str]             # 风险提示
    metadata: Dict                       # 元数据(生成时间、数据版本等)


@dataclass
class SchoolRecommend:
    """单所学校推荐"""
    school_name: str
    risk_level: str           # 冲/稳/保
    score_diff: int           # 分数差
    min_score: int            # 最低录取分
    avg_score: int            # 平均录取分
    reasons: List[str]        # 推荐理由(多条)
    major_advantage: str      # 专业优势说明
    location_advantage: str   # 地域优势
    risk_warning: str         # 风险提示
    raw_data: Dict            # 原始数据
```

#### 3.8.2 推荐理由生成模板

```python
def generate_recommendation_reason(
    school: Dict,
    user_profile: UserProfile,
    rerank_breakdown: Dict
) -> List[str]:
    """生成推荐理由"""
    reasons = []
    
    # 1. 分数匹配理由
    score_diff = user_profile.score - school.get("avg_score", 0)
    if -5 <= score_diff <= 15:
        reasons.append(f"分数匹配度高（{'高出' if score_diff > 0 else '略低'}{abs(score_diff)}分）")
    
    # 2. 学科评估理由
    if school.get("subject_eval"):
        eval_result = school["subject_eval"]
        for major, grade in eval_result.items():
            if grade in ["A+", "A", "A-"]:
                reasons.append(f"{major}专业全国顶尖（评估{grade}）")
            elif grade in ["B+", "B"]:
                reasons.append(f"{major}专业较强（评估{grade}）")
    
    # 3. 地域匹配理由
    if user_profile.city_preference:
        for city in user_profile.city_preference:
            if city in school.get("province", "") or city in school.get("location", ""):
                reasons.append(f"位于{school.get('location', school.get('province'))}，符合地域偏好")
                break
    
    # 4. 学校类型理由
    if school.get("tier") in user_profile.school_types:
        reasons.append(f"{school['tier']}院校，符合学校类型要求")
    
    # 5. 就业优势理由
    if school.get("employment_rate"):
        reasons.append(f"就业率高（{school['employment_rate']}%）")
    
    return reasons[:4]  # 最多4条理由
```

#### 3.8.3 输出示例

```
===============================================
      2025年高考志愿填报推荐报告
===============================================

考生信息:
  省份: 安徽 | 科类: 理科 | 分数: 580
  目标专业: 计算机相关 | 偏好: 本地院校
  
批次线参考:
  本科一批理科: 515分 | 您的分数高出省控线65分

-----------------------------------------------
【冲刺院校】(分数略低，但有机会)
-----------------------------------------------

1. 浙江大学 ⚡
   - 往年录取分: 585-595 | 您的分数差: -5~-15分
   - 推荐理由:
     • 计算机科学与技术评估A+，全国顶尖
     • 工科强势，就业认可度极高
     • 安徽近年偶有断档，可冲刺
   - 风险提示: 录取概率较低，建议放在志愿前面位置

-----------------------------------------------
【稳妥院校】(分数匹配，录取概率高)
-----------------------------------------------

2. 合肥工业大学 ⭐ (推荐)
   - 往年录取分: 572-578 | 您的分数差: +2~+8分
   - 推荐理由:
     • 分数匹配度高（高出2-8分）
     • 计算机科学与技术评估B+，专业较强
     • 位于安徽合肥，符合地域偏好
     • 211院校，性价比高
   - 录取概率: 70-80%

3. 安徽大学 ⭐
   - 往年录取分: 555-560 | 您的分数差: +20~+25分
   - 推荐理由:
     • 分数优势明显，录取把握大
     • 本地211，认可度高
     • 计算机专业评估B
   - 录取概率: 90%+

-----------------------------------------------
【保底院校】(分数优势，确保录取)
-----------------------------------------------

4. 安徽工业大学
   - 往年录取分: 535-545 | 您的分数差: +35~+45分
   - 推荐理由:
     • 分数优势极大，录取几乎确定
     • 计算机专业省内认可
     • 可作为最后保底
   - 录取概率: 99%

-----------------------------------------------
总体建议:
-----------------------------------------------
• 您的分数在安徽理科属于中上水平，可冲击211院校
• 建议志愿顺序: 冲刺2-3所 → 稳妥4-5所 → 保底2-3所
• 计算机专业热门，建议适当降低学校层次保专业
• 关注征集志愿机会

数据版本: 2025年数据 | 生成时间: 2025-07-15
===============================================
```

***

## 4. 数据模型总览

### 4.1 核心数据结构

```python
# 向量文档
VectorDocument:
  id: str                    # 唯一标识
  category: str              # 类别
  text: str                  # 文档化文本
  variant_type: str          # 变体类型(basic/score_based/...)
  embedding: List[float]     # 向量
  metadata: Dict             # 原始结构化数据

# 搜索结果
SearchResult:
  document: VectorDocument
  score: float               # 综合得分
  breakdown: Dict            # 各因子得分明细

# 冲稳保信息
RiskInfo:
  result: SearchResult
  level: str                 # 冲/稳/保
  score_diff: int            # 分数差
  description: str           # 风险描述

# 推荐报告
RecommendationReport:
  user_profile: UserProfile
  charge: List[SchoolRecommend]
  stable: List[SchoolRecommend]
  safe: List[SchoolRecommend]
  advice: str
  warnings: List[str]
```

### 4.2 数据存储格式

```
backend/data/
├── universities.json          # 已有: 核心院校数据
├── universities_list.json     # 新增: 完整高校名单
├── gaokao_scores.json         # 新增: 高考分数线(批次+学校+专业)
├── subject_review.json        # 新增: 学科评估数据
├── majors.json                # 已有
├── industries.json            # 已有
└── decision_rules.json        # 已有

backend/cache/
├── embeddings_cache.db        # Embedding缓存
├── vector_index/              # 向量索引文件
│   ├── university_basic.index
│   ├── score_batch.index
│   ├── score_school.index
│   ├── score_major.index
│   └── subject_eval.index
└── knowledge_base_snapshot.pkl # 知识库快照
```

***

## 5. 性能与扩展性设计

### 5.1 三阶段扩展路线

| 阶段  | 数据量   | 技术方案          | 查询延迟目标  |
| --- | ----- | ------------- | ------- |
| 当前  | < 1万  | 内存+暴力搜索       | < 100ms |
| 扩展  | 1-10万 | FAISS IVF索引   | < 200ms |
| 大规模 | > 10万 | Milvus/Qdrant | < 100ms |

### 5.2 增量更新机制

```python
class IncrementalUpdater:
    """增量更新器"""
    
    def update_document(self, doc_id: str, new_data: Dict):
        """更新单个文档"""
        old_doc = self.kb.get_document(doc_id)
        new_text = self._render_text(new_data)
        new_embedding = self.embedding_service.get_embedding(new_text)
        self.kb.update_embedding(doc_id, new_embedding)
    
    def batch_update(self, changes: List[Dict]):
        """批量更新"""
        for change in changes:
            self.update_document(change["id"], change["data"])
        self._rebuild_index()  # 重建索引
```

### 5.3 并发处理

```python
class AsyncVectorKB:
    """异步向量知识库"""
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """异步搜索"""
        query_embedding = await self._async_get_embedding(query)
        results = await self._async_search(query_embedding, **kwargs)
        return results
    
    async def _async_get_embedding(self, text: str) -> List[float]:
        """异步获取embedding(查缓存或计算)"""
        ...
```

***

## 6. 测试策略

### 6.1 单元测试

| 模块               | 测试用例         |
| ---------------- | ------------ |
| UserProfiler     | 各种输入格式的解析正确性 |
| StructuredFilter | 各种过滤条件的过滤结果  |
| VectorSearch     | 相似度计算准确性     |
| Reranker         | 多因子评分计算      |
| RiskClassifier   | 冲稳保分类边界      |
| DocumentEnhancer | 变体生成数量和质量    |

### 6.2 集成测试

* 完整推荐流程端到端测试

* 模拟用户输入 → 推荐报告输出

### 6.3 性能测试

* 1万条数据查询延迟

* 10万条数据查询延迟

* 并发查询性能

***

## 7. 实施计划

### Phase 1: 基础向量库 (当前文档范围)

* [ ] VectorKnowledgeBase 核心实现

* [ ] 数据转换脚本

* [ ] Embedding缓存

### Phase 2: 用户画像+混合检索

* [ ] UserProfiler 实现

* [ ] StructuredFilter 实现

* [ ] 多路召回融合

### Phase 3: Rerank+冲稳保

* [ ] Reranker 实现

* [ ] RiskClassifier 实现

* [ ] 推荐理由生成

### Phase 4: Agent集成

* [ ] 工具注册表

* [ ] Orchestrator升级

* [ ] 端到端流程

### Phase 5: 性能优化

* [ ] FAISS索引支持

* [ ] 增量更新

* [ ] 异步支持

***

## 8. 风险与缓解

| 风险           | 影响    | 缓解措施       |
| ------------ | ----- | ---------- |
| Embedding计算慢 | 启动时间长 | 缓存+持久化     |
| 向量索引内存大      | OOM   | FAISS/HNSW |
| 数据不准确        | 推荐错误  | 数据验证+人工审核  |
| LLM输出不稳定     | 推荐理由差 | 模板+约束      |

***

## 9. 附录

### 9.1 术语表

| 术语        | 说明                                          |
| --------- | ------------------------------------------- |
| RAG       | Retrieval-Augmented Generation, 检索增强生成      |
| Embedding | 文本的向量表示                                     |
| FAISS     | Facebook AI Similarity Search, 向量检索库        |
| HNSW      | Hierarchical Navigable Small World, 近似最近邻算法 |
| Rerank    | 重排序, 多因子综合评分                                |

### 9.2 参考

* 现有系统: `backend/services/knowledge_base.py`, `backend/services/embedding_service.py`

* Agent框架: `backend/agents/orchestrator.py` 等

* 数据源: CnOpenData 高考相关数据集

