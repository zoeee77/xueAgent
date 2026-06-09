# 张雪峰 AI 志愿填报顾问 — 多智能体 Agent 系统设计文档（工程增强版）

**日期**: 2026-06-07
**版本**: v3.0 — 工程增强版
**作者**: AI Agent System Design

---

## 1. 概述

将现有"张雪峰 AI 志愿填报顾问"从单一 RAG 管道升级为**工程级多智能体决策引擎**，在 v2.0 多 Agent 架构基础上增加：

- Agent 状态控制系统（防崩溃/可恢复）
- 强制结构化输出（Pydantic Schema + JSON 自动修复）
- 向量检索升级（embedding + FAISS 语义匹配 + filter DSL）
- LLM 驱动的 Refiner（自然语言 → 结构化意图 → filter）
- Tool 调用能力（LangChain Agent 模式）
- Memory 系统（SQLite 持久化用户历史偏好）
- 动态评分机制（根据用户偏好调整权重）
- 反对 Agent（Devil's Advocate 提升可信度）
- 全链路可观测性（trace_id + 结构化日志）
- 容错与降级机制（LLM 失败 → 规则 fallback）

---

## 2. 架构设计

### 2.1 整体架构图（v3.0）

```
──────────────────────────────────────────────────────────────────┐
│                       FastAPI Server                              │
│  ┌──────────┐   ┌──────────────────┐   ┌──────────────────────┐  │
│  │ /advise  │──▶│   Orchestrator   │──▶│  Agent Pipeline v3   │  │
│  │ /refine  │   │   (编排器)        │   │                      │  │
│  │ /history │   └─────────────────┘   │ ┌──────────────────┐ │  │
│  └──────────┘            │             │ │ AgentStateManager│ │  │
│                          │             │ │ (状态控制/重试)  │ │  │
│             ┌────────────┴──────┐     │ │ timeout/retry    │ │  │
│             │  Agent Pipeline   │     │ ────────┬─────────┘ │  │
│             │  UserProfiler     │              │             │  │
│             │  DataRetriever+   │   ┌──────────┴──────────  │  │
│             │  MultiRoleReasoner│   │  Structured Output   │  │  │
│             │  Planner          │   │  Engine              │  │  │
│             │  DevilAdvocate    │   │  Pydantic Schema     │  │  │
│             │  Ranker           │   │  + JSON retry/repair │  │  │
│             │  Explainer        │   └──────────┬──────────┘  │  │
│             │  Refiner          │              │             │  │
│             │  IntentParser     │   ┌──────────┴──────────┐  │  │
│             │  ToolAgent        │   │  Logging System      │  │  │
│             └─────────┬─────────┘   │  trace_id 全链路追踪  │  │  │
│                       │             │  Agent 级别日志       │  │  │
│             ┌─────────┴─────────┐   └──────────────────────┘  │  │
│             │  Memory Manager   │                              │  │
│             │  (SQLite 持久化)   │   ┌──────────────────────┐  │  │
│             └───────────────────┘   │  Fallback System     │  │  │
│                                     │  LLM失败→规则降级     │  │  │
│  ┌───────────────────────────────┐  │  数据缺失→提示用户    │  │  │
│  │  Existing Infrastructure      │  │  JSON错误→自动修复    │  │  │
│  │  KnowledgeBase / PromptBuilder│  └──────────────────────┘  │  │
│  │  majors.json / industries.json│                              │  │
│  │  universities.json            │                              │  │
│  └───────────────────────────────┘                              │  │
│                          │                                      │  │
│                          ▼                                      │  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流（v3.0 增强版）

```
用户输入 (score, province, interest, personality, family_resource)
    │
    ▼
[AgentStateManager] ← 初始化 trace_id + AgentState
    │
    ▼
[UserProfiler] ───────────────────────────────────────────────────
  输入: 原始表单数据 + Memory.history_preferences                   │
  输出: UserProfile (Pydantic) + risk_preference                    │
  逻辑: 分数段划分 + 兴趣映射 + 资源评估 + 风险偏好判定              │
  增强: 读取长期 Memory，合并历史偏好到兴趣关键词                     │
    │                                                               │
    ▼                                                               │
[DataRetriever v3] ──────────────────────────────────────────────
  输入: UserProfile + filter DSL                                    │
  输出: Top 10 DataCandidate                                        │
  逻辑: 语义向量检索 (embedding + FAISS) + 结构化过滤                │
  增强: filter 支持 score_range / city_preference / exclude_majors  │
    │                                                               │
    ▼                                                               │
[MultiRoleReasoner] ──────────────────────────────────────────────
  输入: Top Candidates + UserProfile                                │
  输出: 5 × RoleAnalysis (Pydantic, 强制结构化)                     │
  逻辑: asyncio.gather 并行调用 5 角色 LLM 分析                      │
  增强: JSON 解析失败 → 自动重试 (≤2次) → 规则降级                   │
    │                                                               │
    ▼                                                               │
[Planner] ────────────────────────────────────────────────────────
  输入: RoleAnalyses + UserProfile                                  │
  输出: 3 × PlanOption (Pydantic)                                   │
  逻辑: 根据 risk_preference 分配不同风险等级                         │
    │                                                               │
    ▼                                                               │
[Ranker v3] ──────────────────────────────────────────────────────
  输入: PlanOptions + UserProfile                                   │
  输出: RankedResult list (Pydantic)                                │
  逻辑: 动态权重评分                                                │
  增强: 根据 user_pref 动态调整权重                                   │
        例: "想稳定" → risk_weight += 0.1                           │
    │                                                               │
    ▼                                                               │
[DevilAdvocate] ──────────────────────────────────────────────────
  输入: RankedResults + RoleAnalyses                                │
  输出: DevilReport (Pydantic)                                      │
  逻辑: 并行 LLM 调用，输出反对意见/潜在风险                          │
    │                                                               │
    ▼                                                               │
[Explainer] ──────────────────────────────────────────────────────
  输入: RankedResults + RoleAnalyses + DevilReport                  │
  输出: Explanation (Pydantic)                                      │
    │                                                               │
    ▼                                                               │
[AgentOutput] ────────────────────────────────────────────────────
  存储到 Memory (SQLite) + SessionStore                              │
    │                                                               │
    ◀──────────────────────────────────────────────────────────────
    │ 用户反馈
    ▼
[Refiner v3] ─────────────────────────────────────────────────────
  输入: feedback → IntentParser(LLM) → FilterDSL                     │
  输出: new AgentOutput + 差异对比                                   │
  增强: LLM 解析自然语言意图，转换为 filter 再重新走 pipeline          │
```

### 2.3 文件结构（v3.0 完整）

```
backend/
── agents/                              # Agent 层（12 个模块）
│   ├── __init__.py
│   ├── orchestrator.py                  # 总编排器
│   ├── user_profiler.py                 # 用户画像 Agent
│   ├── data_retriever.py                # 数据检索 Agent（增强）
│   ├── multi_role_reasoner.py           # 多角色决策 Agent
│   ├── planner.py                       # 方案生成 Agent
│   ├── ranker.py                        # 排序评分 Agent（动态权重）
│   ├── explainer.py                     # 可解释性 Agent
│   ├── refiner.py                       # 多轮优化 Agent（LLM 驱动）
│   ├── devil_advocate.py                # 反对 Agent（新增）
│   ├── intent_parser.py                 # 意图解析 Agent（新增）
│   └── tool_agent.py                    # 工具调用 Agent（新增）
── state/                               # 状态管理系统（新增）
│   ├── __init__.py
│   ├── agent_state.py                   # AgentState 模型
│   └── state_manager.py                 # AgentStateManager
├── models/
│   ├── __init__.py
│   ├── config.py
│   ├── message.py
│   └── agent_output.py                  # 统一输出模型（Pydantic）
├── memory/                              # 持久化 Memory（新增）
│   ├── __init__.py
│   └── memory_manager.py                # SQLite 存储
├── tools/                               # 工具定义（新增）
│   ├── __init__.py
│   ├── query_university.py              # 查询院校分数线
│   └── query_industry.py                # 查询行业数据
├── fallback/                            # 降级策略（新增）
│   ├── __init__.py
│   └── fallback_handler.py              # LLM 失败降级规则
├── data/
│   ├── majors.json                      # 专业数据
│   ├── industries.json                  # 行业数据
│   ├── universities.json                # 院校数据
│   ├── decision_rules.json              # 决策规则
│   └── vector_store/                    # FAISS 向量存储（新增）
│       └── major_embeddings.faiss
── services/
│   ├── __init__.py
│   ├── knowledge_base.py
│   ├── prompt_builder.py
│   ├── llm_chain.py
│   ├── embedding_service.py             # embedding 服务（新增）
│   └── agent_service.py                 # Agent 服务封装
├── prompts/
│   ├── mental_models.txt
│   ├── decision_heuristics.txt
│   ├── expression_dna.txt
│   ├── roles/                           # 角色 Prompt（新增）
│   │   ├── zhang_xuefeng.txt
│   │   ├── academic_mentor.txt
│   │   ├── industry_expert.txt
│   │   ├── hr.txt
│   │   ├── parent.txt
│   │   └── devil_advocate.txt
│   └── intent_parser.txt                # 意图解析 Prompt（新增）
── __init__.py
├── main.py                              # 修改：新增 /advise 等端点
└── logging_config.py                    # 日志配置（新增）

tests/
├── test_user_profiler.py
── test_data_retriever.py
├── test_multi_role_reasoner.py
├── test_planner.py
├── test_ranker.py
── test_explainer.py
├── test_refiner.py
├── test_orchestrator.py
├── test_state_manager.py                # 新增
├── test_intent_parser.py                # 新增
├── test_devil_advocate.py               # 新增
├── test_tool_agent.py                   # 新增
├── test_memory_manager.py               # 新增
└── test_fallback.py                     # 新增

memory/                                  # SQLite 数据库目录
└── user_memory.db

frontend/
└── app.py                               # 修改：新增高级咨询页面
```

---

## 3. 核心模块详细设计

### 3.1 AgentStateManager（状态控制系统）

**文件**: `backend/state/state_manager.py`

**职责**: 为每个 Agent 步骤提供状态追踪、超时控制、重试机制

**AgentState 模型**:
```python
from enum import Enum
from pydantic import BaseModel
from typing import Optional

class StepName(str, Enum):
    USER_PROFILE = "user_profile"
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    PLANNING = "planning"
    RANKING = "ranking"
    OPPOSING = "opposing"
    EXPLAINING = "explaining"

class AgentStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DEGRADED = "degraded"

class AgentState(BaseModel):
    trace_id: str                          # 全链路追踪 ID
    step: StepName
    status: AgentStatus
    retry_count: int = 0
    max_retries: int = 2
    timeout_seconds: int = 30
    error_message: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
```

**StateManager 核心方法**:
```python
class AgentStateManager:
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.states: dict[StepName, AgentState] = {}
        self.logger = get_logger(trace_id)

    def start_step(self, step: StepName) -> AgentState:
        """开始一个步骤，记录开始时间"""

    def complete_step(self, step: StepName) -> None:
        """标记步骤成功"""

    def fail_step(self, step: StepName, error: str) -> AgentState:
        """标记步骤失败，检查是否可重试"""

    def should_retry(self, step: StepName) -> bool:
        """判断是否应该重试（retry_count < max_retries）"""

    def get_trace_log(self) -> dict:
        """返回全链路追踪日志"""
```

**使用模式**:
```python
state_mgr = AgentStateManager(trace_id=uuid4().hex)

# 每个 Agent 调用包裹在状态管理中
result = await execute_with_state(
    agent_func=DataRetriever().retrieve,
    step=StepName.RETRIEVAL,
    state_mgr=state_mgr,
    timeout=30
)
```

### 3.2 强制结构化输出引擎

**文件**: `backend/agents/structured_output.py`

**职责**: 确保所有 LLM 输出符合 Pydantic Schema，自动重试/修复

**核心模型**:
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class UserProfile(BaseModel):
    score: int = Field(ge=0, le=750, description="高考分数")
    province: str = Field(..., description="省份")
    interest: str = Field(..., description="兴趣方向")
    personality: str = Field(default="未知", description="性格特点")
    family_resource: str = Field(default="普通", description="家庭资源水平")
    risk_preference: str = Field(..., description="风险偏好: aggressive/balanced/conservative")
    score_tier: str = Field(..., description="分数段: top/high/mid/low")
    interest_keywords: List[str] = Field(default=[], description="兴趣关键词")
    resource_level: str = Field(..., description="资源等级: high/medium/low")

class DataCandidate(BaseModel):
    major: str
    employment_rate: float
    avg_salary: int
    resource_threshold: str
    industry: str
    match_score: float
    source: str = Field(default="rule", description="来源: rule/llm/vector")

class RoleAnalysis(BaseModel):
    role_name: str = Field(..., description="角色名称")
    perspective: str = Field(..., description="关注视角")
    recommendation: str = Field(..., description="推荐结论")
    reasoning: str = Field(..., description="推理过程")
    pros: List[str] = Field(default=[], description="优点")
    cons: List[str] = Field(default=[], description="缺点")

class PlanOption(BaseModel):
    tier: str = Field(..., description="风险等级: 冲刺/稳妥/保底")
    major: str
    universities: List[str]
    industry: str
    reason: str
    risk_level: str

class RankedResult(BaseModel):
    rank: int
    plan: PlanOption
    score: float = Field(ge=0, le=100)
    dimension_scores: dict = Field(description="各维度得分")

class DevilReport(BaseModel):
    max_concern: str = Field(..., description="当前推荐的最大问题")
    potential_risks: List[str] = Field(default=[], description="潜在风险")
    opposing_reasons: List[str] = Field(default=[], description="反对理由")

class Explanation(BaseModel):
    why_recommended: str
    why_ranked_first: str
    not_recommended_reasons: List[str]
    risk_warnings: List[str]

class IntentFilter(BaseModel):
    intent: str = Field(..., description="意图类型: exclude_major/include_major/city_preference/stability_focus/risk_focus")
    values: List[str] = Field(default=[], description="意图对应的值")
    description: str = Field(default="", description="用户原始反馈的描述")

class AgentOutput(BaseModel):
    user_profile: UserProfile
    multi_role_analysis: List[RoleAnalysis]
    plans: List[PlanOption]
    ranked_results: List[RankedResult]
    devil_report: DevilReport
    explanation: Explanation
    session_id: str
    iteration: int
    trace_id: str

class AdviseRequest(BaseModel):
    score: int = Field(ge=0, le=750)
    province: str
    interest: str
    personality: str = "未知"
    family_resource: str = "普通"
    session_id: str = ""

class RefineRequest(BaseModel):
    session_id: str
    feedback: str
```

**结构化输出引擎**:
```python
class StructuredOutputEngine:
    """确保 LLM 输出符合 Pydantic Schema"""

    def parse_with_retry(
        self,
        model_class: Type[BaseModel],
        llm_response: str,
        max_retries: int = 2
    ) -> BaseModel:
        """解析 LLM 输出为 Pydantic 模型"""
        for attempt in range(max_retries):
            try:
                # 尝试直接解析 JSON
                data = json.loads(llm_response)
                return model_class(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt < max_retries - 1:
                    # 使用 LLM 修复 JSON
                    llm_response = self._repair_json(
                        model_class, llm_response, str(e)
                    )
                else:
                    raise
```

### 3.3 DataRetriever v3（向量检索升级）

**文件**: `backend/agents/data_retriever.py`

**新增能力**:

#### 语义向量检索
- 使用 `text-embedding-3-small` 或 BGE 模型生成 embedding
- FAISS 索引存储专业/院校/行业描述向量
- 支持语义相似度检索（不仅仅是关键词匹配）

#### 结构化过滤 DSL
```python
class SearchFilter(BaseModel):
    """结构化过滤条件"""
    score_range: Optional[tuple[int, int]] = None  # (min, max)
    city_preference: Optional[list[str]] = None     # ["北京", "上海"]
    exclude_majors: Optional[list[str]] = None      # ["计算机"]
    exclude_industries: Optional[list[str]] = None
    min_employment_rate: Optional[float] = None
    resource_threshold: Optional[str] = None        # "low" | "medium" | "high"
    tier: Optional[str] = None                      # "985" | "211" | "双一流"
```

**检索流程**:
1. 语义检索：用 interest 的 embedding 搜索 majors.json，返回 Top 20 语义匹配
2. 规则过滤：应用 filter DSL 排除不匹配的候选
3. 综合打分：`match_score = semantic_score×0.40 + employment×0.30 + salary×0.15 + resource_fit×0.15`
4. 返回 Top 10

### 3.4 Refiner v3（LLM 驱动）

**文件**: `backend/agents/refiner.py` + `intent_parser.py`

**架构升级**:
```
用户反馈 → IntentParser(LLM) → IntentFilter → DataRetriever(filter) → 完整 Pipeline
```

**IntentParser** 职责:
- 用 LLM 解析自然语言反馈
- 输出结构化 IntentFilter
- 支持复杂/多条件反馈

**IntentParser Prompt**:
```
你是一个意图解析器，负责将用户对志愿填报建议的反馈转换为结构化过滤条件。

用户反馈: "{feedback}"

请分析用户意图，返回 JSON 格式的 IntentFilter：
- intent: 意图类型
  - "exclude_major": 排除某些专业
  - "include_major": 偏好某些专业
  - "city_preference": 城市偏好
  - "stability_focus": 追求稳定
  - "risk_focus": 追求高薪/冒险
  - "family_factor": 家庭因素
- values: 具体值列表
- description: 用户原始反馈的简要描述
```

**示例**:
```
输入: "不喜欢计算机，想去一线城市"
输出: [
  {"intent": "exclude_major", "values": ["计算机", "软件", "AI"]},
  {"intent": "city_preference", "values": ["北京", "上海", "广州", "深圳"]}
]
```

### 3.5 DevilAdvocate（反对 Agent）

**文件**: `backend/agents/devil_advocate.py`

**职责**: 对推荐方案提出反对意见，提升系统可信度

**System Prompt**:
```
你是一个严格的反对者（Devil's Advocate）。你的任务是找出推荐方案中的问题和风险。
基于以下候选方案和数据分析，指出：
1. 当前推荐的最大问题是什么
2. 有哪些潜在风险被忽略了
3. 有什么理由反对这个推荐

请保持客观，用事实和数据说话。
```

**输出**: `DevilReport` (Pydantic 模型)

### 3.6 ToolAgent（工具调用）

**文件**: `backend/agents/tool_agent.py`

**职责**: 让 Agent 能够调用外部工具获取额外信息

**已实现 Tool**:
| Tool | 功能 | 示例 |
|---|---|---|
| `query_university_score` | 查询院校在各省份的分数线 | 查"北邮在河南的投档线" |
| `query_industry_data` | 查询行业就业/薪资数据 | 查"人工智能行业薪资" |
| `query_city_info` | 查询城市发展/生活成本 | 查"深圳生活成本" |

**Tool 定义**:
```python
from langchain.tools import tool

@tool
def query_university_score(university: str, province: str) -> str:
    """查询指定大学在指定省份的录取分数线。
    参数:
        university: 大学名称，如"北京邮电大学"
        province: 省份，如"河南"
    返回: 该校在该省的最低投档线和平均分
    """
    kb = KnowledgeBase()
    uni = kb.query_university(university)
    if uni and uni.get("province") == province:
        return f"{university}在{province}：最低{uni['min_score_2025']}分，平均{uni['avg_score_2025']}分"
    return f"暂无{university}在{province}的分数线数据"
```

### 3.7 Memory Manager（用户长期记忆）

**文件**: `backend/memory/memory_manager.py`

**职责**: 持久化存储用户历史偏好和交互记录

**SQLite 表结构**:
```sql
CREATE TABLE user_sessions (
    session_id TEXT PRIMARY KEY,
    user_profile JSON,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE interaction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    iteration INTEGER,
    user_feedback TEXT,
    agent_output JSON,
    created_at TIMESTAMP
);

CREATE TABLE user_preferences (
    session_id TEXT,
    preference_type TEXT,      -- "rejected_major", "preferred_city", "stability_focus"
    preference_value TEXT,
    confidence FLOAT,          -- 置信度（基于反馈强度）
    created_at TIMESTAMP,
    PRIMARY KEY (session_id, preference_type, preference_value)
);
```

**API**:
```python
class MemoryManager:
    def save_session(self, session_id: str, output: AgentOutput) -> None
    def load_session(self, session_id: str) -> Optional[AgentOutput]
    def save_feedback(self, session_id: str, feedback: str, output: AgentOutput) -> None
    def get_preferences(self, session_id: str) -> dict
    def update_preferences(self, session_id: str, preferences: dict) -> None
    def get_history(self, session_id: str) -> list[dict]
```

### 3.8 Ranker v3（动态权重）

**文件**: `backend/agents/ranker.py`

**动态权重逻辑**:
```python
class Ranker:
    BASE_WEIGHTS = {
        "employment": 0.30,
        "match": 0.25,
        "salary": 0.20,
        "growth": 0.15,
        "risk": 0.10
    }

    def get_dynamic_weights(self, user_profile: UserProfile) -> dict:
        """根据用户偏好动态调整权重"""
        weights = self.BASE_WEIGHTS.copy()

        if "稳定" in user_profile.personality or user_profile.risk_preference == "conservative":
            weights["risk"] += 0.10
            weights["employment"] += 0.05
            weights["salary"] -= 0.10
            weights["growth"] -= 0.05

        if user_profile.resource_level == "low":
            weights["employment"] += 0.10
            weights["salary"] -= 0.05
            weights["match"] -= 0.05

        # 归一化
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
```

### 3.9 FallbackHandler（容错降级）

**文件**: `backend/fallback/fallback_handler.py`

**降级策略**:
```python
class FallbackHandler:
    """LLM 失败时的降级处理"""

    def fallback_role_analysis(self, candidates: list, profile: UserProfile) -> list[RoleAnalysis]:
        """LLM 多角色分析失败 → 基于规则生成分析"""
        # 张雪峰视角：就业率优先
        zhang = self._zhang_rule_analysis(candidates, profile)
        # 家长视角：稳定性优先
        parent = self._parent_rule_analysis(candidates, profile)
        # 其他角色返回简化分析
        ...

    def fallback_explanation(self, ranked: list) -> Explanation:
        """LLM 解释失败 → 基于数据生成解释"""
        ...
```

### 3.10 LoggingSystem（可观测性）

**文件**: `backend/logging_config.py`

**全链路 trace_id**:
```python
import logging
import uuid

def get_logger(trace_id: str = None):
    """获取带 trace_id 的 logger"""
    logger = logging.getLogger("agent")
    if trace_id:
        return logging.LoggerAdapter(logger, {"trace_id": trace_id})
    return logger

# 使用示例
logger = get_logger(trace_id="abc123")
logger.info("UserProfiler started", extra={"step": "user_profile", "status": "running"})
```

**日志格式**:
```
2026-06-07 14:30:01 | trace_id=abc123 | step=user_profile | status=success | duration=0.5s | retry=0
```

---

## 4. API 设计（v3.0）

### 4.1 端点列表

| 方法 | 路径 | 说明 | 请求体 | 响应 |
|---|---|---|---|---|
| POST | `/advise` | 高级咨询 | AdviseRequest | AgentOutput |
| POST | `/advise/refine` | 反馈优化 | RefineRequest | AgentOutput |
| GET | `/advise/history/{sid}` | 历史会话 | — | List[AgentOutput] |
| GET | `/advise/trace/{sid}` | 追踪日志 | — | TraceLog |

### 4.2 错误响应

```python
class ErrorResponse(BaseModel):
    error_code: str        # "llm_failed" | "data_not_found" | "invalid_input" | "timeout"
    message: str
    fallback_available: bool
    suggestions: list[str]
```

---

## 5. 前端改造（v3.0）

### 5.1 新增展示区域

在 v2.0 基础上新增：

1. **反对意见卡片**: 红色边框，展示 DevilAdvocate 的输出
2. **评分权重条**: 展示动态权重的分布（就业 30%、匹配 25%...）
3. **多轮对话历史**: 展示迭代次数和每次反馈
4. **反馈快捷按钮**: "排除计算机" / "想去北京" / "要稳定" 等

### 5.2 交互流程

```
用户填表 → 点击"开始咨询" → 后端执行 Pipeline → 返回结构化结果
    ↓
用户看到推荐结果 + 反对意见
    ↓
用户输入反馈 → 点击"优化推荐" → Refiner 重新计算
    ↓
对比前后差异（高亮变化）
```

---

## 6. 测试策略（v3.0）

### 6.1 新增测试

| 测试文件 | 测试内容 |
|---|---|
| `test_state_manager.py` | 超时控制、重试机制、状态追踪 |
| `test_intent_parser.py` | 自然语言 → IntentFilter 转换 |
| `test_devil_advocate.py` | 反对意见生成、JSON 格式 |
| `test_tool_agent.py` | Tool 调用、结果格式 |
| `test_memory_manager.py` | SQLite 存储/读取、偏好更新 |
| `test_fallback.py` | LLM 失败降级、规则推荐 |
| `test_structured_output.py` | JSON 解析、重试、修复 |

### 6.2 集成测试

- `test_full_pipeline.py`: 完整 Pipeline 测试（含状态管理）
- `test_refine_loop.py`: 多轮优化循环测试
- `test_concurrent_requests.py`: 并发请求测试

---

## 7. 依赖变更

### 新增依赖

```txt
# requirements.txt 新增
faiss-cpu>=1.7.4                  # 向量检索
numpy>=1.24.0                     # 向量计算
aiosqlite>=0.19.0                 # SQLite 异步支持
tiktoken>=0.5.0                   # token 计数
```

---

## 8. 错误处理矩阵（v3.0）

| 错误类型 | 检测方式 | 处理策略 | 降级方案 |
|---|---|---|---|
| LLM API 超时 | timeout 异常 | retry ≤2 次 | FallbackHandler 规则推荐 |
| LLM JSON 解析失败 | ValidationError | StructuredOutputEngine 重试 | 请求 LLM 修复 → 仍失败则规则降级 |
| 知识库无匹配 | 空候选列表 | 返回空结果 + 解释 | 扩大兴趣关键词范围重试 |
| 数据库连接失败 | SQLite 异常 | 降级为内存存储 | 提示用户持久化不可用 |
| embedding 失败 | API 错误 | 降级为关键词检索 | 纯规则检索 |
| 反馈解析失败 | IntentParser 超时 | 规则解析（关键字匹配） | 提示用户重新表述 |

---

## 9. 扩展性设计（v3.0）

| 扩展点 | 实现方式 | 示例 |
|---|---|---|
| 新角色 | 注册表模式 + 新 prompt 文件 | 添加"职业规划师"角色 |
| 新数据源 | DataRetriever 注册新 loader | 添加 salaries.json 薪资数据 |
| 新评分维度 | Ranker 增加维度 + 权重 | 添加"考公友好度"维度 |
| 新反馈类型 | IntentParser 增加 intent 类型 | 添加"exclude_province" |
| 新 Tool | tools/ 下新增模块 | 添加"query_living_cost" |
| 新 embedding 模型 | embedding_service.py 切换模型 | 从 OpenAI 切到 BGE |
| 持久化后端 | MemoryManager 替换存储引擎 | 从 SQLite 切到 Redis/PostgreSQL |
