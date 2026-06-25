#  xueAgent — AI 志愿填报顾问系统

> 「选择比努力更重要，但'有得选'的前提是你足够努力。」

基于 [zhangxuefeng-skill](https://github.com/alchaincyf/zhangxuefeng-skill) 思维框架构建的多角色 AI 志愿填报顾问系统，模拟张雪峰及其团队的专业视角，为考生和家长提供高考志愿填报建议。

## 项目简介

xueAgent 是一个基于 FastAPI + Streamlit 的智能志愿填报顾问。它采用 **RAG（Retrieval-Augmented Generation，检索增强生成）** 架构，通过多智能体协作（Multi-Agent）实现「检索 → 上下文注入 → LLM 推理生成」的完整流程，结合向量检索与 LLM 推理，为用户提供：

- **专业分析**：就业率、平均薪资、职业发展方向
- **院校推荐**：基于分数段、省份、偏好的智能匹配
- **行业解读**：行业进入门槛、家庭资源依赖度、雇主分布
- **多角色视角**：张雪峰、学术导师、行业专家、HR经理、家长代表五种角色协同分析
- **多轮对话**：支持上下文记忆和 Session 隔离
- **用户认证**：JWT 注册/登录，多会话管理

## RAG 架构

```
────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐
│ 用户提问    │────▶│ 检索层 (Retrieval) │────▶│ 注入层 (Context) │────▶│ 生成层 (LLM)  │
│            │     │ 混合检索引擎    │     │ Prompt Builder  │     │ 流式输出     │
└────────────┘     └──────┬───────┘     └──────┬───────┘     └────────────┘
                          │                     │
              ┌───────────┼───────────┐         │
              ▼           ▼           ▼         │
        语义向量检索  规则匹配过滤  关键词检索    │
              ───────────┴───────────         │
                          ▼                     │
                  召回 → 粗排 → 精排 → 融合     │
                                                │
              ┌─────────────────────────────────┘
              ▼
        心智模型/决策启发式/表达DNA
        用户画像/历史对话/候选数据
```

**RAG 流程说明**：

1. **检索（Retrieval）**：用户输入经过意图解析后，触发混合检索管道（四阶段：过滤 → 向量召回 → 粗排 → 融合），从 JSON 知识库或 Qdrant 向量数据库中召回候选专业和院校
2. **注入（Context Injection）**：Prompt Builder 将检索结果与用户画像、对话历史、角色约束文件（心智模型/决策启发式/表达DNA）动态组装为上下文
3. **生成（Generation）**：多角色推理智能体并行分析候选方案，总协调器聚合输出，通过流式 SSE 返回给用户

## 技术栈

### 后端

| 技术 | 说明 |
|------|------|
| **Python 3.11+** | 运行环境 |
| **FastAPI** | Web 框架，提供 RESTful API + SSE 流式输出 |
| **LangChain** | LLM 调用链管理 |
| **Pydantic** | Agent 间类型安全通信协议 — 所有智能体的输入/输出均通过 12 个结构化 Pydantic 模型传递（`UserProfile`、`PlanResult`、`RankResult` 等），确保数据格式严格校验、类型安全、跨 Agent 无缝对接 |
| **psycopg2 + aiosqlite** | PostgreSQL / SQLite 数据库驱动 |
| **Qdrant Client** | 向量数据库客户端 |
| **FAISS / numpy** | 本地向量索引引擎 |
| **python-jose + passlib** | JWT 认证与密码哈希 |
| **uvicorn** | ASGI 服务器 |

### 前端

| 技术 | 说明 |
|------|------|
| **Streamlit** | 纯 Python Web UI 框架 |
| **httpx** | HTTP 客户端，与后端通信 |

### 数据存储

| 组件 | 用途 | 必填 |
|------|------|------|
| **JSON 文件** | 专业/院校/行业知识库 | ✅ 是（内置） |
| **SQLite** | 轻量级本地存储 — 存储用户画像（`user_profiles`）、对话历史（`chat_history`）、偏好设置（`preferences`），使用 `aiosqlite` 异步驱动，离线可用 | 否（可选） |
| **PostgreSQL** | 会话持久化、用户认证存储 — 支持多会话管理、跨设备同步 | 否（可选） |
| **Qdrant** | 云向量语义检索 — 支持 payload 过滤查询、分布式部署 | 否（可选） |
| **FAISS / numpy** | 本地向量索引 — FAISS 优先（自动检测可用性），numpy 回退 | 否（可选） |

**混合存储策略**：
- **开发/离线模式**：SQLite + 本地 FAISS/numpy 索引，零外部依赖即可运行
- **生产/多用户模式**：PostgreSQL（会话持久化）+ Qdrant（云向量检索），支持高并发和分布式部署
- 两者可通过环境变量自由切换，互不干扰

### LLM 支持

支持任何 **OpenAI 兼容 API**，已验证的模型：

| 服务商 | API Base | 推荐模型 |
|--------|----------|----------|
| OpenAI | `https://api.openai.com/v1` | gpt-4o-mini |
| DeepSeek | `https://api.deepseek.com/v1` | deepseek-chat |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen-plus |

## 系统架构

```
┌──────────────────┐     ──────────────────────────────────┐
│   Streamlit UI   │────▶│         FastAPI Backend          │
│   (frontend/)    │────│         (backend/)               │
──────────────────┘     └───────┬──────────────────────────┘
                                 │
                    ┌────────────┼────────────
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │PostgreSQL│ │ Qdrant   │ │ JSON KB  │
              │ 会话存储  │ │ 向量检索  │ │ 知识库   │
              └────────── └──────────┘ └──────────┘
                    ▲
              ┌─────┴──────┐
              │   SQLite   │
              │ 用户画像/   │
              │ 偏好存储   │
              └────────────┘
```

## 功能模块

### 多智能体协作流程

```
用户提问
   │
   ▼
─────────────────────────────────────────────────────────┐
│                 8 步 Orchestrator 编排                   │
│                                                         │
│  Step 1: IntentParser    → 意图解析                     │
│  Step 2: UserProfiler    → 用户画像分析                 │
│  Step 3: DataRetriever   → 四阶段混合检索               │
│  Step 4: MultiRoleReasoner → 5 角色并行推理             │
│  Step 5: Planner         → 冲/稳/保方案生成             │
│  Step 6: Ranker          → 多维度打分排序               │
│  Step 7: DevilAdvocate   → 风险质疑与挑战               │
│  Step 8: Explainer       → 自然语言解释输出             │
│                                                         │
│  → Refine（可选）: Refiner → 根据反馈精炼方案            │
│                                                         │
│  共调度 11 个 Agent（含 ToolAgent、FallbackHandler）      │
─────────────────────────────────────────────────────────┘
   │
   ▼
LLM Stream Response（SSE 流式输出）
```

**编排层级说明**：

| 层级 | Agent 列表 | 说明 |
|------|-----------|------|
| **编排层**（1 个） | `Orchestrator` | 8 步流程串联、超时控制、状态追踪、全局异常处理 |
| **执行层**（11 个） | `IntentParser`、`UserProfiler`、`DataRetrieverV4`、`MultiRoleReasoner`（含 5 子角色）、`Planner`、`Ranker`、`DevilAdvocate`、`Explainer`、`Refiner`、`ToolAgent`、`FallbackHandler` | 各司其职，通过 Pydantic 结构化模型传递数据 |

### 模块详解

| 模块 | 文件 | 功能 |
|------|------|------|
| **意图解析** | `agents/intent_parser.py` | 识别用户问题类型（专业/院校/行业/通用），提取分数、省份、偏好等关键信息。**LLM 优先 + 关键词规则降级**，支持排除专业/偏好省份/风险偏好等多维度筛选 |
| **用户画像** | `agents/user_profiler.py` | 基于对话历史构建考生画像（成绩、地域、家庭背景、兴趣、性格、风险偏好），输出 `UserProfile` Pydantic 模型 |
| **数据检索** | `agents/data_retriever.py` | **四阶段混合检索管道**：① 结构化过滤（按省份/分数段/院校类型）→ ② 语义向量召回（FAISS/Qdrant）→ ③ 关键词匹配 → ④ 可配置权重融合（默认 filter:1.0 + vector:0.3，支持自定义） |
| **多角色推理** | `agents/multi_role_reasoner.py` | **5 角色并行推理**：张雪峰、学术导师、行业专家、HR经理、家长代表。每个角色加载独立约束文件（心智模型 mental_models、决策启发式 decision_heuristics、表达 DNA expression_dna），实现差异化决策。内置**共识评分**（≥2 角色推荐同一专业自动聚合）与**分歧检测**（评分差 > 30 分、推荐方案 ≥ 3 个不同、风险态度冲突三种触发条件） |
| **总协调器** | `agents/orchestrator.py` | 8 步流程编排，每个子 Agent 带独立超时控制（单 Agent 30s，多角色 120s），通过 `asyncio.wait_for` 实现，超时/失败自动触发降级 |
| **方案生成** | `agents/planner.py` | 基于多角色分析和数据检索结果，生成冲/稳/保三套志愿方案。**LLM 优先 + 规则降级**（分数偏移 + 知识库匹配） |
| **排序器** | `agents/orchestrator.py:Ranker` | 多维度打分排序（就业率、专业匹配度、风险适配、薪资前景、发展空间），**LLM 优先 + 规则降级**（基础分 + 风险适配 + 兴趣匹配） |
| **反对者** | `agents/orchestrator.py:DevilAdvocate` | 对方案提出质疑和风险提醒，**LLM 优先 + 规则降级**（按冲/稳/保档位分析落榜风险、兴趣匹配度） |
| **解释器** | `agents/orchestrator.py:Explainer` | 将排序结果和风险提醒翻译为自然语言，**LLM 优先 + 规则降级** |
| **优化器** | `agents/refiner.py` | 根据用户反馈意图（排除专业/偏好省份/调整风险）动态调整方案，支持多轮精炼。**LLM 优先 + 规则降级** |
| **工具代理** | `agents/tool_agent.py` | 院校查询、行业查询等工具的执行入口 |
| **降级处理** | `fallback/fallback_handler.py` | **全局兜底**：当任一决策节点 LLM 超时/失败且规则降级也失败时，FallbackHandler 基于 KnowledgeBase 直接生成冲/稳/保推荐方案 |
| **记忆管理** | `memory/memory_manager.py` | SQLite 持久化用户画像、对话历史、偏好设置，支持异步操作（`aiosqlite`） |
| **状态管理** | `state/agent_state.py` | Agent 执行步骤的状态追踪（RUNNING → SUCCESS/FAILED/TIMEOUT） |
| **Prompt 构建** | `services/prompt_builder.py` | 模块化加载心智模型/决策启发式/表达DNA，按角色和问题类型动态注入 Prompt 上下文 |
| **知识库** | `services/knowledge_base.py` | JSON 数据读写 + 内存缓存管理，支持热加载 |
| **向量检索** | `services/vector_index.py` / `qdrant_index.py` | **双引擎向量索引**：VectorIndex 支持 FAISS（优先，自动检测可用性）/ numpy（回退），QdrantIndex 支持云端向量库。两者接口兼容，可通过环境变量切换。支持**增量索引**（`add_by_name` / `remove_by_name` + MD5 文档哈希检测，避免重复插入） |
| **混合检索** | `services/hybrid_search.py` | 结构化过滤 + 向量召回 + 可配置权重融合，四阶段检索管道的核心实现 |
| **Embedding** | `services/embedding_service.py` | 支持 hash（确定性哈希，零依赖）/ local（本地模型）/ api（云端 API）三种策略 |
| **LLM 链** | `services/llm_chain.py` | 流式/同步 LLM 调用封装，支持 SSE 输出 |
| **JWT 认证** | `auth/jwt_auth.py` | Token 生成/验证 + bcrypt 密码哈希 |
| **会话存储** | `session/lru_session_store.py` + `session/postgres_session_store.py` | **LRU+TTL 内存缓存 + PostgreSQL 持久化双写架构** — `LRUSessionStore` 基于 `OrderedDict` 实现 LRU 淘汰（max_size=1000, ttl=1800s, 满时淘汰最久未访问 10%），`PostgreSQLSessionStore` 读操作优先命中缓存、未命中回填，写操作同步更新缓存和 DB，delete 操作同步使缓存失效 |
| **推荐引擎** | `services/recommend_engine.py` | 专业/院校推荐算法 |

### 双路径降级保障

**每个决策节点均实现 LLM + 规则双路径**：

| 决策节点 | LLM 路径 | 规则降级路径 | 超时阈值 |
|----------|----------|-------------|----------|
| IntentParser | 意图识别 + 条件提取 | 关键词正则匹配 | 30s |
| UserProfiler | 画像推理 | — | 30s |
| DataRetriever | — | 结构化过滤 + 混合检索 | 30s |
| MultiRoleReasoner | 5 角色并行推理 | 返回默认意见（50 分） | 120s |
| Planner | 方案生成 | 分数偏移 + 知识库匹配 | 30s |
| Ranker | 多维度排序 | 基础分 + 风险/兴趣匹配 | 30s |
| DevilAdvocate | 风险质疑 | 档位分析 + 关键词判断 | 30s |
| Explainer | 自然语言解释 | 模板填充 | 30s |
| Refiner | 方案调整 | 规则引擎逐项应用 | 30s |

任一节点超时（`asyncio.wait_for`）或异常 → 自动回退规则路径 → 若规则路径也失败 → 全局 FallbackHandler 兜底。

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/chat` | 流式对话（SSE） |
| POST | `/chat/sync` | 同步对话 |
| POST | `/advise` | 志愿填报建议 |
| POST | `/refine` | 方案优化 |
| POST | `/auth/register` | 用户注册 |
| POST | `/auth/login` | 用户登录 |
| POST | `/session/create` | 创建会话 |
| GET | `/session/list` | 获取会话列表 |
| GET | `/session/history` | 获取会话历史 |
| DELETE | `/session/delete` | 删除会话 |

## 快速开始

### 环境要求

- Python 3.11+
- SQLite（内置，无需额外安装）
- PostgreSQL 14+（可选，不配置时部分功能不可用）
- Qdrant（可选，不配置时使用本地 FAISS/numpy 向量索引）
- OpenAI 兼容 API Key（DeepSeek / 通义千问 / OpenAI 等）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的配置：

```bash
# 必填项 — LLM API 配置
OPENAI_API_KEY=sk-your-api-key        # 你的 LLM API Key
OPENAI_API_BASE=https://api.openai.com/v1   # API 地址
OPENAI_MODEL=gpt-4o-mini              # 模型名称

# 可选 — Embedding 向量模型（不使用向量检索可不填）
EMBEDDING_STRATEGY=hash               # hash | local | api
EMBEDDING_API_KEY=your-key            # 使用 api 策略时需要
EMBEDDING_API_BASE=https://...        # Embedding API 地址
EMBEDDING_MODEL=text-embedding-3-small # Embedding 模型

# 可选 — Qdrant 向量数据库（不使用 Qdrant 可不填）
QDRANT_URL=https://...                # Qdrant 实例地址
QDRANT_API_KEY=your-key               # Qdrant API Key
QDRANT_COLLECTION=majors              # 集合名称

# 可选 — PostgreSQL 数据库（不配置时使用 SQLite）
DB_HOST=localhost
DB_PORT=5432
DB_NAME=xueAgent
DB_USER=postgres
DB_PASSWORD=your-password

# 可选 — JWT 认证密钥（生产环境务必更换）
JWT_SECRET_KEY=your-random-secret
```

> **最小启动配置**：只需填写 `OPENAI_API_KEY`、`OPENAI_API_BASE`、`OPENAI_MODEL` 三项即可运行。SQLite 会自动初始化，无需额外配置。

### 3. 启动服务

```bash
# 启动后端（终端 1）
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 启动前端（终端 2）
streamlit run frontend/app.py
```

浏览器访问 http://localhost:8501

## Docker 部署

### 单容器模式（开发/演示）

```bash
docker build -t xueagent .
docker run -p 8501:8501 --env-file .env xueagent
```

### Docker Compose 多服务编排（推荐生产）

项目包含 [`docker-compose.yml`](file:///e:/vibecodeing/claude/xuefeng/docker-compose.yml)，支持后端 + 前端分离部署：

```bash
docker-compose up -d
```

**服务说明**：

| 服务 | 端口 | 说明 |
|------|------|------|
| `backend` | 8000 | FastAPI 后端，提供 REST API + SSE 流式输出 |
| `frontend` | 8501 | Streamlit 前端，依赖 backend 服务 |

如需完整生产环境（含 PostgreSQL + Qdrant），可自行扩展 docker-compose.yml 添加对应的服务定义。

## 模型切换

编辑 `.env` 文件中的 LLM 配置，重启即可生效：

```bash
# DeepSeek
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat

# 通义千问
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-plus

# OpenAI
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

## 项目结构

```
├── backend/                  # FastAPI 后端
│   ├── main.py               # 应用入口 + API 路由
│   ├── agents/               # 多智能体（意图解析/用户画像/数据检索/多角色推理/方案生成/精炼/工具）
│   │   ├── orchestrator.py   # 总协调器（8 步编排 + Ranker/DevilAdvocate/Explainer）
│   │   ├── intent_parser.py  # 意图解析（LLM + 关键词规则双路径）
│   │   ├── user_profiler.py  # 用户画像分析
│   │   ├── data_retriever.py # 四阶段混合检索管道
│   │   ├── multi_role_reasoner.py  # 5 角色并行推理 + 共识评分/分歧检测
│   │   ├── planner.py        # 冲/稳/保方案生成（LLM + 规则降级）
│   │   ├── refiner.py        # 方案精炼（LLM + 规则降级）
│   │   └── tool_agent.py     # 工具执行代理
│   ├── auth/                 # JWT 认证 + 用户存储
│   ├── session/              # 会话存储（LRU 内存缓存 + PostgreSQL 持久化双写）
│   ├── memory/               # SQLite 用户画像/对话历史/偏好存储
│   ├── state/                # Agent 状态管理（RUNNING/SUCCESS/FAILED/TIMEOUT）
│   ├── services/             # 业务逻辑
│   │   ├── prompt_builder.py      # Prompt 构建（心智模型/决策启发式/表达DNA）
│   │   ├── knowledge_base.py      # JSON 知识库读写 + 缓存
│   │   ├── vector_index.py        # FAISS/NumPy 双引擎向量索引 + 增量更新
│   │   ├── qdrant_index.py        # Qdrant 云向量库封装
│   │   ├── hybrid_search.py       # 混合检索（结构化过滤 + 向量召回 + 权重融合）
│   │   ├── embedding_service.py   # 哈希/本地/API 三种 Embedding 策略
│   │   ├── llm_chain.py           # LLM 调用链（同步/流式）
│   │   └── recommend_engine.py    # 推荐引擎
│   ├── tools/                # 查询工具（院校/行业/推荐）
│   ├── models/               # 数据模型 + 配置 + Agent 间 Pydantic 通信模型
│   ├── data/                 # JSON 知识库数据（运行时生成，不提交）
│   ├── cache/                # 向量索引缓存（运行时生成，不提交）
│   ├── prompts/              # Prompt 模板（角色心智模型/决策启发式/表达DNA）
│   ├── fallback/             # 全局降级处理
│   └── scripts/              # 工具脚本
├── frontend/                 # Streamlit 前端
│   ├── app.py                # 多轮对话主界面
│   └── advise_app.py         # 建议顾问界面
├── tests/                    # 单元测试
├── scripts/                  # 数据库建表脚本
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量模板
├── Dockerfile
├── docker-compose.yml        # 后端 + 前端多服务编排
── README.md
```

## 运行测试

```bash
pytest tests/ -v --tb=short
```

## API 文档

启动后端后访问 http://localhost:8000/docs 查看 Swagger API 文档。

## License

MIT
