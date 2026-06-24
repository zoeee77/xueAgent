# 🎓 xueAgent — AI 志愿填报顾问系统

> 「选择比努力更重要，但'有得选'的前提是你足够努力。」

基于 [zhangxuefeng-skill](https://github.com/alchaincyf/zhangxuefeng-skill) 思维框架构建的多角色 AI 志愿填报顾问系统，模拟张雪峰及其团队的专业视角，为考生和家长提供高考志愿填报建议。

## 项目简介

xueAgent 是一个基于 FastAPI + Streamlit 的智能志愿填报顾问。它通过多智能体协作架构（Multi-Agent），结合向量检索与 LLM 推理，为用户提供：

- **专业分析**：就业率、平均薪资、职业发展方向
- **院校推荐**：基于分数段、省份、偏好的智能匹配
- **行业解读**：行业进入门槛、家庭资源依赖度、雇主分布
- **多角色视角**：张雪峰、学术导师、行业专家、HR经理、家长代表五种角色协同分析
- **多轮对话**：支持上下文记忆和 Session 隔离
- **用户认证**：JWT 注册/登录，多会话管理

## 技术栈

### 后端

| 技术 | 说明 |
|------|------|
| **Python 3.11+** | 运行环境 |
| **FastAPI** | Web 框架，提供 RESTful API + SSE 流式输出 |
| **LangChain** | LLM 调用链管理 |
| **Pydantic** | 数据模型验证与配置管理 |
| **psycopg2** | PostgreSQL 数据库驱动 |
| **Qdrant Client** | 向量数据库客户端 |
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
| **PostgreSQL** | 会话持久化、用户管理 | 否（可选） |
| **Qdrant** | 向量语义检索 | 否（可选） |
| **numpy / faiss** | 本地向量索引（Qdrant 替代方案） | 否（可选） |

### LLM 支持

支持任何 **OpenAI 兼容 API**，已验证的模型：

| 服务商 | API Base | 推荐模型 |
|--------|----------|----------|
| OpenAI | `https://api.openai.com/v1` | gpt-4o-mini |
| DeepSeek | `https://api.deepseek.com/v1` | deepseek-chat |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen-plus |

## 系统架构

```
┌──────────────────┐     ┌──────────────────────────────────┐
│   Streamlit UI   │────▶│         FastAPI Backend          │
│   (frontend/)    │◀────│         (backend/)               │
└──────────────────┘     └───────┬──────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │PostgreSQL│ │ Qdrant   │ │ JSON KB  │
              │ 会话存储  │ │ 向量检索  │ │ 知识库   │
              └──────────┘ └──────────┘ └──────────┘
```

## 功能模块

### 多智能体协作流程

```
用户提问 → IntentParser(意图解析)
              │
              ▼
         UserProfiler(用户画像)
              │
              ▼
       DataRetriever(数据检索) ────混合检索───→ JSON KB / Qdrant
              │
              ▼
        MultiRoleReasoner(多角色推理)
         ┌──────┼──────┬──────┬──────┐
         ▼      ▼      ▼      ▼      ▼
      张雪峰  学术导师 行业专家 HR经理 家长代表
              │
              ▼
       Orchestrator(总协调器) → Plan + Rank
              │
              ▼
          Refiner(优化器)
              │
              ▼
        FallbackHandler(降级处理)
              │
              ▼
         LLM Stream Response
```

### 模块详解

| 模块 | 文件 | 功能 |
|------|------|------|
| **意图解析** | `agents/intent_parser.py` | 识别用户问题类型（专业/院校/行业/通用），提取分数、省份、偏好等关键信息 |
| **用户画像** | `agents/user_profiler.py` | 基于对话历史构建考生画像（成绩、地域、家庭背景、兴趣） |
| **数据检索** | `agents/data_retriever.py` | 混合检索引擎，结合语义向量检索 + 规则匹配 + 关键词检索 |
| **多角色推理** | `agents/multi_role_reasoner.py` | 5 种角色并行分析，各自输出专业视角的观点 |
| **总协调器** | `agents/orchestrator.py` | 串联所有子智能体，生成志愿方案 + 多维度打分排序 |
| **优化器** | `agents/refiner.py` | 对用户反馈进行方案优化，支持追问和修正 |
| **降级处理** | `fallback/fallback_handler.py` | LLM 异常时的兜底响应机制 |
| **记忆管理** | `memory/memory_manager.py` | 对话上下文记忆，支持跨轮次信息保留 |
| **状态管理** | `state/agent_state.py` | Agent 执行步骤的状态追踪 |
| **Prompt 构建** | `services/prompt_builder.py` | 模块化加载心智模型/决策启发式/表达DNA，按角色和问题类型动态注入 |
| **知识库** | `services/knowledge_base.py` | JSON 数据读写 + 缓存管理 |
| **向量检索** | `services/vector_index.py` / `qdrant_index.py` | numpy/faiss 本地索引 + Qdrant 云向量库 |
| **Embedding** | `services/embedding_service.py` | 支持 hash/local/api 三种 Embedding 策略 |
| **LLM 链** | `services/llm_chain.py` | 流式/同步 LLM 调用封装 |
| **JWT 认证** | `auth/jwt_auth.py` | Token 生成/验证 + 密码哈希 |
| **会话存储** | `session/postgres_session_store.py` | PostgreSQL 多会话管理 |
| **推荐引擎** | `services/recommend_engine.py` | 专业/院校推荐算法 |

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
- PostgreSQL 14+（可选，不配置时部分功能不可用）
- Qdrant（可选，不配置时使用本地 numpy 向量索引）
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

# 可选 — PostgreSQL 数据库（不配置时使用内存会话）
DB_HOST=localhost
DB_PORT=5432
DB_NAME=xueAgent
DB_USER=postgres
DB_PASSWORD=your-password

# 可选 — JWT 认证密钥（生产环境务必更换）
JWT_SECRET_KEY=your-random-secret
```

> **最小启动配置**：只需填写 `OPENAI_API_KEY`、`OPENAI_API_BASE`、`OPENAI_MODEL` 三项即可运行。

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
│   ├── agents/               # 多智能体（意图解析/数据检索/多角色推理/优化）
│   ├── auth/                 # JWT 认证 + 用户存储
│   ├── session/              # PostgreSQL 会话存储
│   ├── memory/               # 对话记忆管理
│   ├── state/                # Agent 状态管理
│   ├── services/             # 业务逻辑（知识库/Prompt/LLM/向量检索/推荐）
│   ├── tools/                # 查询工具（院校/行业/推荐）
│   ├── models/               # 数据模型 + 配置
│   ├── data/                 # JSON 知识库数据（运行时生成，不提交）
│   ├── cache/                # 向量索引缓存（运行时生成，不提交）
│   ├── prompts/              # Prompt 模板（角色心智模型/决策启发式/表达DNA）
│   ├── fallback/             # 降级处理
│   └── scripts/              # 工具脚本
├── frontend/                 # Streamlit 前端
│   ├── app.py                # 多轮对话主界面
│   └── advise_app.py         # 建议顾问界面
├── tests/                    # 单元测试
├── scripts/                  # 数据库建表脚本
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量模板
├── Dockerfile
└── README.md
```

## 运行测试

```bash
pytest tests/ -v --tb=short
```

## API 文档

启动后端后访问 http://localhost:8000/docs 查看 Swagger API 文档。

## License

MIT
