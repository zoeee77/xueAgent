---
name: zhangxuefeng-ai-app-design
date: 2026-06-07
status: approved
---

# 张雪峰 AI 志愿填报顾问 · 设计文档

> 「选择比努力更重要，但'有得选'的前提是你足够努力。」

## 1. 系统架构

```
┌─────────────────────────────────────────────────┐
│            Streamlit Frontend                   │
│  - 多轮对话界面 (session state)                  │
│  - 历史聊天记录（可折叠/展开）                    │
│  - 示例问题快捷按钮（预设6个常见场景）             │
│  - 输入建议 placeholder（引导用户提问）            │
│  - Loading spinner（长响应时实时动画提示）         │
│  - 多轮对话上下文可视化（角色标签 + 时间线）       │
└──────────────┬──────────────────────────────────┘
               │ HTTP POST /chat (streaming SSE)
┌──────────────▼──────────────────────────────────┐
│          FastAPI Backend                        │
│                                                 │
│  /chat          → 流式聊天接口 (SSE)             │
│  /health        → 健康检查                       │
│                                                 │
│  ┌────────────┐ ┌────────────────────┐          │
│  │ LLM Chain  │ │ Prompt Builder     │          │
│  │(LangChain) │ │ (动态模板构建)      │          │
│  └─────┬──────┘ └──────────┬─────────┘          │
│        │                   │                    │
│  ┌─────▼───────────────────▼─────────┐          │
│  │        Knowledge Base             │          │
│  │  - majors.json (专业数据)         │          │
│  │  - universities.json (院校数据)    │          │
│  │  - industries.json (行业数据)      │          │
│  │  - decision_rules.json (规则)      │          │
│  │  - data_updater.py (增量更新脚本)  │          │
│  └───────────────────────────────────┘          │
│                                                 │
│  Cache: 内存缓存 (dict+TTL) + 热门查询缓存       │
│         - 相同问题5分钟内直接返回缓存结果         │
│         - 按 session_id 隔离对话历史             │
│                                                 │
│  Config: Pydantic BaseSettings (.env)           │
└─────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│       OpenAI 兼容 LLM API                       │
│  环境变量: OPENAI_API_BASE, OPENAI_API_KEY,     │
│            OPENAI_MODEL, LLM_PROVIDER           │
│  支持热切换模型，无需重启服务                    │
└─────────────────────────────────────────────────┘
```

## 2. Prompt 模块化设计

System Prompt 由三个独立模块拼接而成，便于单独修改和测试：

### 2.1 思维模型 (`prompts/mental_models.txt`)
- 现实优先原则（先看数据再开口）
- 阶层流动视角（教育是跃迁通道）
- 就业导向决策（毕业去哪比学什么重要）
- 家庭资源评估（没有资源就别碰资源密集型行业）
- 地域杠杆效应（一线城市>家乡城市）

### 2.2 决策启发式 (`prompts/decision_heuristics.txt`)
- 8条核心决策规则（如"先选城市再选学校再选专业"）
- 分数段差异化策略
- 专业避坑指南

### 2.3 表达风格 DNA (`prompts/expression_dna.txt`)
- 东北大哥语气
- 快节奏、段子化、接地气
- 数据+比喻双驱动
- 免责声明仅首次出现

### 2.4 动态注入
`prompt_builder.py` 根据问题分类自动注入：
- 专业咨询 → 注入专业就业率数据 + 对应启发式
- 院校选择 → 注入分数线数据 + 地域规则
- 行业前景 → 注入行业就业分布数据
- 综合规划 → 注入全部5个思维模型

## 3. 知识库设计

### 3.1 majors.json
```json
{
  "金融学": {
    "employment_rate": 0.72,
    "avg_salary": 8500,
    "top_directions": ["银行柜员", "理财顾问", "财务"],
    "resource_threshold": "high",
    "description": "金融行业门槛高，985/211优先"
  }
}
```

### 3.2 universities.json
```json
{
  "郑州大学": {
    "province": "河南",
    "tier": "211",
    "min_score_2024": 580,
    "avg_score_2024": 610
  }
}
```

### 3.3 industries.json
```json
{
  "金融": {
    "entry_barrier": "high",
    "family_resource_dependent": true,
    "top_employers": ["银行", "证券", "保险"],
    "graduate_distribution": {
      "top_tier": 0.15,
      "mid_tier": 0.35,
      "grassroots": 0.50
    }
  }
}
```

### 3.4 decision_rules.json
```json
{
  "score_range_strategies": {
    "550-600": "优先选城市，其次专业，最后学校",
    "600-650": "可以兼顾学校和专业",
    "500-550": "专业>城市>学校，学一门手艺最重要"
  }
}
```

### 3.5 知识库扩展性

**增量更新机制** (`data_updater.py`):
- 支持单条新增/批量导入 JSON 数据
- 数据校验：确保必填字段完整
- 增量合并：相同 key 覆盖，新 key 追加
- 备份：更新前自动备份原文件

**热门查询缓存策略**:
- 一级缓存：内存 dict + TTL (5分钟)，适合高频查询
- 二级缓存：热门查询结果预计算，启动时加载
- 缓存 key：`hash(question_type + score + province + major)`

## 4. API 接口

### POST /chat (streaming SSE)
```json
{
  "message": "河南560分想学金融",
  "history": [{"role": "user", "content": "..."}],
  "session_id": "uuid"
}
```
Response: Server-Sent Events (streaming chunks)

### GET /health
Response: `{"status": "ok"}`

## 5. 配置管理

### 环境变量 (.env)

通过 `pydantic-settings` 的 `BaseSettings` 统一管理：

```python
class AppSettings(BaseSettings):
    openai_api_key: str
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"  # openai / deepseek / qwen
    cache_ttl_seconds: int = 300
    max_history_length: int = 20

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
```

**模型切换**:
- `OPENAI_API_BASE`: API 地址（切换服务商时修改）
- `OPENAI_MODEL`: 模型名称（如 `gpt-4o`, `deepseek-chat`, `qwen-plus`）
- `LLM_PROVIDER`: 提供商标识（用于日志和错误提示）
- 修改环境变量后自动热加载，无需重启服务

### 数据安全
- `.env` 文件在 `.gitignore` 中排除
- 提供 `.env.example` 模板
- API Key 不在日志中打印

## 6. 测试策略

| 模块 | 测试内容 |
|------|----------|
| KnowledgeBase | 按分数/地区/专业查询、无匹配返回空、边界值 |
| PromptBuilder | 不同问题类型生成正确模板、模块可独立加载 |
| LLMChain | mock LLM 调用、流式输出格式正确、异常处理 |
| API | /chat 接受请求返回流、/health 返回ok、无效请求返回400 |
| 多轮对话 | 历史消息正确传递、session 隔离 |

### Prompt 回归验证

`test_prompt_builder.py` 必须包含：
- **完整性测试**: 验证 System Prompt 包含全部三个模块（思维模型、决策启发式、表达DNA）
- **动态注入测试**: 验证专业咨询/院校选择/行业前景/综合规划各注入正确的数据片段
- **模板回归测试**: 对已知输入，断言生成的 prompt 包含关键锚点字符串（如"现实优先"、"就业率"等）
- **边界测试**: 无匹配数据时 prompt 仍完整，只缺少数据注入部分
- **模块独立性**: 每个 prompt 模块可单独加载，不依赖其他文件存在

## 7. 文件结构

```
.
├── backend/
│   ├── __init__.py
│   ├── main.py                      # FastAPI 入口 + 路由
│   ├── models/
│   │   ├── __init__.py
│   │   ├── message.py               # Pydantic 消息模型
│   │   └── config.py                # AppSettings (BaseSettings)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── knowledge_base.py        # 知识库查询 + 缓存
│   │   ├── prompt_builder.py        # Prompt 构建 + 模块加载
│   │   ├── llm_chain.py             # LangChain 链
│   │   └── data_updater.py          # 数据增量更新脚本
│   ├── data/
│   │   ├── majors.json
│   │   ├── universities.json
│   │   ├── industries.json
│   │   └── decision_rules.json
│   └── prompts/
│       ├── mental_models.txt
│       ├── decision_heuristics.txt
│       └── expression_dna.txt
├── frontend/
│   └── app.py                       # Streamlit 主程序
├── tests/
│   ├── __init__.py
│   ├── test_knowledge_base.py
│   ├── test_prompt_builder.py       # 含 Prompt 回归验证
│   ├── test_llm_chain.py
│   └── test_api.py
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 8. 运行流程

### 本地开发
1. `pip install -r requirements.txt`
2. `cp .env.example .env` 配置 API key 和模型
3. `uvicorn backend.main:app --reload` 启动后端 (默认 8000)
4. `streamlit run frontend/app.py` 启动前端 (默认 8501)
5. 访问 http://localhost:8501 使用

### Docker 部署

**单容器模式** (推荐开发/演示):
- 使用 `Dockerfile` 构建，同时包含后端和前端
- Streamlit 通过 `http://localhost:8000` 调用后端 API
- `docker build -t zhangxuefeng-ai . && docker run -p 8501:8501 --env-file .env zhangxuefeng-ai`

**分离容器模式** (推荐生产):
- 使用 `docker-compose.yml`
- `backend` 容器: uvicorn (8000)
- `frontend` 容器: streamlit (8501), 通过环境变量 `BACKEND_URL` 连接后端
- `docker-compose up -d`

### 模型切换示例
```bash
# 切换到 DeepSeek
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat

# 切换到通义千问
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-plus
```
修改 `.env` 后重启服务即可生效。
