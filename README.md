# 🎓 张雪峰 AI 志愿填报顾问

> 「选择比努力更重要，但'有得选'的前提是你足够努力。」

基于 [zhangxuefeng-skill](https://github.com/alchaincyf/zhangxuefeng-skill) 思维框架构建的 AI 志愿填报顾问系统。

## 功能特点

- **数据驱动**：内置专业就业率、院校分数线、行业就业分布等 JSON 知识库
- **思维模型**：5大核心心智模型 + 8条决策启发式 + 完整表达DNA
- **动态 Prompt**：根据问题类型自动注入相关数据和策略
- **多轮对话**：支持上下文记忆，session 隔离
- **模型切换**：支持任何 OpenAI 兼容 API（OpenAI/DeepSeek/通义千问等）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 3. 启动后端

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 启动前端

```bash
streamlit run frontend/app.py
```

### 5. 访问

打开浏览器访问 http://localhost:8501

## Docker 部署

### 单容器模式（开发/演示）

```bash
docker build -t zhangxuefeng-ai .
docker run -p 8501:8501 --env-file .env zhangxuefeng-ai
```

### 分离容器模式（生产）

```bash
docker-compose up -d
```

## 模型切换

编辑 `.env` 文件：

```bash
# OpenAI
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# DeepSeek
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat

# 通义千问
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-plus
```

修改后重启服务即可生效。

## 项目结构

```
├── backend/              # FastAPI 后端
│   ├── main.py           # 应用入口 + API 路由
│   ├── models/           # 数据模型
│   ├── services/         # 业务逻辑（知识库/Prompt/LLM）
│   ├── data/             # JSON 知识库数据
│   └── prompts/          # Prompt 模板模块
├── frontend/             # Streamlit 前端
│   └── app.py            # 多轮对话界面
├── tests/                # 测试
├── requirements.txt      # Python 依赖
├── .env.example          # 环境变量模板
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 运行测试

```bash
pytest tests/ -v --tb=short
```

## 知识库扩展

新增专业数据：

```python
from backend.services.data_updater import add_or_update_data, validate_major

new_major = {
    "人工智能": {
        "employment_rate": 0.93,
        "avg_salary": 15000,
        "top_directions": ["算法工程师", "数据科学家"],
        "resource_threshold": "low",
        "description": "新兴热门专业，薪资高但门槛也高"
    }
}

errors = validate_major(new_major["人工智能"])
if not errors:
    add_or_update_data("majors.json", new_major)
```

## License

MIT
