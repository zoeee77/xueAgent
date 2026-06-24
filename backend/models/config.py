from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """应用配置，通过环境变量或 .env 文件加载。"""

    # LLM 配置
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"

    # Embedding 配置
    embedding_strategy: str = "hash"        # hash | local | api
    embedding_api_key: str = ""
    embedding_api_base: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_local_model: str = "BAAI/bge-small-zh"
    embedding_cache_max_size: int = 10000

    # Vector Index 配置
    vector_index_engine: str = "auto"       # faiss | numpy | auto | qdrant

    # Qdrant 向量数据库配置
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "majors"

    # 应用配置
    cache_ttl_seconds: int = 300
    max_history_length: int = 20
    backend_url: str = "http://localhost:8000"

    # 检索权重
    retrieval_weight_semantic: float = 0.40
    retrieval_weight_rule: float = 0.35
    retrieval_weight_keyword: float = 0.25

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# 全局单例，应用启动时加载
settings = AppSettings()
