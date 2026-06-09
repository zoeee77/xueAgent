"""API 端点测试。"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock modules before importing backend.main
sys.modules.setdefault("sse_starlette", MagicMock())
sys.modules.setdefault("sse_starlette.sse", MagicMock())

# Mock langchain modules
_mock_lc_openai = MagicMock()
_mock_lc_openai.ChatOpenAI = MagicMock()
sys.modules.setdefault("langchain_openai", _mock_lc_openai)

_mock_lc_core = MagicMock()
_mock_lc_core.messages = MagicMock()
sys.modules.setdefault("langchain_core", _mock_lc_core)
sys.modules.setdefault("langchain_core.messages", _mock_lc_core.messages)

# Mock pydantic_settings
_mock_ps = MagicMock()
_mock_ps.BaseSettings = type("BaseSettings", (), {"__init__": lambda self, **kw: None})
_mock_ps.SettingsConfigDict = MagicMock(return_value={})
sys.modules.setdefault("pydantic_settings", _mock_ps)

# Mock backend.models.config to provide settings with required attributes
class MockSettings:
    openai_api_key: str = "test-key"
    openai_api_base: str = "https://api.example.com/v1"
    openai_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    cache_ttl_seconds: int = 300
    max_history_length: int = 20
    backend_url: str = "http://localhost:8000"

    # Embedding 配置
    embedding_strategy: str = "hash"
    embedding_api_key: str = ""
    embedding_api_base: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_local_model: str = "BAAI/bge-small-zh"
    embedding_cache_max_size: int = 10000

    # Vector Index 配置
    vector_index_engine: str = "auto"

    # 检索权重
    retrieval_weight_semantic: float = 0.40
    retrieval_weight_rule: float = 0.35
    retrieval_weight_keyword: float = 0.25

_mock_config = MagicMock()
_mock_config.settings = MockSettings()
_mock_config.AppSettings = MagicMock()
_mock_config.SettingsConfigDict = MagicMock()
sys.modules.setdefault("backend.models.config", _mock_config)

from fastapi.testclient import TestClient
from backend.main import app, sessions


@pytest.fixture
def client():
    """创建测试客户端。"""
    # 在测试中初始化 kb 和 prompt_builder
    from backend.services.knowledge_base import KnowledgeBase
    from backend.services.prompt_builder import PromptBuilder
    import backend.main

    backend.main.kb = KnowledgeBase()
    backend.main.prompt_builder = PromptBuilder(backend.main.kb)

    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    """每个测试后清空 sessions。"""
    yield
    sessions.clear()


class TestHealthEndpoint:
    """健康检查测试。"""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestChatEndpoint:
    """聊天接口测试。"""

    @patch("backend.main.stream_chat")
    def test_chat_accepts_request(self, mock_stream, client):
        """测试聊天接口能接受请求。"""
        async def mock_generator():
            yield {"event": "message", "data": '{"chunk": "测试回复"}'}
            yield {"event": "done", "data": '{"complete": true}'}

        mock_stream.return_value = mock_generator()

        response = client.post(
            "/chat",
            json={"message": "金融学怎么样", "session_id": "test-1"},
        )
        # SSE 响应，status code 应该是 200
        assert response.status_code == 200

    def test_chat_invalid_request(self, client):
        """测试无效请求（缺少 message 字段）。"""
        response = client.post("/chat", json={})
        # pydantic 验证失败返回 422
        assert response.status_code == 422

    @patch("backend.main.stream_chat")
    def test_chat_stores_session(self, mock_stream, client):
        """测试聊天接口存储 session 历史。"""
        async def mock_generator():
            yield {"event": "message", "data": '{"chunk": "回复"}'}
            yield {"event": "done", "data": '{"complete": true}'}

        mock_stream.return_value = mock_generator()

        client.post("/chat", json={"message": "你好", "session_id": "test-session"})

        assert "test-session" in sessions
        assert sessions["test-session"][0]["role"] == "user"
        assert sessions["test-session"][0]["content"] == "你好"

    def test_chat_default_session(self, client):
        """测试不提供 session_id 时使用 default。"""
        with patch("backend.main.stream_chat") as mock_stream:
            async def mock_generator():
                yield {"event": "message", "data": '{"chunk": "回复"}'}
                yield {"event": "done", "data": '{"complete": true}'}
            mock_stream.return_value = mock_generator()

            client.post("/chat", json={"message": "你好"})
            assert "default" in sessions


class TestSyncChatEndpoint:
    """同步聊天接口测试。"""

    @patch("backend.services.llm_chain.chat_sync")
    def test_sync_chat_returns_message(self, mock_chat, client):
        mock_chat.return_value = "测试回复"
        response = client.post("/chat/sync", json={"message": "你好"})
        assert response.status_code == 200
        assert "message" in response.json()
