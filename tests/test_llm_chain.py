"""LLM Chain 测试：使用 mock 测试 LLM 调用。"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Create message classes that store content properly
class _MockMessage:
    def __init__(self, content=""):
        self.content = content

class _MockHumanMessage(_MockMessage):
    pass

class _MockAIMessage(_MockMessage):
    pass

class _MockSystemMessage(_MockMessage):
    pass

# Mock pydantic_settings
class _MockBaseSettings:
    pass

class _MockSettingsConfigDict(dict):
    pass

_mock_pydantic_settings = MagicMock()
_mock_pydantic_settings.BaseSettings = _MockBaseSettings
_mock_pydantic_settings.SettingsConfigDict = _MockSettingsConfigDict

# Mock backend.models.config
_mock_settings = MagicMock()
_mock_settings.openai_api_key = "test-key"
_mock_settings.openai_api_base = "https://api.test.com/v1"
_mock_settings.openai_model = "gpt-4o-mini"

import types
_config_mod = types.ModuleType("backend.models.config")
_config_mod.AppSettings = MagicMock
_config_mod.settings = _mock_settings
_config_mod.BaseSettings = _MockBaseSettings
_config_mod.SettingsConfigDict = _MockSettingsConfigDict

# Mock langchain modules
_langchain_messages_module = MagicMock()
_langchain_messages_module.HumanMessage = _MockHumanMessage
_langchain_messages_module.AIMessage = _MockAIMessage
_langchain_messages_module.SystemMessage = _MockSystemMessage

_mock_langchain_core = MagicMock()

_mock_openai = MagicMock()

# Remove any previously cached modules from other test files
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith(("backend.services.llm_chain", "langchain", "pydantic")):
        del sys.modules[mod_name]

# Set fresh mocks
sys.modules["langchain_openai"] = _mock_openai
sys.modules["langchain_core"] = _mock_langchain_core
sys.modules["langchain_core.messages"] = _langchain_messages_module
sys.modules["pydantic_settings"] = _mock_pydantic_settings
sys.modules["backend.models.config"] = _config_mod

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.services.llm_chain import format_history, stream_chat


class TestFormatHistory:
    """历史消息格式化测试。"""

    def test_format_empty(self):
        result = format_history([])
        assert len(result) == 0

    def test_format_single_message(self):
        result = format_history([{"role": "user", "content": "你好"}])
        assert len(result) == 1
        assert result[0].content == "你好"

    def test_format_multiple_messages(self):
        history = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
            {"role": "user", "content": "问题2"},
        ]
        result = format_history(history)
        assert len(result) == 3
        assert result[0].content == "问题1"
        assert result[1].content == "回答1"
        assert result[2].content == "问题2"

    def test_format_unknown_role_treated_as_user(self):
        history = [{"role": "system", "content": "系统消息"}]
        result = format_history(history)
        # system role 不添加到历史（由 system_prompt 处理）
        assert len(result) == 0


@pytest.mark.asyncio
class TestStreamChat:
    """流式聊天测试（mock）。"""

    async def test_stream_yields_content(self):
        """测试流式输出能产生内容（需要 mock LLM）。"""
        mock_chunk = MagicMock()
        mock_chunk.content = "测试回复"

        async def mock_stream(_):
            yield mock_chunk

        with patch("backend.services.llm_chain.create_llm") as mock_create:
            mock_llm = MagicMock()
            mock_llm.astream = mock_stream
            mock_create.return_value = mock_llm

            chunks = []
            async for chunk in stream_chat("测试问题", "测试system prompt"):
                chunks.append(chunk)

            assert len(chunks) > 0
            assert chunks[0] == "测试回复"

    async def test_stream_with_history(self):
        """测试带历史消息的流式调用。"""
        mock_chunk = MagicMock()
        mock_chunk.content = "带历史的回复"

        async def mock_stream(_):
            yield mock_chunk

        with patch("backend.services.llm_chain.create_llm") as mock_create:
            mock_llm = MagicMock()
            mock_llm.astream = mock_stream
            mock_create.return_value = mock_llm

            history = [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好啊"},
            ]
            chunks = []
            async for chunk in stream_chat("继续问", "system", history=history):
                chunks.append(chunk)

            assert len(chunks) > 0
