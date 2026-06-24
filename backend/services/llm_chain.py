"""LangChain LLM 链模块：构建和管理 LLM 调用。"""

import threading
from typing import Optional, AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.models.config import settings

# -- LLM 单例 ----------------------------------------------------------------

_llm_instance: Optional[ChatOpenAI] = None
_llm_lock = threading.Lock()


def create_llm() -> ChatOpenAI:
    """创建/返回 LLM 单例实例（线程安全懒加载）。

    避免每次调用都重新创建 ChatOpenAI 实例带来的初始化开销。
    配置变化时需重启服务。
    """
    global _llm_instance
    if _llm_instance is None:
        with _llm_lock:
            if _llm_instance is None:
                _llm_instance = ChatOpenAI(
                    model=settings.openai_model,
                    openai_api_key=settings.openai_api_key,
                    openai_api_base=settings.openai_api_base,
                    streaming=True,
                    temperature=0.7,
                )
    return _llm_instance


def format_history(history: list) -> list:
    """将历史消息转换为 LangChain 消息格式。"""
    messages = []
    for msg in history:
        # Support both dict-like and Pydantic model objects
        if hasattr(msg, "model_dump"):
            # Pydantic v2 model
            msg_dict = msg.model_dump()
        elif hasattr(msg, "dict"):
            # Pydantic v1 model
            msg_dict = msg.dict()
        elif isinstance(msg, dict):
            msg_dict = msg
        else:
            continue
        role = msg_dict.get("role", "user")
        content = msg_dict.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


async def stream_chat(
    question: str,
    system_prompt: str,
    history: Optional[list[dict]] = None,
) -> AsyncGenerator[str, None]:
    """流式调用 LLM。

    Args:
        question: 用户问题
        system_prompt: 构建好的 System Prompt
        history: 历史消息列表

    Yields:
        流式文本片段
    """
    llm = create_llm()

    messages = [SystemMessage(content=system_prompt)]
    if history:
        messages.extend(format_history(history))
    messages.append(HumanMessage(content=question))

    async for chunk in llm.astream(messages):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content


def chat_sync(
    question: str,
    system_prompt: str,
    history: Optional[list[dict]] = None,
) -> str:
    """同步调用 LLM（用于测试）。"""
    llm = create_llm()

    messages = [SystemMessage(content=system_prompt)]
    if history:
        messages.extend(format_history(history))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)
