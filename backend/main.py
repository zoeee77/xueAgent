"""FastAPI 应用入口 + API 路由。"""

import json
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    # Fallback for environments without sse-starlette
    EventSourceResponse = None  # type: ignore

from backend.logging_config import setup_logging, trace_id_ctx
from backend.models.config import settings
from backend.models.message import ChatRequest, HealthResponse, AdviseRequest, AdviseResponse
from backend.services.knowledge_base import KnowledgeBase
from backend.services.prompt_builder import PromptBuilder
from backend.services.llm_chain import stream_chat
from backend.agents.orchestrator import Orchestrator
from backend.agents.refiner import Refiner
from backend.agents.intent_parser import IntentParser
from backend.memory.memory_manager import MemoryManager
from backend.fallback.fallback_handler import FallbackHandler

logger = logging.getLogger(__name__)


# 全局服务实例
kb: KnowledgeBase
prompt_builder: PromptBuilder
orchestrator: Orchestrator
refiner: Refiner
intent_parser: IntentParser
memory_manager: MemoryManager
fallback_handler: FallbackHandler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期管理。"""
    global kb, prompt_builder, orchestrator, refiner, intent_parser, memory_manager, fallback_handler
    kb = KnowledgeBase(cache_ttl=settings.cache_ttl_seconds)
    prompt_builder = PromptBuilder(knowledge_base=kb)
    orchestrator = Orchestrator(kb=kb)
    refiner = Refiner(kb=kb)
    intent_parser = IntentParser()
    memory_manager = MemoryManager()
    fallback_handler = FallbackHandler()

    # 设置日志
    setup_logging()

    # 初始化记忆系统
    try:
        await memory_manager.init()
    except Exception as e:
        logger.warning(f"Memory manager init failed (continuing without memory): {e}")

    logger.info("Multi-agent system initialized successfully")
    yield

    # Cleanup
    try:
        await memory_manager.close()
    except Exception:
        pass


app = FastAPI(title="张雪峰 AI 志愿填报顾问", lifespan=lifespan)

# Session 存储：session_id -> history
sessions: dict[str, list[dict]] = {}


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查端点。"""
    return HealthResponse(status="ok")


@app.post("/chat")
async def chat(request: ChatRequest):
    """流式聊天接口。使用 SSE 返回流式输出。

    Request:
        message: 用户输入
        history: 历史消息列表（可选）
        session_id: 会话ID（可选，不提供则自动生成）

    Response:
        Server-Sent Events stream，每个 event 包含 text chunk
    """
    global kb, prompt_builder

    session_id = request.session_id or "default"

    # 获取或创建 session 历史
    if session_id not in sessions:
        sessions[session_id] = []

    # 合并传入历史和 session 历史
    # Convert Pydantic Message objects to dicts for consistency
    if request.history:
        history = []
        for msg in request.history:
            if hasattr(msg, "model_dump"):
                history.append(msg.model_dump())
            elif hasattr(msg, "dict"):
                history.append(msg.dict())
            elif isinstance(msg, dict):
                history.append(msg)
    else:
        history = sessions[session_id]

    # 截断历史消息（防止 token 超出限制）
    max_len = settings.max_history_length
    if len(history) > max_len:
        history = history[-max_len:]

    # 构建 System Prompt
    system_prompt = prompt_builder.build_system_prompt(
        question=request.message,
        history=history,
    )

    # 保存用户消息到 session
    sessions[session_id].append({"role": "user", "content": request.message})

    # 构建完整历史（包含当前请求）
    full_history = history + [{"role": "user", "content": request.message}]

    async def event_generator():
        assistant_response = ""
        async for chunk in stream_chat(
            question=request.message,
            system_prompt=system_prompt,
            history=full_history,
        ):
            assistant_response += chunk
            yield {"event": "message", "data": json.dumps({"chunk": chunk})}

        # 保存助手回复到 session
        sessions[session_id].append({"role": "assistant", "content": assistant_response})
        yield {"event": "done", "data": json.dumps({"complete": True})}

    if EventSourceResponse is not None:
        return EventSourceResponse(event_generator())
    else:
        # Fallback: return as plain streaming response
        async def text_generator():
            async for item in event_generator():
                yield json.dumps(item) + "\n"
        return StreamingResponse(text_generator(), media_type="text/event-stream")


@app.post("/chat/sync")
async def chat_sync_endpoint(request: ChatRequest):
    """同步聊天接口（非流式，适用于不支持 SSE 的场景）。"""
    from backend.services.llm_chain import chat_sync

    global kb, prompt_builder

    try:
        session_id = request.session_id or "default"

        # Convert Pydantic Message objects to dicts for consistency
        if request.history:
            history = []
            for msg in request.history:
                if hasattr(msg, "model_dump"):
                    history.append(msg.model_dump())
                elif hasattr(msg, "dict"):
                    history.append(msg.dict())
                elif isinstance(msg, dict):
                    history.append(msg)
        else:
            history = sessions.get(session_id, [])

        logger.info(f"Received chat request: session={session_id}, message={request.message[:50]}..., history_len={len(history)}")

        system_prompt = prompt_builder.build_system_prompt(
            question=request.message,
            history=history,
        )

        logger.info(f"Built system prompt, length={len(system_prompt)}")

        response_text = chat_sync(
            question=request.message,
            system_prompt=system_prompt,
            history=history,
        )

        logger.info(f"Got response, length={len(response_text)}")

        return {"message": response_text}
    except Exception as e:
        import traceback
        logger.error(f"Chat sync error: {e}\n{traceback.format_exc()}")
        raise


@app.post("/advise", response_model=AdviseResponse)
async def advise(request: AdviseRequest):
    """多智能体高级咨询端点。

    接收用户分数、地区、兴趣等信息，通过多智能体协作生成报考方案。
    返回完整的决策链结果：用户画像 → 数据检索 → 多角色分析 → 方案生成 → 排序 → 解释。
    """
    global kb, orchestrator, intent_parser, memory_manager

    # 生成 trace_id
    import uuid
    trace_id = str(uuid.uuid4())
    trace_id_ctx.set(trace_id)

    session_id = request.session_id or trace_id

    logger.info(f"Advise request: trace_id={trace_id}, score={request.score}, province={request.province}, interests={request.interests}")

    try:
        # 读取用户历史偏好（如果有）
        history_preferences = {}
        try:
            history_preferences = await memory_manager.get_preferences(session_id)
        except Exception as e:
            logger.warning(f"Failed to get preferences: {e}")

        # 构建用户输入
        user_input = {
            "score": request.score,
            "province": request.province,
            "interests": request.interests,
            "personality": request.personality,
            "family_resources": request.family_resources,
        }

        # 执行多智能体决策链
        result = await orchestrator.execute(user_input, trace_id=trace_id)

        # 保存用户偏好到记忆
        try:
            await memory_manager.set_preference(session_id, "province", request.province)
            for interest in request.interests:
                await memory_manager.set_preference(session_id, f"interest_{interest}", "1")
        except Exception as e:
            logger.warning(f"Failed to save preferences: {e}")

        response = AdviseResponse(
            success=True,
            is_fallback=result.get("is_fallback", False),
            trace_id=trace_id,
            user_profile=result.get("profile"),
            data_result=result.get("data_retrieval"),
            multi_role_result=result.get("multi_role_analysis"),
            plans=result.get("plan"),
            ranked_plans=result.get("rank"),
            devil_advocate=result.get("devil_advocate"),
            explanation=result.get("explanation"),
        )

        logger.info(f"Advise completed: trace_id={trace_id}")
        return response

    except Exception as e:
        logger.error(f"Advise error: {e}\n{traceback.format_exc()}")
        return AdviseResponse(
            success=False,
            trace_id=trace_id,
            error=str(e),
        )


@app.post("/refine", response_model=AdviseResponse)
async def refine_endpoint(request: AdviseRequest):
    """多轮优化端点。

    接收用户反馈（如"不喜欢计算机"、"想去一线城市"），基于反馈重新生成推荐方案。
    """
    global kb, orchestrator, intent_parser, refiner

    import uuid
    trace_id = str(uuid.uuid4())
    trace_id_ctx.set(trace_id)

    session_id = request.session_id or trace_id

    logger.info(f"Refine request: trace_id={trace_id}, message from interests: {request.interests}")

    try:
        # 解析用户意图
        feedback_text = " ".join(request.interests) if request.interests else ""
        intent_result = await intent_parser.parse(feedback_text)

        # 获取当前用户画像
        current_profile = None
        try:
            profile_data = await memory_manager.get_profile(session_id)
            if profile_data:
                from backend.models.agent_output import UserProfile
                current_profile = UserProfile(**profile_data)
        except Exception as e:
            logger.warning(f"Failed to get profile: {e}")

        # 执行优化
        result = await orchestrator.refine(
            user_input={
                "score": request.score,
                "province": request.province,
                "interests": request.interests,
                "personality": request.personality,
                "family_resources": request.family_resources,
            },
            intent_result=intent_result,
            trace_id=trace_id,
        )

        response = AdviseResponse(
            success=True,
            is_fallback=result.get("is_fallback", False),
            trace_id=trace_id,
            user_profile=result.get("profile"),
            plans=result.get("plan"),
            ranked_plans=result.get("rank"),
            explanation=result.get("explanation"),
        )

        logger.info(f"Refine completed: trace_id={trace_id}")
        return response

    except Exception as e:
        logger.error(f"Refine error: {e}\n{traceback.format_exc()}")
        return AdviseResponse(
            success=False,
            trace_id=trace_id,
            error=str(e),
        )
