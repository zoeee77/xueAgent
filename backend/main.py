"""FastAPI 应用入口 + API 路由。"""

import asyncio
import json
import logging
import time
import traceback
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.responses import StreamingResponse

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    # Fallback for environments without sse-starlette
    EventSourceResponse = None  # type: ignore

from backend.logging_config import setup_logging, trace_id_ctx
from backend.models.config import settings
from backend.models.message import ChatRequest, HealthResponse, AdviseRequest, AdviseResponse, CreateSessionRequest, SessionListResponse, SessionHistoryResponse, LoginRequest, RegisterRequest, AuthResponse, SessionMessage
from backend.services.knowledge_base import KnowledgeBase
from backend.services.prompt_builder import PromptBuilder
from backend.services.llm_chain import stream_chat
from backend.agents.orchestrator import Orchestrator
from backend.agents.refiner import Refiner
from backend.agents.intent_parser import IntentParser
from backend.memory.memory_manager import MemoryManager
from backend.fallback.fallback_handler import FallbackHandler
from backend.session.postgres_session_store import PostgreSQLSessionStore
from backend.auth.jwt_auth import get_current_user, create_access_token
from backend.auth.user_store import create_user, authenticate_user

logger = logging.getLogger(__name__)


# 全局服务实例
kb: KnowledgeBase
prompt_builder: PromptBuilder
orchestrator: Orchestrator
refiner: Refiner
intent_parser: IntentParser
memory_manager: MemoryManager
fallback_handler: FallbackHandler
pg_session_store: PostgreSQLSessionStore


async def ensure_session_owner_impl(user_id: str, session_id: str, store: PostgreSQLSessionStore) -> None:
    """校验 session 是否属于当前用户，否则抛出 403。"""
    owned = await asyncio.to_thread(store.check_ownership, user_id, session_id)
    if not owned:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权访问该会话",
        )


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
    pg_session_store = PostgreSQLSessionStore()

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


class SessionStore:
    """带 TTL + LRU 淘汰的 session 存储，防止内存泄漏。

    - 最大 1000 个 session
    - 30 分钟无访问自动淘汰
    - 满时淘汰最旧的 10%
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 1800):
        self._cache: dict[str, tuple[list[dict], float]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def get(self, key: str) -> list[dict]:
        if key in self._cache:
            data, ts = self._cache[key]
            if time.time() - ts < self._ttl:
                return data
            del self._cache[key]
        return []

    def set(self, key: str, value: list[dict]):
        if len(self._cache) >= self._max_size:
            # 淘汰最久未访问的 10%
            n = max(1, self._max_size // 10)
            oldest = sorted(self._cache.items(), key=lambda x: x[1][1])[:n]
            for k, _ in oldest:
                del self._cache[k]
        self._cache[key] = (value, time.time())

    def append(self, key: str, item: dict):
        data = self.get(key)
        data.append(item)
        self.set(key, data)

    def cleanup(self):
        now = time.time()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts >= self._ttl]
        for k in expired:
            del self._cache[k]


# Session 存储：带 TTL + LRU 淘汰
session_store = SessionStore(max_size=1000, ttl_seconds=1800)


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查端点。"""
    return HealthResponse(status="ok")


@app.post("/chat")
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
):
    """流式聊天接口。使用 SSE 返回流式输出。需要 JWT 认证。

    Request:
        message: 用户输入
        history: 历史消息列表（可选）
        session_id: 会话ID（可选，不提供则自动生成）

    Response:
        Server-Sent Events stream，每个 event 包含 text chunk
    """
    global kb, prompt_builder, pg_session_store

    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"

    # 会话所有权校验（如果 session 已存在）
    if request.session_id:
        await ensure_session_owner_impl(user_id, session_id, pg_session_store)

    # 从 PostgreSQL 读取历史消息
    if request.history is not None:
        history = []
        for msg in request.history:
            if hasattr(msg, "model_dump"):
                history.append(msg.model_dump())
            elif hasattr(msg, "dict"):
                history.append(msg.dict())
            elif isinstance(msg, dict):
                history.append(msg)
    else:
        history = await asyncio.to_thread(pg_session_store.get, session_id, settings.max_history_length)

    # 截断历史消息（防止 token 超出限制）
    max_len = settings.max_history_length
    if len(history) > max_len:
        history = history[-max_len:]

    # 构建 System Prompt
    system_prompt = prompt_builder.build_system_prompt(
        question=request.message,
        history=history,
    )

    # 保存用户消息到 PostgreSQL
    await asyncio.to_thread(pg_session_store.append, session_id, {"role": "user", "content": request.message})

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

        # 保存助手回复到 PostgreSQL
        await asyncio.to_thread(pg_session_store.append, session_id, {"role": "assistant", "content": assistant_response})
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
async def chat_sync_endpoint(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
):
    """同步聊天接口（非流式，适用于不支持 SSE 的场景）。需要 JWT 认证。"""
    from backend.services.llm_chain import chat_sync

    global kb, prompt_builder, pg_session_store

    try:
        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"

        # 会话所有权校验
        if request.session_id:
            await ensure_session_owner_impl(user_id, session_id, pg_session_store)

        # 从 PostgreSQL 读取历史消息
        if request.history is not None:
            history = []
            for msg in request.history:
                if hasattr(msg, "model_dump"):
                    history.append(msg.model_dump())
                elif hasattr(msg, "dict"):
                    history.append(msg.dict())
                elif isinstance(msg, dict):
                    history.append(msg)
        else:
            history = await asyncio.to_thread(pg_session_store.get, session_id, settings.max_history_length)

        logger.info(f"Received chat request: session={session_id}, user={user_id}, message={request.message[:50]}..., history_len={len(history)}")

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

        # 保存完整对话到 PostgreSQL
        await asyncio.to_thread(pg_session_store.append, session_id, {"role": "user", "content": request.message})
        await asyncio.to_thread(pg_session_store.append, session_id, {"role": "assistant", "content": response_text})

        return {"message": response_text}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Chat sync error: {e}\n{traceback.format_exc()}")
        raise


@app.post("/advise", response_model=AdviseResponse)
async def advise(
    request: AdviseRequest,
    user_id: str = Depends(get_current_user),
):
    """多智能体高级咨询端点。需要 JWT 认证。

    接收用户分数、地区、兴趣等信息，通过多智能体协作生成报考方案。
    返回完整的决策链结果：用户画像 → 数据检索 → 多角色分析 → 方案生成 → 排序 → 解释。
    """
    global kb, orchestrator, intent_parser, memory_manager

    # 生成 trace_id
    trace_id = str(uuid.uuid4())
    trace_id_ctx.set(trace_id)

    # 使用 JWT 中的 user_id 作为 session_id 基础
    session_id = request.session_id or f"sess_{user_id}_{uuid.uuid4().hex[:8]}"

    # 会话所有权校验
    if request.session_id:
        await ensure_session_owner_impl(user_id, session_id, pg_session_store)

    logger.info(f"Advise request: trace_id={trace_id}, user={user_id}, score={request.score}, province={request.province}, interests={request.interests}")

    try:
        # 读取用户历史偏好（基于 user_id）
        history_preferences = {}
        try:
            history_preferences = await memory_manager.get_preferences(user_id)
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

        # 保存用户偏好到记忆（使用 user_id）
        try:
            await memory_manager.set_preference(user_id, "province", request.province)
            for interest in request.interests:
                await memory_manager.set_preference(user_id, f"interest_{interest}", "1")
        except Exception as e:
            logger.warning(f"Failed to save preferences: {e}")

        # 处理 fallback 模式
        if result.get("is_fallback", False):
            fallback_data = result.get("fallback_result", {})
            response = AdviseResponse(
                success=True,
                is_fallback=True,
                trace_id=trace_id,
                user_profile=result.get("profile"),
                data_result=result.get("data_retrieval"),
                multi_role_result=result.get("multi_role_analysis"),
                plans=fallback_data.get("plans"),
                ranked_plans=result.get("rank"),
                devil_advocate=result.get("devil_advocate"),
                explanation=fallback_data.get("explanation"),
            )
        else:
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

        logger.info(f"Advise completed: trace_id={trace_id}, user={user_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Advise error: {e}\n{traceback.format_exc()}")
        return AdviseResponse(
            success=False,
            trace_id=trace_id,
            error=str(e),
        )


@app.post("/refine", response_model=AdviseResponse)
async def refine_endpoint(
    request: AdviseRequest,
    user_id: str = Depends(get_current_user),
):
    """多轮优化端点。需要 JWT 认证。

    接收用户反馈（如"不喜欢计算机"、"想去一线城市"），基于反馈重新生成推荐方案。
    """
    global kb, orchestrator, intent_parser, refiner

    trace_id = str(uuid.uuid4())
    trace_id_ctx.set(trace_id)

    # 使用 JWT 中的 user_id
    session_id = request.session_id or f"sess_{user_id}_{uuid.uuid4().hex[:8]}"

    # 会话所有权校验
    if request.session_id:
        await ensure_session_owner_impl(user_id, session_id, pg_session_store)

    logger.info(f"Refine request: trace_id={trace_id}, user={user_id}, message from interests: {request.interests}")

    try:
        # 解析用户意图
        feedback_text = " ".join(request.interests) if request.interests else ""
        intent_result = await intent_parser.parse(feedback_text)

        # 获取当前用户画像（使用 user_id）
        current_profile = None
        try:
            profile_data = await memory_manager.get_profile(user_id)
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

        logger.info(f"Refine completed: trace_id={trace_id}, user={user_id}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refine error: {e}\n{traceback.format_exc()}")
        return AdviseResponse(
            success=False,
            trace_id=trace_id,
            error=str(e),
        )


# ─── 认证 API ─────────────────────────────────────────────────────

@app.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """用户注册。

    Request:
        username: 用户名
        password: 密码

    Response:
        access_token, user_id, username
    """
    user_id = create_user(request.username, request.password)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="用户名已存在",
        )
    token = create_access_token({"sub": user_id, "username": request.username})
    return AuthResponse(access_token=token, user_id=user_id, username=request.username)


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """用户登录。

    Request:
        username: 用户名
        password: 密码

    Response:
        access_token, user_id, username
    """
    user_id = authenticate_user(request.username, request.password)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    token = create_access_token({"sub": user_id, "username": request.username})
    return AuthResponse(access_token=token, user_id=user_id, username=request.username)


# ─── 会话管理 API（受 JWT 保护）────────────────────────────────


@app.post("/session/create", response_model=SessionInfo)
async def create_session(
    request: CreateSessionRequest,
    user_id: str = Depends(get_current_user),
):
    """创建新会话。需要 JWT 认证，user_id 从 Token 自动提取。

    Request:
        title: 会话标题（可选）

    Response:
        创建的会话信息
    """
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    success = await asyncio.to_thread(pg_session_store.create_session, user_id, session_id, request.title)

    if not success:
        # session_id 冲突，重试一次
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        await asyncio.to_thread(pg_session_store.create_session, user_id, session_id, request.title)

    sessions = await asyncio.to_thread(pg_session_store.list_sessions, user_id, limit=1)
    return sessions[0] if sessions else SessionInfo(
        session_id=session_id,
        user_id=user_id,
        title=request.title or "新对话",
        created_at="",
        updated_at="",
    )


@app.get("/session/list", response_model=SessionListResponse)
async def list_sessions(
    user_id: str = Depends(get_current_user),
    limit: int = 20,
):
    """获取当前用户的会话列表。需要 JWT 认证。

    Query:
        limit: 返回数量（默认20）

    Response:
        会话列表
    """
    sessions = await asyncio.to_thread(pg_session_store.list_sessions, user_id, limit)
    return SessionListResponse(
        sessions=[SessionInfo(**s) for s in sessions]
    )


@app.get("/session/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: str,
    user_id: str = Depends(get_current_user),
    limit: int = 50,
):
    """获取某个会话的历史消息。需要 JWT 认证，只能查看自己的会话。

    Query:
        session_id: 会话ID
        limit: 返回消息数量（默认50）

    Response:
        会话历史消息列表
    """
    await ensure_session_owner_impl(user_id, session_id, pg_session_store)
    messages = await asyncio.to_thread(pg_session_store.get, session_id, limit)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[SessionMessage(**m) for m in messages],
    )


@app.delete("/session/delete")
async def delete_session(
    session_id: str,
    user_id: str = Depends(get_current_user),
):
    """删除某个会话。需要 JWT 认证，只能删除自己的会话。

    Query:
        session_id: 会话ID

    Response:
        {"success": true/false}
    """
    await ensure_session_owner_impl(user_id, session_id, pg_session_store)
    success = await asyncio.to_thread(pg_session_store.delete_session, session_id)
    return {"success": success}
