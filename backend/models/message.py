from pydantic import BaseModel, Field
from typing import Optional, Any


class Message(BaseModel):
    """单条聊天消息。"""

    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    """聊天请求。"""

    message: str
    history: list[Message] = Field(default_factory=list)
    session_id: Optional[str] = None


class AdviseRequest(BaseModel):
    """多智能体高级咨询请求。"""

    score: int
    province: str
    interests: list[str] = Field(default_factory=list)
    personality: Optional[str] = None
    family_resources: Optional[str] = None
    session_id: Optional[str] = None


class AdviseResponse(BaseModel):
    """多智能体高级咨询响应。
    
    正常模式下 plans/explanation 为 dict 结构，
    fallback 模式下 plans 为 list，explanation 为 str。
    """

    success: bool
    is_fallback: bool = False
    trace_id: str
    user_profile: Optional[dict] = None
    data_result: Optional[dict] = None
    multi_role_result: Optional[dict] = None
    plans: Optional[Any] = None
    ranked_plans: Optional[Any] = None
    devil_advocate: Optional[Any] = None
    explanation: Optional[Any] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = "ok"


class CreateSessionRequest(BaseModel):
    """创建会话请求。"""

    title: Optional[str] = "新对话"


class SessionInfo(BaseModel):
    """会话信息。"""

    session_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


class SessionListResponse(BaseModel):
    """会话列表响应。"""

    sessions: list[SessionInfo]


class SessionMessage(BaseModel):
    """会话中的消息。"""

    role: str
    content: str
    timestamp: str


class SessionHistoryResponse(BaseModel):
    """会话历史响应。"""

    session_id: str
    messages: list[SessionMessage]


class LoginRequest(BaseModel):
    """登录请求。"""

    username: str
    password: str


class RegisterRequest(BaseModel):
    """注册请求。"""

    username: str
    password: str


class AuthResponse(BaseModel):
    """认证响应。"""

    access_token: str
    token_type: str = "Bearer"
    user_id: str
    username: str
