from pydantic import BaseModel, Field
from typing import Optional


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
    """多智能体高级咨询响应。"""

    success: bool
    is_fallback: bool = False
    trace_id: str
    user_profile: Optional[dict] = None
    data_result: Optional[dict] = None
    multi_role_result: Optional[dict] = None
    plans: Optional[dict] = None
    ranked_plans: Optional[dict] = None
    devil_advocate: Optional[dict] = None
    explanation: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = "ok"
