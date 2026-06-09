from enum import Enum
from typing import Optional

from pydantic import BaseModel


class StepName(str, Enum):
    USER_PROFILE = "USER_PROFILE"
    DATA_RETRIEVE = "DATA_RETRIEVE"
    MULTI_ROLE_REASON = "MULTI_ROLE_REASON"
    PLAN = "PLAN"
    RANK = "RANK"
    EXPLAIN = "EXPLAIN"
    INTENT_PARSE = "INTENT_PARSE"
    REFINE = "REFINE"
    TOOL_CALL = "TOOL_CALL"
    FALLBACK = "FALLBACK"
    ORCHESTRATE = "ORCHESTRATE"


class AgentStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    RETRYING = "RETRYING"


class AgentState(BaseModel):
    trace_id: str
    step: StepName
    status: AgentStatus = AgentStatus.PENDING
    retry_count: int = 0
    max_retries: int = 2
    timeout_seconds: int = 30
    error_message: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.completed_at if self.completed_at is not None else None
        if end is None:
            import time

            end = time.time()
        return end - self.started_at
