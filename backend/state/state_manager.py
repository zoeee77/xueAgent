import asyncio
from typing import Optional

from backend.state.agent_state import AgentState, AgentStatus, StepName


class StateManager:
    """Thread-safe store for AgentState instances keyed by trace_id."""

    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}
        self._lock = asyncio.Lock()

    async def create_state(self, trace_id: str, step: StepName) -> AgentState:
        async with self._lock:
            state = AgentState(trace_id=trace_id, step=step)
            self._states[trace_id] = state
            return state

    async def update_state(self, trace_id: str, **kwargs) -> Optional[AgentState]:
        async with self._lock:
            state = self._states.get(trace_id)
            if state is None:
                return None
            for key, value in kwargs.items():
                setattr(state, key, value)
            return state

    async def get_state(self, trace_id: str) -> Optional[AgentState]:
        async with self._lock:
            return self._states.get(trace_id)
