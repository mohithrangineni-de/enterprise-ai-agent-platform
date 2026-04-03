"""
memory/short_term.py

Conversation buffer memory — stores the last N turns per session.
Used by the Planner agent for multi-turn context awareness.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationMemory:
    session_id: str
    max_turns: int = 10
    _buffer: deque = field(default_factory=deque)

    def add(self, role: str, content: str) -> None:
        self._buffer.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep only last max_turns pairs
        while len(self._buffer) > self.max_turns * 2:
            self._buffer.popleft()

    def get_history(self, last_n: int | None = None) -> list[dict]:
        history = list(self._buffer)
        if last_n:
            history = history[-(last_n * 2):]
        return [{"role": m["role"], "content": m["content"]} for m in history]

    def clear(self) -> None:
        self._buffer.clear()


# ─── Session registry (in-process; use Redis in production) ───────────────

_sessions: dict[str, ConversationMemory] = {}


def get_session_memory(session_id: str, max_turns: int = 10) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory(session_id=session_id, max_turns=max_turns)
    return _sessions[session_id]
