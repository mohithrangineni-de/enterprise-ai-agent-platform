"""
core/state.py

Shared state schema that flows through the LangGraph agent graph.
Every agent reads from and writes to this state object.
"""

from __future__ import annotations
from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field
import operator
from datetime import datetime


# ─── Sub-task produced by the Planner ──────────────────────────────────────

class AgentTask(BaseModel):
    task_id: str
    agent_type: Literal["sql", "rag", "python", "response"]
    description: str
    dependencies: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "failed"] = "pending"


# ─── Output from each specialist agent ────────────────────────────────────

class AgentResult(BaseModel):
    task_id: str
    agent_type: str
    output: Any
    sources: list[str] = Field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=dict)


# ─── A single trace span (used by observability layer) ────────────────────

class TraceSpan(BaseModel):
    span_id: str
    agent: str
    action: str
    input_summary: str
    output_summary: str
    started_at: datetime
    ended_at: datetime | None = None
    status: Literal["ok", "error"] = "ok"


# ─── Main graph state — passed between all nodes ──────────────────────────

class AgentState(BaseModel):
    """
    The single source of truth flowing through the LangGraph graph.
    Uses Annotated[list, operator.add] so parallel branches can append
    without overwriting each other.
    """

    # User input
    session_id: str
    question: str
    conversation_history: list[dict] = Field(default_factory=list)

    # Planner outputs
    tasks: list[AgentTask] = Field(default_factory=list)
    routing_rationale: str = ""

    # Accumulated results from all agents
    # operator.add enables concurrent agents to append safely
    results: Annotated[list[AgentResult], operator.add] = Field(default_factory=list)

    # Final synthesized answer
    final_answer: str = ""
    answer_confidence: float = 0.0
    cited_sources: list[str] = Field(default_factory=list)

    # Observability
    trace_id: str = ""
    spans: Annotated[list[TraceSpan], operator.add] = Field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: float = 0.0

    # Control flow
    error: str | None = None
    completed: bool = False


# ─── Convenience helpers ───────────────────────────────────────────────────

def get_results_by_type(state: AgentState, agent_type: str) -> list[AgentResult]:
    return [r for r in state.results if r.agent_type == agent_type]


def state_summary(state: AgentState) -> dict:
    """Compact dict for logging — avoids dumping full content."""
    return {
        "session_id": state.session_id,
        "trace_id": state.trace_id,
        "question_len": len(state.question),
        "tasks": len(state.tasks),
        "results": len(state.results),
        "completed": state.completed,
        "total_tokens": state.total_tokens,
        "total_latency_ms": state.total_latency_ms,
        "error": state.error,
    }
