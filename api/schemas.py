"""
api/schemas.py

Pydantic v2 models for all API inputs and outputs.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None
    conversation_history: list[dict] | None = None

    model_config = {"json_schema_extra": {
        "example": {
            "question": "Why did revenue drop last quarter?",
            "session_id": "user_abc",
        }
    }}


class QueryResponse(BaseModel):
    session_id: str
    trace_id: str
    answer: str
    confidence: float
    sources: list[str]
    agents_used: list[str]
    routing_rationale: str
    total_tokens: int
    total_latency_ms: float


class TraceSpanOut(BaseModel):
    span_id: str
    trace_id: str
    name: str
    parent_span_id: str | None
    duration_ms: float
    status: str
    error: str | None
    attributes: dict


class TraceResponse(BaseModel):
    trace_id: str
    span_count: int = 0
    total_ms: float = 0.0
    spans: list[TraceSpanOut] = []
    errors: list[str] = []


class HealthResponse(BaseModel):
    status: str
    version: str
