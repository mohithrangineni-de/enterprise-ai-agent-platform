"""
api/main.py

FastAPI application — the external interface to the agent platform.
"""

from __future__ import annotations
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import QueryRequest, QueryResponse, TraceResponse, HealthResponse
from core.graph import agent_graph
from core.state import AgentState
from core.config import settings
from observability.logger import configure_logging, bind_trace_context, clear_trace_context, get_logger
from observability.tracer import build_trace_summary

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(log_level=settings.log_level, json_logs=settings.json_logs)
    log.info("app.startup", version="1.0.0", model=settings.llm_model)
    yield
    log.info("app.shutdown")


app = FastAPI(
    title="Enterprise AI Agent Platform",
    description="Multi-agent analytics system with RAG, SQL, and Python execution",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Main endpoint — runs the full agent pipeline for a question.
    """
    session_id = req.session_id or str(uuid.uuid4())[:8]
    trace_id = str(uuid.uuid4())[:12]

    bind_trace_context(session_id=session_id, trace_id=trace_id)
    log.info("api.query.start", question=req.question[:80])

    initial_state = AgentState(
        session_id=session_id,
        question=req.question,
        conversation_history=req.conversation_history or [],
        trace_id=trace_id,
    )

    try:
        final_state: AgentState = await agent_graph.ainvoke(initial_state)
    except Exception as e:
        log.error("api.query.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_trace_context()

    return QueryResponse(
        session_id=session_id,
        trace_id=trace_id,
        answer=final_state.final_answer,
        confidence=final_state.answer_confidence,
        sources=final_state.cited_sources,
        agents_used=[r.agent_type for r in final_state.results],
        routing_rationale=final_state.routing_rationale,
        total_tokens=final_state.total_tokens,
        total_latency_ms=round(final_state.total_latency_ms, 2),
    )


@app.get("/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str):
    """
    Retrieve the full trace for a completed query.
    Useful for the observability dashboard and debugging.
    """
    summary = build_trace_summary(trace_id)
    return TraceResponse(**summary)
