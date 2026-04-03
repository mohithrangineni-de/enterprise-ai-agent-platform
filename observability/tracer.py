"""
observability/tracer.py

Lightweight tracing layer compatible with OpenTelemetry.
Wraps agent steps in named spans, captures latency + status,
and stores spans in the shared AgentState for the dashboard.

For production: replace the in-process span store with
an OTLP exporter (Jaeger, Tempo, Honeycomb, Datadog).
"""

from __future__ import annotations
import uuid
import time
import contextlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable
from functools import wraps

from observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_span_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        end = self.ended_at or time.monotonic()
        return (end - self.started_at) * 1000

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "parent_span_id": self.parent_span_id,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


# ─── In-process span registry (replace with OTLP exporter in prod) ────────

_trace_store: dict[str, list[Span]] = {}


def get_spans(trace_id: str) -> list[Span]:
    return _trace_store.get(trace_id, [])


def _record_span(span: Span) -> None:
    _trace_store.setdefault(span.trace_id, []).append(span)
    log.debug("tracer.span_complete",
              name=span.name,
              trace_id=span.trace_id,
              duration_ms=round(span.duration_ms, 2),
              status=span.status)


# ─── Context manager for manual spans ─────────────────────────────────────

@contextlib.contextmanager
def trace_span(name: str, trace_id: str, attributes: dict | None = None):
    span = Span(name=name, trace_id=trace_id, attributes=attributes or {})
    try:
        yield span
    except Exception as e:
        span.status = "error"
        span.error = str(e)
        raise
    finally:
        span.ended_at = time.monotonic()
        _record_span(span)


# ─── Decorator for agent methods ───────────────────────────────────────────

def traced(span_name: str | None = None):
    """
    Decorator that wraps an async function in a trace span.
    Reads trace_id from the first AgentState argument if present.

    Usage:
        @traced("sql_agent.run")
        async def run(self, state: AgentState) -> AgentState:
            ...
    """
    def decorator(fn: Callable):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            # Try to extract trace_id from the state arg
            trace_id = "unknown"
            for arg in args:
                if hasattr(arg, "trace_id"):
                    trace_id = arg.trace_id or str(uuid.uuid4())[:8]
                    break

            name = span_name or f"{fn.__module__}.{fn.__qualname__}"
            span = Span(name=name, trace_id=trace_id)

            try:
                result = await fn(*args, **kwargs)
                span.ended_at = time.monotonic()
                _record_span(span)
                return result
            except Exception as e:
                span.status = "error"
                span.error = str(e)
                span.ended_at = time.monotonic()
                _record_span(span)
                raise

        return wrapper
    return decorator


# ─── Trace summary for the API response ───────────────────────────────────

def build_trace_summary(trace_id: str) -> dict:
    spans = get_spans(trace_id)
    if not spans:
        return {"trace_id": trace_id, "spans": [], "total_ms": 0}

    total = sum(s.duration_ms for s in spans)
    return {
        "trace_id": trace_id,
        "span_count": len(spans),
        "total_ms": round(total, 2),
        "spans": [s.to_dict() for s in spans],
        "errors": [s.name for s in spans if s.status == "error"],
    }
