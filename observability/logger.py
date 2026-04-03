"""
observability/logger.py

Structured JSON logging using structlog.
Every agent step emits a log event with:
- timestamp, level, logger name
- session_id, trace_id (when available)
- event-specific fields

Usage:
    from observability.logger import get_logger
    log = get_logger(__name__)
    log.info("sql_agent.done", rows=42, latency_ms=120.5)
"""

from __future__ import annotations
import logging
import sys
import structlog
from structlog.types import Processor


def _add_app_context(logger, method_name, event_dict):
    """Add static app context to every log event."""
    event_dict.setdefault("app", "enterprise-ai-agent-platform")
    return event_dict


def configure_logging(log_level: str = "INFO", json_logs: bool = True) -> None:
    """Call once at app startup."""
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _add_app_context,
    ]

    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def bind_trace_context(session_id: str, trace_id: str) -> None:
    """Bind IDs to the current async context so all logs in a request carry them."""
    structlog.contextvars.bind_contextvars(
        session_id=session_id,
        trace_id=trace_id,
    )


def clear_trace_context() -> None:
    structlog.contextvars.clear_contextvars()
