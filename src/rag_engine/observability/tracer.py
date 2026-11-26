"""Pipeline tracing for request timing breakdown and W3C Server-Timing headers.

Allows callers, operators, and dashboard tools to inspect exactly where time
goes across retrieval, rank fusion, re-ranking, LLM generation, and guardrails.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar, Token

logger = logging.getLogger(__name__)

STAGE_DESCRIPTIONS: dict[str, str] = {
    "guardrails_input": "Input Guardrails",
    "retrieve": "Retrieval Pipeline",
    "dense": "Dense Vector Search",
    "keyword": "BM25 Keyword Search",
    "fuse": "Reciprocal Rank Fusion",
    "rerank": "Re-ranking Pass",
    "generate": "LLM Answer Generation",
    "guardrails_output": "Output Guardrails",
    "total": "Total Query Latency",
}


class PipelineTracer:
    """Collects millisecond durations for each pipeline stage."""

    def __init__(self) -> None:
        self._timings: dict[str, float] = {}
        self._active_starts: dict[str, float] = {}

    def start_stage(self, name: str) -> None:
        """Mark the start timestamp for a stage."""
        self._active_starts[name] = time.perf_counter()

    def end_stage(self, name: str) -> float:
        """End an active stage and record its elapsed time in milliseconds."""
        start = self._active_starts.pop(name, None)
        if start is None:
            return 0.0
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        self._timings[name] = elapsed_ms
        return elapsed_ms

    def record_stage(self, name: str, duration_ms: float) -> None:
        """Explicitly record a stage duration in milliseconds."""
        self._timings[name] = round(duration_ms, 2)

    def get_stage_duration(self, name: str) -> float | None:
        """Return the recorded duration in milliseconds, or None if unrecorded."""
        return self._timings.get(name)

    def to_dict(self) -> dict[str, float]:
        """Return a copy of all recorded stage timings in milliseconds."""
        return dict(self._timings)

    def to_server_timing_header(self) -> str:
        """Format timings into a valid W3C Server-Timing header string."""
        parts: list[str] = []
        for name, dur_ms in self._timings.items():
            desc = STAGE_DESCRIPTIONS.get(name)
            if desc:
                parts.append(f'{name};dur={dur_ms:.2f};desc="{desc}"')
            else:
                parts.append(f"{name};dur={dur_ms:.2f}")
        return ", ".join(parts)


_CURRENT_TRACER: ContextVar[PipelineTracer | None] = ContextVar("_CURRENT_TRACER", default=None)


def get_current_tracer() -> PipelineTracer | None:
    """Return the active pipeline tracer for the current async context, if any."""
    return _CURRENT_TRACER.get()


def set_current_tracer(tracer: PipelineTracer) -> Token[PipelineTracer | None]:
    """Set the active pipeline tracer for the current async context."""
    return _CURRENT_TRACER.set(tracer)


def reset_current_tracer(token: Token[PipelineTracer | None]) -> None:
    """Reset the pipeline tracer back to its previous state."""
    _CURRENT_TRACER.reset(token)


@contextmanager
def trace_stage(name: str) -> Iterator[None]:
    """Context manager to measure and record a pipeline stage duration."""
    tracer = get_current_tracer()
    if tracer is None:
        yield
        return

    tracer.start_stage(name)
    try:
        yield
    finally:
        tracer.end_stage(name)


@asynccontextmanager
async def trace_stage_async(name: str) -> AsyncIterator[None]:
    """Async context manager to measure and record a pipeline stage duration."""
    tracer = get_current_tracer()
    if tracer is None:
        yield
        return

    tracer.start_stage(name)
    try:
        yield
    finally:
        tracer.end_stage(name)
