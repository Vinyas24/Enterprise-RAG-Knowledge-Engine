"""Unit tests for pipeline tracing and W3C Server-Timing header generation."""

from __future__ import annotations

import asyncio
import time

import pytest

from rag_engine.observability.tracer import (
    PipelineTracer,
    get_current_tracer,
    reset_current_tracer,
    set_current_tracer,
    trace_stage,
    trace_stage_async,
)


def test_tracer_records_stages_and_durations() -> None:
    tracer = PipelineTracer()
    tracer.start_stage("dense")
    time.sleep(0.005)
    dur = tracer.end_stage("dense")

    assert dur > 0.0
    assert tracer.get_stage_duration("dense") == dur
    assert tracer.get_stage_duration("nonexistent") is None

    tracer.record_stage("keyword", 12.345)
    assert tracer.get_stage_duration("keyword") == 12.35
    timings = tracer.to_dict()
    assert "dense" in timings
    assert "keyword" in timings


def test_tracer_formats_server_timing_header() -> None:
    tracer = PipelineTracer()
    tracer.record_stage("retrieve", 15.25)
    tracer.record_stage("generate", 120.50)
    tracer.record_stage("custom_stage", 3.14)

    header = tracer.to_server_timing_header()
    assert 'retrieve;dur=15.25;desc="Retrieval Pipeline"' in header
    assert 'generate;dur=120.50;desc="LLM Answer Generation"' in header
    assert "custom_stage;dur=3.14" in header


def test_tracer_handles_unknown_end_gracefully() -> None:
    tracer = PipelineTracer()
    assert tracer.end_stage("never_started") == 0.0


def test_trace_stage_context_manager() -> None:
    # No tracer active: no-op
    with trace_stage("retrieve"):
        pass

    tracer = PipelineTracer()
    token = set_current_tracer(tracer)
    try:
        assert get_current_tracer() is tracer
        with trace_stage("dense"):
            time.sleep(0.002)

        dur = tracer.get_stage_duration("dense")
        assert dur is not None
        assert dur > 0.0
    finally:
        reset_current_tracer(token)

    assert get_current_tracer() is None


@pytest.mark.asyncio
async def test_trace_stage_async_context_manager() -> None:
    # No tracer active: no-op
    async with trace_stage_async("generate"):
        pass

    tracer = PipelineTracer()
    token = set_current_tracer(tracer)
    try:
        async with trace_stage_async("generate"):
            await asyncio.sleep(0.002)

        dur = tracer.get_stage_duration("generate")
        assert dur is not None
        assert dur > 0.0
    finally:
        reset_current_tracer(token)
