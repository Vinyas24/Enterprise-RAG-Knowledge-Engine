"""Unit tests for the performance and load test harness."""

from __future__ import annotations

from uuid import uuid4

import pytest

from rag_engine.core.enums import AccessLevel
from rag_engine.evaluation.dataset import EvalQuestion
from rag_engine.evaluation.load_test import (
    LatencyStats,
    LoadTestReport,
    LoadTestRunResult,
    compute_latency_stats,
    execute_concurrent_requests,
    render_load_test_markdown,
)
from rag_engine.retrieval.base import RetrievedChunk, Retriever


class ConstantTimeRetriever(Retriever):
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    async def retrieve(
        self,
        *,
        tenant_id: object,
        query: str,
        top_k: int = 5,
        clearance: AccessLevel = AccessLevel.INTERNAL,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        del tenant_id, query, top_k, clearance, min_score
        if self.should_fail:
            raise RuntimeError("Synthetic failure")
        return [
            RetrievedChunk(
                chunk_id=uuid4(),
                document_id=uuid4(),
                document_title="Mock Doc",
                chunk_index=0,
                content="Mock content",
                score=1.0,
                access_level=AccessLevel.INTERNAL,
            )
        ]


def test_compute_latency_stats() -> None:
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    stats = compute_latency_stats(latencies)
    assert stats.min_ms == 10.0
    assert stats.max_ms == 100.0
    assert stats.p50_ms == 50.0
    assert stats.p90_ms == 90.0
    assert stats.p95_ms == 100.0
    assert stats.mean_ms == 55.0

    empty_stats = compute_latency_stats([])
    assert empty_stats.p50_ms == 0.0


@pytest.mark.asyncio
async def test_execute_concurrent_requests() -> None:
    retriever = ConstantTimeRetriever()
    questions = [
        EvalQuestion(id="q1", question="What is Python?", relevant_documents=["p1"]),
        EvalQuestion(id="q2", question="What is Docker?", relevant_documents=["d1"]),
    ]

    latencies, ok, fail, dur = await execute_concurrent_requests(
        retriever,
        tenant_id=uuid4(),
        questions=questions,
        total_requests=10,
        concurrency=3,
        repeat_ratio=0.5,
    )
    assert ok == 10
    assert fail == 0
    assert len(latencies) == 10
    assert dur > 0.0


@pytest.mark.asyncio
async def test_execute_concurrent_requests_with_failures() -> None:
    failing_retriever = ConstantTimeRetriever(should_fail=True)
    questions = [EvalQuestion(id="q1", question="Question", relevant_documents=["d1"])]

    latencies, ok, fail, dur = await execute_concurrent_requests(
        failing_retriever,
        tenant_id=uuid4(),
        questions=questions,
        total_requests=5,
        concurrency=2,
    )
    assert ok == 0
    assert fail == 5
    assert len(latencies) == 0
    assert dur > 0.0


def test_render_load_test_markdown() -> None:
    report = LoadTestReport(
        generated_at="2026-11-26T12:00:00Z",
        dataset_name="test-corpus",
        document_count=5,
        chunk_count=20,
        question_count=10,
        runs=[
            LoadTestRunResult(
                name="Scenario 1",
                concurrency=5,
                total_requests=50,
                successful_requests=50,
                failed_requests=0,
                error_rate=0.0,
                duration_seconds=1.23,
                qps=40.7,
                latency=LatencyStats(
                    min_ms=5.0,
                    p50_ms=10.0,
                    p90_ms=15.0,
                    p95_ms=18.0,
                    p99_ms=20.0,
                    max_ms=22.0,
                    mean_ms=11.5,
                ),
            )
        ],
    )
    markdown = render_load_test_markdown(report)
    assert "# Performance and Concurrency Benchmark" in markdown
    assert "Scenario 1" in markdown
    assert "40.7" in markdown
