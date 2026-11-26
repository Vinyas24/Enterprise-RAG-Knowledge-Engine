"""Integration tests verifying performance, caching, and latency targets."""

from __future__ import annotations

import pytest

from rag_engine.core.config import Settings
from rag_engine.core.db import PostgresDatabase
from rag_engine.embedding import Embedder
from rag_engine.evaluation.dataset import EvalDataset
from rag_engine.evaluation.load_test import run_performance_suite
from rag_engine.vectorstore import VectorStore

pytestmark = pytest.mark.integration


def _benchmark_dataset() -> EvalDataset:
    return EvalDataset.model_validate(
        {
            "name": "perf-eval",
            "version": 1,
            "documents": [
                {
                    "key": "doc-a",
                    "title": "Alpha Platform Architecture",
                    "filename": "arch.md",
                    "content": (
                        "# Architecture Overview\n\n"
                        "The Alpha platform microservices communicate asynchronously via gRPC "
                        "and process streaming telemetry with low latency bounded buffer queues."
                    ),
                },
                {
                    "key": "doc-b",
                    "title": "Beta Deployment Guidelines",
                    "filename": "deploy.md",
                    "content": (
                        "# Deployment Guidelines\n\n"
                        "Canary releases gradually roll out new versions to a 5 percent subset "
                        "of production traffic before expanding globally."
                    ),
                },
            ],
            "questions": [
                {
                    "id": "q1",
                    "question": "asynchronous microservices telemetry",
                    "relevant_documents": ["doc-a"],
                },
                {
                    "id": "q2",
                    "question": "canary releases production traffic subset",
                    "relevant_documents": ["doc-b"],
                },
                {
                    "id": "q3",
                    "question": "low latency bounded buffer queues",
                    "relevant_documents": ["doc-a"],
                },
            ],
        }
    )


@pytest.mark.asyncio
async def test_performance_targets_and_concurrency(
    test_settings: Settings,
    database: PostgresDatabase,
    vector_store: VectorStore,
    embedder: Embedder,
) -> None:
    dataset = _benchmark_dataset()

    report = await run_performance_suite(
        dataset,
        settings=test_settings,
        database=database,
        vector_store=vector_store,
        embedder=embedder,
    )

    assert len(report.runs) >= 4

    # 1. Verify zero errors across all load tiers
    for run in report.runs:
        assert run.failed_requests == 0
        assert run.error_rate == 0.0

    # 2. Verify Cache Acceleration Target:
    # Warm retrieval (cached) p50 latency must be < 5.0ms (typically < 1.0ms)
    warm_run = next(run for run in report.runs if "Warm Retrieval" in run.name)
    assert warm_run.latency.p50_ms < 5.0

    # 3. Verify Concurrency Scaling:
    # High concurrency run (concurrency=10) completed successfully with healthy QPS
    c10_run = next(run for run in report.runs if "Concurrency=10" in run.name)
    assert c10_run.successful_requests == 200
    assert c10_run.qps > 10.0
