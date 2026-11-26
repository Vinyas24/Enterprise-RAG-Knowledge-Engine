"""Unit tests for audit logging."""

from __future__ import annotations

from uuid import uuid4

import pytest

from rag_engine.core.enums import AccessLevel
from rag_engine.guardrails.audit import AuditRecord, InMemoryAuditLogger, hash_query


def test_hash_query() -> None:
    h1 = hash_query("What is semantic search?")
    h2 = hash_query("What is semantic search?  ")
    h3 = hash_query("Different question")

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64


@pytest.mark.asyncio
async def test_in_memory_audit_logger() -> None:
    logger = InMemoryAuditLogger()
    record = AuditRecord(
        tenant_id=uuid4(),
        question="What is the policy for [EMAIL_ADDRESS]?",
        raw_question_hash=hash_query("What is the policy for test@example.com?"),
        clearance=AccessLevel.INTERNAL,
        answer="Policy details [1].",
        guardrail_action="allowed",
        retrieved_chunk_ids=[{"chunk_id": str(uuid4()), "score": 0.8}],
        citations=[{"snippet": "Snippet"}],
        latency_ms=12.5,
    )

    await logger.record(record)

    assert len(logger.records) == 1
    logged = logger.records[0]
    assert logged.question == "What is the policy for [EMAIL_ADDRESS]?"
    assert logged.guardrail_action == "allowed"
    assert logged.latency_ms == 12.5

    logger.clear()
    assert len(logger.records) == 0
