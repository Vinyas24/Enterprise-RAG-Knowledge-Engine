"""Audit logging for queries, retrieval provenance, answers, and guardrails."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from rag_engine.core.db import DatabaseProtocol
from rag_engine.core.enums import AccessLevel
from rag_engine.core.models import AuditLog

logger = logging.getLogger(__name__)


def hash_query(query: str) -> str:
    """Compute deterministic SHA-256 hash of raw query for correlation."""
    return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """A single audit entry capturing query lifecycle and guardrail actions."""

    tenant_id: UUID
    question: str
    raw_question_hash: str
    clearance: AccessLevel
    answer: str
    guardrail_action: str
    id: UUID = field(default_factory=uuid4)
    request_id: str | None = None
    user_id: UUID | None = None
    retrieved_chunk_ids: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    guardrail_details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class AuditLogger(Protocol):
    """Protocol for recording audit entries."""

    async def record(self, entry: AuditRecord) -> None:
        """Persist or emit an audit entry."""
        ...


class DatabaseAuditLogger:
    """Audit logger backed by PostgreSQL ``audit_logs`` table."""

    def __init__(self, database: DatabaseProtocol) -> None:
        self._database = database

    async def record(self, entry: AuditRecord) -> None:
        try:
            async with self._database.session() as session:
                log_entry = AuditLog(
                    id=entry.id,
                    tenant_id=entry.tenant_id,
                    request_id=entry.request_id,
                    user_id=entry.user_id,
                    question=entry.question,
                    raw_question_hash=entry.raw_question_hash,
                    clearance=entry.clearance,
                    retrieved_chunk_ids=entry.retrieved_chunk_ids,
                    answer=entry.answer,
                    citations=entry.citations,
                    guardrail_action=entry.guardrail_action,
                    guardrail_details=entry.guardrail_details,
                    latency_ms=entry.latency_ms,
                    created_at=entry.created_at,
                )
                session.add(log_entry)
                await session.commit()
        except Exception:
            logger.exception("audit.record_failed", extra={"audit_id": str(entry.id)})


class InMemoryAuditLogger:
    """In-memory audit recorder for testing and offline evaluation."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    async def record(self, entry: AuditRecord) -> None:
        self.records.append(entry)

    def clear(self) -> None:
        self.records.clear()
