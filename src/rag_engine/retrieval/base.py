"""Retrieval contracts and data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import UUID

from rag_engine.core.enums import AccessLevel


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A passage retrieved from the knowledge base with provenance."""

    chunk_id: UUID
    document_id: UUID
    document_title: str
    chunk_index: int
    content: str
    score: float
    access_level: AccessLevel
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None


@runtime_checkable
class Retriever(Protocol):
    """Retrieval backend interface."""

    async def retrieve(
        self,
        *,
        tenant_id: UUID,
        query: str,
        top_k: int = 5,
        clearance: AccessLevel = AccessLevel.INTERNAL,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Retrieve top passages matching the query respecting tenant and clearance."""
        ...
