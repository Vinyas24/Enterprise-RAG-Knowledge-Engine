"""Unit tests for retrieval resilience and outage fallback behavior."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from rag_engine.core.enums import AccessLevel
from rag_engine.retrieval.base import RetrievedChunk, Retriever
from rag_engine.retrieval.hybrid import HybridRetriever


class FailingRetriever(Retriever):
    def __init__(self, error_message: str = "Provider outage") -> None:
        self.error_message = error_message

    async def retrieve(
        self,
        *,
        tenant_id: UUID,
        query: str,
        top_k: int = 5,
        clearance: AccessLevel = AccessLevel.INTERNAL,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        del tenant_id, query, top_k, clearance, min_score
        raise RuntimeError(self.error_message)


class SuccessfulRetriever(Retriever):
    def __init__(self, name: str) -> None:
        self.name = name

    async def retrieve(
        self,
        *,
        tenant_id: UUID,
        query: str,
        top_k: int = 5,
        clearance: AccessLevel = AccessLevel.INTERNAL,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        del query, top_k, clearance, min_score
        return [
            RetrievedChunk(
                chunk_id=uuid4(),
                document_id=uuid4(),
                document_title=f"{self.name} Doc",
                chunk_index=0,
                content=f"Content from {self.name}",
                score=0.95,
                access_level=AccessLevel.INTERNAL,
            )
        ]


@pytest.mark.asyncio
async def test_hybrid_dense_outage_fallback() -> None:
    dense = FailingRetriever("Pinecone connection timed out")
    keyword = SuccessfulRetriever("Keyword")

    hybrid = HybridRetriever(dense=dense, keyword=keyword, fallback_enabled=True)
    results = await hybrid.retrieve(tenant_id=uuid4(), query="test query")

    # Seamless fallback: returned keyword results without failing the query
    assert len(results) == 1
    assert results[0].document_title == "Keyword Doc"


@pytest.mark.asyncio
async def test_hybrid_keyword_outage_fallback() -> None:
    dense = SuccessfulRetriever("Dense")
    keyword = FailingRetriever("PostgreSQL connection pool exhausted")

    hybrid = HybridRetriever(dense=dense, keyword=keyword, fallback_enabled=True)
    results = await hybrid.retrieve(tenant_id=uuid4(), query="test query")

    # Seamless fallback: returned dense results without failing the query
    assert len(results) == 1
    assert results[0].document_title == "Dense Doc"


@pytest.mark.asyncio
async def test_hybrid_both_fail_propagates_error() -> None:
    dense = FailingRetriever("Dense provider down")
    keyword = FailingRetriever("Keyword database down")

    hybrid = HybridRetriever(dense=dense, keyword=keyword, fallback_enabled=True)
    with pytest.raises(RuntimeError):
        await hybrid.retrieve(tenant_id=uuid4(), query="test query")


@pytest.mark.asyncio
async def test_hybrid_fallback_disabled_raises() -> None:
    dense = FailingRetriever("Dense failed")
    keyword = SuccessfulRetriever("Keyword")

    hybrid = HybridRetriever(dense=dense, keyword=keyword, fallback_enabled=False)
    with pytest.raises(RuntimeError, match="Dense failed"):
        await hybrid.retrieve(tenant_id=uuid4(), query="test query")
