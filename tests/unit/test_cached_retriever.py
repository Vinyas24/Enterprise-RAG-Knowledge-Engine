"""Unit tests for the CachedRetriever decorator."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from rag_engine.cache.retrieval import CachedRetriever
from rag_engine.core.enums import AccessLevel
from rag_engine.retrieval.base import RetrievedChunk, Retriever


class FakeRetriever(Retriever):
    def __init__(self) -> None:
        self.call_count = 0
        self.last_query: str | None = None

    async def retrieve(
        self,
        *,
        tenant_id: UUID,
        query: str,
        top_k: int = 5,
        clearance: AccessLevel = AccessLevel.INTERNAL,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        self.call_count += 1
        self.last_query = query
        return [
            RetrievedChunk(
                chunk_id=uuid4(),
                document_id=uuid4(),
                document_title="Test Doc",
                chunk_index=0,
                content=f"Content for {query}",
                score=0.9,
                access_level=clearance,
            )
        ]


@pytest.mark.asyncio
async def test_cached_retriever_caching_and_normalization() -> None:
    inner = FakeRetriever()
    cached = CachedRetriever(inner, max_size=10)
    tenant_id = uuid4()

    # 1. Initial retrieval -> miss
    r1 = await cached.retrieve(
        tenant_id=tenant_id,
        query="What is Semantic Search?",
        top_k=5,
        clearance=AccessLevel.INTERNAL,
    )
    assert len(r1) == 1
    assert inner.call_count == 1

    # 2. Same query with different casing and spacing -> hit!
    r2 = await cached.retrieve(
        tenant_id=tenant_id,
        query="  what   is semantic  SEARCH? ",
        top_k=5,
        clearance=AccessLevel.INTERNAL,
    )
    assert len(r2) == 1
    assert inner.call_count == 1  # No extra call to inner retriever!

    stats = cached.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.hit_ratio == 0.5


@pytest.mark.asyncio
async def test_cached_retriever_clearance_sensitivity() -> None:
    inner = FakeRetriever()
    cached = CachedRetriever(inner)
    tenant_id = uuid4()

    # Same query, different clearance -> different cache keys!
    await cached.retrieve(
        tenant_id=tenant_id,
        query="secret",
        clearance=AccessLevel.INTERNAL,
    )
    await cached.retrieve(
        tenant_id=tenant_id,
        query="secret",
        clearance=AccessLevel.RESTRICTED,
    )
    assert inner.call_count == 2


@pytest.mark.asyncio
async def test_cached_retriever_tenant_invalidation() -> None:
    inner = FakeRetriever()
    cached = CachedRetriever(inner)
    tenant_a = uuid4()
    tenant_b = uuid4()

    await cached.retrieve(tenant_id=tenant_a, query="query a")
    await cached.retrieve(tenant_id=tenant_b, query="query b")
    assert inner.call_count == 2
    assert cached.stats().size == 2

    # Invalidate Tenant A
    removed = cached.invalidate_tenant(tenant_a)
    assert removed == 1
    assert cached.stats().size == 1

    # Tenant B query is still cached
    await cached.retrieve(tenant_id=tenant_b, query="query b")
    assert inner.call_count == 2  # Hit!

    # Tenant A query must re-fetch
    await cached.retrieve(tenant_id=tenant_a, query="query a")
    assert inner.call_count == 3  # Miss


@pytest.mark.asyncio
async def test_cached_retriever_empty_query_and_disabled() -> None:
    inner = FakeRetriever()
    cached = CachedRetriever(inner, enabled=False)
    tenant_id = uuid4()

    assert await cached.retrieve(tenant_id=tenant_id, query="   ") == []
    assert inner.call_count == 0

    await cached.retrieve(tenant_id=tenant_id, query="q")
    await cached.retrieve(tenant_id=tenant_id, query="q")
    assert inner.call_count == 2
