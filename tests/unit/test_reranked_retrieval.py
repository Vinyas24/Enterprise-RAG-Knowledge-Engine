"""The re-ranking wrapper: pool sizing, trimming, deadline and fallback."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any
from uuid import UUID

import pytest

from rag_engine.core.enums import AccessLevel
from rag_engine.ranking import (
    RerankAttempt,
    RerankedRetriever,
    RerankError,
    RerankTransientError,
)
from rag_engine.retrieval.base import RetrievedChunk


def _chunk(index: int, *, content: str = "content", score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=index + 1),
        document_id=UUID(int=100 + index),
        document_title=f"Doc {index}",
        chunk_index=index,
        content=content,
        score=score,
        access_level=AccessLevel.INTERNAL,
    )


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def retrieve(self, **kwargs: Any) -> list[RetrievedChunk]:
        self.calls.append(kwargs)
        return self.chunks[: kwargs["top_k"]]


class StubReranker:
    name = "stub"

    def __init__(
        self,
        *,
        order: list[int] | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
        override: list[RetrievedChunk] | None = None,
    ) -> None:
        self._order = order
        self._error = error
        self._delay = delay
        self._override = override
        self.calls: list[dict[str, Any]] = []

    async def rerank(
        self, *, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "candidates": list(candidates), "top_n": top_n})
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        if self._override is not None:
            return self._override[:top_n]
        if self._order is not None:
            return [
                replace(candidates[index], score=10.0 - rank)
                for rank, index in enumerate(self._order)
            ][:top_n]
        return [replace(chunk, score=10.0 - index) for index, chunk in enumerate(candidates)][
            :top_n
        ]

    async def warmup(self) -> None:
        pass

    async def close(self) -> None:
        pass


def _retriever(
    base: StubRetriever,
    reranker: StubReranker,
    *,
    pool: int = 20,
    timeout: float = 5.0,
    attempts: list[RerankAttempt] | None = None,
) -> RerankedRetriever:
    return RerankedRetriever(
        base=base,
        reranker=reranker,
        candidate_pool=pool,
        timeout_seconds=timeout,
        observer=attempts.append if attempts is not None else None,
    )


async def test_asks_the_inner_retriever_for_the_candidate_pool() -> None:
    base = StubRetriever([_chunk(index) for index in range(10)])
    reranker = StubReranker()

    results = await _retriever(base, reranker, pool=6).retrieve(
        tenant_id=UUID(int=1), query="q", top_k=2
    )

    assert len(results) == 2
    assert base.calls[0]["top_k"] == 6
    assert len(reranker.calls[0]["candidates"]) == 6
    assert reranker.calls[0]["top_n"] == 2


async def test_pool_never_shrinks_below_the_final_size() -> None:
    base = StubRetriever([_chunk(index) for index in range(30)])
    reranker = StubReranker()

    await _retriever(base, reranker, pool=5).retrieve(tenant_id=UUID(int=1), query="q", top_k=12)

    assert base.calls[0]["top_k"] == 12
    assert reranker.calls[0]["top_n"] == 12


async def test_returns_the_reranked_order_and_scores() -> None:
    base = StubRetriever([_chunk(0), _chunk(1), _chunk(2)])
    reranker = StubReranker(order=[2, 0])

    results = await _retriever(base, reranker, pool=3).retrieve(
        tenant_id=UUID(int=1), query="q", top_k=2
    )

    assert [chunk.chunk_index for chunk in results] == [2, 0]
    assert [chunk.score for chunk in results] == [10.0, 9.0]


async def test_ignores_duplicate_and_unknown_chunks_from_a_provider() -> None:
    base = StubRetriever([_chunk(0), _chunk(1)])
    unknown = replace(_chunk(1), chunk_id=UUID(int=999))
    reranker = StubReranker(override=[_chunk(1), unknown, _chunk(1), _chunk(0)])

    # top_k matches the override length so the stub does not pre-trim the
    # duplicate before the wrapper's provenance filter sees it.
    results = await _retriever(base, reranker, pool=2).retrieve(
        tenant_id=UUID(int=1), query="q", top_k=4
    )

    assert [chunk.chunk_index for chunk in results] == [1, 0]


async def test_keeps_retrieved_metadata_not_the_provider_copy() -> None:
    original = _chunk(0, content="original")
    tampered = replace(original, content="rewritten", access_level=AccessLevel.PUBLIC)
    base = StubRetriever([original])
    reranker = StubReranker(override=[tampered])

    results = await _retriever(base, reranker, pool=1).retrieve(
        tenant_id=UUID(int=1), query="q", top_k=1
    )

    assert results[0] is not tampered
    assert results[0].content == "original"
    assert results[0].access_level is AccessLevel.INTERNAL
    assert results[0].score == tampered.score


async def test_falls_back_to_the_unreranked_order_on_an_error() -> None:
    base = StubRetriever([_chunk(0), _chunk(1), _chunk(2)])
    reranker = StubReranker(error=RerankTransientError("provider unavailable"))
    attempts: list[RerankAttempt] = []

    results = await _retriever(base, reranker, attempts=attempts).retrieve(
        tenant_id=UUID(int=1), query="q", top_k=2
    )

    assert [chunk.chunk_index for chunk in results] == [0, 1]
    assert len(attempts) == 1
    assert attempts[0].used_fallback is True
    assert attempts[0].reason == "provider unavailable"
    assert attempts[0].latency_ms >= 0.0


async def test_falls_back_when_the_deadline_is_exceeded() -> None:
    base = StubRetriever([_chunk(0), _chunk(1)])
    reranker = StubReranker(delay=0.5)
    attempts: list[RerankAttempt] = []

    started = asyncio.get_running_loop().time()
    results = await _retriever(base, reranker, timeout=0.01, attempts=attempts).retrieve(
        tenant_id=UUID(int=1), query="q", top_k=2
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert [chunk.chunk_index for chunk in results] == [0, 1]
    assert attempts[0].used_fallback is True
    assert attempts[0].reason == "TimeoutError"
    # The caller waits for the deadline, not for the slow provider.
    assert elapsed < 0.4


async def test_falls_back_when_the_provider_returns_nothing_usable() -> None:
    base = StubRetriever([_chunk(0), _chunk(1)])
    reranker = StubReranker(override=[])
    attempts: list[RerankAttempt] = []

    results = await _retriever(base, reranker, attempts=attempts).retrieve(
        tenant_id=UUID(int=1), query="q", top_k=2
    )

    assert [chunk.chunk_index for chunk in results] == [0, 1]
    assert attempts[0].used_fallback is True
    assert "no usable chunks" in (attempts[0].reason or "")


async def test_records_a_successful_attempt() -> None:
    base = StubRetriever([_chunk(0)])
    attempts: list[RerankAttempt] = []

    await _retriever(base, StubReranker(), attempts=attempts).retrieve(
        tenant_id=UUID(int=1), query="q", top_k=1
    )

    assert len(attempts) == 1
    assert attempts[0].used_fallback is False
    assert attempts[0].reason is None
    assert attempts[0].latency_ms >= 0.0


async def test_skips_the_reranker_when_nothing_was_retrieved() -> None:
    base = StubRetriever([])
    reranker = StubReranker()
    attempts: list[RerankAttempt] = []

    results = await _retriever(base, reranker, attempts=attempts).retrieve(
        tenant_id=UUID(int=1), query="q", top_k=3
    )

    assert results == []
    assert reranker.calls == []
    assert attempts == []


async def test_skips_both_stages_for_a_blank_query() -> None:
    base = StubRetriever([_chunk(0)])
    reranker = StubReranker()

    results = await _retriever(base, reranker).retrieve(tenant_id=UUID(int=1), query="   ", top_k=3)

    assert results == []
    assert base.calls == []
    assert reranker.calls == []


async def test_forwards_the_scope_and_score_floor_to_the_inner_retriever() -> None:
    base = StubRetriever([_chunk(0)])
    reranker = StubReranker()
    tenant_id = UUID(int=42)

    await _retriever(base, reranker).retrieve(
        tenant_id=tenant_id,
        query="q",
        top_k=1,
        clearance=AccessLevel.PUBLIC,
        min_score=0.25,
    )

    call = base.calls[0]
    assert call["tenant_id"] == tenant_id
    assert call["clearance"] is AccessLevel.PUBLIC
    assert call["min_score"] == 0.25
    assert reranker.calls[0]["query"] == "q"


def test_rejects_an_unusable_configuration() -> None:
    base = StubRetriever([])
    reranker = StubReranker()

    with pytest.raises(ValueError, match="candidate_pool"):
        RerankedRetriever(base=base, reranker=reranker, candidate_pool=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        RerankedRetriever(base=base, reranker=reranker, timeout_seconds=0)


async def test_propagates_unexpected_programming_errors() -> None:
    """Only expected provider failures are hidden by the fallback."""

    class Broken(StubReranker):
        async def rerank(self, **_kwargs: Any) -> list[RetrievedChunk]:
            raise TypeError("bug in provider")

    with pytest.raises(TypeError, match="bug in provider"):
        await _retriever(StubRetriever([_chunk(0)]), Broken()).retrieve(
            tenant_id=UUID(int=1), query="q", top_k=1
        )


async def test_wraps_errors_raised_by_the_reranker_as_fallbacks() -> None:
    base = StubRetriever([_chunk(0)])
    reranker = StubReranker(error=RerankError("boom"))

    results = await _retriever(base, reranker).retrieve(tenant_id=UUID(int=1), query="q", top_k=1)

    assert [chunk.chunk_index for chunk in results] == [0]
