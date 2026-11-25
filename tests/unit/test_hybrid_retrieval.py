"""Unit tests for reciprocal rank fusion and the hybrid retriever."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4

import pytest

from rag_engine.core.config import RetrievalMode, Settings
from rag_engine.core.enums import AccessLevel
from rag_engine.retrieval.base import RetrievedChunk
from rag_engine.retrieval.dense import DenseRetriever
from rag_engine.retrieval.factory import build_retriever
from rag_engine.retrieval.fusion import reciprocal_rank_fusion
from rag_engine.retrieval.hybrid import HybridRetriever


def _chunk(index: int, *, title: str = "Handbook", score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=index + 1),
        document_id=UUID(int=100 + index),
        document_title=title,
        chunk_index=index,
        content=f"content {index}",
        score=score,
        access_level=AccessLevel.INTERNAL,
    )


def test_rrf_sums_contributions_for_chunks_found_by_both_paths() -> None:
    dense = [_chunk(0), _chunk(1)]
    keyword = [_chunk(1), _chunk(2)]

    fused = reciprocal_rank_fusion(rankings=[dense, keyword], weights=[1.0, 1.0], limit=5, k=60)

    assert [item.chunk_index for item in fused] == [1, 0, 2]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)


def test_rrf_weights_change_the_order() -> None:
    dense = [_chunk(0)]
    keyword = [_chunk(1)]

    keyword_heavy = reciprocal_rank_fusion(
        rankings=[dense, keyword], weights=[0.1, 1.0], limit=2, k=60
    )
    dense_heavy = reciprocal_rank_fusion(
        rankings=[dense, keyword], weights=[1.0, 0.1], limit=2, k=60
    )

    assert [item.chunk_index for item in keyword_heavy] == [1, 0]
    assert [item.chunk_index for item in dense_heavy] == [0, 1]


def test_rrf_skips_zero_weight_lists() -> None:
    fused = reciprocal_rank_fusion(rankings=[[_chunk(0)], [_chunk(1)]], weights=[1.0, 0.0], limit=5)

    assert [item.chunk_index for item in fused] == [0]


def test_rrf_keeps_the_same_chunk_once() -> None:
    shared = _chunk(0)

    fused = reciprocal_rank_fusion(rankings=[[shared], [shared]], weights=[1.0, 1.0], limit=5)

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(2 / 61)


def test_rrf_is_deterministic_for_equal_scores() -> None:
    zulu = _chunk(0, title="Zulu")
    alpha = _chunk(1, title="Alpha")

    fused = reciprocal_rank_fusion(rankings=[[zulu], [alpha]], weights=[1.0, 1.0], limit=2)

    assert [item.document_title for item in fused] == ["Alpha", "Zulu"]


def test_rrf_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        reciprocal_rank_fusion(rankings=[[_chunk(0)]], weights=[], limit=1)
    with pytest.raises(ValueError, match="k must be"):
        reciprocal_rank_fusion(rankings=[], weights=[], limit=1, k=0)


def test_rrf_returns_nothing_for_a_non_positive_limit() -> None:
    assert reciprocal_rank_fusion(rankings=[[_chunk(0)]], weights=[1.0], limit=0) == []


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk], *, fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def retrieve(self, **kwargs: Any) -> list[RetrievedChunk]:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("retrieval path failed")
        return list(self.chunks)


class RendezvousRetriever:
    """Blocks until the other path has started, proving the calls overlap."""

    def __init__(self, chunk: RetrievedChunk, own: asyncio.Event, peer: asyncio.Event) -> None:
        self.chunk = chunk
        self._own = own
        self._peer = peer

    async def retrieve(self, **_kwargs: Any) -> list[RetrievedChunk]:
        self._own.set()
        await asyncio.wait_for(self._peer.wait(), timeout=1.0)
        return [self.chunk]


async def test_hybrid_runs_both_paths_concurrently() -> None:
    dense_started, keyword_started = asyncio.Event(), asyncio.Event()
    retriever = HybridRetriever(
        dense=RendezvousRetriever(_chunk(0), dense_started, keyword_started),
        keyword=RendezvousRetriever(_chunk(1), keyword_started, dense_started),
    )

    results = await retriever.retrieve(tenant_id=uuid4(), query="deployment", top_k=2)

    assert len(results) == 2


async def test_hybrid_merges_both_rankings() -> None:
    dense = StubRetriever([_chunk(0), _chunk(1)])
    keyword = StubRetriever([_chunk(2)])
    retriever = HybridRetriever(dense=dense, keyword=keyword, dense_weight=1.0, keyword_weight=1.0)

    results = await retriever.retrieve(tenant_id=uuid4(), query="deployment", top_k=3)

    assert {item.chunk_index for item in results} == {0, 1, 2}
    assert len(dense.calls) == 1 and len(keyword.calls) == 1


async def test_hybrid_skips_a_zero_weight_path() -> None:
    dense = StubRetriever([_chunk(0)])
    keyword = StubRetriever([_chunk(1)])
    retriever = HybridRetriever(dense=dense, keyword=keyword, dense_weight=1.0, keyword_weight=0.0)

    results = await retriever.retrieve(tenant_id=uuid4(), query="deployment", top_k=3)

    assert [item.chunk_index for item in results] == [0]
    assert keyword.calls == []


async def test_hybrid_can_run_keyword_only() -> None:
    dense = StubRetriever([_chunk(0)])
    keyword = StubRetriever([_chunk(1)])
    retriever = HybridRetriever(dense=dense, keyword=keyword, dense_weight=0.0, keyword_weight=1.0)

    results = await retriever.retrieve(tenant_id=uuid4(), query="deployment", top_k=3)

    assert [item.chunk_index for item in results] == [1]
    assert dense.calls == []


async def test_hybrid_asks_each_path_for_a_deeper_candidate_pool() -> None:
    dense = StubRetriever([])
    keyword = StubRetriever([])
    retriever = HybridRetriever(dense=dense, keyword=keyword)
    tenant_id = uuid4()

    await retriever.retrieve(tenant_id=tenant_id, query="q", top_k=2, clearance=AccessLevel.PUBLIC)

    # max(top_k * 3, 20) so fusion can promote single-path results.
    assert dense.calls[0]["top_k"] == 20
    assert keyword.calls[0]["top_k"] == 20
    # Both paths receive the same scope; clearance is not applied by fusion.
    assert dense.calls[0]["clearance"] is AccessLevel.PUBLIC
    assert keyword.calls[0]["clearance"] is AccessLevel.PUBLIC
    assert dense.calls[0]["tenant_id"] == tenant_id
    assert keyword.calls[0]["tenant_id"] == tenant_id


async def test_hybrid_forwards_min_score_only_to_the_dense_path() -> None:
    dense = StubRetriever([])
    keyword = StubRetriever([])
    retriever = HybridRetriever(dense=dense, keyword=keyword)

    await retriever.retrieve(tenant_id=uuid4(), query="q", min_score=0.42)

    assert dense.calls[0]["min_score"] == 0.42
    assert "min_score" not in keyword.calls[0]


async def test_hybrid_returns_nothing_for_a_blank_query() -> None:
    dense = StubRetriever([])
    keyword = StubRetriever([])
    retriever = HybridRetriever(dense=dense, keyword=keyword)

    assert await retriever.retrieve(tenant_id=uuid4(), query="   ") == []
    assert dense.calls == [] and keyword.calls == []


async def test_hybrid_propagates_a_path_failure() -> None:
    retriever = HybridRetriever(
        dense=StubRetriever([_chunk(0)]),
        keyword=StubRetriever([], fail=True),
    )

    with pytest.raises(RuntimeError, match="retrieval path failed"):
        await retriever.retrieve(tenant_id=uuid4(), query="deployment")


def test_hybrid_rejects_an_unusable_configuration() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever(
            dense=StubRetriever([]),
            keyword=StubRetriever([]),
            dense_weight=0.0,
            keyword_weight=0.0,
        )
    with pytest.raises(ValueError, match="negative"):
        HybridRetriever(dense=StubRetriever([]), keyword=StubRetriever([]), keyword_weight=-1.0)
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(dense=StubRetriever([]), keyword=StubRetriever([]), rrf_k=0)


def _build(mode: RetrievalMode) -> Any:
    return build_retriever(
        Settings(_env_file=None, retrieval_mode=mode),
        embedder=object(),
        vector_store=object(),
        database=object(),
    )


def test_factory_returns_dense_when_configured() -> None:
    assert isinstance(_build(RetrievalMode.DENSE), DenseRetriever)


def test_factory_returns_hybrid_by_default() -> None:
    retriever = _build(RetrievalMode.HYBRID)

    assert isinstance(retriever, HybridRetriever)


def test_factory_passes_the_configured_fusion_weights() -> None:
    settings = Settings(
        _env_file=None,
        retrieval_mode="hybrid",
        hybrid_dense_weight=2.0,
        hybrid_keyword_weight=0.5,
        hybrid_rrf_k=10,
        retrieval_keyword_candidates=42,
    )
    retriever = build_retriever(
        settings, embedder=object(), vector_store=object(), database=object()
    )

    assert isinstance(retriever, HybridRetriever)
    assert retriever._dense_weight == 2.0
    assert retriever._keyword_weight == 0.5
    assert retriever._rrf_k == 10
    assert retriever._keyword._candidate_limit == 42
