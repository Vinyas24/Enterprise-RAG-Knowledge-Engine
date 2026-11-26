"""Re-ranker providers: ordering, failure classification and selection."""

from __future__ import annotations

import json
import sys
from typing import Any
from uuid import UUID

import httpx
import pytest

from rag_engine.core.config import RerankProvider, Settings
from rag_engine.core.enums import AccessLevel
from rag_engine.ranking import (
    CohereReranker,
    CrossEncoderReranker,
    LexicalReranker,
    RerankResponseError,
    RerankTransientError,
    RerankUnavailableError,
    build_reranker,
    resolve_rerank_model,
)
from rag_engine.retrieval.base import RetrievedChunk


def _chunk(index: int, content: str, *, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=index + 1),
        document_id=UUID(int=100 + index),
        document_title="Handbook",
        chunk_index=index,
        content=content,
        score=score,
        access_level=AccessLevel.INTERNAL,
    )


def _settings(**updates: Any) -> Settings:
    return Settings(_env_file=None, **updates)


async def test_lexical_prefers_the_chunk_matching_more_query_terms() -> None:
    pool = [
        _chunk(0, "unrelated receipts mileage"),
        _chunk(1, "deployment rollback runbook"),
        _chunk(2, "deployment only"),
    ]

    ranked = await LexicalReranker().rerank(query="deployment rollback", candidates=pool, top_n=2)

    assert [chunk.chunk_index for chunk in ranked] == [1, 2]
    assert ranked[0].score > ranked[1].score > 0.0


async def test_lexical_trims_to_top_n_and_replaces_scores() -> None:
    pool = [_chunk(index, "deployment runbook", score=0.5) for index in range(5)]

    ranked = await LexicalReranker().rerank(query="deployment runbook", candidates=pool, top_n=2)

    assert len(ranked) == 2
    assert all(chunk.score != 0.5 for chunk in ranked)


async def test_lexical_keeps_the_merged_order_when_nothing_matches() -> None:
    pool = [_chunk(0, "alpha"), _chunk(1, "beta"), _chunk(2, "gamma")]

    ranked = await LexicalReranker().rerank(query="nothing-here", candidates=pool, top_n=3)

    assert [chunk.chunk_index for chunk in ranked] == [0, 1, 2]
    assert all(chunk.score == 0.0 for chunk in ranked)


async def test_lexical_handles_a_blank_query_and_empty_pool() -> None:
    pool = [_chunk(0, "alpha"), _chunk(1, "beta")]

    assert await LexicalReranker().rerank(query="   ", candidates=pool, top_n=1) == [
        _chunk(0, "alpha")
    ]
    assert await LexicalReranker().rerank(query="alpha", candidates=[], top_n=3) == []
    assert await LexicalReranker().rerank(query="alpha", candidates=pool, top_n=0) == []


async def test_cross_encoder_sorts_by_score_and_trims() -> None:
    pool = [_chunk(0, "a"), _chunk(1, "b"), _chunk(2, "c")]
    seen: list[list[tuple[str, str]]] = []

    def scorer(pairs: Any) -> list[float]:
        seen.append(list(pairs))
        return [0.2, 0.9, 0.5]

    reranker = CrossEncoderReranker(scorer=scorer)

    ranked = await reranker.rerank(query="question", candidates=pool, top_n=2)

    assert seen == [[("question", "a"), ("question", "b"), ("question", "c")]]
    assert [chunk.chunk_index for chunk in ranked] == [1, 2]
    assert [chunk.score for chunk in ranked] == [0.9, 0.5]


async def test_cross_encoder_rejects_a_wrong_number_of_scores() -> None:
    reranker = CrossEncoderReranker(scorer=lambda _pairs: [0.5])

    with pytest.raises(RerankResponseError, match="1 scores for 2 candidates"):
        await reranker.rerank(query="q", candidates=[_chunk(0, "a"), _chunk(1, "b")], top_n=2)


async def test_cross_encoder_warmup_needs_no_model_when_a_scorer_is_injected() -> None:
    reranker = CrossEncoderReranker(scorer=lambda pairs: [0.0 for _ in pairs])

    await reranker.warmup()
    await reranker.close()


async def test_cross_encoder_reports_the_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``sys.modules[name] = None`` makes the next import raise ImportError.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    reranker = CrossEncoderReranker()

    with pytest.raises(RerankUnavailableError, match=r"rerank.*extra"):
        await reranker.warmup()


def test_cross_encoder_rejects_an_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        CrossEncoderReranker(batch_size=0)


async def test_cohere_maps_indexes_back_to_candidates() -> None:
    pool = [_chunk(0, "first"), _chunk(1, "second"), _chunk(2, "third")]
    reranker = CohereReranker(
        api_key="test-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 2, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.22},
                    ]
                },
            )
        ),
    )

    ranked = await reranker.rerank(query="question", candidates=pool, top_n=2)

    # Provider order is not trusted; scores decide, and metadata is preserved.
    assert [chunk.chunk_index for chunk in ranked] == [2, 0]
    assert [chunk.score for chunk in ranked] == [0.91, 0.22]
    assert ranked[0].content == "third"
    await reranker.close()


async def test_cohere_sends_the_documented_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    reranker = CohereReranker(
        api_key="secret",
        model="rerank-v3.5",
        base_url="https://api.example.test/v2",
        transport=httpx.MockTransport(handler),
    )

    await reranker.rerank(query="q", candidates=[_chunk(0, "doc")], top_n=1)

    assert captured["url"] == "https://api.example.test/v2/rerank"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"] == {
        "model": "rerank-v3.5",
        "query": "q",
        "documents": ["doc"],
        # Never ask for more results than documents sent.
        "top_n": 1,
    }
    await reranker.close()


async def test_cohere_classifies_transient_failures() -> None:
    for status in (429, 503):
        reranker = CohereReranker(
            api_key="key",
            transport=httpx.MockTransport(lambda _r, code=status: httpx.Response(code)),
        )
        with pytest.raises(RerankTransientError):
            await reranker.rerank(query="q", candidates=[_chunk(0, "doc")], top_n=1)
        await reranker.close()


async def test_cohere_classifies_timeouts() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow provider")

    reranker = CohereReranker(api_key="key", transport=httpx.MockTransport(handler))

    with pytest.raises(RerankTransientError, match="timed out"):
        await reranker.rerank(query="q", candidates=[_chunk(0, "doc")], top_n=1)
    await reranker.close()


async def test_cohere_classifies_rejections_and_bad_payloads() -> None:
    reranker = CohereReranker(
        api_key="key",
        transport=httpx.MockTransport(lambda _request: httpx.Response(400, text="invalid model")),
    )
    with pytest.raises(RerankResponseError, match="invalid model"):
        await reranker.rerank(query="q", candidates=[_chunk(0, "doc")], top_n=1)
    await reranker.close()

    payloads = [
        {"results": []},
        {"results": [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.4}]},
        {"results": [{"index": 5, "relevance_score": 0.5}]},
        {"results": [{"index": 0, "relevance_score": "high"}]},
        {"unexpected": True},
    ]
    for payload in payloads:
        reranker = CohereReranker(
            api_key="key",
            transport=httpx.MockTransport(lambda _r, body=payload: httpx.Response(200, json=body)),
        )
        with pytest.raises(RerankResponseError):
            await reranker.rerank(query="q", candidates=[_chunk(0, "doc")], top_n=1)
        await reranker.close()


def test_factory_returns_none_when_disabled() -> None:
    assert build_reranker(_settings()) is None
    assert resolve_rerank_model(_settings()) is None


def test_factory_builds_each_enabled_provider() -> None:
    lexical = build_reranker(_settings(rerank_provider="lexical"))
    cross_encoder = build_reranker(_settings(rerank_provider="cross-encoder"))
    cohere = build_reranker(_settings(rerank_provider="cohere", rerank_api_key="key"))

    assert isinstance(lexical, LexicalReranker)
    assert isinstance(cross_encoder, CrossEncoderReranker)
    assert isinstance(cohere, CohereReranker)


def test_factory_uses_the_provider_default_model() -> None:
    assert (
        resolve_rerank_model(_settings(rerank_provider="cross-encoder"))
        == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    assert (
        resolve_rerank_model(_settings(rerank_provider="cohere", rerank_api_key="key"))
        == "rerank-v3.5"
    )
    assert (
        resolve_rerank_model(_settings(rerank_provider="lexical", rerank_model="ignored")) is None
    )


def test_factory_rejects_cohere_without_a_key() -> None:
    with pytest.raises(ValueError, match="rerank_api_key is required"):
        _settings(rerank_provider="cohere")


def test_rerank_provider_values_are_stable() -> None:
    assert {provider.value for provider in RerankProvider} == {
        "off",
        "cohere",
        "cross-encoder",
        "lexical",
    }
