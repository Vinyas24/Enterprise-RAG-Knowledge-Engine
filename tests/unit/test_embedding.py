"""Unit tests for embedding providers, batching, and embedder factory."""

from __future__ import annotations

import json
from collections.abc import Sequence

import httpx
import pytest
from pydantic import SecretStr

from rag_engine.core.config import Settings
from rag_engine.embedding.base import EmbeddingProvider
from rag_engine.embedding.batching import BatchedEmbedder
from rag_engine.embedding.deterministic import DeterministicEmbeddingProvider
from rag_engine.embedding.errors import (
    EmbeddingRateLimitError,
    EmbeddingResponseError,
    EmbeddingTransientError,
)
from rag_engine.embedding.factory import build_embedder
from rag_engine.embedding.openai import OpenAIEmbeddingProvider, _retry_after_seconds

# ---------------------------------------------------------------------------
# DeterministicEmbeddingProvider
# ---------------------------------------------------------------------------


def test_deterministic_dimensions_validation() -> None:
    with pytest.raises(ValueError, match="at least 8 dimensions"):
        DeterministicEmbeddingProvider(dimensions=7)


async def test_deterministic_embed_basic() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=16)
    assert provider.name == "deterministic"
    assert provider.dimensions == 16

    texts = ["hello world", "test embedding", "hello world"]
    vectors = await provider.embed(texts)

    assert len(vectors) == 3
    assert all(len(v) == 16 for v in vectors)
    # Determinism: identical texts produce identical vectors
    assert vectors[0] == vectors[2]
    # Unit norm check
    for v in vectors:
        norm = sum(x * x for x in v) ** 0.5
        assert pytest.approx(norm, abs=1e-5) == 1.0

    await provider.close()


async def test_deterministic_empty_or_punctuation() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=16)
    vectors = await provider.embed(["", "!!!???"])
    assert len(vectors) == 2
    for v in vectors:
        norm = sum(x * x for x in v) ** 0.5
        assert pytest.approx(norm, abs=1e-5) == 1.0


async def test_deterministic_similarity() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=64)
    v1, v2, v3 = await provider.embed(
        [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown dog jumps over the lazy fox",
            "quantum physics and general relativity in cosmology",
        ]
    )

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=False))

    sim_related = cosine(v1, v2)
    sim_unrelated = cosine(v1, v3)
    assert sim_related > sim_unrelated


# ---------------------------------------------------------------------------
# OpenAIEmbeddingProvider
# ---------------------------------------------------------------------------


def test_openai_api_key_required() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        OpenAIEmbeddingProvider(api_key="", model="text-embedding-3-small", dimensions=4)


async def test_openai_embed_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.read())
        assert data["model"] == "text-embedding-3-small"
        assert data["input"] == ["first text", "second text"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4, 0.0, 0.0]},
                    {"index": 0, "embedding": [0.1, 0.2, 0.0, 0.0]},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        model="text-embedding-3-small",
        dimensions=4,
        transport=transport,
    )

    vectors = await provider.embed(["first text", "second text"])
    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.0, 0.0]
    assert vectors[1] == [0.3, 0.4, 0.0, 0.0]
    await provider.close()


async def test_openai_embed_rate_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "2.5"})

    transport = httpx.MockTransport(handler)
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        model="text-embedding-3-small",
        dimensions=4,
        transport=transport,
    )

    with pytest.raises(EmbeddingRateLimitError) as exc_info:
        await provider.embed(["text"])
    assert exc_info.value.retry_after == 2.5
    await provider.close()


async def test_openai_embed_transient_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    transport = httpx.MockTransport(handler)
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        model="text-embedding-3-small",
        dimensions=4,
        transport=transport,
    )

    with pytest.raises(EmbeddingTransientError, match="503"):
        await provider.embed(["text"])
    await provider.close()


async def test_openai_embed_bad_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Invalid input parameter")

    transport = httpx.MockTransport(handler)
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        model="text-embedding-3-small",
        dimensions=4,
        transport=transport,
    )

    with pytest.raises(EmbeddingResponseError, match="400"):
        await provider.embed(["text"])
    await provider.close()


async def test_openai_embed_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    transport = httpx.MockTransport(handler)
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        model="text-embedding-3-small",
        dimensions=4,
        transport=transport,
    )

    with pytest.raises(EmbeddingTransientError, match="timed out"):
        await provider.embed(["text"])
    await provider.close()


async def test_openai_embed_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    transport = httpx.MockTransport(handler)
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        model="text-embedding-3-small",
        dimensions=4,
        transport=transport,
    )

    with pytest.raises(EmbeddingResponseError, match="invalid JSON"):
        await provider.embed(["text"])
    await provider.close()


async def test_openai_embed_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    transport = httpx.MockTransport(handler)
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        model="text-embedding-3-small",
        dimensions=4,
        transport=transport,
    )

    with pytest.raises(EmbeddingResponseError, match="dimensions"):
        await provider.embed(["text"])
    await provider.close()


def test_retry_after_parsing() -> None:
    resp1 = httpx.Response(429, headers={"retry-after": "5.0"})
    assert _retry_after_seconds(resp1) == 5.0

    resp2 = httpx.Response(429, headers={"retry-after": "invalid"})
    assert _retry_after_seconds(resp2) is None

    resp3 = httpx.Response(429)
    assert _retry_after_seconds(resp3) is None


# ---------------------------------------------------------------------------
# BatchedEmbedder
# ---------------------------------------------------------------------------


class MockProvider(EmbeddingProvider):
    name = "mock"

    def __init__(self, dimensions: int = 4) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []
        self.errors: list[Exception] = []
        self.closed = False

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.errors:
            err = self.errors.pop(0)
            raise err
        return [[1.0] * self.dimensions for _ in texts]

    async def close(self) -> None:
        self.closed = True


async def test_batched_embedder_validation() -> None:
    provider = MockProvider()
    with pytest.raises(ValueError, match="batch_size"):
        BatchedEmbedder(provider, batch_size=0)
    with pytest.raises(ValueError, match="max_retries"):
        BatchedEmbedder(provider, max_retries=-1)


async def test_batched_embedder_batching() -> None:
    provider = MockProvider(dimensions=4)
    delays: list[float] = []

    async def fake_sleep(d: float) -> None:
        delays.append(d)

    embedder = BatchedEmbedder(provider, batch_size=2, sleep=fake_sleep)
    assert embedder.dimensions == 4

    # Empty inputs
    assert await embedder.embed([]) == []
    assert len(provider.calls) == 0

    # 5 inputs with batch_size 2
    inputs = ["a", "b", "c", "d", "e"]
    results = await embedder.embed(inputs)
    assert len(results) == 5
    assert len(provider.calls) == 3
    assert provider.calls[0] == ["a", "b"]
    assert provider.calls[1] == ["c", "d"]
    assert provider.calls[2] == ["e"]
    assert len(delays) == 0

    await embedder.close()
    assert provider.closed


async def test_batched_embedder_transient_retry() -> None:
    provider = MockProvider(dimensions=4)
    provider.errors = [EmbeddingTransientError("error 1"), EmbeddingTransientError("error 2")]
    delays: list[float] = []

    async def fake_sleep(d: float) -> None:
        delays.append(d)

    embedder = BatchedEmbedder(
        provider,
        batch_size=4,
        max_retries=3,
        retry_base_seconds=0.1,
        sleep=fake_sleep,
    )

    results = await embedder.embed(["hello"])
    assert len(results) == 1
    assert len(delays) == 2
    assert delays[0] == 0.1
    assert delays[1] == 0.2


async def test_batched_embedder_rate_limit_retry() -> None:
    provider = MockProvider(dimensions=4)
    provider.errors = [EmbeddingRateLimitError("rate limited", retry_after=1.5)]
    delays: list[float] = []

    async def fake_sleep(d: float) -> None:
        delays.append(d)

    embedder = BatchedEmbedder(provider, batch_size=4, max_retries=2, sleep=fake_sleep)
    results = await embedder.embed(["hello"])
    assert len(results) == 1
    assert delays == [1.5]


async def test_batched_embedder_permanent_error_no_retry() -> None:
    provider = MockProvider(dimensions=4)
    provider.errors = [EmbeddingResponseError("bad request")]
    delays: list[float] = []

    async def fake_sleep(d: float) -> None:
        delays.append(d)

    embedder = BatchedEmbedder(provider, batch_size=4, sleep=fake_sleep)
    with pytest.raises(EmbeddingResponseError):
        await embedder.embed(["hello"])
    assert len(delays) == 0


async def test_batched_embedder_max_retries_exceeded() -> None:
    provider = MockProvider(dimensions=4)
    provider.errors = [EmbeddingTransientError("err")] * 5
    delays: list[float] = []

    async def fake_sleep(d: float) -> None:
        delays.append(d)

    embedder = BatchedEmbedder(provider, batch_size=4, max_retries=2, sleep=fake_sleep)
    with pytest.raises(EmbeddingTransientError):
        await embedder.embed(["hello"])
    assert len(delays) == 2


# ---------------------------------------------------------------------------
# Factory: build_embedder
# ---------------------------------------------------------------------------


def test_build_embedder_deterministic() -> None:
    settings = Settings(embedding_provider="deterministic", embedding_dimensions=32)
    embedder = build_embedder(settings)
    assert isinstance(embedder, BatchedEmbedder)
    assert embedder.dimensions == 32


def test_build_embedder_openai() -> None:
    settings = Settings(
        embedding_provider="openai",
        embedding_api_key=SecretStr("sk-test"),
        embedding_dimensions=64,
        embedding_model="text-embedding-3-small",
    )
    embedder = build_embedder(settings)
    assert isinstance(embedder, BatchedEmbedder)
    assert embedder.dimensions == 64


def test_build_embedder_unknown() -> None:
    settings = Settings()
    object.__setattr__(settings, "embedding_provider", "invalid")
    with pytest.raises(ValueError, match="unknown embedding provider"):
        build_embedder(settings)
