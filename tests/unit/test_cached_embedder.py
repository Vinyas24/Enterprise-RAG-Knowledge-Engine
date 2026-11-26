"""Unit tests for the CachedEmbedder decorator."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_engine.cache.embedding import CachedEmbedder
from rag_engine.embedding.base import Embedder


class FakeEmbedder(Embedder):
    def __init__(self, dimensions: int = 4) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []
        self.closed = False

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t))] * self.dimensions for t in texts]

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_cached_embedder_caching_and_order() -> None:
    inner = FakeEmbedder(dimensions=4)
    cached = CachedEmbedder(inner, max_size=100)

    # 1. Initial embed call
    v1 = await cached.embed(["apple", "banana"])
    assert len(v1) == 2
    assert v1[0] == [5.0, 5.0, 5.0, 5.0]
    assert v1[1] == [6.0, 6.0, 6.0, 6.0]
    assert len(inner.calls) == 1
    assert inner.calls[0] == ["apple", "banana"]

    # 2. Second call with overlap and new text in different order
    v2 = await cached.embed(["banana", "cherry", "apple"])
    assert len(v2) == 3
    assert v2[0] == [6.0, 6.0, 6.0, 6.0]  # banana (cached)
    assert v2[1] == [6.0, 6.0, 6.0, 6.0]  # cherry (computed)
    assert v2[2] == [5.0, 5.0, 5.0, 5.0]  # apple (cached)

    # Underlying embedder was only called for "cherry"
    assert len(inner.calls) == 2
    assert inner.calls[1] == ["cherry"]

    stats = cached.stats()
    assert stats.hits == 2
    assert stats.misses == 3  # apple, banana, cherry
    assert stats.hit_ratio == 2 / 5


@pytest.mark.asyncio
async def test_cached_embedder_empty_input() -> None:
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner)
    assert await cached.embed([]) == []
    assert len(inner.calls) == 0


@pytest.mark.asyncio
async def test_cached_embedder_disabled() -> None:
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner, enabled=False)

    await cached.embed(["hello"])
    await cached.embed(["hello"])
    assert len(inner.calls) == 2


@pytest.mark.asyncio
async def test_cached_embedder_clear_and_close() -> None:
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner)

    await cached.embed(["test"])
    assert cached.stats().size == 1
    cached.clear()
    assert cached.stats().size == 0

    await cached.close()
    assert inner.closed is True
