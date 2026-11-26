"""Embedding cache: transparently caches text embeddings to eliminate provider latency."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from rag_engine.embedding.base import Embedder

from .base import CacheStats, LruCache

logger = logging.getLogger(__name__)


class CachedEmbedder(Embedder):
    """Embedder decorator that caches text-to-vector mappings in a bounded LRU cache."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        max_size: int = 10000,
        ttl_seconds: float = 3600.0,
        enabled: bool = True,
    ) -> None:
        self._embedder = embedder
        self._enabled = enabled
        self._cache: LruCache[str, list[float]] = LruCache(
            max_size=max_size,
            default_ttl_seconds=ttl_seconds,
        )
        self.dimensions = embedder.dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        if not self._enabled:
            return await self._embedder.embed(texts)

        results: list[list[float] | None] = [None] * len(texts)
        missing_indices_by_text: dict[str, list[int]] = {}

        for idx, raw_text in enumerate(texts):
            clean_text = raw_text.strip()
            cached_vector = self._cache.get(clean_text)
            if cached_vector is not None:
                results[idx] = cached_vector
            else:
                missing_indices_by_text.setdefault(clean_text, []).append(idx)

        if missing_indices_by_text:
            distinct_missing = list(missing_indices_by_text.keys())
            computed_vectors = await self._embedder.embed(distinct_missing)

            for clean_text, vector in zip(distinct_missing, computed_vectors, strict=True):
                self._cache.set(clean_text, vector)
                for idx in missing_indices_by_text[clean_text]:
                    results[idx] = vector

        # Every slot is guaranteed populated by this point
        return [vec for vec in results if vec is not None]

    async def close(self) -> None:
        await self._embedder.close()

    def stats(self) -> CacheStats:
        """Return embedding cache hit/miss statistics."""
        return self._cache.stats()

    def clear(self) -> None:
        """Clear cached embeddings."""
        self._cache.clear()
