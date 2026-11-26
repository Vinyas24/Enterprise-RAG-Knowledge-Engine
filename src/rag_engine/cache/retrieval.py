"""Retrieval cache: transparently caches query retrieval results."""

from __future__ import annotations

import logging
from dataclasses import replace
from uuid import UUID

from rag_engine.core.enums import AccessLevel
from rag_engine.retrieval.base import RetrievedChunk, Retriever

from .base import CacheStats, LruCache

logger = logging.getLogger(__name__)

RetrievalCacheKey = tuple[UUID, str, int, str, float]


class CachedRetriever(Retriever):
    """Retriever decorator that caches query results in an in-memory LRU cache."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        max_size: int = 2000,
        ttl_seconds: float = 300.0,
        enabled: bool = True,
    ) -> None:
        self._retriever = retriever
        self._enabled = enabled
        self._cache: LruCache[RetrievalCacheKey, list[RetrievedChunk]] = LruCache(
            max_size=max_size,
            default_ttl_seconds=ttl_seconds,
        )

    def _normalize_query(self, query: str) -> str:
        return " ".join(query.strip().lower().split())

    async def retrieve(
        self,
        *,
        tenant_id: UUID,
        query: str,
        top_k: int = 5,
        clearance: AccessLevel = AccessLevel.INTERNAL,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        clean_query = query.strip()
        if not clean_query:
            return []

        if not self._enabled:
            return await self._retriever.retrieve(
                tenant_id=tenant_id,
                query=query,
                top_k=top_k,
                clearance=clearance,
                min_score=min_score,
            )

        key: RetrievalCacheKey = (
            tenant_id,
            self._normalize_query(clean_query),
            top_k,
            clearance.value,
            round(min_score, 4),
        )

        cached_chunks = self._cache.get(key)
        if cached_chunks is not None:
            # Return fresh copies so mutations by the caller do not corrupt cache
            return [replace(c) for c in cached_chunks]

        results = await self._retriever.retrieve(
            tenant_id=tenant_id,
            query=query,
            top_k=top_k,
            clearance=clearance,
            min_score=min_score,
        )

        self._cache.set(key, results)
        return [replace(c) for c in results]

    def invalidate_tenant(self, tenant_id: UUID) -> int:
        """Evict all cached query results belonging to a specific tenant."""
        return self._cache.invalidate_where(lambda k: k[0] == tenant_id)

    def stats(self) -> CacheStats:
        """Return retrieval cache hit/miss statistics."""
        return self._cache.stats()

    def clear(self) -> None:
        """Clear all cached query results."""
        self._cache.clear()
