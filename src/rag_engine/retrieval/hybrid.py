"""Hybrid retrieval: dense and keyword search in parallel, fused by rank.

Both paths are independent (one talks to the vector store, the other to
PostgreSQL), so they run concurrently. A weight of zero removes a path from the
fan-out entirely rather than paying for a call whose result would be discarded.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from uuid import UUID

from rag_engine.core.enums import AccessLevel

from .base import RetrievedChunk, Retriever
from .fusion import DEFAULT_RRF_K, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# Each path is asked for more results than the caller needs so fusion has room
# to promote a chunk that only one path found. Three times the requested depth
# is the same overfetch factor dense retrieval already used.
_MIN_CANDIDATE_DEPTH = 20
_CANDIDATE_MULTIPLIER = 3


class HybridRetriever(Retriever):
    """Fuse a dense retriever and a keyword retriever with weighted RRF."""

    def __init__(
        self,
        *,
        dense: Retriever,
        keyword: Retriever,
        dense_weight: float = 1.0,
        keyword_weight: float = 1.0,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if dense_weight == 0.0 and keyword_weight == 0.0:
            raise ValueError("at least one of dense_weight and keyword_weight must be non-zero")
        if dense_weight < 0.0 or keyword_weight < 0.0:
            raise ValueError("fusion weights cannot be negative")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        self._dense = dense
        self._keyword = keyword
        self._dense_weight = dense_weight
        self._keyword_weight = keyword_weight
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        *,
        tenant_id: UUID,
        query: str,
        top_k: int = 5,
        clearance: AccessLevel = AccessLevel.INTERNAL,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Run the enabled paths concurrently and fuse their rankings.

        ``min_score`` is forwarded to the dense path only: it is a similarity
        floor and has no meaning for BM25 scores.
        """
        if not query.strip():
            return []

        depth = max(top_k * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATE_DEPTH)
        weights: list[float] = []
        calls: list[Awaitable[list[RetrievedChunk]]] = []
        if self._dense_weight > 0.0:
            weights.append(self._dense_weight)
            calls.append(
                self._dense.retrieve(
                    tenant_id=tenant_id,
                    query=query,
                    top_k=depth,
                    clearance=clearance,
                    min_score=min_score,
                )
            )
        if self._keyword_weight > 0.0:
            weights.append(self._keyword_weight)
            calls.append(
                self._keyword.retrieve(
                    tenant_id=tenant_id,
                    query=query,
                    top_k=depth,
                    clearance=clearance,
                )
            )

        # ``gather`` propagates the first failure; a silent half-result would
        # be worse than a failed query, because the caller cannot tell. The
        # constructor guarantees at least one call, so ``calls`` is never empty.
        rankings = await asyncio.gather(*calls)
        return reciprocal_rank_fusion(
            rankings=rankings,
            weights=weights,
            limit=top_k,
            k=self._rrf_k,
        )
