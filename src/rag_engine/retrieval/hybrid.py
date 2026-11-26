"""Hybrid retrieval: dense and keyword search in parallel, fused by rank.

Both paths are independent (one talks to the vector store, the other to
PostgreSQL), so they run concurrently. A weight of zero removes a path from the
fan-out entirely rather than paying for a call whose result would be discarded.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable
from uuid import UUID

from rag_engine.core.enums import AccessLevel
from rag_engine.observability import get_metrics_registry, trace_stage

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
        fallback_enabled: bool = False,
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
        self._fallback_enabled = fallback_enabled

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
        path_names: list[str] = []
        weights: list[float] = []
        calls: list[Awaitable[list[RetrievedChunk]]] = []

        async def _traced(name: str, coro: Awaitable[list[RetrievedChunk]]) -> list[RetrievedChunk]:
            t0 = time.perf_counter()
            with trace_stage(name):
                res = await coro
            dur = time.perf_counter() - t0
            get_metrics_registry().record_timing(
                "rag_pipeline_stage_duration_seconds", dur, labels={"stage": name}
            )
            return res

        if self._dense_weight > 0.0:
            path_names.append("dense")
            weights.append(self._dense_weight)
            calls.append(
                _traced(
                    "dense",
                    self._dense.retrieve(
                        tenant_id=tenant_id,
                        query=query,
                        top_k=depth,
                        clearance=clearance,
                        min_score=min_score,
                    ),
                )
            )
        if self._keyword_weight > 0.0:
            path_names.append("keyword")
            weights.append(self._keyword_weight)
            calls.append(
                _traced(
                    "keyword",
                    self._keyword.retrieve(
                        tenant_id=tenant_id,
                        query=query,
                        top_k=depth,
                        clearance=clearance,
                    ),
                )
            )

        def _fuse(rk: list[list[RetrievedChunk]], wt: list[float]) -> list[RetrievedChunk]:
            t0 = time.perf_counter()
            with trace_stage("fuse"):
                out = reciprocal_rank_fusion(
                    rankings=rk,
                    weights=wt,
                    limit=top_k,
                    k=self._rrf_k,
                )
            dur = time.perf_counter() - t0
            get_metrics_registry().record_timing(
                "rag_pipeline_stage_duration_seconds", dur, labels={"stage": "fuse"}
            )
            return out

        if not self._fallback_enabled:
            rankings = await asyncio.gather(*calls)
            return _fuse(rankings, weights)

        # Graceful outage fallback: if one path times out or fails,
        # log and continue with the surviving path
        results = await asyncio.gather(*calls, return_exceptions=True)
        valid_rankings: list[list[RetrievedChunk]] = []
        valid_weights: list[float] = []
        first_error: BaseException | None = None

        for name, weight, result in zip(path_names, weights, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "retrieval.hybrid_path_failed",
                    extra={"path": name, "error": str(result)},
                    exc_info=result,
                )
                if first_error is None:
                    first_error = result
            else:
                valid_rankings.append(result)
                valid_weights.append(weight)

        if not valid_rankings:
            # Both paths failed: re-raise the error so the caller is aware
            if first_error is not None:
                raise first_error
            return []

        if len(valid_rankings) == 1:
            # Graceful single-path fallback: return top_k directly from surviving path
            return valid_rankings[0][:top_k]

        return _fuse(valid_rankings, valid_weights)
