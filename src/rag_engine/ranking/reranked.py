"""Retrieval wrapper that re-ranks candidates and trims to the final top-n.

The wrapper owns the policy the provider must not:

* **Candidate pool vs final size.** The inner retriever is asked for
  ``max(rerank_candidates, top_k)`` results; the re-ranker trims them to
  ``top_k``. Over-fetching is what gives a second pass something to promote.
* **Deadline.** A re-ranker that exceeds ``timeout_seconds`` is abandoned.
* **Fallback.** On timeout or a provider error the caller gets the un-reranked
  order, never an error page. Falling back silently is acceptable because the
  candidate order is already a valid answer; the failure is logged and, in
  evaluation, recorded per question.

Provenance is taken from the original candidates, not from what the provider
returns. A re-ranker chooses an order and a score; it does not get to rewrite
chunk content, ids or access levels.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from uuid import UUID

from rag_engine.core.enums import AccessLevel
from rag_engine.observability import get_metrics_registry, trace_stage
from rag_engine.retrieval.base import RetrievedChunk, Retriever

from .base import RerankAttempt, Reranker
from .errors import RerankError

logger = logging.getLogger(__name__)


class RerankedRetriever(Retriever):
    """Wrap a retriever with a re-ranking second pass."""

    def __init__(
        self,
        *,
        base: Retriever,
        reranker: Reranker,
        candidate_pool: int = 20,
        timeout_seconds: float = 5.0,
        observer: Callable[[RerankAttempt], None] | None = None,
    ) -> None:
        if candidate_pool < 1:
            raise ValueError("candidate_pool must be at least 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base = base
        self._reranker = reranker
        self._candidate_pool = candidate_pool
        self._timeout_seconds = timeout_seconds
        self._observer = observer

    async def retrieve(
        self,
        *,
        tenant_id: UUID,
        query: str,
        top_k: int = 5,
        clearance: AccessLevel = AccessLevel.INTERNAL,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        if top_k < 1:
            return []
        if not query.strip():
            return []

        pool_size = max(self._candidate_pool, top_k)
        candidates = await self._base.retrieve(
            tenant_id=tenant_id,
            query=query,
            top_k=pool_size,
            clearance=clearance,
            min_score=min_score,
        )
        if not candidates:
            return []

        started = time.perf_counter()
        try:
            with trace_stage("rerank"):
                async with asyncio.timeout(self._timeout_seconds):
                    ranked = await self._reranker.rerank(
                        query=query,
                        candidates=candidates,
                        top_n=top_k,
                    )
            selected = _keep_known_chunks(ranked, candidates, limit=top_k)
            if not selected:
                raise RerankError(f"{self._reranker.name} returned no usable chunks")
        except (RerankError, TimeoutError) as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            get_metrics_registry().record_timing(
                "rag_pipeline_stage_duration_seconds",
                elapsed_ms / 1000.0,
                labels={"stage": "rerank"},
            )
            reason = str(exc) or type(exc).__name__
            logger.warning(
                "rerank.fallback",
                extra={
                    "reranker": self._reranker.name,
                    "reason": reason,
                    "candidate_count": len(candidates),
                    "latency_ms": round(elapsed_ms, 2),
                },
            )
            self._notify(RerankAttempt(latency_ms=elapsed_ms, used_fallback=True, reason=reason))
            return candidates[:top_k]

        elapsed_sec = time.perf_counter() - started
        get_metrics_registry().record_timing(
            "rag_pipeline_stage_duration_seconds",
            elapsed_sec,
            labels={"stage": "rerank"},
        )
        self._notify(
            RerankAttempt(
                latency_ms=elapsed_sec * 1000.0,
                used_fallback=False,
            )
        )
        return selected

    def _notify(self, attempt: RerankAttempt) -> None:
        if self._observer is not None:
            self._observer(attempt)


def _keep_known_chunks(
    ranked: Sequence[RetrievedChunk],
    candidates: Sequence[RetrievedChunk],
    *,
    limit: int,
) -> list[RetrievedChunk]:
    """Filter a provider's ranking down to unique, retrievable chunks.

    Duplicates and ids that were never in the pool are dropped rather than
    trusted. A provider that returns nothing usable triggers the fallback in
    the caller, because an empty prompt is worse than an un-reranked one.
    """
    original = {chunk.chunk_id: chunk for chunk in candidates}
    seen: set[UUID] = set()
    selected: list[RetrievedChunk] = []
    for item in ranked:
        chunk = original.get(item.chunk_id)
        if chunk is None or item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        selected.append(_with_score(chunk, item.score))
        if len(selected) >= limit:
            break
    return selected


def _with_score(chunk: RetrievedChunk, score: float) -> RetrievedChunk:
    return replace(chunk, score=score)
