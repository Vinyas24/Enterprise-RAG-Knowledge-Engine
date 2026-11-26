"""Re-ranking contracts.

A re-ranker scores (query, chunk) pairs jointly and returns the best ``top_n``.
The interface is deliberately narrow: providers do not fetch chunks, apply
tenant filters, or decide what to do on failure. Candidate selection and the
fallback policy live in :class:`rag_engine.ranking.reranked.RerankedRetriever`,
so every provider degrades identically when it times out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_engine.retrieval.base import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RerankAttempt:
    """What one re-rank call cost, and whether the fallback was used.

    The evaluation harness collects these so it can report re-rank latency
    separately from total retrieval latency instead of inferring the split
    from two wall-clock numbers.
    """

    latency_ms: float
    used_fallback: bool
    reason: str | None = None


@runtime_checkable
class Reranker(Protocol):
    """A joint scorer for retrieved candidates."""

    name: str

    async def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        """Return at most ``top_n`` candidates, best first.

        Only chunks from ``candidates`` may be returned, and the returned
        ``score`` is the re-ranker's relevance score. It is not comparable with
        cosine similarity, BM25 or a fused RRF score.
        """
        ...

    async def warmup(self) -> None:
        """Load models or establish clients before the first query.

        Implementations may raise :class:`RerankUnavailableError` so a bad
        configuration fails at startup instead of silently falling back on
        every request.
        """
        ...

    async def close(self) -> None:
        """Release resources. Must be idempotent."""
        ...
