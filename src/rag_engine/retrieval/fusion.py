"""Reciprocal rank fusion of ranked retrieval lists.

Fusing ranks instead of scores is the point: a cosine similarity and a BM25
score live on unrelated scales, and normalising them would make the weights
depend on the result set. A rank is comparable across any two retrievers, so a
weight means the same thing regardless of the query.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from uuid import UUID

from .base import RetrievedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    *,
    rankings: Sequence[Sequence[RetrievedChunk]],
    weights: Sequence[float],
    limit: int,
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Merge ranked lists with ``weight / (k + rank)``.

    ``rankings`` and ``weights`` are positionally aligned. A zero weight skips
    the list entirely, so the retriever that produced it need not be called at
    all. Chunks are keyed by ``chunk_id``: both retrieval paths hydrate from the
    same rows, so the same chunk appearing in both lists is one result whose
    fused score is the sum of both contributions.
    """
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights must have the same length")
    if limit < 1:
        return []
    if k < 1:
        raise ValueError("k must be at least 1")

    fused: dict[UUID, float] = {}
    chunks: dict[UUID, RetrievedChunk] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        if weight <= 0.0:
            continue
        for rank, chunk in enumerate(ranking, start=1):
            chunks.setdefault(chunk.chunk_id, chunk)
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + weight / (k + rank)

    merged = [replace(chunks[chunk_id], score=score) for chunk_id, score in fused.items()]
    # RRF produces ties routinely (the same rank pair from both lists), so the
    # tie-break has to be stable or evaluation numbers drift between runs.
    merged.sort(
        key=lambda item: (
            -item.score,
            item.document_title,
            item.chunk_index,
            item.content,
            str(item.chunk_id),
        )
    )
    return merged[:limit]
