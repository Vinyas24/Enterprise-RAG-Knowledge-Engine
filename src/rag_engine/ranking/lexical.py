"""Deterministic lexical re-ranking for offline evaluation.

This is **not** a cross-encoder. It re-applies BM25 to the merged candidate
pool, which is enough to test the re-ranking plumbing, produce a reproducible
offline evaluation, and keep CI hermetic. Production deployments should use
Cohere or the local neural cross-encoder; the numbers from this provider are a
floor, not a claim about what a learned re-ranker will do.

Scoring uses the same metric as the keyword path (``bm25_score``) with document
frequencies measured **inside the candidate pool**. That is the point of a
second pass: the first pass selects for recall across two retrievers, the
second ranks the survivors against each other. No parameter is tuned on the
evaluation set.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

from rag_engine.retrieval.base import RetrievedChunk
from rag_engine.retrieval.keyword import bm25_score, tokenize


class LexicalReranker:
    """BM25 over the candidate pool. Offline, deterministic, no model."""

    name = "lexical"

    async def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        pool = list(candidates)
        if top_n < 1 or not pool:
            return []

        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return pool[:top_n]

        counters = [(chunk, Counter(tokenize(chunk.content))) for chunk in pool]
        document_count = len(pool)
        average_length = sum(len(chunk.content) for chunk, _ in counters) / document_count
        document_frequencies = {
            term: sum(1 for _, frequencies in counters if frequencies.get(term, 0) > 0)
            for term in terms
        }

        scored: list[tuple[float, int, RetrievedChunk]] = []
        for position, (chunk, frequencies) in enumerate(counters):
            score = bm25_score(
                term_frequencies=frequencies,
                length=len(chunk.content),
                average_length=average_length,
                document_frequencies=document_frequencies,
                document_count=document_count,
            )
            scored.append((score, position, chunk))

        # Position breaks ties, so a query that matches nothing returns the
        # merged order unchanged rather than an arbitrary permutation.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [replace(chunk, score=score) for score, _, chunk in scored[:top_n]]

    async def warmup(self) -> None:
        """Nothing to load: this provider is pure Python."""

    async def close(self) -> None:
        """No resources to release."""
