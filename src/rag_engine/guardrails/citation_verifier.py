"""Output guardrail: Citation verification against retrieved candidate passages."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_engine.generation.base import Citation
from rag_engine.retrieval.base import RetrievedChunk

from .base import CitationValidationResult

_CITATION_RE = re.compile(r"\[(\d+)\]")


def validate_citations(
    answer: str,
    passages: Sequence[RetrievedChunk],
    citations: Sequence[Citation],
) -> CitationValidationResult:
    """Verify that cited bracket references exist within candidate passages.

    Strips citations to non-existent chunk indices or ungrounded chunks.
    """
    cited_indices = [int(m) for m in _CITATION_RE.findall(answer)]
    pool_size = len(passages)

    valid_citations: list[Citation] = []
    dropped_citations: list[int] = []

    # Map candidate (document_id, chunk_index) -> chunk
    candidate_chunks = {(chunk.document_id, chunk.chunk_index): chunk for chunk in passages}

    # Verify existing citations
    for citation in citations:
        pair = (citation.document_id, citation.chunk_index)
        if pair in candidate_chunks:
            valid_citations.append(citation)
        else:
            dropped_citations.append(citation.chunk_index)

    # Check for hallucinated indices in text (e.g. [99] when pool_size is 3)
    for idx in cited_indices:
        if (idx < 1 or idx > pool_size) and idx not in dropped_citations:
            dropped_citations.append(idx)

    is_valid = len(dropped_citations) == 0

    return CitationValidationResult(
        is_valid=is_valid,
        valid_citations=valid_citations,
        dropped_citations=dropped_citations,
    )
