"""Citation parsing and provenance mapping."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_engine.retrieval.base import RetrievedChunk

from .base import Citation

_CITATION_RE = re.compile(r"\[(\d+)\]")
_MAX_SNIPPET_CHARS = 240


def extract_citations(
    answer: str,
    passages: Sequence[RetrievedChunk],
) -> list[Citation]:
    """Map bracketed references in the answer back to passage chunks."""
    cited_numbers = {int(match) for match in _CITATION_RE.findall(answer)}

    citations: list[Citation] = []
    seen_chunks: set[tuple[object, int]] = set()

    # If the answer explicitly references [N], cite those specific passages
    if cited_numbers:
        for num in sorted(cited_numbers):
            index = num - 1
            if 0 <= index < len(passages):
                chunk = passages[index]
                pair = (chunk.document_id, chunk.chunk_index)
                if pair not in seen_chunks:
                    seen_chunks.add(pair)
                    citations.append(_to_citation(chunk))
    elif passages and not answer.startswith("I cannot answer"):
        # If the LLM answered without explicit bracket markers, cite the top passage
        top_chunk = passages[0]
        citations.append(_to_citation(top_chunk))

    return citations


def _to_citation(chunk: RetrievedChunk) -> Citation:
    snippet = chunk.content.strip().replace("\n", " ")
    if len(snippet) > _MAX_SNIPPET_CHARS:
        snippet = snippet[:_MAX_SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."

    return Citation(
        document_id=chunk.document_id,
        document_title=chunk.document_title,
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        snippet=snippet,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_path=chunk.section_path,
    )
