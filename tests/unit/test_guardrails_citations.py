"""Unit tests for citation verification."""

from __future__ import annotations

from uuid import uuid4

from rag_engine.core.enums import AccessLevel
from rag_engine.generation.base import Citation
from rag_engine.guardrails.citation_verifier import validate_citations
from rag_engine.retrieval.base import RetrievedChunk


def _make_chunk(index: int, doc_id: object, chunk_id: object, title: str = "Doc") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,  # type: ignore[arg-type]
        document_id=doc_id,  # type: ignore[arg-type]
        document_title=title,
        chunk_index=index,
        content="Sample passage text.",
        score=0.9,
        access_level=AccessLevel.INTERNAL,
    )


def test_validate_valid_citations() -> None:
    doc_id = uuid4()
    chunk_id = uuid4()
    chunk = _make_chunk(0, doc_id, chunk_id)
    citation = Citation(
        document_id=doc_id,
        document_title="Doc",
        chunk_id=chunk_id,
        chunk_index=0,
        snippet="Sample passage text.",
    )

    answer = "This is verified fact [1]."
    result = validate_citations(answer, [chunk], [citation])

    assert result.is_valid
    assert len(result.valid_citations) == 1
    assert len(result.dropped_citations) == 0


def test_drops_hallucinated_out_of_bounds_citation() -> None:
    doc_id = uuid4()
    chunk_id = uuid4()
    chunk = _make_chunk(0, doc_id, chunk_id)
    citation = Citation(
        document_id=doc_id,
        document_title="Doc",
        chunk_id=chunk_id,
        chunk_index=0,
        snippet="Sample passage text.",
    )

    # Answer hallucinates bracket [99] when only 1 passage exists
    answer = "This is verified fact [1], and another claim [99]."
    result = validate_citations(answer, [chunk], [citation])

    assert not result.is_valid
    assert 99 in result.dropped_citations
    assert len(result.valid_citations) == 1


def test_drops_citation_to_unretrieved_chunk() -> None:
    doc_id = uuid4()
    chunk_id = uuid4()
    chunk = _make_chunk(0, doc_id, chunk_id)

    # Citation refers to a different, unretrieved document
    phantom_citation = Citation(
        document_id=uuid4(),
        document_title="Phantom Doc",
        chunk_id=uuid4(),
        chunk_index=5,
        snippet="Phantom snippet.",
    )

    answer = "Some claim [1]."
    result = validate_citations(answer, [chunk], [phantom_citation])

    assert len(result.valid_citations) == 0
    assert 5 in result.dropped_citations
