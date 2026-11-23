"""Chunker invariants.

The important properties are not "it splits" but:

* no chunk is larger than the configured size (retrieval and embedding models
  have hard limits),
* consecutive chunks share context (overlap),
* a chunk never crosses a page or heading boundary (provenance),
* character offsets reproduce each chunk from the normalised source text
  (traceability),
* natural boundaries are preferred over hard cuts (chunk quality).
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from rag_engine.core.config import Settings
from rag_engine.ingestion.chunking import (
    RecursiveChunker,
    build_chunker,
    normalise_blocks,
    split_text,
    split_text_spans,
)
from rag_engine.ingestion.parsers.base import ParsedBlock, ParsedDocument


def _chunker(size: int = 200, overlap: int = 20) -> RecursiveChunker:
    return RecursiveChunker(chunk_size=size, chunk_overlap=overlap)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0, "chunk_overlap": 0},
        {"chunk_size": 100, "chunk_overlap": -1},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 150},
    ],
)
def test_invalid_chunk_configuration_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(**kwargs)


def test_short_text_stays_in_one_chunk() -> None:
    drafts = _chunker().chunk(ParsedDocument(blocks=(ParsedBlock(text="Short sentence."),)))

    assert len(drafts) == 1
    assert drafts[0].content == "Short sentence."
    assert drafts[0].char_count == len("Short sentence.")


def test_empty_document_produces_no_chunks() -> None:
    assert _chunker().chunk(ParsedDocument(blocks=())) == []
    assert _chunker().chunk(ParsedDocument(blocks=(ParsedBlock(text="   "),))) == []


def test_no_chunk_exceeds_the_configured_size() -> None:
    paragraph = "This is a sentence about retrieval quality. " * 80
    drafts = _chunker(size=300, overlap=60).chunk(
        ParsedDocument(blocks=(ParsedBlock(text=paragraph),))
    )

    assert len(drafts) > 1
    assert max(draft.char_count for draft in drafts) <= 300
    assert all(draft.char_count == len(draft.content) for draft in drafts)


def test_consecutive_chunks_overlap() -> None:
    words = " ".join(f"word{index}" for index in range(300))
    chunks = split_text(words, size=200, overlap=40)

    assert len(chunks) >= 2
    for previous, current in pairwise(chunks):
        # The tail of one chunk reappears at the head of the next.
        assert previous[-20:] in current


def test_overlap_is_capped_so_size_is_never_exceeded() -> None:
    # A chunk that exactly fills the budget cannot take an overlap prefix.
    text = "a" * 100 + " b" * 100
    chunks = split_text(text, size=100, overlap=99)

    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunks_do_not_cross_page_boundaries() -> None:
    document = ParsedDocument(
        blocks=(
            ParsedBlock(text="page one " * 40, page=1),
            ParsedBlock(text="page two " * 40, page=2),
        ),
        page_count=2,
    )

    drafts = _chunker(size=120, overlap=20).chunk(document)

    page_one = [draft for draft in drafts if draft.page_start == 1]
    page_two = [draft for draft in drafts if draft.page_start == 2]
    assert page_one and page_two
    assert all(draft.page_end == 1 for draft in page_one)
    assert all(draft.page_end == 2 for draft in page_two)


def test_chunks_do_not_cross_section_boundaries() -> None:
    document = ParsedDocument(
        blocks=(
            ParsedBlock(text="alpha " * 30, section_path="Alpha"),
            ParsedBlock(text="beta " * 30, section_path="Beta"),
        )
    )

    drafts = _chunker(size=100, overlap=10).chunk(document)

    sections = {draft.section_path for draft in drafts}
    assert sections == {"Alpha", "Beta"}
    for draft in drafts:
        if draft.section_path == "Alpha":
            assert "beta" not in draft.content
        else:
            assert "alpha" not in draft.content


def test_blocks_with_the_same_provenance_are_joined() -> None:
    document = ParsedDocument(
        blocks=(
            ParsedBlock(text="one", page=3, section_path="Intro"),
            ParsedBlock(text="two", page=3, section_path="Intro"),
        )
    )

    drafts = _chunker().chunk(document)

    assert len(drafts) == 1
    assert drafts[0].content == "one\n\ntwo"
    assert drafts[0].page_start == 3


def test_unbroken_text_is_hard_split() -> None:
    chunks = split_text("x" * 1000, size=100, overlap=0)

    assert [len(chunk) for chunk in chunks] == [100] * 10
    assert "".join(chunks) == "x" * 1000


def test_split_text_returns_nothing_for_blank_input() -> None:
    assert split_text("   \n\n ", size=100, overlap=10) == []


# ---------------------------------------------------------------------------
# Phase 2: boundary quality, traceability and strategy selection
# ---------------------------------------------------------------------------


def test_factory_builds_the_configured_strategy() -> None:
    settings = Settings(
        chunk_strategy="recursive",
        chunk_size_chars=300,
        chunk_overlap_chars=50,
        _env_file=None,
    )

    chunker = build_chunker(settings)
    drafts = chunker.chunk(ParsedDocument(blocks=(ParsedBlock(text="word " * 200),)))

    assert chunker.name == "recursive"
    assert max(draft.char_count for draft in drafts) <= 300


def test_factory_rejects_an_unknown_strategy() -> None:
    settings = Settings(_env_file=None).model_copy(update={"chunk_strategy": "semantic"})

    with pytest.raises(ValueError, match="unknown chunking strategy"):
        build_chunker(settings)


def test_sentence_boundaries_are_preferred_over_word_boundaries() -> None:
    chunks = split_text("First sentence. Second sentence. Third sentence.", size=40, overlap=0)

    assert chunks == ["First sentence. Second sentence. ", "Third sentence."]


def test_abbreviations_do_not_end_a_sentence() -> None:
    chunks = split_text("Dr. Smith arrived. The lab was empty.", size=25, overlap=0)

    assert chunks == ["Dr. Smith arrived. ", "The lab was empty."]


@pytest.mark.parametrize(
    "text",
    [
        "Pi is 3.14 today. Next sentence.",
        "J. Smith wrote the RFC. Then he left.",
    ],
)
def test_initials_and_decimals_do_not_create_false_sentence_ends(text: str) -> None:
    chunks = split_text(text, size=25, overlap=0)

    # No chunk may end in the middle of an initial, an acronym or a number.
    assert all(chunk.strip() not in {"Pi", "J.", "3."} for chunk in chunks)
    assert any(chunk.rstrip().endswith(".") for chunk in chunks)


def test_offsets_reproduce_each_chunk_exactly() -> None:
    document = ParsedDocument(
        blocks=(
            ParsedBlock(text="First page. " * 30, page=1),
            # An empty page must not produce a chunk or disturb offsets.
            ParsedBlock(text="   ", page=2),
            ParsedBlock(text="Second page. " * 30, page=3),
        ),
        page_count=3,
    )

    drafts = _chunker(size=120, overlap=30).chunk(document)
    normalised = normalise_blocks(document.blocks)

    assert drafts
    assert all(draft.page_start != 2 for draft in drafts)
    for draft in drafts:
        assert draft.start_char is not None
        assert draft.end_char is not None
        assert draft.end_char > draft.start_char
        assert draft.end_char - draft.start_char == draft.char_count
        assert normalised[draft.start_char : draft.end_char] == draft.content


def test_overlap_starts_on_a_word_boundary_and_shares_context() -> None:
    text = " ".join(f"token{index}" for index in range(400))

    spans = split_text_spans(text, size=200, overlap=50)

    assert len(spans) >= 2
    for span in spans:
        assert text[span.start : span.end] == span.text
    for previous, current in pairwise(spans):
        assert current.start < previous.end
        assert current.start == 0 or text[current.start - 1].isspace()


def test_table_rows_are_not_split_when_they_fit() -> None:
    table = "| name | value |\n| alpha | 1 |\n| beta | 2 |\n| gamma | 3 |"
    source_rows = set(table.splitlines())

    drafts = _chunker(size=40, overlap=0).chunk(ParsedDocument(blocks=(ParsedBlock(text=table),)))

    assert len(drafts) > 1
    for draft in drafts:
        rows = draft.content.rstrip("\n").splitlines()
        assert rows
        assert set(rows) <= source_rows
    assert "| beta | 2 |" in "".join(draft.content for draft in drafts)


def test_very_long_paragraph_keeps_offsets_and_word_boundaries() -> None:
    paragraph = "Retrieval quality depends on chunking, and boundaries matter. " * 100
    document = ParsedDocument(blocks=(ParsedBlock(text=paragraph),))

    drafts = _chunker(size=250, overlap=30).chunk(document)
    normalised = normalise_blocks(document.blocks)

    assert len(drafts) > 10
    assert all(draft.char_count <= 250 for draft in drafts)
    for draft in drafts:
        assert draft.start_char is not None
        assert draft.end_char is not None
        assert normalised[draft.start_char : draft.end_char] == draft.content
        assert draft.start_char == 0 or normalised[draft.start_char - 1].isspace()
