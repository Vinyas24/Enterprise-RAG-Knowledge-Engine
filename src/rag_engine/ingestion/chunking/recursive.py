"""Recursive, boundary-aware chunking.

The splitter descends a hierarchy of natural boundaries:

1. blank line (paragraph),
2. single newline (list item, table row, heading),
3. sentence end (abbreviation- and decimal-aware),
4. clause punctuation,
5. word,
6. hard character cut only when no boundary exists at all.

That order is what keeps ideas intact: a table row or a sentence is never cut
while a coarser boundary is available. Overlap is also taken from a sentence or
word boundary rather than an arbitrary character offset, so a chunk does not
begin with the tail of a word.

Chunks are grouped by ``(page, section_path)`` before splitting, so a chunk never
spans two pages or two headings. ``start_char``/``end_char`` point into the
normalised text formed by joining non-empty blocks with a blank line; tests
assert that slicing that text at the offsets reproduces the chunk exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Protocol

from rag_engine.ingestion.parsers.base import ParsedBlock, ParsedDocument

from .base import ChunkDraft

# Sentence endings that are almost never a real sentence boundary. Lowercase
# comparison; the token includes its trailing period.
_ABBREVIATIONS = frozenset(
    {
        "al.",
        "approx.",
        "dept.",
        "dr.",
        "e.g.",
        "est.",
        "etc.",
        "fig.",
        "i.e.",
        "inc.",
        "ltd.",
        "mr.",
        "mrs.",
        "ms.",
        "no.",
        "prof.",
        "st.",
        "vs.",
    }
)
_DOTTED_INITIALS = re.compile(r"(?:[A-Za-z]\.)+")
_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]?(?=\s|$)")
_SENTENCE_STARTERS = frozenset("\"'\u201c\u2018([{")


@dataclass(frozen=True, slots=True)
class _Span:
    text: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class _Segment:
    text: str
    start_char: int
    page_start: int | None
    page_end: int | None
    section_path: str | None


class RecursiveChunker:
    """Boundary-aware recursive splitter with overlap."""

    name = "recursive"

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for segment in _segments(document.blocks):
            for span in split_text_spans(
                segment.text,
                size=self._chunk_size,
                overlap=self._chunk_overlap,
                offset=segment.start_char,
            ):
                drafts.append(
                    ChunkDraft(
                        content=span.text,
                        char_count=len(span.text),
                        page_start=segment.page_start,
                        page_end=segment.page_end,
                        section_path=segment.section_path,
                        start_char=span.start,
                        end_char=span.end,
                    )
                )
        return drafts


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def normalise_blocks(blocks: tuple[ParsedBlock, ...]) -> str:
    """The text that chunk offsets refer to.

    Parsers may emit blank blocks (an empty PDF page) and incidental leading or
    trailing whitespace. Both are dropped here so offsets are stable and a
    chunk never starts with invisible padding.
    """
    return "\n\n".join(text for block in blocks if (text := block.text.strip()))


def _segments(blocks: tuple[ParsedBlock, ...]) -> list[_Segment]:
    segments: list[_Segment] = []
    buffer: list[str] = []
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None

    def flush() -> None:
        if buffer:
            segments.append(
                _Segment(
                    text="\n\n".join(buffer),
                    start_char=0,
                    page_start=page_start,
                    page_end=page_end,
                    section_path=section_path,
                )
            )
            buffer.clear()

    for block in blocks:
        text = block.text.strip()
        if not text:
            # Empty pages and blank paragraphs contribute no chunk and no noise.
            continue
        if buffer and (block.page != page_start or block.section_path != section_path):
            flush()
        if not buffer:
            page_start = block.page
            page_end = block.page
            section_path = block.section_path
        else:
            page_end = block.page if block.page is not None else page_end
        buffer.append(text)

    flush()

    # Assign offsets in the normalised document: segments joined by a blank
    # line, blocks inside a segment joined the same way.
    cursor = 0
    positioned: list[_Segment] = []
    for index, segment in enumerate(segments):
        if index:
            cursor += 2
        positioned.append(replace(segment, start_char=cursor))
        cursor += len(segment.text)
    return positioned


# ---------------------------------------------------------------------------
# Boundary-aware splitting
# ---------------------------------------------------------------------------


class _Boundary(Protocol):
    def cuts(self, text: str) -> list[int]:
        """Positions *after which* a split is allowed."""
        ...


@dataclass(frozen=True, slots=True)
class _LiteralBoundary:
    separator: str

    def cuts(self, text: str) -> list[int]:
        cuts: list[int] = []
        cursor = 0
        while (index := text.find(self.separator, cursor)) != -1:
            cursor = index + len(self.separator)
            cuts.append(cursor)
        return cuts


@dataclass(frozen=True, slots=True)
class _SentenceBoundary:
    def cuts(self, text: str) -> list[int]:
        cuts: list[int] = []
        for match in _SENTENCE_END.finditer(text):
            punctuation = match.start()
            cursor = match.end()
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text):
                continue
            if not _starts_sentence(text[cursor]):
                continue
            if _is_false_sentence_end(text, punctuation):
                continue
            cuts.append(cursor)
        return cuts


# Most specific first. An empty separator is intentionally absent: it is the
# hard-split fallback, not a boundary.
BOUNDARIES: tuple[_Boundary, ...] = (
    _LiteralBoundary("\n\n"),
    _LiteralBoundary("\n"),
    _SentenceBoundary(),
    _LiteralBoundary("; "),
    _LiteralBoundary(", "),
    _LiteralBoundary(" "),
)


def _starts_sentence(char: str) -> bool:
    return char.isupper() or char.isdigit() or char in _SENTENCE_STARTERS


def _is_false_sentence_end(text: str, punctuation: int) -> bool:
    """Abbreviations, initials, acronyms and decimals are not sentence ends."""
    if punctuation > 0 and text[punctuation - 1].isdigit():
        # A period between digits (3.14) never reaches here because the regex
        # requires whitespace after it; numbered headings like "1. Intro" do.
        return False

    token_start = max(
        text.rfind(" ", 0, punctuation),
        text.rfind("\n", 0, punctuation),
        text.rfind("\t", 0, punctuation),
    )
    token = text[token_start + 1 : punctuation + 1]
    if token.lower() in _ABBREVIATIONS:
        return True
    return bool(_DOTTED_INITIALS.fullmatch(token))


def split_text_spans(text: str, *, size: int, overlap: int, offset: int = 0) -> list[_Span]:
    """Split ``text`` into spans of at most ``size`` characters.

    ``offset`` shifts the returned spans into the caller's coordinate space, so
    a strategy can keep exact document offsets while splitting one segment.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    body = text.strip()
    if not body:
        return []

    base = offset + (len(text) - len(text.lstrip()))
    if len(body) <= size:
        return [_Span(body, base, base + len(body))]

    # Reserve room for the overlap prefix before splitting. Doing it afterwards
    # only works when a chunk happens to be short, which is rarely true.
    budget = size if overlap <= 0 else size - overlap
    pieces = _split_spans(body, size=budget, boundaries=BOUNDARIES, offset=base)
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    return _apply_overlap(pieces, overlap=overlap)


def split_text(text: str, *, size: int, overlap: int) -> list[str]:
    """String-only convenience wrapper used by tests and callers."""
    return [span.text for span in split_text_spans(text, size=size, overlap=overlap)]


def _split_spans(
    text: str, *, size: int, boundaries: tuple[_Boundary, ...], offset: int
) -> list[_Span]:
    if not text:
        return []
    if len(text) <= size:
        return [_Span(text, offset, offset + len(text))]

    for index, boundary in enumerate(boundaries):
        cuts = boundary.cuts(text)
        if not cuts:
            continue

        parts = _parts(text, cuts, offset)
        chunks: list[_Span] = []
        buffer: _Span | None = None
        for part in parts:
            if part.length > size:
                if buffer is not None:
                    chunks.append(buffer)
                    buffer = None
                chunks.extend(
                    _split_spans(
                        part.text,
                        size=size,
                        boundaries=boundaries[index + 1 :],
                        offset=part.start,
                    )
                )
            elif buffer is None:
                buffer = part
            elif buffer.length + part.length <= size:
                buffer = _Span(buffer.text + part.text, buffer.start, part.end)
            else:
                chunks.append(buffer)
                buffer = part
        if buffer is not None:
            chunks.append(buffer)
        if chunks:
            return chunks

    # No boundary helped (one enormous URL, base64 blob, ...). Cut hard so the
    # size limit still holds; this is the only case that can split a word.
    return [
        _Span(
            text[position : position + size],
            offset + position,
            offset + min(position + size, len(text)),
        )
        for position in range(0, len(text), size)
    ]


def _parts(text: str, cuts: list[int], offset: int) -> list[_Span]:
    bounds = [0, *(cut for cut in cuts if 0 < cut < len(text)), len(text)]
    return [
        _Span(text[start:end], offset + start, offset + end)
        for start, end in pairwise(bounds)
        if end > start
    ]


def _apply_overlap(pieces: list[_Span], *, overlap: int) -> list[_Span]:
    result = [pieces[0]]
    for previous, current in pairwise(pieces):
        prefix = _overlap_prefix(previous.text, overlap)
        start = previous.end - len(prefix)
        result.append(_Span(prefix + current.text, start, current.end))
    return result


def _overlap_prefix(previous: str, overlap: int) -> str:
    """Take context from a sentence or word boundary, never mid-word."""
    if overlap <= 0 or not previous:
        return ""
    if len(previous) <= overlap:
        return previous

    snippet_start = len(previous) - overlap
    snippet = previous[snippet_start:]

    # A clean sentence start is the most coherent context.
    sentence_cuts = _SentenceBoundary().cuts(snippet)
    if sentence_cuts:
        return snippet[sentence_cuts[0] :]

    # Otherwise skip a partial leading word.
    if snippet_start > 0 and not previous[snippet_start - 1].isspace():
        space = snippet.find(" ")
        if 0 <= space < len(snippet) - 1:
            snippet = snippet[space + 1 :]
    return snippet
