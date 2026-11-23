"""Chunking contract.

Every strategy turns a parsed document into drafts. The draft is deliberately
transport-free: it carries content plus everything retrieval and citations will
need (location, source offsets), but no ORM object or tenant-specific state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_engine.ingestion.parsers.base import ParsedDocument


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """One retrievable passage before it is assigned a document position."""

    content: str
    char_count: int
    page_start: int | None
    page_end: int | None
    section_path: str | None
    # Offsets into the parser's normalised text. Optional because a future
    # strategy may not be able to produce them; the recursive strategy always
    # does, and the persistence layer stores whatever it receives.
    start_char: int | None = None
    end_char: int | None = None


class ChunkingStrategy(Protocol):
    """Swappable chunking algorithm.

    The protocol exists so variants (fixed-size, sentence-window,
    embedding-based semantic splitting) can be benchmarked without touching the
    worker or the persistence layer.
    """

    name: str

    def chunk(self, document: ParsedDocument) -> list[ChunkDraft]:
        """Return drafts in reading order."""
        ...
