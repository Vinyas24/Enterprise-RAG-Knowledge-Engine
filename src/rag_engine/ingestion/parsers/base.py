"""Parser contract shared by every file type.

A parser turns bytes into ordered blocks of text. Blocks, not a single string,
because provenance is part of the contract: a chunk must be traceable to a page
(PDF) or a heading path (DOCX/Markdown) later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    text: str
    page: int | None = None
    section_path: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    blocks: tuple[ParsedBlock, ...]
    page_count: int | None = None


class DocumentParser(Protocol):
    """One implementation per supported file type."""

    name: str
    extensions: tuple[str, ...]

    def validate_signature(self, data: bytes) -> None:
        """Reject bytes that clearly are not this file type.

        Runs at upload time, before any expensive work is queued.
        """
        ...

    def parse(self, data: bytes) -> ParsedDocument:
        """Extract provenance-tagged text blocks."""
        ...
