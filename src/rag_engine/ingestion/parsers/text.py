"""Plain-text parser."""

from __future__ import annotations

import re

from rag_engine.ingestion.errors import InvalidDocumentContent, ParseError
from rag_engine.ingestion.parsers.base import ParsedBlock, ParsedDocument

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class TextParser:
    name: str = "text"
    extensions: tuple[str, ...] = (".txt",)

    def validate_signature(self, data: bytes) -> None:
        # A text file with NUL bytes is almost always a renamed binary.
        if b"\x00" in data[:8192]:
            raise InvalidDocumentContent("file contains binary data and is not valid text")

    def parse(self, data: bytes) -> ParsedDocument:
        text = decode_text(data)
        blocks = tuple(
            ParsedBlock(text=paragraph.strip())
            for paragraph in _PARAGRAPH_BREAK.split(text)
            if paragraph.strip()
        )
        if not blocks:
            raise ParseError("document contains no extractable text")
        return ParsedDocument(blocks=blocks)


def decode_text(data: bytes) -> str:
    """Decode UTF-8 (with optional BOM), or fail loudly."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InvalidDocumentContent("file is not valid UTF-8 text") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")
