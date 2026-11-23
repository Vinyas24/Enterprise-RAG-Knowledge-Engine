"""Markdown parser.

Headings become the section path for every block beneath them, so a chunk from
a long document still answers "where in the document is this?" without the
reader having to open the file. Fenced code blocks are tracked so a ``#``
comment inside a shell snippet is not mistaken for a heading.
"""

from __future__ import annotations

import re

from rag_engine.ingestion.errors import InvalidDocumentContent, ParseError
from rag_engine.ingestion.parsers.base import ParsedBlock, ParsedDocument
from rag_engine.ingestion.parsers.text import decode_text

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_MAX_SECTION_CHARS = 1024


class MarkdownParser:
    name: str = "markdown"
    extensions: tuple[str, ...] = (".md", ".markdown")

    def validate_signature(self, data: bytes) -> None:
        if b"\x00" in data[:8192]:
            raise InvalidDocumentContent("file contains binary data and is not valid Markdown")

    def parse(self, data: bytes) -> ParsedDocument:
        blocks: list[ParsedBlock] = []
        headings: list[tuple[int, str]] = []
        paragraph: list[str] = []
        in_fence = False

        def flush() -> None:
            if paragraph:
                text = "\n".join(paragraph).strip()
                if text:
                    blocks.append(ParsedBlock(text=text, section_path=_section_path(headings)))
                paragraph.clear()

        for line in decode_text(data).split("\n"):
            if _FENCE.match(line):
                in_fence = not in_fence
                paragraph.append(line)
                continue

            heading = None if in_fence else _HEADING.match(line)
            if heading:
                flush()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, title[:200]))
                # The heading is content too: a query about "Retry policy"
                # should be able to match the heading itself.
                blocks.append(ParsedBlock(text=title, section_path=_section_path(headings)))
                continue

            if not line.strip():
                flush()
                continue
            paragraph.append(line)

        flush()
        if not blocks:
            raise ParseError("document contains no extractable text")
        return ParsedDocument(blocks=tuple(blocks))


def _section_path(headings: list[tuple[int, str]]) -> str | None:
    if not headings:
        return None
    return " > ".join(title for _, title in headings)[:_MAX_SECTION_CHARS]
