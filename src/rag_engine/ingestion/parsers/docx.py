"""DOCX parser (Office Open XML).

Word documents have no fixed pagination, so provenance here is the heading
path. Tables are flattened row-by-row with ``|`` separators: losing them would
silently drop content from most real-world documents.
"""

from __future__ import annotations

import io
import re
import zipfile

from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError

from rag_engine.ingestion.errors import InvalidDocumentContent, ParseError
from rag_engine.ingestion.parsers.base import ParsedBlock, ParsedDocument

_HEADING_STYLE = re.compile(r"heading\s*([1-6])", re.IGNORECASE)
_MAX_SECTION_CHARS = 1024


class DocxParser:
    name: str = "docx"
    extensions: tuple[str, ...] = (".docx",)

    def validate_signature(self, data: bytes) -> None:
        """A DOCX file is a ZIP container that must hold ``word/document.xml``."""
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile as exc:
            raise InvalidDocumentContent("file is not a valid DOCX (zip) container") from exc
        if "word/document.xml" not in names:
            raise InvalidDocumentContent("file is not a valid DOCX document")

    def parse(self, data: bytes) -> ParsedDocument:
        try:
            document = DocxDocument(io.BytesIO(data))
        except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:
            raise ParseError("DOCX document could not be read") from exc

        blocks: list[ParsedBlock] = []
        headings: list[tuple[int, str]] = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            level = _heading_level(paragraph)
            if level is not None:
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, text[:200]))
            blocks.append(ParsedBlock(text=text, section_path=_section_path(headings)))

        for table in document.tables:
            rows = [
                " | ".join(cell.text.strip() for cell in row.cells).strip() for row in table.rows
            ]
            text = "\n".join(row for row in rows if row)
            if text:
                blocks.append(ParsedBlock(text=text, section_path=_section_path(headings)))

        if not blocks:
            raise ParseError("DOCX document contains no extractable text")
        return ParsedDocument(blocks=tuple(blocks))


def _heading_level(paragraph: object) -> int | None:
    style = getattr(paragraph, "style", None)
    if style is None:
        return None
    for candidate in (getattr(style, "name", ""), getattr(style, "style_id", "")):
        match = _HEADING_STYLE.match(str(candidate))
        if match:
            return int(match.group(1))
    return None


def _section_path(headings: list[tuple[int, str]]) -> str | None:
    if not headings:
        return None
    return " > ".join(title for _, title in headings)[:_MAX_SECTION_CHARS]
