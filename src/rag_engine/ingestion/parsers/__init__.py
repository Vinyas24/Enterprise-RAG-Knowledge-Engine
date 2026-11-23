"""Parser registry: one lookup point for "which parser handles this file?"."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rag_engine.ingestion.errors import UnsupportedDocumentType
from rag_engine.ingestion.parsers.base import DocumentParser
from rag_engine.ingestion.parsers.docx import DocxParser
from rag_engine.ingestion.parsers.markdown import MarkdownParser
from rag_engine.ingestion.parsers.pdf import PdfParser
from rag_engine.ingestion.parsers.text import TextParser


class ParserRegistry:
    """Maps file extensions to parsers.

    Extension, not declared content type: the browser-supplied MIME type is
    advisory and frequently wrong. The parser's signature check then verifies
    that the bytes match.
    """

    def __init__(self, parsers: Iterable[DocumentParser]) -> None:
        self._by_extension: dict[str, DocumentParser] = {}
        for parser in parsers:
            for extension in parser.extensions:
                self._by_extension[extension.lower()] = parser

    @classmethod
    def default(cls) -> ParserRegistry:
        return cls([TextParser(), MarkdownParser(), DocxParser(), PdfParser()])

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_extension))

    def parser_for(self, filename: str) -> DocumentParser:
        suffix = Path(filename).suffix.lower()
        parser = self._by_extension.get(suffix)
        if parser is None:
            raise UnsupportedDocumentType(filename, self.supported_extensions)
        return parser
