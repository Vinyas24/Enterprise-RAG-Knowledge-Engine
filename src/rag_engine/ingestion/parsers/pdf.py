"""PDF parser.

One block per page, so a chunk always knows which page it came from. Scanned
PDFs have no text layer; rather than returning an empty document (which would
look like a successful ingestion that produces zero chunks), parsing fails with
an explicit message. OCR is deliberately out of scope.
"""

from __future__ import annotations

import io
import re

from pypdf import PdfReader

from rag_engine.ingestion.errors import InvalidDocumentContent, ParseError
from rag_engine.ingestion.parsers.base import ParsedBlock, ParsedDocument

_MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")


class PdfParser:
    name: str = "pdf"
    extensions: tuple[str, ...] = (".pdf",)

    def validate_signature(self, data: bytes) -> None:
        # The spec tolerates leading junk; the header must appear early.
        if b"%PDF-" not in data[:1024]:
            raise InvalidDocumentContent("file is not a valid PDF (missing %PDF header)")

    def parse(self, data: bytes) -> ParsedDocument:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                # Empty-password PDFs are common; anything else cannot be read.
                raise ParseError("PDF is password protected")
            pages = [
                (number, _normalise(page.extract_text() or ""))
                for number, page in enumerate(reader.pages, start=1)
            ]
            page_count = len(reader.pages)
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError("PDF document could not be read") from exc

        blocks = tuple(ParsedBlock(text=text, page=number) for number, text in pages if text)
        if not blocks:
            raise ParseError(
                "PDF contains no extractable text; scanned documents require OCR, "
                "which is not supported"
            )
        return ParsedDocument(blocks=blocks, page_count=page_count)


def _normalise(text: str) -> str:
    text = _TRAILING_SPACES.sub("\n", text)
    return _MULTIPLE_BLANK_LINES.sub("\n\n", text).strip()
