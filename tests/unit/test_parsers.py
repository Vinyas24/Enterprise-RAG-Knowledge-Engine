"""Parser contract tests for TXT, Markdown, DOCX and PDF.

The assertions deliberately check provenance, not only text: a chunk that
cannot say which page or heading it came from cannot produce a trustworthy
citation later.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from docx import Document as DocxDocument

from rag_engine.ingestion.errors import (
    InvalidDocumentContent,
    ParseError,
    UnsupportedDocumentType,
)
from rag_engine.ingestion.parsers import ParserRegistry
from rag_engine.ingestion.parsers.docx import DocxParser
from rag_engine.ingestion.parsers.markdown import MarkdownParser
from rag_engine.ingestion.parsers.pdf import PdfParser
from rag_engine.ingestion.parsers.text import TextParser

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _docx_bytes(*, heading: str | None = "Overview", body: str = "Body text") -> bytes:
    document = DocxDocument()
    if heading is not None:
        document.add_heading(heading, level=1)
    document.add_paragraph(body)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_covers_every_supported_extension() -> None:
    registry = ParserRegistry.default()

    assert registry.supported_extensions == (".docx", ".markdown", ".md", ".pdf", ".txt")


@pytest.mark.parametrize(
    ("filename", "parser_type"),
    [
        ("notes.txt", TextParser),
        ("NOTES.TXT", TextParser),
        ("readme.md", MarkdownParser),
        ("readme.markdown", MarkdownParser),
        ("spec.docx", DocxParser),
        ("spec.pdf", PdfParser),
    ],
)
def test_registry_resolves_by_extension_case_insensitively(
    filename: str, parser_type: type[object]
) -> None:
    assert isinstance(ParserRegistry.default().parser_for(filename), parser_type)


def test_registry_rejects_unsupported_extension_with_guidance() -> None:
    with pytest.raises(UnsupportedDocumentType) as excinfo:
        ParserRegistry.default().parser_for("archive.zip")

    assert ".pdf" in excinfo.value.supported
    assert "archive.zip" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def test_text_parser_splits_paragraphs_and_normalises_newlines() -> None:
    parsed = TextParser().parse(b"First paragraph.\r\nStill first.\r\n\r\nSecond paragraph.")

    assert [block.text for block in parsed.blocks] == [
        "First paragraph.\nStill first.",
        "Second paragraph.",
    ]
    assert all(block.page is None for block in parsed.blocks)


def test_text_parser_accepts_a_utf8_bom() -> None:
    parsed = TextParser().parse("\ufeffhello".encode())

    assert parsed.blocks[0].text == "hello"


def test_text_parser_rejects_nul_bytes() -> None:
    with pytest.raises(InvalidDocumentContent, match="binary"):
        TextParser().validate_signature(b"text\x00more")


def test_text_parser_rejects_invalid_utf8() -> None:
    with pytest.raises(InvalidDocumentContent, match="UTF-8"):
        TextParser().parse(b"\xff\xfe\x00broken")


def test_text_parser_rejects_whitespace_only_documents() -> None:
    with pytest.raises(ParseError, match="no extractable text"):
        TextParser().parse(b"  \n\n \t \n")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_markdown_headings_become_section_paths() -> None:
    source = b"""# Guide

Intro text.

## Installation

Run the installer.

### Details

Deep text.
"""

    parsed = MarkdownParser().parse(source)

    paths = [block.section_path for block in parsed.blocks]
    assert "Guide" in paths
    assert "Guide > Installation" in paths
    assert "Guide > Installation > Details" in paths
    assert any(block.text == "Run the installer." for block in parsed.blocks)


def test_markdown_heading_levels_close_deeper_sections() -> None:
    source = b"# A\n\n## B\n\ntext\n\n# C\n\ntail"

    parsed = MarkdownParser().parse(source)

    tail = next(block for block in parsed.blocks if block.text == "tail")
    assert tail.section_path == "C"


def test_markdown_comments_inside_code_fences_are_not_headings() -> None:
    source = b"# Real heading\n\n```bash\n# not a heading\nls -la\n```\n"

    parsed = MarkdownParser().parse(source)

    assert all(block.section_path == "Real heading" for block in parsed.blocks)


def test_markdown_parser_rejects_binary_and_empty_documents() -> None:
    with pytest.raises(InvalidDocumentContent):
        MarkdownParser().validate_signature(b"#\x00heading")
    with pytest.raises(ParseError, match="no extractable text"):
        MarkdownParser().parse(b"   \n\n")


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def test_docx_parser_extracts_paragraphs_with_heading_provenance() -> None:
    parsed = DocxParser().parse(_docx_bytes(heading="Deployment", body="Use containers."))

    assert parsed.blocks[0].text == "Deployment"
    body = next(block for block in parsed.blocks if block.text == "Use containers.")
    assert body.section_path == "Deployment"


def test_docx_parser_flattens_tables() -> None:
    document = DocxDocument()
    document.add_paragraph("Pricing")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Plan"
    table.cell(0, 1).text = "Price"
    table.cell(1, 0).text = "Pro"
    table.cell(1, 1).text = "99"
    buffer = io.BytesIO()
    document.save(buffer)

    parsed = DocxParser().parse(buffer.getvalue())

    rendered = "\n".join(block.text for block in parsed.blocks)
    assert "Plan | Price" in rendered
    assert "Pro | 99" in rendered


def test_docx_parser_rejects_non_zip_and_zip_without_document() -> None:
    with pytest.raises(InvalidDocumentContent, match="zip"):
        DocxParser().validate_signature(b"not a zip")

    buffer = io.BytesIO()
    import zipfile

    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("other.txt", "hello")
    with pytest.raises(InvalidDocumentContent, match="not a valid DOCX"):
        DocxParser().validate_signature(buffer.getvalue())


def test_docx_parser_rejects_empty_documents() -> None:
    document = DocxDocument()
    buffer = io.BytesIO()
    document.save(buffer)

    with pytest.raises(ParseError, match="no extractable text"):
        DocxParser().parse(buffer.getvalue())


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def test_pdf_parser_extracts_pages_with_page_numbers() -> None:
    parsed = PdfParser().parse((FIXTURES / "simple.pdf").read_bytes())

    assert parsed.page_count == 2
    assert [block.page for block in parsed.blocks] == [1, 2]
    assert "retrieval step" in parsed.blocks[0].text
    assert "chunk overlap" in parsed.blocks[1].text


def test_pdf_parser_rejects_bytes_without_a_pdf_header() -> None:
    with pytest.raises(InvalidDocumentContent, match="%PDF"):
        PdfParser().validate_signature(b"<html>not a pdf</html>")


def test_pdf_parser_rejects_corrupt_pdf_content() -> None:
    with pytest.raises(ParseError, match="could not be read"):
        PdfParser().parse(b"%PDF-1.4\ngarbage that pypdf cannot parse")
