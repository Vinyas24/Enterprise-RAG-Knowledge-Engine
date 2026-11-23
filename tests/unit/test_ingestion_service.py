"""Upload name/title hygiene and pre-body validation."""

from __future__ import annotations

import pytest

from rag_engine.core.config import Settings
from rag_engine.ingestion.errors import InvalidFilename, UnsupportedDocumentType
from rag_engine.ingestion.parsers import ParserRegistry
from rag_engine.ingestion.service import IngestionService, resolve_title, sanitize_filename


def _service() -> IngestionService:
    settings = Settings(_env_file=None)

    class _NoopStore:
        async def put(self, key: str, data: bytes) -> None:
            raise AssertionError("validation must not write blobs")

        async def get(self, key: str) -> bytes:
            raise AssertionError("validation must not read blobs")

    return IngestionService(
        parsers=ParserRegistry.default(), blob_store=_NoopStore(), settings=settings
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("/srv/reports/report.pdf", "report.pdf"),
        ("C:\\Users\\analyst\\report.pdf", "report.pdf"),
        ("  spaced name.txt  ", "spaced name.txt"),
    ],
)
def test_client_directory_components_are_stripped(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", ".", "..", "..\\"])
def test_unusable_names_are_rejected(raw: str) -> None:
    with pytest.raises(InvalidFilename):
        sanitize_filename(raw)


def test_control_characters_are_rejected() -> None:
    with pytest.raises(InvalidFilename, match="control"):
        sanitize_filename("bad\nname.txt")


def test_overlong_names_are_rejected() -> None:
    with pytest.raises(InvalidFilename, match="too long"):
        sanitize_filename("a" * 252 + ".txt")


def test_title_defaults_to_the_filename_stem() -> None:
    assert resolve_title(None, "quarterly-report.pdf") == "quarterly-report"


def test_title_whitespace_is_normalised() -> None:
    assert resolve_title("  Quarterly   Report  ", "ignored.pdf") == "Quarterly Report"


def test_blank_title_falls_back_to_the_stem() -> None:
    assert resolve_title("   ", "notes.md") == "notes"


def test_title_is_bounded_to_the_column_width() -> None:
    assert len(resolve_title("x" * 900, "a.txt")) == 512


def test_ensure_supported_returns_the_sanitised_name() -> None:
    assert _service().ensure_supported("/data/Notes.TXT") == "Notes.TXT"


def test_ensure_supported_rejects_other_types() -> None:
    with pytest.raises(UnsupportedDocumentType):
        _service().ensure_supported("sheet.xlsx")
