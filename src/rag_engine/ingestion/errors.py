"""Domain errors for ingestion.

These carry no HTTP status: the transport layer decides how to map them. That
keeps parsing and persistence testable without FastAPI.
"""

from __future__ import annotations

from uuid import UUID


class IngestionError(Exception):
    """Base class for expected ingestion failures."""


class UnsupportedDocumentType(IngestionError):
    def __init__(self, filename: str, supported: tuple[str, ...]) -> None:
        self.filename = filename
        self.supported = supported
        super().__init__(
            f"unsupported document type for {filename!r}; supported: {', '.join(supported)}"
        )


class InvalidFilename(IngestionError):
    pass


class EmptyDocument(IngestionError):
    pass


class DocumentTooLarge(IngestionError):
    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(f"document exceeds the maximum size of {max_bytes} bytes")


class InvalidDocumentContent(IngestionError):
    """The bytes do not look like the declared file type."""


class ParseError(IngestionError):
    """The document could not be parsed into text."""


class DuplicateDocument(IngestionError):
    def __init__(self, document_id: UUID) -> None:
        self.document_id = document_id
        super().__init__("an identical document already exists for this tenant")
