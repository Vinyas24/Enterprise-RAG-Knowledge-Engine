"""Upload orchestration: validate, deduplicate, store, enqueue.

This is the boundary between the HTTP layer and ingestion. It knows nothing
about FastAPI, and it owns the ordering that matters:

1. Cheap validation first (name, extension, signature).
2. Deduplication before any blob or row is written.
3. Blob first, then rows: an orphaned blob is harmless (content-addressed and
   shareable), whereas a document row pointing at missing bytes is not.
4. Document and job in one transaction, so an accepted upload always has a job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.core.config import Settings
from rag_engine.core.enums import AccessLevel
from rag_engine.core.models import Document, IngestionJob, Tenant
from rag_engine.ingestion import repository
from rag_engine.ingestion.blobstore import BlobStore, LocalBlobStore
from rag_engine.ingestion.errors import (
    DuplicateDocument,
    EmptyDocument,
    InvalidFilename,
)
from rag_engine.ingestion.hashing import hash_content
from rag_engine.ingestion.parsers import ParserRegistry

_MAX_TITLE_CHARS = 512
_MAX_CONTENT_TYPE_CHARS = 127


@dataclass(frozen=True, slots=True)
class UploadResult:
    document: Document
    job: IngestionJob


class IngestionService:
    def __init__(
        self, *, parsers: ParserRegistry, blob_store: BlobStore, settings: Settings
    ) -> None:
        self._parsers = parsers
        self._blob_store = blob_store
        self._settings = settings

    def ensure_supported(self, filename: str) -> str:
        """Validate the extension before the request body is read.

        Returns the sanitized file name so the caller can reuse it.
        """
        safe_name = sanitize_filename(filename)
        self._parsers.parser_for(safe_name)
        return safe_name

    async def create_upload(
        self,
        session: AsyncSession,
        *,
        tenant: Tenant,
        filename: str,
        content_type: str | None,
        data: bytes,
        title: str | None,
        access_level: AccessLevel,
    ) -> UploadResult:
        safe_name = sanitize_filename(filename)
        if not data:
            raise EmptyDocument("uploaded file is empty")

        parser = self._parsers.parser_for(safe_name)
        parser.validate_signature(data)

        content_hash = hash_content(data)
        existing = await repository.find_document_by_hash(
            session, tenant_id=tenant.id, content_hash=content_hash
        )
        if existing is not None:
            raise DuplicateDocument(existing.id)

        suffix = Path(safe_name).suffix.lower()
        storage_key = LocalBlobStore.build_key(
            tenant_slug=tenant.slug, content_hash=content_hash, suffix=suffix
        )
        await self._blob_store.put(storage_key, data)

        document, job = await repository.create_document_with_job(
            session,
            tenant_id=tenant.id,
            title=resolve_title(title, safe_name),
            filename=safe_name,
            content_type=(content_type[:_MAX_CONTENT_TYPE_CHARS] if content_type else None),
            size_bytes=len(data),
            content_hash=content_hash,
            storage_key=storage_key,
            access_level=access_level,
            max_attempts=self._settings.worker_max_attempts,
        )
        try:
            await session.commit()
        except IntegrityError:
            # Lost a race against a concurrent upload of identical bytes. The
            # unique constraint, not the pre-check, is the source of truth.
            await session.rollback()
            existing = await repository.find_document_by_hash(
                session, tenant_id=tenant.id, content_hash=content_hash
            )
            if existing is not None:
                raise DuplicateDocument(existing.id) from None
            raise

        return UploadResult(document=document, job=job)


def sanitize_filename(raw: str | None) -> str:
    """Reduce an untrusted upload name to a bare file name.

    Strips any client-side directory components (POSIX and Windows), rejects
    path traversals and control characters, and bounds the length to the
    column width.
    """
    candidate = (raw or "").replace("\\", "/").strip()
    name = Path(candidate).name
    if not name or name in {".", ".."}:
        raise InvalidFilename("uploaded file has no usable filename")
    if any(ord(char) < 32 for char in name):
        raise InvalidFilename("uploaded file name contains control characters")
    if len(name) > 255:
        raise InvalidFilename("uploaded file name is too long")
    return name


def resolve_title(title: str | None, filename: str) -> str:
    candidate = " ".join((title or "").split()) or Path(filename).stem
    candidate = candidate.strip() or "untitled"
    return candidate[:_MAX_TITLE_CHARS]
