"""Persistence operations for ingestion.

Deliberately plain functions over an ``AsyncSession``: no repository objects,
no hidden commits. The caller owns the transaction, which makes the upload
flow (document + job atomically) explicit.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from uuid import UUID

from sqlalchemy import and_, delete, func, insert, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.core.enums import AccessLevel, DocumentStatus, JobStatus
from rag_engine.core.ids import chunk_id, chunk_vector_id
from rag_engine.core.models import Chunk, Document, IngestionJob, Tenant, utcnow
from rag_engine.ingestion.chunking import ChunkDraft
from rag_engine.ingestion.retry import next_attempt_at

# ---------------------------------------------------------------------------
# Tenants
# ---------------------------------------------------------------------------


async def get_tenant_by_slug(session: AsyncSession, slug: str) -> Tenant | None:
    return (await session.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none()


async def create_tenant(session: AsyncSession, *, slug: str, name: str) -> Tenant:
    tenant = Tenant(slug=slug, name=name)
    session.add(tenant)
    await session.flush()
    return tenant


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


async def find_document_by_hash(
    session: AsyncSession, *, tenant_id: UUID, content_hash: str
) -> Document | None:
    statement = select(Document).where(
        Document.tenant_id == tenant_id, Document.content_hash == content_hash
    )
    return (await session.execute(statement)).scalar_one_or_none()


async def get_document(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    document_id: UUID,
    for_update: bool = False,
) -> Document | None:
    statement = select(Document).where(Document.id == document_id, Document.tenant_id == tenant_id)
    if for_update:
        # Serialises reindex requests for the same document, so two callers
        # cannot both observe "no active job" and enqueue duplicates.
        statement = statement.with_for_update()
    return (await session.execute(statement)).scalar_one_or_none()


async def list_documents(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    limit: int,
    offset: int,
    status: DocumentStatus | None = None,
    access_level: AccessLevel | None = None,
) -> Sequence[Document]:
    statement = select(Document).where(Document.tenant_id == tenant_id)
    if status is not None:
        statement = statement.where(Document.status == status)
    if access_level is not None:
        statement = statement.where(Document.access_level == access_level)
    statement = (
        statement.order_by(Document.created_at.desc(), Document.id).limit(limit).offset(offset)
    )
    return (await session.execute(statement)).scalars().all()


async def count_documents(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    status: DocumentStatus | None = None,
    access_level: AccessLevel | None = None,
) -> int:
    statement = select(func.count()).select_from(Document).where(Document.tenant_id == tenant_id)
    if status is not None:
        statement = statement.where(Document.status == status)
    if access_level is not None:
        statement = statement.where(Document.access_level == access_level)
    return int((await session.execute(statement)).scalar_one())


async def list_chunks(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    document_id: UUID,
    limit: int,
    offset: int,
) -> Sequence[Chunk]:
    """Chunks of one document in reading order, tenant-scoped."""
    statement = (
        select(Chunk)
        .where(Chunk.tenant_id == tenant_id, Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .limit(limit)
        .offset(offset)
    )
    return (await session.execute(statement)).scalars().all()


async def count_chunks(session: AsyncSession, *, tenant_id: UUID, document_id: UUID) -> int:
    statement = (
        select(func.count())
        .select_from(Chunk)
        .where(Chunk.tenant_id == tenant_id, Chunk.document_id == document_id)
    )
    return int((await session.execute(statement)).scalar_one())


async def create_document_with_job(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    title: str,
    filename: str,
    content_type: str | None,
    size_bytes: int,
    content_hash: str,
    storage_key: str,
    access_level: AccessLevel,
    max_attempts: int,
) -> tuple[Document, IngestionJob]:
    """Insert a document and its first ingestion job in one transaction."""
    document = Document(
        tenant_id=tenant_id,
        title=title,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        content_hash=content_hash,
        storage_key=storage_key,
        access_level=access_level,
    )
    session.add(document)
    await session.flush()

    job = IngestionJob(
        document_id=document.id,
        tenant_id=tenant_id,
        max_attempts=max_attempts,
    )
    session.add(job)
    await session.flush()
    return document, job


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


async def enqueue_job(
    session: AsyncSession, *, document_id: UUID, tenant_id: UUID, max_attempts: int
) -> IngestionJob:
    """Queue a fresh ingestion attempt for an existing document.

    Used by reindex: the content-addressed blob is unchanged, but chunks and
    vectors are rebuilt (for example after an embedding-model change or a
    previously failed attempt).
    """
    now = utcnow()
    job = IngestionJob(
        document_id=document_id,
        tenant_id=tenant_id,
        max_attempts=max_attempts,
        available_at=now,
    )
    session.add(job)
    await session.execute(
        update(Document)
        .where(Document.id == document_id, Document.tenant_id == tenant_id)
        .values(status=DocumentStatus.PENDING, updated_at=now)
    )
    await session.flush()
    return job


async def get_job(session: AsyncSession, *, tenant_id: UUID, job_id: UUID) -> IngestionJob | None:
    statement = select(IngestionJob).where(
        IngestionJob.id == job_id, IngestionJob.tenant_id == tenant_id
    )
    return (await session.execute(statement)).scalar_one_or_none()


async def list_jobs(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    limit: int,
    offset: int,
    status: JobStatus | None = None,
) -> Sequence[IngestionJob]:
    statement = select(IngestionJob).where(IngestionJob.tenant_id == tenant_id)
    if status is not None:
        statement = statement.where(IngestionJob.status == status)
    statement = (
        statement.order_by(IngestionJob.created_at.desc(), IngestionJob.id)
        .limit(limit)
        .offset(offset)
    )
    return (await session.execute(statement)).scalars().all()


async def count_jobs(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    status: JobStatus | None = None,
) -> int:
    statement = (
        select(func.count()).select_from(IngestionJob).where(IngestionJob.tenant_id == tenant_id)
    )
    if status is not None:
        statement = statement.where(IngestionJob.status == status)
    return int((await session.execute(statement)).scalar_one())


async def latest_job_for_document(
    session: AsyncSession, *, document_id: UUID
) -> IngestionJob | None:
    statement = (
        select(IngestionJob)
        .where(IngestionJob.document_id == document_id)
        .order_by(IngestionJob.created_at.desc())
        .limit(1)
    )
    return (await session.execute(statement)).scalar_one_or_none()


async def claim_next_job(
    session: AsyncSession, *, visibility_timeout_seconds: float
) -> IngestionJob | None:
    """Atomically take ownership of one available job.

    ``FOR UPDATE SKIP LOCKED`` is what makes this safe with several workers:
    each transaction locks a different row instead of blocking, and a crashed
    worker's job becomes claimable again once ``started_at`` is older than the
    visibility timeout.
    """
    now = utcnow()
    stale_before = now - timedelta(seconds=visibility_timeout_seconds)

    statement = (
        select(IngestionJob)
        .where(
            IngestionJob.attempts < IngestionJob.max_attempts,
            or_(
                and_(
                    IngestionJob.status == JobStatus.QUEUED,
                    IngestionJob.available_at <= now,
                ),
                and_(
                    IngestionJob.status == JobStatus.PROCESSING,
                    IngestionJob.started_at < stale_before,
                ),
            ),
        )
        .order_by(IngestionJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = (await session.execute(statement)).scalar_one_or_none()
    if job is None:
        return None

    job.status = JobStatus.PROCESSING
    job.attempts += 1
    job.started_at = now
    job.error = None
    job.updated_at = now

    await session.execute(
        update(Document)
        .where(Document.id == job.document_id)
        .values(status=DocumentStatus.PROCESSING, updated_at=now)
    )
    await session.flush()
    return job


async def fail_abandoned_jobs(session: AsyncSession, *, visibility_timeout_seconds: float) -> int:
    """Terminally fail jobs abandoned by dead workers after exhausting retries."""
    now = utcnow()
    stale_before = now - timedelta(seconds=visibility_timeout_seconds)
    statement = (
        update(IngestionJob)
        .where(
            IngestionJob.status == JobStatus.PROCESSING,
            IngestionJob.started_at < stale_before,
            IngestionJob.attempts >= IngestionJob.max_attempts,
        )
        .values(
            status=JobStatus.FAILED,
            error="ingestion worker stopped responding",
            finished_at=now,
            updated_at=now,
        )
        .returning(IngestionJob.document_id)
    )
    document_ids = (await session.execute(statement)).scalars().all()
    if document_ids:
        await session.execute(
            update(Document)
            .where(Document.id.in_(document_ids))
            .values(status=DocumentStatus.FAILED, updated_at=now)
        )
    return len(document_ids)


async def replace_chunks(
    session: AsyncSession,
    *,
    document_id: UUID,
    tenant_id: UUID,
    drafts: Sequence[ChunkDraft],
    access_level: AccessLevel,
) -> list[tuple[UUID, str]]:
    """Replace a document's chunk rows with ``drafts``.

    Returns ``(chunk_uuid, vector_id)`` pairs so callers can upsert vectors
    with the same deterministic ids PostgreSQL stores. The caller owns the
    transaction and decides whether the document is marked indexed.
    """
    now = utcnow()
    await session.execute(delete(Chunk).where(Chunk.document_id == document_id))

    id_pairs: list[tuple[UUID, str]] = []
    if drafts:
        rows = []
        for index, draft in enumerate(drafts):
            c_id = chunk_id(document_id, index)
            v_id = chunk_vector_id(document_id, index)
            id_pairs.append((c_id, v_id))
            rows.append(
                {
                    "id": c_id,
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "chunk_index": index,
                    "content": draft.content,
                    "char_count": draft.char_count,
                    "page_start": draft.page_start,
                    "page_end": draft.page_end,
                    "section_path": draft.section_path,
                    "access_level": access_level,
                    "start_char": draft.start_char,
                    "end_char": draft.end_char,
                    "created_at": now,
                }
            )
        await session.execute(insert(Chunk), rows)

    await session.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(chunk_count=len(drafts), updated_at=now)
    )
    return id_pairs


async def complete_job(
    session: AsyncSession,
    *,
    job_id: UUID,
    document_id: UUID,
    tenant_id: UUID,
    drafts: Sequence[ChunkDraft],
    access_level: AccessLevel,
) -> list[tuple[UUID, str]]:
    """Replace the document's chunks and mark both document and job successful."""
    id_pairs = await replace_chunks(
        session,
        document_id=document_id,
        tenant_id=tenant_id,
        drafts=drafts,
        access_level=access_level,
    )
    now = utcnow()
    await session.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(
            status=DocumentStatus.INDEXED,
            updated_at=now,
        )
    )
    await session.execute(
        update(IngestionJob)
        .where(IngestionJob.id == job_id)
        .values(
            status=JobStatus.DONE,
            error=None,
            finished_at=now,
            updated_at=now,
        )
    )
    return id_pairs


async def fail_job(
    session: AsyncSession,
    *,
    job_id: UUID,
    error: str,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> bool:
    """Record a failed attempt. Returns True when the job is terminally failed."""
    job = await session.get(IngestionJob, job_id)
    if job is None:
        return False

    now = utcnow()
    terminal = job.attempts >= job.max_attempts
    if terminal:
        job.status = JobStatus.FAILED
        job.finished_at = now
        document_status = DocumentStatus.FAILED
    else:
        job.status = JobStatus.QUEUED
        job.started_at = None
        job.available_at = next_attempt_at(
            job.attempts,
            base_seconds=retry_base_seconds,
            max_seconds=retry_max_seconds,
            now=now,
        )
        document_status = DocumentStatus.PENDING

    job.error = error
    job.updated_at = now
    await session.execute(
        update(Document)
        .where(Document.id == job.document_id)
        .values(status=document_status, updated_at=now)
    )
    return terminal


async def get_chunk_vector_ids(session: AsyncSession, *, document_id: UUID) -> list[str]:
    """Return the vector-store ids for every chunk of a document.

    Used during re-ingestion to identify stale vectors for cleanup.
    """
    rows = (
        (
            await session.execute(
                select(Chunk.chunk_index)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.chunk_index)
            )
        )
        .scalars()
        .all()
    )
    return [chunk_vector_id(document_id, index) for index in rows]


async def delete_document_if_idle(
    session: AsyncSession, *, tenant_id: UUID, document_id: UUID
) -> bool:
    """Delete a document only when no queued/processing job can write to it.

    The caller deletes vectors before calling this. Making the final delete
    conditional closes the race where a reindex is enqueued (or a worker claims
    a job) between the API's initial check and the commit. ``False`` means the
    document still exists because a job became active; the caller reports 409.
    """
    active_documents = select(IngestionJob.document_id).where(
        IngestionJob.status.in_((JobStatus.QUEUED, JobStatus.PROCESSING))
    )
    statement = (
        delete(Document)
        .where(
            Document.id == document_id,
            Document.tenant_id == tenant_id,
            Document.id.notin_(active_documents),
        )
        .returning(Document.id)
    )
    deleted_id = (await session.execute(statement)).scalar_one_or_none()
    return deleted_id is not None
