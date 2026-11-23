"""HTTP response models.

Kept separate from the ORM models on purpose: the database schema and the
public contract should be free to diverge (renames, hidden columns, extra
fields) without one change forcing the other.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from rag_engine.core.enums import AccessLevel, DocumentStatus, JobStatus
from rag_engine.core.models import Chunk, Document, IngestionJob


class JobSummary(BaseModel):
    id: UUID
    status: JobStatus
    attempts: int
    max_attempts: int
    error: str | None
    available_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class JobRead(JobSummary):
    document_id: UUID


class DocumentSummary(BaseModel):
    id: UUID
    title: str
    filename: str
    status: DocumentStatus
    access_level: AccessLevel
    size_bytes: int
    chunk_count: int
    created_at: datetime


class DocumentRead(DocumentSummary):
    content_type: str | None
    content_hash: str
    updated_at: datetime
    job: JobSummary | None = Field(
        default=None, description="Most recent ingestion job, if one exists."
    )


class DocumentAccepted(BaseModel):
    document_id: UUID
    job_id: UUID
    status: JobStatus


class DocumentListResponse(BaseModel):
    items: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class ChunkRead(BaseModel):
    """A chunk plus everything needed to verify its citation."""

    id: UUID
    document_id: UUID
    tenant_id: UUID
    chunk_index: int
    content: str
    char_count: int
    start_char: int | None
    end_char: int | None
    page_start: int | None
    page_end: int | None
    section_path: str | None
    access_level: AccessLevel
    created_at: datetime


class ChunkListResponse(BaseModel):
    items: list[ChunkRead]
    total: int
    limit: int
    offset: int


class CitationRead(BaseModel):
    """Citation identifying the exact source passage for an answer."""

    document_id: UUID
    document_title: str
    chunk_id: UUID
    chunk_index: int
    snippet: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None


class QueryRequest(BaseModel):
    """User query payload."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    clearance: AccessLevel = AccessLevel.INTERNAL


class QueryResponse(BaseModel):
    """Synthesized answer and source citations."""

    question: str
    answer: str
    citations: list[CitationRead]
    weak_retrieval: bool


def job_summary(job: IngestionJob) -> JobSummary:
    return JobSummary(
        id=job.id,
        status=job.status,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        error=job.error,
        available_at=job.available_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
    )


def job_read(job: IngestionJob) -> JobRead:
    return JobRead(**job_summary(job).model_dump(), document_id=job.document_id)


def document_summary(document: Document) -> DocumentSummary:
    return DocumentSummary(
        id=document.id,
        title=document.title,
        filename=document.filename,
        status=document.status,
        access_level=document.access_level,
        size_bytes=document.size_bytes,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
    )


def document_read(document: Document, job: IngestionJob | None) -> DocumentRead:
    return DocumentRead(
        **document_summary(document).model_dump(),
        content_type=document.content_type,
        content_hash=document.content_hash,
        updated_at=document.updated_at,
        job=job_summary(job) if job is not None else None,
    )


def chunk_read(chunk: Chunk) -> ChunkRead:
    return ChunkRead(
        id=chunk.id,
        document_id=chunk.document_id,
        tenant_id=chunk.tenant_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        char_count=chunk.char_count,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_path=chunk.section_path,
        access_level=chunk.access_level,
        created_at=chunk.created_at,
    )
