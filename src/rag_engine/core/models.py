"""SQLAlchemy models: tenants, users, documents, chunks, ingestion jobs.

Deliberate choices:

* **Application-generated UUID primary keys.** No sequence contention, ids are
  known before commit (so the upload response can return them), and they are
  safe to expose in URLs.
* **No ORM relationships.** Under async SQLAlchemy a lazy relationship access
  raises at runtime. Explicit queries are boring and predictable; the database
  still enforces referential integrity with ``ON DELETE`` rules.
* **VARCHAR + CHECK instead of native enums.** Adding an access level or status
  is a normal migration rather than a PostgreSQL type alteration.
* **Metadata is denormalised where reads need it.** ``chunks.tenant_id`` and
  ``ingestion_jobs.tenant_id`` duplicate the parent's tenant so tenant-scoped
  queries never need a join, which matters once retrieval filters by tenant.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum as PyEnum
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Computed,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from rag_engine.core.enums import AccessLevel, DocumentStatus, JobStatus, UserRole

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def utcnow() -> datetime:
    return datetime.now(UTC)


def _enum_values(enum_class: type[PyEnum]) -> list[str]:
    return [str(member.value) for member in enum_class]


def enum_column(enum_class: type[StrEnum], *, name: str, length: int = 16) -> Enum:
    """Store a Python enum as a checked string column."""
    return Enum(
        enum_class,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=length,
        values_callable=_enum_values,
        validate_strings=True,
    )


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(String(63), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), nullable=False
    )


class User(Base):
    """Directory record and authorization subject.

    The API does not authenticate yet; this table is the anchor for the
    identity and clearance model wired up in a later phase.
    """

    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),)

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        enum_column(UserRole, name="user_role", length=16), nullable=False, default=UserRole.MEMBER
    )
    clearance: Mapped[AccessLevel] = mapped_column(
        enum_column(AccessLevel, name="access_level", length=16),
        nullable=False,
        default=AccessLevel.INTERNAL,
    )
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), nullable=False
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        # Deduplication scope is (tenant, content): the same bytes uploaded by
        # two tenants are two documents, because existence itself is private.
        UniqueConstraint("tenant_id", "content_hash", name="uq_documents_tenant_content_hash"),
        Index("ix_documents_tenant_created_at", "tenant_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    # Null until authentication exists; documents are owned by the tenant today.
    owner_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(127), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(255), nullable=False)
    access_level: Mapped[AccessLevel] = mapped_column(
        enum_column(AccessLevel, name="access_level", length=16),
        nullable=False,
        default=AccessLevel.INTERNAL,
    )
    status: Mapped[DocumentStatus] = mapped_column(
        enum_column(DocumentStatus, name="document_status", length=16),
        nullable=False,
        default=DocumentStatus.PENDING,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=func.now(),
        onupdate=utcnow,
        nullable=False,
    )


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        # The worker's claim query filters on exactly these columns.
        Index("ix_ingestion_jobs_status_available_at", "status", "available_at"),
        Index("ix_ingestion_jobs_document_id", "document_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        enum_column(JobStatus, name="job_status", length=16),
        nullable=False,
        default=JobStatus.QUEUED,
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Earliest time the job may be claimed; pushed forward by retry backoff.
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=func.now(),
        onupdate=utcnow,
        nullable=False,
    )


class Chunk(Base):
    """A retrievable passage with provenance back to its source document."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_chunk_index"),
        Index("ix_chunks_tenant_document", "tenant_id", "document_id"),
        # Retrieval filters on exactly this pair: tenant isolation first, then
        # the clearance predicate.
        Index("ix_chunks_tenant_access_level", "tenant_id", "access_level"),
        # Keyword retrieval filters with ``content_tsv @@ query``; GIN is the
        # index type that supports that operator.
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        # Offsets are nullable for rows written before Phase 2, but a present
        # range must be ordered and non-negative.
        CheckConstraint(
            "start_char IS NULL OR end_char IS NULL OR (start_char >= 0 AND end_char > start_char)",
            name="char_range",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Generated by PostgreSQL from ``content`` with the ``simple`` config. Kept
    # as a stored column (not an expression index) so keyword retrieval can
    # rank with ``ts_rank_cd`` and so the lexemes are inspectable.
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple'::regconfig, content)", persisted=True),
        nullable=False,
    )
    # Both are optional: DOCX has sections but no fixed pages, plain text has
    # neither, PDF pages have a page but usually no section path.
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # Denormalised from the document at ingestion time. Retrieval can enforce
    # clearance without joining documents, and both filters share one index.
    access_level: Mapped[AccessLevel] = mapped_column(
        enum_column(AccessLevel, name="access_level", length=16),
        nullable=False,
        default=AccessLevel.INTERNAL,
    )
    # Offsets into the parser's normalised text (see ``normalise_blocks``).
    # Nullable so rows from Phase 1 remain valid after the migration.
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), nullable=False
    )


class AuditLog(Base):
    """Immutable audit trail for queries, retrieved chunks, answers, and guardrails."""

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_tenant_created_at", "tenant_id", "created_at"),
        Index("ix_audit_logs_request_id", "request_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    raw_question_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    clearance: Mapped[AccessLevel] = mapped_column(
        enum_column(AccessLevel, name="access_level", length=16),
        nullable=False,
        default=AccessLevel.INTERNAL,
    )
    retrieved_chunk_ids: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False, default=list
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False, default=list
    )
    guardrail_action: Mapped[str] = mapped_column(String(64), nullable=False, default="allowed")
    guardrail_details: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False, default=dict
    )
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), nullable=False
    )
