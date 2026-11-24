"""Job execution: claim → read → parse → chunk → embed → upsert → persist.

Transaction boundaries are deliberate:

* Claiming commits immediately, so a long parse never holds a row lock and
  other workers can pick up other jobs.
* Parsing and embedding happen entirely outside a transaction (CPU/network work
  has no business holding a database connection).
* Vectors are written **before** the PostgreSQL completion transaction. A
  document is only marked ``indexed`` once its vectors are durable, and a
  vector-store failure leaves the previous chunks/vectors in place so the job
  can retry. The alternative (commit first, repair later) marks a document
  searchable when it is not.
* The final write is one transaction, so chunks, document status and job status
  move together.

Cross-store writes cannot be atomic. The failure window is therefore explicit:
if the database commit fails after vectors were upserted, the job retries and
the deterministic ids make the retry an idempotent replacement. A stale-vector
sweep removes ids that the new chunk set no longer contains.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from uuid import UUID

from rag_engine.core.config import Settings
from rag_engine.core.db import DatabaseProtocol
from rag_engine.core.enums import AccessLevel
from rag_engine.core.ids import chunk_vector_id
from rag_engine.core.models import Document
from rag_engine.embedding import Embedder
from rag_engine.ingestion import repository
from rag_engine.ingestion.blobstore import BlobStore, verify_content
from rag_engine.ingestion.chunking import ChunkDraft, ChunkingStrategy
from rag_engine.ingestion.errors import IngestionError
from rag_engine.ingestion.parsers import ParserRegistry
from rag_engine.vectorstore import VectorRecord, VectorStore, metadata_for_vector
from rag_engine.vectorstore.maintenance import delete_stale_vectors

logger = logging.getLogger(__name__)

_MAX_STORED_ERROR_CHARS = 1000


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    processed: bool
    job_id: UUID | None = None
    succeeded: bool = False
    chunk_count: int = 0


@dataclass(frozen=True, slots=True)
class Extraction:
    """Everything the write transaction needs, produced outside it."""

    drafts: list[ChunkDraft]
    access_level: AccessLevel
    vectors: list[list[float]] = field(default_factory=list)


class IngestionProcessor:
    def __init__(
        self,
        *,
        database: DatabaseProtocol,
        blob_store: BlobStore,
        parsers: ParserRegistry,
        chunker: ChunkingStrategy,
        embedder: Embedder,
        vector_store: VectorStore,
        settings: Settings,
    ) -> None:
        self._database = database
        self._blob_store = blob_store
        self._parsers = parsers
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._settings = settings

    async def process_next(self) -> ProcessOutcome:
        """Process at most one ready job. Returns ``processed=False`` if idle."""
        async with self._database.session() as session:
            abandoned = await repository.fail_abandoned_jobs(
                session,
                visibility_timeout_seconds=self._settings.worker_visibility_timeout_seconds,
            )
            job = await repository.claim_next_job(
                session,
                visibility_timeout_seconds=self._settings.worker_visibility_timeout_seconds,
            )
            await session.commit()

        if abandoned:
            logger.warning("ingestion.jobs_abandoned", extra={"count": abandoned})
        if job is None:
            return ProcessOutcome(processed=False)

        logger.info(
            "ingestion.job_started",
            extra={
                "job_id": str(job.id),
                "document_id": str(job.document_id),
                "attempt": job.attempts,
                "max_attempts": job.max_attempts,
            },
        )
        started = time.perf_counter()

        try:
            extraction = await self._extract(job.document_id)
            namespace = str(job.tenant_id)

            # Previous ids come from the chunks currently committed. They let
            # us delete a shrinking document's tail immediately, without
            # depending on list consistency in the vector backend.
            async with self._database.session() as session:
                previous_vector_ids = await repository.get_chunk_vector_ids(
                    session, document_id=job.document_id
                )

            records = self._vector_records(job.document_id, job.tenant_id, extraction)
            new_vector_ids = [record.id for record in records]

            # Vectors first: a document is never marked indexed without them.
            await self._vector_store.upsert(namespace, records)
            await delete_stale_vectors(
                self._vector_store,
                namespace,
                job.document_id,
                keep_ids=new_vector_ids,
                previous_ids=previous_vector_ids,
            )

            async with self._database.session() as session:
                await repository.complete_job(
                    session,
                    job_id=job.id,
                    document_id=job.document_id,
                    tenant_id=job.tenant_id,
                    drafts=extraction.drafts,
                    access_level=extraction.access_level,
                )
                await session.commit()
        except Exception as exc:
            # Any failure before the commit leaves the previous chunks intact
            # and the job retryable. Deterministic ids make the retry replace
            # vectors rather than duplicate them.
            await self._record_failure(job.id, exc)
            logger.error(
                "ingestion.job_failed",
                extra={
                    "job_id": str(job.id),
                    "document_id": str(job.document_id),
                    "attempt": job.attempts,
                    "duration_ms": _elapsed_ms(started),
                },
                exc_info=True,
            )
            return ProcessOutcome(processed=True, job_id=job.id, succeeded=False)

        logger.info(
            "ingestion.job_completed",
            extra={
                "job_id": str(job.id),
                "document_id": str(job.document_id),
                "chunks": len(extraction.drafts),
                "vectors": len(new_vector_ids),
                "duration_ms": _elapsed_ms(started),
            },
        )
        return ProcessOutcome(
            processed=True,
            job_id=job.id,
            succeeded=True,
            chunk_count=len(extraction.drafts),
        )

    async def _extract(self, document_id: UUID) -> Extraction:
        # Snapshot the fields we need, then release the connection before the
        # expensive work starts.
        async with self._database.session() as session:
            document = await session.get(Document, document_id)
            if document is None:
                raise IngestionError("document no longer exists")
            filename, storage_key, expected_hash, access_level = (
                document.filename,
                document.storage_key,
                document.content_hash,
                document.access_level,
            )

        data = await self._blob_store.get(storage_key)
        verify_content(data, expected_hash)

        parsed = self._parsers.parser_for(filename).parse(data)
        drafts = self._chunker.chunk(parsed)
        if not drafts:
            raise IngestionError("document produced no chunks")

        # Embed all chunk texts.
        texts = [draft.content for draft in drafts]
        vectors = await self._embedder.embed(texts)

        return Extraction(drafts=drafts, access_level=access_level, vectors=vectors)

    def _vector_records(
        self, document_id: UUID, tenant_id: UUID, extraction: Extraction
    ) -> list[VectorRecord]:
        """Pair each draft with its embedding and the metadata retrieval needs."""
        if len(extraction.vectors) != len(extraction.drafts):
            raise IngestionError(
                f"embedder returned {len(extraction.vectors)} vectors for "
                f"{len(extraction.drafts)} chunks"
            )

        records: list[VectorRecord] = []
        for index, (draft, vector) in enumerate(
            zip(extraction.drafts, extraction.vectors, strict=True)
        ):
            if len(vector) != self._embedder.dimensions:
                raise IngestionError(
                    f"embedding {index} has {len(vector)} dimensions, "
                    f"expected {self._embedder.dimensions}"
                )
            records.append(
                VectorRecord(
                    id=chunk_vector_id(document_id, index),
                    values=vector,
                    metadata=metadata_for_vector(
                        {
                            "document_id": str(document_id),
                            "tenant_id": str(tenant_id),
                            "chunk_index": index,
                            "access_level": str(extraction.access_level),
                            "page_start": draft.page_start,
                            "page_end": draft.page_end,
                            "section_path": draft.section_path,
                            "start_char": draft.start_char,
                            "end_char": draft.end_char,
                        }
                    ),
                )
            )
        return records

    async def aclose(self) -> None:
        """Release embedding and vector-store clients on worker shutdown.

        Each client is closed independently and failures are logged rather
        than raised: shutdown must not fail because a socket is already gone,
        and one broken client must not prevent the other from closing.
        """
        for component, close in (
            ("embedder", self._embedder.close),
            ("vector_store", self._vector_store.close),
        ):
            try:
                await close()
            except Exception:
                logger.warning(
                    "ingestion.close_failed", extra={"component": component}, exc_info=True
                )

    async def _record_failure(self, job_id: UUID, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"[:_MAX_STORED_ERROR_CHARS]
        try:
            async with self._database.session() as session:
                terminal = await repository.fail_job(
                    session,
                    job_id=job_id,
                    error=error,
                    retry_base_seconds=self._settings.worker_retry_base_seconds,
                    retry_max_seconds=self._settings.worker_retry_max_seconds,
                )
                await session.commit()
        except Exception:  # pragma: no cover - database down while recording failure
            logger.exception("ingestion.failure_not_recorded", extra={"job_id": str(job_id)})
            return

        logger.info(
            "ingestion.job_retry_scheduled" if not terminal else "ingestion.job_exhausted",
            extra={"job_id": str(job_id), "error": error},
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
