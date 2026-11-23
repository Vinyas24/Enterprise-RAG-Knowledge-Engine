"""End-to-end ingestion behaviour against PostgreSQL and the filesystem.

These tests cover the Phase 1 acceptance criteria directly:

* an upload returns a job id (202),
* the worker completes the job in the background,
* status is queryable through the API,
* duplicate content is detected and reported with the existing document.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from docx import Document as DocxDocument
from httpx2 import AsyncClient
from sqlalchemy import func, select, update

from rag_engine.core.db import PostgresDatabase
from rag_engine.core.enums import DocumentStatus, JobStatus
from rag_engine.core.models import Chunk, Document, IngestionJob
from rag_engine.ingestion.chunking import normalise_blocks
from rag_engine.ingestion.parsers import ParserRegistry
from rag_engine.ingestion.processor import IngestionProcessor
from rag_engine.ingestion.service import IngestionService

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TXT = b"Retrieval quality depends on chunking. " * 60


async def _upload(
    client: AsyncClient,
    headers: dict[str, str],
    *,
    filename: str,
    data: bytes,
    content_type: str = "text/plain",
    **form: str,
) -> Any:
    return await client.post(
        "/documents",
        headers=headers,
        files={"file": (filename, data, content_type)},
        data=form,
    )


async def _document(database: PostgresDatabase, document_id: str | UUID) -> Document | None:
    async with database.session() as session:
        return await session.get(Document, UUID(str(document_id)))


async def _job(database: PostgresDatabase, job_id: str | UUID) -> IngestionJob | None:
    async with database.session() as session:
        return await session.get(IngestionJob, UUID(str(job_id)))


async def _chunks(database: PostgresDatabase, document_id: str | UUID) -> list[Chunk]:
    async with database.session() as session:
        result = await session.execute(
            select(Chunk)
            .where(Chunk.document_id == UUID(str(document_id)))
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_upload_returns_a_job_that_the_worker_completes(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
    database: PostgresDatabase,
) -> None:
    response = await _upload(
        api_client,
        auth_headers,
        filename="retrieval-notes.txt",
        data=TXT,
        title="Retrieval notes",
    )

    assert response.status_code == 202, response.text
    accepted = response.json()
    assert accepted["status"] == "queued"
    document_id, job_id = accepted["document_id"], accepted["job_id"]

    # Immediately after acceptance the job exists and is queryable.
    job_response = await api_client.get(f"/ingestion-jobs/{job_id}", headers=auth_headers)
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "queued"
    assert job_response.json()["document_id"] == document_id

    document_response = await api_client.get(f"/documents/{document_id}", headers=auth_headers)
    assert document_response.status_code == 200
    assert document_response.json()["status"] == "pending"

    outcome = await processor.process_next()

    assert outcome.processed is True
    assert outcome.succeeded is True
    assert outcome.job_id == UUID(job_id)
    assert outcome.chunk_count > 1

    # Nothing left to do.
    assert (await processor.process_next()).processed is False

    job_response = await api_client.get(f"/ingestion-jobs/{job_id}", headers=auth_headers)
    job = job_response.json()
    assert job["status"] == "done"
    assert job["attempts"] == 1
    assert job["error"] is None
    assert job["finished_at"] is not None

    document_response = await api_client.get(f"/documents/{document_id}", headers=auth_headers)
    document = document_response.json()
    assert document["status"] == "indexed"
    assert document["title"] == "Retrieval notes"
    assert document["chunk_count"] == outcome.chunk_count
    assert document["job"]["status"] == "done"

    chunks = await _chunks(database, document_id)
    assert len(chunks) == outcome.chunk_count
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.content.strip() for chunk in chunks)
    assert all(chunk.char_count == len(chunk.content) for chunk in chunks)


async def test_uploaded_bytes_are_stored_content_addressed(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    storage: Any,
    database: PostgresDatabase,
) -> None:
    response = await _upload(api_client, auth_headers, filename="policy.txt", data=b"policy body")

    document = await _document(database, response.json()["document_id"])
    assert document is not None
    assert document.storage_key == f"acme/{document.content_hash[:2]}/{document.content_hash}.txt"
    assert (storage.root / document.storage_key).read_bytes() == b"policy body"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


async def test_duplicate_upload_is_rejected_with_the_existing_document(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    first = await _upload(api_client, auth_headers, filename="policy.txt", data=b"same bytes")
    assert first.status_code == 202

    second = await _upload(api_client, auth_headers, filename="policy-copy.txt", data=b"same bytes")

    assert second.status_code == 409, second.text
    error = second.json()["error"]
    assert error["code"] == "duplicate_document"
    assert error["details"]["document_id"] == first.json()["document_id"]
    assert error["request_id"]

    listing = await api_client.get("/documents", headers=auth_headers)
    assert listing.json()["total"] == 1


async def test_identical_bytes_are_allowed_for_a_different_tenant(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    create_tenant: Any,
) -> None:
    other = await create_tenant("globex", "Globex")
    first = await _upload(api_client, auth_headers, filename="policy.txt", data=b"shared content")
    second = await _upload(
        api_client,
        {"X-Tenant-ID": other.slug},
        filename="policy.txt",
        data=b"shared content",
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["document_id"] != first.json()["document_id"]


# ---------------------------------------------------------------------------
# Validation at the edge
# ---------------------------------------------------------------------------


async def test_oversized_upload_is_rejected_before_it_is_stored(
    api_client_factory: Any,
    tenant: Any,
    test_settings: Any,
    database: PostgresDatabase,
) -> None:
    settings = test_settings.model_copy(update={"upload_max_bytes": 1024})

    async with api_client_factory(settings) as client:
        response = await _upload(
            client,
            {"X-Tenant-ID": tenant.slug},
            filename="big.txt",
            data=b"a" * 2048,
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"
    assert response.json()["error"]["details"] == {"max_bytes": 1024}

    async with database.session() as session:
        total = (await session.execute(select(func.count()).select_from(Document))).scalar_one()
    assert total == 0


async def test_unsupported_extension_reports_supported_types(
    api_client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await _upload(
        api_client, auth_headers, filename="sheet.xlsx", data=b"not really a spreadsheet"
    )

    assert response.status_code == 415
    error = response.json()["error"]
    assert error["code"] == "unsupported_document_type"
    assert ".pdf" in error["details"]["supported_extensions"]


async def test_empty_upload_is_rejected(
    api_client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await _upload(api_client, auth_headers, filename="empty.txt", data=b"")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "empty_document"


async def test_content_that_does_not_match_the_extension_is_rejected(
    api_client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await _upload(
        api_client,
        auth_headers,
        filename="pretend.pdf",
        data=b"plain text, not a pdf",
        content_type="application/pdf",
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "invalid_document_content"


async def test_invalid_access_level_is_a_validation_error(
    api_client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await _upload(
        api_client,
        auth_headers,
        filename="policy.txt",
        data=b"policy",
        access_level="top-secret",
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


async def test_tenant_header_is_required(api_client: AsyncClient) -> None:
    response = await api_client.get("/documents")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_tenant"


async def test_unknown_tenant_is_rejected(api_client: AsyncClient) -> None:
    response = await api_client.get("/documents", headers={"X-Tenant-ID": "nobody"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "tenant_not_found"


async def test_documents_and_jobs_are_invisible_across_tenants(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    create_tenant: Any,
) -> None:
    accepted = (
        await _upload(api_client, auth_headers, filename="secret.txt", data=b"secret bytes")
    ).json()
    other = await create_tenant("globex", "Globex")
    other_headers = {"X-Tenant-ID": other.slug}

    document = await api_client.get(f"/documents/{accepted['document_id']}", headers=other_headers)
    job = await api_client.get(f"/ingestion-jobs/{accepted['job_id']}", headers=other_headers)
    chunks = await api_client.get(
        f"/documents/{accepted['document_id']}/chunks", headers=other_headers
    )
    listing = await api_client.get("/documents", headers=other_headers)

    assert document.status_code == 404
    assert job.status_code == 404
    assert chunks.status_code == 404
    assert listing.json()["total"] == 0


async def test_unknown_document_and_job_are_404(
    api_client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    missing = "00000000-0000-4000-8000-000000000000"

    document = await api_client.get(f"/documents/{missing}", headers=auth_headers)
    job = await api_client.get(f"/ingestion-jobs/{missing}", headers=auth_headers)
    chunks = await api_client.get(f"/documents/{missing}/chunks", headers=auth_headers)

    assert document.status_code == 404
    assert document.json()["error"]["code"] == "document_not_found"
    assert job.status_code == 404
    assert job.json()["error"]["code"] == "job_not_found"
    assert chunks.status_code == 404
    assert chunks.json()["error"]["code"] == "document_not_found"


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


async def test_listing_supports_pagination_and_status_filter(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
) -> None:
    accepted = []
    for index in range(3):
        response = await _upload(
            api_client, auth_headers, filename=f"doc-{index}.txt", data=f"content {index}".encode()
        )
        accepted.append(response.json())

    first_page = await api_client.get("/documents?limit=2&offset=0", headers=auth_headers)
    second_page = await api_client.get("/documents?limit=2&offset=2", headers=auth_headers)

    assert first_page.json()["total"] == 3
    assert first_page.json()["limit"] == 2
    assert len(first_page.json()["items"]) == 2
    assert len(second_page.json()["items"]) == 1

    pending = await api_client.get("/documents?status=pending", headers=auth_headers)
    assert pending.json()["total"] == 3

    await processor.process_next()
    indexed = await api_client.get("/documents?status=indexed", headers=auth_headers)
    still_pending = await api_client.get("/documents?status=pending", headers=auth_headers)
    assert indexed.json()["total"] == 1
    assert still_pending.json()["total"] == 2


# ---------------------------------------------------------------------------
# Provenance per file type
# ---------------------------------------------------------------------------


async def test_markdown_ingestion_preserves_the_heading_path(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
    database: PostgresDatabase,
) -> None:
    markdown = b"# Deployment\n\nUse containers.\n\n## Rollback\n\nRun the previous image.\n"
    accepted = (
        await _upload(
            api_client,
            auth_headers,
            filename="runbook.md",
            data=markdown,
            content_type="text/markdown",
        )
    ).json()

    assert (await processor.process_next()).succeeded is True

    chunks = await _chunks(database, accepted["document_id"])
    paths = {chunk.section_path for chunk in chunks}
    assert "Deployment" in paths
    assert "Deployment > Rollback" in paths


async def test_pdf_ingestion_preserves_page_numbers(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
    database: PostgresDatabase,
) -> None:
    accepted = (
        await _upload(
            api_client,
            auth_headers,
            filename="simple.pdf",
            data=(FIXTURES / "simple.pdf").read_bytes(),
            content_type="application/pdf",
        )
    ).json()

    outcome = await processor.process_next()
    assert outcome.succeeded is True

    chunks = await _chunks(database, accepted["document_id"])
    assert any(chunk.page_start == 1 for chunk in chunks)
    assert any(chunk.page_start == 2 for chunk in chunks)
    assert all(chunk.page_start == chunk.page_end for chunk in chunks)


async def test_docx_ingestion_extracts_paragraphs_and_tables(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
    database: PostgresDatabase,
) -> None:
    document = DocxDocument()
    document.add_heading("Architecture", level=1)
    document.add_paragraph("The API writes blobs; the worker reads them.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Queue"
    table.cell(0, 1).text = "PostgreSQL"
    buffer = io.BytesIO()
    document.save(buffer)

    accepted = (
        await _upload(
            api_client,
            auth_headers,
            filename="architecture.docx",
            data=buffer.getvalue(),
            content_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
        )
    ).json()

    assert (await processor.process_next()).succeeded is True

    chunks = await _chunks(database, accepted["document_id"])
    rendered = "\n".join(chunk.content for chunk in chunks)
    assert "The API writes blobs" in rendered
    assert "Queue | PostgreSQL" in rendered
    assert any(chunk.section_path == "Architecture" for chunk in chunks)


# ---------------------------------------------------------------------------
# Retry, terminal failure and abandoned work
# ---------------------------------------------------------------------------


async def test_transient_failure_is_retried_and_can_recover(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
    storage: Any,
    database: PostgresDatabase,
) -> None:
    accepted = (
        await _upload(api_client, auth_headers, filename="recover.txt", data=b"recoverable")
    ).json()
    document = await _document(database, accepted["document_id"])
    assert document is not None
    blob = storage.root / document.storage_key
    blob.unlink()

    first = await processor.process_next()
    assert first.processed is True
    assert first.succeeded is False

    job = await _job(database, accepted["job_id"])
    assert job is not None
    assert job.status is JobStatus.QUEUED
    assert job.attempts == 1
    assert job.error is not None and "FileNotFoundError" in job.error
    assert job.available_at > datetime.now(UTC)

    # Simulate the backoff having elapsed and the storage recovering.
    await storage.put(document.storage_key, b"recoverable")
    async with database.session() as session:
        await session.execute(
            update(IngestionJob)
            .where(IngestionJob.id == job.id)
            .values(available_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        await session.commit()

    second = await processor.process_next()
    assert second.succeeded is True

    job = await _job(database, accepted["job_id"])
    assert job is not None
    assert job.status is JobStatus.DONE
    assert job.attempts == 2
    assert job.error is None

    document = await _document(database, accepted["document_id"])
    assert document is not None
    assert document.status is DocumentStatus.INDEXED


async def test_failure_is_terminal_after_the_attempt_budget(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
    storage: Any,
    database: PostgresDatabase,
) -> None:
    accepted = (
        await _upload(api_client, auth_headers, filename="doomed.txt", data=b"doomed")
    ).json()
    document = await _document(database, accepted["document_id"])
    assert document is not None
    (storage.root / document.storage_key).unlink()

    async with database.session() as session:
        await session.execute(
            update(IngestionJob)
            .where(IngestionJob.id == UUID(accepted["job_id"]))
            .values(max_attempts=1)
        )
        await session.commit()

    assert (await processor.process_next()).succeeded is False

    job = await _job(database, accepted["job_id"])
    assert job is not None
    assert job.status is JobStatus.FAILED
    assert job.finished_at is not None

    document = await _document(database, accepted["document_id"])
    assert document is not None
    assert document.status is DocumentStatus.FAILED


async def test_abandoned_job_is_failed_when_retries_are_exhausted(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
    database: PostgresDatabase,
) -> None:
    accepted = (
        await _upload(api_client, auth_headers, filename="abandoned.txt", data=b"abandoned")
    ).json()

    async with database.session() as session:
        job = await session.get(IngestionJob, UUID(accepted["job_id"]))
        assert job is not None
        job.status = JobStatus.PROCESSING
        job.attempts = job.max_attempts
        job.started_at = datetime.now(UTC) - timedelta(hours=1)
        await session.commit()

    outcome = await processor.process_next()

    assert outcome.processed is False
    job = await _job(database, accepted["job_id"])
    assert job is not None
    assert job.status is JobStatus.FAILED
    assert "stopped responding" in (job.error or "")


async def test_stale_processing_job_is_reclaimed(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
    database: PostgresDatabase,
) -> None:
    accepted = (
        await _upload(api_client, auth_headers, filename="stale.txt", data=b"stale work")
    ).json()

    async with database.session() as session:
        job = await session.get(IngestionJob, UUID(accepted["job_id"]))
        assert job is not None
        job.status = JobStatus.PROCESSING
        job.attempts = 1
        job.started_at = datetime.now(UTC) - timedelta(hours=1)
        await session.commit()

    outcome = await processor.process_next()

    assert outcome.succeeded is True
    assert outcome.job_id == UUID(accepted["job_id"])


# ---------------------------------------------------------------------------
# Access level is persisted, not just accepted
# ---------------------------------------------------------------------------


async def test_access_level_is_persisted(
    api_client: AsyncClient, auth_headers: dict[str, str], database: PostgresDatabase
) -> None:
    accepted = (
        await _upload(
            api_client,
            auth_headers,
            filename="restricted.txt",
            data=b"restricted bytes",
            access_level="confidential",
        )
    ).json()

    document = await _document(database, accepted["document_id"])
    assert document is not None
    assert document.access_level.value == "confidential"

    response = await api_client.get(f"/documents/{accepted['document_id']}", headers=auth_headers)
    assert response.json()["access_level"] == "confidential"


async def test_ingestion_service_is_wired_into_the_app(app: Any) -> None:
    assert isinstance(app.state.ingestion_service, IngestionService)


async def test_chunks_are_traceable_and_expose_access_metadata(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
    database: PostgresDatabase,
    tenant: Any,
) -> None:
    accepted = (
        await _upload(
            api_client,
            auth_headers,
            filename="traceable.txt",
            data=TXT,
            access_level="confidential",
        )
    ).json()
    assert (await processor.process_next()).succeeded is True

    response = await api_client.get(
        f"/documents/{accepted['document_id']}/chunks", headers=auth_headers
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    items = payload["items"]

    assert payload["limit"] == 50
    assert payload["offset"] == 0
    assert payload["total"] == len(items)
    assert len(items) > 1
    assert [item["chunk_index"] for item in items] == list(range(len(items)))

    # Offsets are the contract that makes a citation verifiable: slicing the
    # parser's normalised text at the reported span reproduces the chunk.
    parsed = ParserRegistry.default().parser_for("traceable.txt").parse(TXT)
    normalised = normalise_blocks(parsed.blocks)
    for item in items:
        assert item["document_id"] == accepted["document_id"]
        assert item["tenant_id"] == str(tenant.id)
        assert item["access_level"] == "confidential"
        assert item["start_char"] is not None
        assert item["end_char"] is not None
        assert item["end_char"] > item["start_char"]
        assert item["end_char"] - item["start_char"] == item["char_count"]
        assert normalised[item["start_char"] : item["end_char"]] == item["content"]

    # Adjacent chunks share context, which is what the overlap setting buys.
    assert any(
        current["start_char"] < previous["end_char"] for previous, current in pairwise(items)
    )

    page = await api_client.get(
        f"/documents/{accepted['document_id']}/chunks",
        headers=auth_headers,
        params={"limit": 1, "offset": 1},
    )
    assert page.status_code == 200
    assert page.json()["total"] == payload["total"]
    assert [item["chunk_index"] for item in page.json()["items"]] == [1]
