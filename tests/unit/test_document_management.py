"""Unit tests for document and ingestion job management endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app
from rag_engine.core.config import Settings
from rag_engine.core.enums import AccessLevel, DocumentStatus, JobStatus
from rag_engine.core.models import Document, IngestionJob, Tenant
from tests.conftest import FakeDatabase


def _setup_app() -> TestClient:
    settings = Settings(_env_file=None)
    app = create_app(settings, database=FakeDatabase())
    return TestClient(app)


def test_list_documents_with_access_level_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_engine.ingestion.repository as repo

    tenant = Tenant(id=uuid4(), slug="acme", name="Acme Inc")
    monkeypatch.setattr(repo, "get_tenant_by_slug", AsyncMock(return_value=tenant))

    doc = Document(
        id=uuid4(),
        tenant_id=tenant.id,
        title="Restricted Doc",
        filename="doc.pdf",
        status=DocumentStatus.INDEXED,
        access_level=AccessLevel.RESTRICTED,
        size_bytes=1024,
        chunk_count=5,
        content_hash="abc",
        storage_key="path",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    mock_list = AsyncMock(return_value=[doc])
    mock_count = AsyncMock(return_value=1)
    monkeypatch.setattr(repo, "list_documents", mock_list)
    monkeypatch.setattr(repo, "count_documents", mock_count)

    client = _setup_app()
    with client:
        resp = client.get(
            "/documents",
            headers={"X-Tenant-ID": "acme"},
            params={"access_level": "restricted", "limit": 10, "offset": 0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["access_level"] == "restricted"
        assert mock_list.call_args.kwargs["access_level"] == AccessLevel.RESTRICTED


def test_list_ingestion_jobs_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_engine.ingestion.repository as repo

    tenant = Tenant(id=uuid4(), slug="acme", name="Acme Inc")
    monkeypatch.setattr(repo, "get_tenant_by_slug", AsyncMock(return_value=tenant))

    job = IngestionJob(
        id=uuid4(),
        document_id=uuid4(),
        tenant_id=tenant.id,
        status=JobStatus.DONE,
        attempts=1,
        max_attempts=3,
        error=None,
        available_at=datetime.now(UTC),
        created_at=datetime.now(UTC),
    )

    mock_list_jobs = AsyncMock(return_value=[job])
    mock_count_jobs = AsyncMock(return_value=1)
    monkeypatch.setattr(repo, "list_jobs", mock_list_jobs)
    monkeypatch.setattr(repo, "count_jobs", mock_count_jobs)

    client = _setup_app()
    with client:
        resp = client.get(
            "/ingestion-jobs",
            headers={"X-Tenant-ID": "acme"},
            params={"limit": 5, "offset": 0, "status": "done"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["status"] == "done"
        assert mock_list_jobs.call_args.kwargs["status"] == JobStatus.DONE


def test_get_ingestion_job_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_engine.ingestion.repository as repo

    tenant = Tenant(id=uuid4(), slug="acme", name="Acme Inc")
    monkeypatch.setattr(repo, "get_tenant_by_slug", AsyncMock(return_value=tenant))
    monkeypatch.setattr(repo, "get_job", AsyncMock(return_value=None))

    client = _setup_app()
    with client:
        resp = client.get(f"/ingestion-jobs/{uuid4()}", headers={"X-Tenant-ID": "acme"})
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "job_not_found"
