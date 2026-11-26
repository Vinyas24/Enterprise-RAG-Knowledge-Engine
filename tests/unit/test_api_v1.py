"""Unit tests for API v1 routing, OpenAPI docs, and version discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app
from rag_engine.core.config import Settings
from rag_engine.core.models import Tenant
from rag_engine.generation.base import GenerationResult
from rag_engine.generation.service import QueryEngine
from tests.conftest import FakeDatabase


def test_v1_routes_exposed_in_openapi() -> None:
    settings = Settings(_env_file=None)
    app = create_app(settings, database=FakeDatabase())

    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]

        # Both v1 and unversioned routes exist for smooth client transition
        assert "/v1/query" in paths
        assert "/v1/documents" in paths
        assert "/v1/documents/{document_id}" in paths
        assert "/v1/ingestion-jobs" in paths
        assert "/v1/ingestion-jobs/{job_id}" in paths
        assert "/v1/metrics" in paths

        assert "/query" in paths
        assert "/documents" in paths
        assert "/metrics" in paths


def test_service_info_declares_v1_version() -> None:
    settings = Settings(_env_file=None)
    app = create_app(settings, database=FakeDatabase())

    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "v1" in body["api_versions"]
        assert body["current_version"] == "v1"
        assert body["metrics_url"] == "/metrics"


def test_v1_query_execution_and_timings(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_engine.ingestion.repository as repo

    tenant = Tenant(id=uuid4(), slug="acme", name="Acme Inc")
    monkeypatch.setattr(repo, "get_tenant_by_slug", AsyncMock(return_value=tenant))

    mock_engine = AsyncMock(spec=QueryEngine)
    mock_engine.answer.return_value = GenerationResult(
        answer="Version 1 response.",
        citations=[],
        weak_retrieval=False,
        timings={"total": 24.5, "retrieve": 12.1, "generate": 12.4},
    )

    settings = Settings(server_timing_enabled=True, _env_file=None)
    app = create_app(settings, database=FakeDatabase(), query_engine=mock_engine)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/query",
            headers={"X-Tenant-ID": "acme"},
            json={"question": "Testing v1 query"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Version 1 response."
        assert data["timings"] is not None
        assert data["timings"]["total_ms"] == 24.5
        assert data["timings"]["retrieve_ms"] == 12.1


def test_v1_metrics_endpoint() -> None:
    settings = Settings(_env_file=None)
    app = create_app(settings, database=FakeDatabase())

    with TestClient(app) as client:
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
