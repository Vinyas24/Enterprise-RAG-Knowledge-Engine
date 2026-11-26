"""Unit tests for basic API authentication."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app
from rag_engine.core.config import Environment, Settings
from rag_engine.core.models import Tenant
from rag_engine.generation.base import GenerationResult
from rag_engine.generation.service import QueryEngine
from tests.conftest import FakeDatabase


def _make_app(auth_enabled: bool = False, api_key: str | None = None) -> TestClient:
    settings = Settings(
        environment=Environment.TEST,
        auth_enabled=auth_enabled,
        api_key=api_key,
        admin_api_key="super-admin-key" if api_key else None,
        _env_file=None,
    )
    mock_engine = AsyncMock(spec=QueryEngine)
    mock_engine.answer.return_value = GenerationResult(
        answer="Grounded answer.",
        citations=[],
        weak_retrieval=False,
    )
    app = create_app(settings, database=FakeDatabase(), query_engine=mock_engine)
    return TestClient(app)


def test_auth_permissive_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_engine.ingestion.repository as repo

    tenant = Tenant(id=uuid4(), slug="acme", name="Acme Inc")
    monkeypatch.setattr(repo, "get_tenant_by_slug", AsyncMock(return_value=tenant))

    client = _make_app(auth_enabled=False, api_key=None)
    with client:
        # Protected endpoint without API key succeeds when auth is disabled
        resp = client.post(
            "/query",
            headers={"X-Tenant-ID": "acme"},
            json={"question": "What is RAG?"},
        )
        assert resp.status_code == 200


def test_auth_enforced_blocks_missing_key() -> None:
    client = _make_app(auth_enabled=True, api_key="test-api-key")
    with client:
        resp = client.post(
            "/query",
            headers={"X-Tenant-ID": "acme"},
            json={"question": "What is RAG?"},
        )
        assert resp.status_code == 401
        data = resp.json()
        assert data["error"]["code"] == "unauthorized"
        assert "API key is required" in data["error"]["message"]


def test_auth_enforced_blocks_invalid_key() -> None:
    client = _make_app(auth_enabled=True, api_key="test-api-key")
    with client:
        resp = client.post(
            "/query",
            headers={"X-Tenant-ID": "acme", "X-API-Key": "wrong-key"},
            json={"question": "What is RAG?"},
        )
        assert resp.status_code == 401
        data = resp.json()
        assert data["error"]["code"] == "unauthorized"
        assert data["error"]["message"] == "Invalid API key"


def test_auth_allows_valid_header_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_engine.ingestion.repository as repo

    tenant = Tenant(id=uuid4(), slug="acme", name="Acme Inc")
    monkeypatch.setattr(repo, "get_tenant_by_slug", AsyncMock(return_value=tenant))

    client = _make_app(auth_enabled=True, api_key="test-api-key")
    with client:
        resp = client.post(
            "/query",
            headers={"X-Tenant-ID": "acme", "X-API-Key": "test-api-key"},
            json={"question": "What is RAG?"},
        )
        assert resp.status_code == 200


def test_auth_allows_valid_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_engine.ingestion.repository as repo

    tenant = Tenant(id=uuid4(), slug="acme", name="Acme Inc")
    monkeypatch.setattr(repo, "get_tenant_by_slug", AsyncMock(return_value=tenant))

    client = _make_app(auth_enabled=True, api_key="test-api-key")
    with client:
        resp = client.post(
            "/query",
            headers={"X-Tenant-ID": "acme", "Authorization": "Bearer test-api-key"},
            json={"question": "What is RAG?"},
        )
        assert resp.status_code == 200


def test_auth_allows_admin_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_engine.ingestion.repository as repo

    tenant = Tenant(id=uuid4(), slug="acme", name="Acme Inc")
    monkeypatch.setattr(repo, "get_tenant_by_slug", AsyncMock(return_value=tenant))

    client = _make_app(auth_enabled=True, api_key="test-api-key")
    with client:
        resp = client.post(
            "/query",
            headers={"X-Tenant-ID": "acme", "X-API-Key": "super-admin-key"},
            json={"question": "What is RAG?"},
        )
        assert resp.status_code == 200


def test_unprotected_endpoints_accessible_without_auth() -> None:
    client = _make_app(auth_enabled=True, api_key="test-api-key")
    with client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        assert client.get("/metrics").status_code == 200
        assert client.get("/").status_code == 200
