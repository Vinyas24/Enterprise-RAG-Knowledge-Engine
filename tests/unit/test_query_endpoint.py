"""Unit tests for the /query HTTP endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app
from rag_engine.core.config import Settings
from rag_engine.core.enums import AccessLevel
from rag_engine.generation.base import Citation, GenerationResult
from rag_engine.generation.service import QueryEngine
from tests.conftest import FakeDatabase


def _setup_app_with_query_engine(engine: QueryEngine) -> FastAPI:
    settings = Settings(_env_file=None)
    database = FakeDatabase()
    app = create_app(settings, database=database, query_engine=engine)
    return app


def test_query_requires_tenant_header() -> None:
    engine = AsyncMock(spec=QueryEngine)
    app = _setup_app_with_query_engine(engine)

    with TestClient(app) as client:
        response = client.post("/query", json={"question": "What is RAG?"})
        assert response.status_code == 400
        assert "X-Tenant-ID header is required" in response.json()["error"]["message"]


def test_query_question_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_engine.ingestion.repository as repo
    from rag_engine.core.models import Tenant

    tenant = Tenant(id=uuid4(), slug="acme", name="Acme Inc")
    monkeypatch.setattr(repo, "get_tenant_by_slug", AsyncMock(return_value=tenant))

    engine = AsyncMock(spec=QueryEngine)
    app = _setup_app_with_query_engine(engine)

    with TestClient(app) as client:
        # Empty question
        resp = client.post("/query", headers={"X-Tenant-ID": "acme"}, json={"question": ""})
        assert resp.status_code == 422


def test_query_success_with_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = AsyncMock(spec=QueryEngine)
    doc_id = uuid4()
    chunk_id = uuid4()

    engine.answer.return_value = GenerationResult(
        answer="RAG stands for Retrieval-Augmented Generation [1].",
        citations=[
            Citation(
                document_id=doc_id,
                document_title="Glossary",
                chunk_id=chunk_id,
                chunk_index=0,
                snippet="RAG stands for Retrieval-Augmented Generation.",
                page_start=1,
                page_end=1,
                section_path="Definitions",
            )
        ],
        weak_retrieval=False,
    )

    app = _setup_app_with_query_engine(engine)

    # Mock get_tenant_by_slug to return a dummy tenant
    from rag_engine.core.models import Tenant

    tenant = Tenant(id=uuid4(), slug="acme", name="Acme Inc")

    import rag_engine.ingestion.repository as repo

    monkeypatch.setattr(repo, "get_tenant_by_slug", AsyncMock(return_value=tenant))

    with TestClient(app) as client:
        response = client.post(
            "/query",
            headers={"X-Tenant-ID": "acme"},
            json={"question": "What does RAG stand for?", "top_k": 3, "clearance": "internal"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What does RAG stand for?"
        assert data["answer"] == "RAG stands for Retrieval-Augmented Generation [1]."
        assert data["weak_retrieval"] is False
        assert len(data["citations"]) == 1
        citation = data["citations"][0]
        assert citation["document_title"] == "Glossary"
        assert citation["chunk_index"] == 0
        assert citation["page_start"] == 1
        assert citation["section_path"] == "Definitions"

    engine.answer.assert_called_once_with(
        tenant_id=tenant.id,
        question="What does RAG stand for?",
        top_k=3,
        clearance=AccessLevel.INTERNAL,
    )
