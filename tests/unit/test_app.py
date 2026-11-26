"""App factory: metadata endpoint, docs exposure and route wiring."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_engine import __version__
from rag_engine.api import app as app_module
from rag_engine.api.app import create_app
from rag_engine.core.config import Environment, Settings
from rag_engine.ranking import RerankedRetriever, RerankUnavailableError
from rag_engine.retrieval.base import RetrievedChunk
from tests.conftest import FakeDatabase


class StubRetriever:
    """Minimal retriever for the injection seam."""

    async def retrieve(self, **_kwargs: Any) -> list[RetrievedChunk]:
        return []


def test_root_describes_the_service(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "rag-engine"
    assert body["version"] == __version__
    assert body["environment"] == "test"
    assert body["docs_url"] == "/docs"
    assert body["health"] == {"liveness": "/health/live", "readiness": "/health/ready"}


def test_openapi_lists_the_public_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/health/live" in paths
    assert "/health/ready" in paths
    assert "/documents" in paths
    assert "/documents/{document_id}" in paths


def test_production_hides_docs_and_openapi() -> None:
    settings = Settings(environment=Environment.PRODUCTION, _env_file=None)
    app = create_app(settings, database=FakeDatabase())

    with TestClient(app) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        root = client.get("/").json()

    assert root["docs_url"] is None


def test_app_version_matches_package_version(client: TestClient) -> None:
    assert client.get("/").json()["version"] == __version__


def test_reranker_wraps_the_retriever_when_configured() -> None:
    settings = Settings(
        environment=Environment.TEST,
        rerank_provider="lexical",
        _env_file=None,
    )
    app = create_app(settings, database=FakeDatabase())

    with TestClient(app) as started:
        assert isinstance(started.app.state.retriever, RerankedRetriever)
        # The API still answers while the wrapper is in place.
        assert started.get("/health/live").status_code == 200


def test_injected_retriever_is_not_wrapped_by_a_reranker() -> None:
    """The injected seam must stay the object under test."""
    settings = Settings(
        environment=Environment.TEST,
        rerank_provider="lexical",
        _env_file=None,
    )
    injected = StubRetriever()
    app = create_app(settings, database=FakeDatabase(), retriever=injected)

    with TestClient(app) as started:
        assert started.app.state.retriever is injected


def test_a_reranker_that_cannot_warm_up_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing model or extra is a configuration error, not a silent fallback."""

    class Unavailable:
        name = "unavailable"

        async def warmup(self) -> None:
            raise RerankUnavailableError("model is not installed")

        async def rerank(self, **_kwargs: object) -> list[RetrievedChunk]:
            raise AssertionError("should not be called")

        async def close(self) -> None:
            pass

    monkeypatch.setattr(app_module, "build_reranker", lambda _settings: Unavailable())
    settings = Settings(
        environment=Environment.TEST,
        rerank_provider="lexical",
        _env_file=None,
    )
    app = create_app(settings, database=FakeDatabase())

    with pytest.raises(RerankUnavailableError, match="model is not installed"), TestClient(app):
        pass
