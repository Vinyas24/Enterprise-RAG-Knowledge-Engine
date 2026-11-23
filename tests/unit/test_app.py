"""App factory: metadata endpoint, docs exposure and route wiring."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_engine import __version__
from rag_engine.api.app import create_app
from rag_engine.core.config import Environment, Settings
from tests.conftest import FakeDatabase


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
