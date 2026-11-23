"""Health probes: shape, dependency reporting and failure handling."""

from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_engine import __version__
from rag_engine.api.app import create_app
from rag_engine.core.config import Settings
from tests.conftest import FakeDatabase


def test_liveness_does_not_touch_the_database(database: FakeDatabase, client: TestClient) -> None:
    database.error = RuntimeError("database is down")

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["cache-control"] == "no-store"


def test_readiness_reports_healthy_database(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["environment"] == "test"
    assert body["checks"]["database"] == {"status": "ok", "latency_ms": 1.0, "detail": None}


def test_readiness_reports_unreachable_database(settings: Settings) -> None:
    database = FakeDatabase(error=RuntimeError("could not connect to postgresql://user:hunter2@db"))
    app = create_app(settings, database=database)

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["database"]["status"] == "error"
    assert body["checks"]["database"]["detail"] == "database unreachable"
    # Driver errors can embed credentials; they must never reach the client.
    assert "hunter2" not in response.text


def test_readiness_is_bounded_by_the_configured_timeout(settings: Settings) -> None:
    database = FakeDatabase(delay_seconds=2.0)
    slow_settings = settings.model_copy(update={"health_check_timeout_seconds": 0.05})
    app = create_app(slow_settings, database=database)

    started = time.perf_counter()
    with TestClient(app) as client:
        response = client.get("/health/ready")
    elapsed = time.perf_counter() - started

    assert response.status_code == 503
    assert response.json()["checks"]["database"]["status"] == "error"
    assert elapsed < 1.0


def test_lifespan_disposes_the_database(database: FakeDatabase, app: FastAPI) -> None:
    with TestClient(app):
        assert database.disposed is False
    assert database.disposed is True
