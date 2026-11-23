"""Integration tests against a real PostgreSQL instance.

Run automatically in CI (a service container is provisioned) and locally via
``docker compose up -d postgres && make test-integration``. Without
``RAG_TEST_DATABASE_URL`` the module is skipped, so the unit suite stays
hermetic.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app
from rag_engine.core.config import Environment, Settings
from rag_engine.core.db import PostgresDatabase

DATABASE_URL = os.getenv("RAG_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(DATABASE_URL is None, reason="RAG_TEST_DATABASE_URL is not set"),
]


def _settings() -> Settings:
    assert DATABASE_URL is not None
    return Settings(environment=Environment.TEST, database_url=DATABASE_URL, _env_file=None)


async def test_ping_round_trip() -> None:
    database = PostgresDatabase(_settings())
    try:
        latency_ms = await database.ping()
    finally:
        await database.dispose()

    assert latency_ms > 0


def test_readiness_reports_a_real_database() -> None:
    app = create_app(_settings())

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"]["status"] == "ok"
    assert body["checks"]["database"]["latency_ms"] is not None
