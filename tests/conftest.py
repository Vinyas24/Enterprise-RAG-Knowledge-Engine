"""Shared fixtures.

Tests never touch a real database or network: dependencies are injected at the
``create_app`` seam, which is the same seam production uses.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.api.app import create_app
from rag_engine.core.config import Environment, LogFormat, Settings
from rag_engine.core.context import reset_request_id, set_request_id


class FakeDatabase:
    """In-memory stand-in for :class:`rag_engine.core.db.DatabaseProtocol`."""

    def __init__(
        self,
        *,
        latency_ms: float = 1.0,
        error: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.latency_ms = latency_ms
        self.error = error
        self.delay_seconds = delay_seconds
        self.disposed = False

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        mock_session = AsyncMock(spec=AsyncSession)
        yield mock_session

    async def ping(self) -> float:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error
        return self.latency_ms

    async def dispose(self) -> None:
        self.disposed = True


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from developer ``RAG_*`` environment variables."""
    for key in list(os.environ):
        if key.startswith("RAG_") and key != "RAG_TEST_DATABASE_URL":
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Keep ``configure_logging`` calls in one test from leaking into others."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def bound_request_id() -> Iterator[str]:
    """Bind a request id to the current task for the duration of a test."""
    token = set_request_id("test-request-id")
    yield "test-request-id"
    reset_request_id(token)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment=Environment.TEST,
        log_level="DEBUG",
        log_format=LogFormat.JSON,
        database_url="postgresql+asyncpg://test:test@localhost:5432/test",
        _env_file=None,
    )


@pytest.fixture
def database() -> FakeDatabase:
    return FakeDatabase()


@pytest.fixture
def app(settings: Settings, database: FakeDatabase) -> FastAPI:
    return create_app(settings, database=database)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
