"""Integration fixtures: real PostgreSQL, real filesystem, real worker code.

Design notes:

* Migrations run once per session through Alembic, exactly as they do in
  production. Tests never create tables with ``metadata.create_all``, so a
  missing migration fails the suite instead of going unnoticed.
* Every test starts from truncated tables. ``TRUNCATE ... CASCADE`` keeps the
  reset fast and dependency-aware.
* The ASGI app is exercised through its real lifespan, so the same wiring used
  by Compose is under test. Only cross-process concerns (a second worker
  process) are out of scope.
* Without ``RAG_TEST_DATABASE_URL`` the whole directory is skipped, leaving the
  unit suite hermetic.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from rag_engine.api.app import create_app
from rag_engine.core.config import Environment, Settings
from rag_engine.core.db import PostgresDatabase
from rag_engine.core.models import Tenant
from rag_engine.embedding import Embedder, build_embedder
from rag_engine.ingestion.blobstore import LocalBlobStore
from rag_engine.ingestion.chunking import build_chunker
from rag_engine.ingestion.parsers import ParserRegistry
from rag_engine.ingestion.processor import IngestionProcessor
from rag_engine.vectorstore import MemoryVectorStore

DATABASE_URL = os.getenv("RAG_TEST_DATABASE_URL")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_TABLES = "tenants, users, documents, chunks, ingestion_jobs"


@pytest.fixture(scope="session", autouse=True)
def _require_database() -> None:
    if DATABASE_URL is None:
        pytest.skip(
            "integration tests need a PostgreSQL instance; set RAG_TEST_DATABASE_URL "
            "to an asyncpg URL (see README)"
        )


@pytest.fixture(scope="session")
def database_url() -> str:
    assert DATABASE_URL is not None
    return DATABASE_URL


@pytest.fixture(scope="session", autouse=True)
def _migrated_database(database_url: str) -> Iterator[None]:
    """Apply migrations once, the same way the ``migrate`` service does."""
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    yield


@pytest.fixture(scope="session")
def test_settings(database_url: str, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        environment=Environment.TEST,
        log_level="WARNING",
        database_url=database_url,
        blob_store_path=tmp_path_factory.mktemp("uploads"),
        _env_file=None,
    )


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(database_url, poolclass=NullPool)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture(autouse=True)
async def _truncate_tables(engine: AsyncEngine) -> AsyncIterator[None]:
    async with engine.begin() as connection:
        await connection.execute(text(f"TRUNCATE {_TABLES} CASCADE"))
    yield


@pytest.fixture
async def database(test_settings: Settings) -> AsyncIterator[PostgresDatabase]:
    """One database handle per test; the fixture, not the app, owns closing it.

    The app also disposes it on lifespan shutdown. Disposing twice is a no-op,
    and owning it here covers tests that exercise the worker without an app.
    """
    database = PostgresDatabase(test_settings)
    try:
        yield database
    finally:
        await database.dispose()


@pytest.fixture
def storage(tmp_path: Path) -> LocalBlobStore:
    return LocalBlobStore(tmp_path / "uploads")


@pytest.fixture
def vector_store(test_settings: Settings) -> MemoryVectorStore:
    return MemoryVectorStore(dimension=test_settings.embedding_dimensions)


@pytest.fixture
def embedder(test_settings: Settings) -> Embedder:
    return build_embedder(test_settings)


@pytest.fixture
async def app(
    test_settings: Settings,
    database: PostgresDatabase,
    storage: LocalBlobStore,
    vector_store: MemoryVectorStore,
    embedder: Embedder,
) -> AsyncIterator[Any]:
    """App with its real lifespan running, exactly as the server starts it."""
    application = create_app(
        test_settings,
        database=database,
        blob_store=storage,
        vector_store=vector_store,
        embedder=embedder,
    )
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def api_client(app: Any) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
async def api_client_factory(
    database: PostgresDatabase,
    storage: LocalBlobStore,
    vector_store: MemoryVectorStore,
    embedder: Embedder,
) -> Any:
    """Build a client with non-default settings (e.g. a tiny upload limit)."""

    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[AsyncClient]:
        application = create_app(
            settings,
            database=database,
            blob_store=storage,
            vector_store=vector_store,
            embedder=embedder,
        )
        async with application.router.lifespan_context(application):
            transport = ASGITransport(app=application)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                yield client

    return factory


@pytest.fixture
async def create_tenant(database: PostgresDatabase) -> Any:
    async def create(slug: str, name: str | None = None) -> Tenant:
        tenant = Tenant(slug=slug, name=name or slug.title())
        async with database.session() as session:
            session.add(tenant)
            await session.commit()
        return tenant

    return create


@pytest.fixture
async def tenant(create_tenant: Any) -> Tenant:
    return await create_tenant("acme", "Acme Corporation")


@pytest.fixture
def auth_headers(tenant: Tenant) -> dict[str, str]:
    return {"X-Tenant-ID": tenant.slug}


@pytest.fixture
async def processor(
    test_settings: Settings,
    database: PostgresDatabase,
    storage: LocalBlobStore,
    vector_store: MemoryVectorStore,
    embedder: Embedder,
) -> IngestionProcessor:
    return IngestionProcessor(
        database=database,
        blob_store=storage,
        parsers=ParserRegistry.default(),
        chunker=build_chunker(test_settings),
        embedder=embedder,
        vector_store=vector_store,
        settings=test_settings,
    )
