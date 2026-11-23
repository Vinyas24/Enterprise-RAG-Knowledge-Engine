"""PostgreSQL access: engine, session factory and a bounded health probe."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag_engine.core.config import Settings


class DatabaseProtocol(Protocol):
    """Contract the API and worker depend on.

    Depending on the protocol rather than the concrete class lets tests inject
    a fake with no network access, and lets the implementation change later.
    """

    def session(self) -> AbstractAsyncContextManager[AsyncSession]:
        """Open a session; closing it rolls back anything uncommitted."""
        ...

    async def ping(self) -> float:
        """Round-trip to the database; returns latency in milliseconds."""
        ...

    async def dispose(self) -> None:
        """Release pooled connections."""
        ...


class PostgresDatabase:
    """Async SQLAlchemy engine wrapper."""

    def __init__(self, settings: Settings) -> None:
        self._engine: AsyncEngine = create_async_engine(
            settings.database_url,
            echo=settings.database_echo,
            pool_size=settings.database_pool_size,
            pool_timeout=settings.database_pool_timeout_seconds,
            pool_pre_ping=True,
            connect_args={
                "timeout": settings.database_connect_timeout_seconds,
                # Shows up in pg_stat_activity; makes connection attribution
                # possible when several services share one database.
                "server_settings": {"application_name": settings.app_name},
            },
        )
        # expire_on_commit=False: objects stay readable after commit, which
        # avoids a hidden refresh round-trip when serialising a response.
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._session_factory() as session:
            yield session

    async def ping(self) -> float:
        """Execute ``SELECT 1`` and return the measured round-trip time."""
        started = time.perf_counter()
        async with self._engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return (time.perf_counter() - started) * 1000

    async def dispose(self) -> None:
        await self._engine.dispose()
