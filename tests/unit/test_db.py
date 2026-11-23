"""Database plumbing that can be verified without a live server."""

from __future__ import annotations

from rag_engine.core.config import Settings
from rag_engine.core.db import PostgresDatabase


async def test_construction_does_not_open_a_connection(settings: Settings) -> None:
    """Startup must not depend on the database being up.

    The engine is lazy, so the API can boot, serve ``/health/live`` and report
    ``/health/ready`` as degraded while PostgreSQL is still starting.
    """
    database = PostgresDatabase(settings)

    await database.dispose()
