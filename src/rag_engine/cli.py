"""Operator CLI.

Tenants have no self-service signup yet, so creating one is an explicit
administrative action::

    rag-engine create-tenant --slug acme --name "Acme Corp"
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from rag_engine.core.config import get_settings
from rag_engine.core.db import PostgresDatabase
from rag_engine.core.logging import configure_logging
from rag_engine.core.models import Tenant
from rag_engine.ingestion import repository


async def _create_tenant(slug: str, name: str) -> int:
    settings = get_settings()
    database = PostgresDatabase(settings)
    try:
        async with database.session() as session:
            try:
                tenant = await repository.create_tenant(session, slug=slug, name=name)
                await session.commit()
            except IntegrityError:
                await session.rollback()
                print(f"tenant {slug!r} already exists")
                return 1
        print(f"created tenant {tenant.slug} ({tenant.id})")
        return 0
    finally:
        await database.dispose()


async def _list_tenants() -> int:
    settings = get_settings()
    database = PostgresDatabase(settings)
    try:
        async with database.session() as session:
            tenants = (await session.execute(select(Tenant).order_by(Tenant.slug))).scalars().all()
        if not tenants:
            print("no tenants")
            return 0
        for tenant in tenants:
            print(f"{tenant.slug}\t{tenant.name}\t{tenant.id}")
        return 0
    finally:
        await database.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-engine", description="RAG Knowledge Engine admin CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-tenant", help="create a tenant")
    create.add_argument("--slug", required=True, help="lowercase URL-safe identifier")
    create.add_argument("--name", required=True, help="human-readable tenant name")

    subparsers.add_parser("list-tenants", help="list tenants")

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(level=settings.log_level, log_format=settings.log_format.value)

    if args.command == "create-tenant":
        return asyncio.run(_create_tenant(args.slug, args.name))
    return asyncio.run(_list_tenants())


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
