"""CLI behaviour against the real test database."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select

from rag_engine import cli
from rag_engine.core.db import PostgresDatabase
from rag_engine.core.models import Tenant

pytestmark = pytest.mark.integration


async def test_create_list_and_duplicate_tenant(
    monkeypatch: pytest.MonkeyPatch,
    test_settings: Any,
    database: PostgresDatabase,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: test_settings)
    monkeypatch.setattr(cli, "configure_logging", lambda **_kwargs: None)

    assert await cli._create_tenant("cli-acme", "CLI Acme") == 0
    assert "created tenant cli-acme" in capsys.readouterr().out

    async with database.session() as session:
        tenant = (await session.execute(select(Tenant))).scalar_one()
    assert tenant.slug == "cli-acme"

    # A duplicate is reported without a traceback and with a non-zero exit.
    assert await cli._create_tenant("cli-acme", "CLI Acme") == 1
    assert "already exists" in capsys.readouterr().out

    assert await cli._list_tenants() == 0
    listing = capsys.readouterr().out
    assert "cli-acme" in listing
    assert "CLI Acme" in listing


async def test_list_tenants_reports_an_empty_database(
    monkeypatch: pytest.MonkeyPatch,
    test_settings: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: test_settings)
    monkeypatch.setattr(cli, "configure_logging", lambda **_kwargs: None)

    assert await cli._list_tenants() == 0
    assert "no tenants" in capsys.readouterr().out
