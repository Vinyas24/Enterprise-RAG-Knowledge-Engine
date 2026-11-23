"""CLI entry point: argument dispatch without touching a database."""

from __future__ import annotations

import pytest

from rag_engine import cli
from rag_engine.core.config import Settings


def _install_common(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "configure_logging", lambda **_kwargs: None)


def test_create_tenant_arguments_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    async def fake_create(slug: str, name: str) -> int:
        captured.update(slug=slug, name=name)
        return 0

    _install_common(monkeypatch)
    monkeypatch.setattr(cli, "_create_tenant", fake_create)

    assert cli.main(["create-tenant", "--slug", "acme", "--name", "Acme Corp"]) == 0
    assert captured == {"slug": "acme", "name": "Acme Corp"}


def test_list_tenants_is_the_default_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    async def fake_list() -> int:
        nonlocal called
        called = True
        return 0

    _install_common(monkeypatch)
    monkeypatch.setattr(cli, "_list_tenants", fake_list)

    assert cli.main(["list-tenants"]) == 0
    assert called is True


def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_create_tenant_requires_a_slug_and_name() -> None:
    with pytest.raises(SystemExit):
        cli.main(["create-tenant", "--slug", "acme"])
