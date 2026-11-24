"""CLI entry point: argument dispatch without touching a database."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

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


def test_evaluate_arguments_are_forwarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(args: Namespace, settings: Settings) -> int:
        captured["args"] = args
        captured["settings"] = settings
        return 0

    _install_common(monkeypatch)
    monkeypatch.setattr(cli, "run_evaluation_command", fake_run_command)

    exit_code = cli.main(
        [
            "evaluate",
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--top-k",
            "3",
            "--min-score",
            "0.25",
            "--baseline",
            str(tmp_path / "baseline.json"),
            "--save-baseline",
            str(tmp_path / "new-baseline.json"),
            "--json-output",
            str(tmp_path / "report.json"),
            "--markdown-output",
            str(tmp_path / "report.md"),
        ]
    )

    assert exit_code == 0
    args = captured["args"]
    assert isinstance(args, Namespace)
    assert args.dataset == tmp_path / "dataset.json"
    assert args.top_k == 3
    assert args.min_score == 0.25
    assert args.baseline == tmp_path / "baseline.json"
    assert args.save_baseline == tmp_path / "new-baseline.json"
    assert args.json_output == tmp_path / "report.json"
    assert args.markdown_output == tmp_path / "report.md"
    assert isinstance(captured["settings"], Settings)


def test_evaluate_defaults_to_the_committed_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(args: Namespace, settings: Settings) -> int:
        captured["args"] = args
        return 0

    _install_common(monkeypatch)
    monkeypatch.setattr(cli, "run_evaluation_command", fake_run_command)

    assert cli.main(["evaluate"]) == 0
    args = captured["args"]
    assert isinstance(args, Namespace)
    assert args.dataset == Path("eval/datasets/enterprise_baseline.json")
    assert args.baseline is None
