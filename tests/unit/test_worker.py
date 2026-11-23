"""Worker loop and entry point behaviour with a fake processor."""

from __future__ import annotations

import signal
import types
from typing import Any

import pytest

from rag_engine.core.config import Settings
from rag_engine.ingestion.processor import IngestionProcessor, ProcessOutcome
from rag_engine.ingestion.worker import _main, build_processor, main, run_forever


class _FakeDatabase:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


class _FakeProcessor:
    def __init__(self, *, processed: bool = False) -> None:
        self.processed = processed
        self.calls = 0

    async def process_next(self) -> ProcessOutcome:
        self.calls += 1
        return ProcessOutcome(processed=self.processed)


class _FakeLoop:
    def __init__(self) -> None:
        self.handlers: dict[int, Any] = {}

    def add_signal_handler(self, sig: int, callback: Any) -> None:
        self.handlers[sig] = callback


class _StopAfterOneWait:
    """An event that reports 'stopped' on the second check."""

    def __init__(self) -> None:
        self.checks = 0

    def is_set(self) -> bool:
        self.checks += 1
        return self.checks > 1

    def set(self) -> None:  # pragma: no cover - registered, never invoked
        pass

    async def wait(self) -> None:
        return None


async def _noop_run_forever(processor: object, *, poll_interval: float) -> None:
    return None


def _settings() -> Settings:
    return Settings(environment="test", log_level="DEBUG", _env_file=None)


def _install_entrypoint_doubles(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    processor: _FakeProcessor,
    *,
    database: _FakeDatabase | None = None,
) -> _FakeDatabase:
    import rag_engine.ingestion.worker as worker_module

    resolved_database = database or _FakeDatabase(settings)
    monkeypatch.setattr(worker_module, "get_settings", lambda: settings)
    monkeypatch.setattr(worker_module, "configure_logging", lambda **_kwargs: None)
    monkeypatch.setattr(worker_module, "PostgresDatabase", lambda _settings: resolved_database)
    monkeypatch.setattr(worker_module, "build_processor", lambda _settings, database: processor)
    return resolved_database


def test_build_processor_wires_every_component() -> None:
    settings = _settings()

    processor = build_processor(settings, database=_FakeDatabase(settings))  # type: ignore[arg-type]

    assert isinstance(processor, IngestionProcessor)


async def test_run_forever_polls_then_exits_when_signalled() -> None:
    loop = _FakeLoop()

    async def fake_wait_for(awaitable: Any, timeout: float) -> None:  # noqa: ASYNC109
        # Signal shutdown the way the OS would, then let the loop observe it.
        loop.handlers[signal.SIGINT]()
        awaitable.close()

    fake_asyncio = types.SimpleNamespace(
        Event=_StopAfterOneWait,
        get_running_loop=lambda: loop,
        wait_for=fake_wait_for,
    )

    import rag_engine.ingestion.worker as worker_module

    original = worker_module.asyncio
    worker_module.asyncio = fake_asyncio  # type: ignore[assignment]
    try:
        processor = _FakeProcessor(processed=False)
        await run_forever(processor, poll_interval=0.01)  # type: ignore[arg-type]
    finally:
        worker_module.asyncio = original

    assert processor.calls == 1
    assert loop.handlers


async def test_main_once_processes_and_disposes(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    processor = _FakeProcessor(processed=True)
    database = _install_entrypoint_doubles(monkeypatch, settings, processor)

    assert await _main(settings, once=True) == 0

    assert processor.calls == 1
    assert database.disposed is True


def test_main_once_from_the_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    processor = _FakeProcessor()
    _install_entrypoint_doubles(monkeypatch, settings, processor)

    assert main(["--once"]) == 0
    assert processor.calls == 1


def test_main_runs_the_loop_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    _install_entrypoint_doubles(monkeypatch, settings, _FakeProcessor())

    import rag_engine.ingestion.worker as worker_module

    monkeypatch.setattr(worker_module, "run_forever", _noop_run_forever)

    assert main([]) == 0


def test_keyboard_interrupt_exits_with_130(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    _install_entrypoint_doubles(monkeypatch, settings, _FakeProcessor())

    import rag_engine.ingestion.worker as worker_module

    async def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(worker_module, "run_forever", interrupt)

    assert main([]) == 130
