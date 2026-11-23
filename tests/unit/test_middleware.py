"""Request correlation, access logging and the error envelope."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app
from rag_engine.api.middleware import REQUEST_ID_HEADER
from rag_engine.core.config import Settings
from tests.conftest import FakeDatabase


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    request_id = response.headers[REQUEST_ID_HEADER]
    assert len(request_id) == 32
    assert request_id.isalnum()


def test_valid_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health/live", headers={REQUEST_ID_HEADER: "client-supplied-123"})

    assert response.headers[REQUEST_ID_HEADER] == "client-supplied-123"


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "has spaces",
        "semi;colon",
        "line\nbreak",
        "x" * 65,
        "-leading-dash",
        "",
    ],
)
def test_invalid_request_id_is_replaced(client: TestClient, unsafe_value: str) -> None:
    response = client.get("/health/live", headers={REQUEST_ID_HEADER: unsafe_value})

    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id != unsafe_value
    assert len(request_id) == 32


def test_request_ids_are_unique_per_request(client: TestClient) -> None:
    first = client.get("/health/live").headers[REQUEST_ID_HEADER]
    second = client.get("/health/live").headers[REQUEST_ID_HEADER]

    assert first != second


def test_not_found_uses_the_error_envelope(client: TestClient) -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"
    assert body["error"]["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_unexpected_exception_returns_safe_error(
    settings: Settings, database: FakeDatabase
) -> None:
    app = create_app(settings, database=database)

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("kaboom: internal detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["message"] == "Internal server error"
    assert "kaboom" not in response.text
    assert response.headers[REQUEST_ID_HEADER] == body["error"]["request_id"]


def test_unexpected_exception_is_logged_once_with_traceback(
    settings: Settings,
    database: FakeDatabase,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app(settings, database=database)

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    with caplog.at_level(logging.INFO), TestClient(app, raise_server_exceptions=False) as client:
        client.get("/boom")

    failures = [
        record for record in caplog.records if record.getMessage() == "request.unhandled_exception"
    ]
    assert len(failures) == 1
    assert failures[0].exc_info is not None


def test_access_log_line_has_request_fields(
    settings: Settings,
    database: FakeDatabase,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app(settings, database=database)

    with caplog.at_level(logging.INFO), TestClient(app) as client:
        client.get("/health/live?token=should-not-be-logged")

    completed = [record for record in caplog.records if record.getMessage() == "request.completed"]
    assert len(completed) == 1
    record = completed[0]
    assert record.status_code == 200  # type: ignore[attr-defined]
    assert record.method == "GET"  # type: ignore[attr-defined]
    assert record.path == "/health/live"  # type: ignore[attr-defined]
    assert record.duration_ms >= 0  # type: ignore[attr-defined]
    assert "should-not-be-logged" not in repr(record.__dict__)


def test_json_logging_end_to_end(
    settings: Settings, database: FakeDatabase, capsys: pytest.CaptureFixture[str]
) -> None:
    # Build the app inside the test so the log handler binds to the stream that
    # pytest is capturing.
    app = create_app(settings, database=database)
    with TestClient(app) as client:
        response = client.get("/health/live")

    request_id = response.headers[REQUEST_ID_HEADER]
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    completed = [line for line in lines if line.get("message") == "request.completed"]

    assert completed, "expected one access log line"
    entry = completed[-1]
    assert entry["request_id"] == request_id
    assert entry["path"] == "/health/live"
    assert entry["status_code"] == 200
    assert entry["level"] == "info"
    assert entry["duration_ms"] >= 0
