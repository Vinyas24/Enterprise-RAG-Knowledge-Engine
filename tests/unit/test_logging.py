"""Logging: schema, request-id correlation, extras and idempotency."""

from __future__ import annotations

import io
import json
import logging

import pytest

from rag_engine.core.logging import configure_logging


def _records(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_json_log_line_contains_core_fields() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", log_format="json", stream=stream)

    logging.getLogger("rag_engine.test").info("hello world")

    [record] = _records(stream)
    assert record["level"] == "info"
    assert record["logger"] == "rag_engine.test"
    assert record["message"] == "hello world"
    assert str(record["timestamp"]).endswith("Z")


def test_request_id_is_taken_from_context(bound_request_id: str) -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", log_format="json", stream=stream)

    logging.getLogger("rag_engine.test").info("correlated")

    [record] = _records(stream)
    assert record["request_id"] == bound_request_id


def test_request_id_is_absent_outside_a_request() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", log_format="json", stream=stream)

    logging.getLogger("rag_engine.test").info("background job")

    [record] = _records(stream)
    assert "request_id" not in record


def test_extra_fields_are_merged_and_unserialisable_values_are_stringified() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", log_format="json", stream=stream)

    logging.getLogger("rag_engine.test").info(
        "indexed", extra={"document_id": "doc-1", "chunks": 42, "marker": object()}
    )

    [record] = _records(stream)
    assert record["document_id"] == "doc-1"
    assert record["chunks"] == 42
    assert isinstance(record["marker"], str)


def test_exceptions_are_serialised() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", log_format="json", stream=stream)
    logger = logging.getLogger("rag_engine.test")

    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")

    [record] = _records(stream)
    assert "ValueError: boom" in str(record["exception"])


def test_console_format_is_human_readable(bound_request_id: str) -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", log_format="console", stream=stream)

    logging.getLogger("rag_engine.test").info("readable")

    output = stream.getvalue()
    assert "readable" in output
    assert "INFO" in output
    assert bound_request_id in output


def test_console_format_renders_structured_extras() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", log_format="console", stream=stream)

    logging.getLogger("rag_engine.test").info(
        "request.completed", extra={"path": "/health/ready", "status_code": 200}
    )

    output = stream.getvalue()
    assert 'path="/health/ready"' in output
    assert "status_code=200" in output


def test_configure_logging_is_idempotent() -> None:
    configure_logging(level="INFO", log_format="json", stream=io.StringIO())
    configure_logging(level="INFO", log_format="json", stream=io.StringIO())

    root = logging.getLogger()
    marked = [
        handler for handler in root.handlers if getattr(handler, "_rag_engine_handler", False)
    ]
    assert len(marked) == 1


def test_unknown_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported log format"):
        configure_logging(log_format="xml", stream=io.StringIO())
