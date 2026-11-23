"""Structured logging built on the standard library.

Two formatters are available:

* ``json``    one JSON object per line, including ``request_id`` when a record
              is emitted inside a request. This is the production default.
* ``console`` human-readable output for local development.

We deliberately avoid a logging framework dependency: the requirements are a
stable schema, correlation ids and no duplicated access logs, and the stdlib
covers all three in a few dozen lines that are easy to test.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import IO, Any

from rag_engine.core.context import get_request_id

# Keys that ``logging`` puts on every record. Anything else in ``record.__dict__``
# came from ``logger.info(..., extra={...})`` and belongs in the JSON payload.
_RESERVED_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message",
    "asctime",
    "taskName",
    "request_id",
}

_MARKER = "_rag_engine_handler"


def _iso_timestamp(created: float) -> str:
    return (
        datetime.fromtimestamp(created, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _iso_timestamp(record.created),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None)
        if request_id and request_id != "-":
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # ``default=str`` keeps the formatter resilient: an unexpected extra
        # value (UUID, Decimal, ...) must never break log emission.
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable output for local development, with structured extras."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s [%(request_id)s] %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_ATTRS and not key.startswith("_")
        }
        if not extras:
            return line
        # ``default=str`` keeps the formatter resilient for non-JSON extras.
        rendered = " ".join(
            f"{key}={json.dumps(value, default=str)}" for key, value in extras.items()
        )
        return f"{line} {rendered}"


class RequestIdFilter(logging.Filter):
    """Fall back to the ambient request id so every handler line stays correlated."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = get_request_id() or "-"
        return True


def configure_logging(
    *,
    level: str = "INFO",
    log_format: str = "json",
    stream: IO[str] | None = None,
) -> None:
    """Install the application log handler on the root logger.

    Idempotent: previously installed handlers (identified by a marker) are
    replaced, while unrelated handlers such as pytest's ``caplog`` are left
    untouched. Uvicorn's own handlers are removed so each request produces
    exactly one access line (ours, which carries the request id and timing).
    """
    if log_format == "json":
        formatter: logging.Formatter = JsonFormatter()
    elif log_format == "console":
        formatter = ConsoleFormatter()
    else:
        raise ValueError(f"unsupported log format: {log_format!r}")

    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    setattr(handler, _MARKER, True)

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _MARKER, False):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    # Requests are logged by RequestContextMiddleware instead.
    logging.getLogger("uvicorn.access").disabled = True

    logging.captureWarnings(True)
