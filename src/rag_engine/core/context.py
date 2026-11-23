"""Request-scoped context: a per-request identifier shared by logs and errors.

A :class:`~contextvars.ContextVar` (not a module global) keeps the value
correct under concurrent requests: every ASGI task gets its own copy.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Return the request id bound to the current task, if any."""
    return _request_id.get()


def set_request_id(value: str) -> Token[str | None]:
    """Bind ``value`` to the current task and return a reset token."""
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the previous request id, even if the request failed."""
    _request_id.reset(token)
