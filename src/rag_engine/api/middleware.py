"""Request correlation middleware.

Responsibilities:

1. Assign a request id (accept a caller-supplied one only if it is well formed,
   otherwise generate a fresh one) and expose it to logs, error responses and
   the response header.
2. Emit exactly one structured access log line per request with method, path,
   status and duration. Query strings are deliberately excluded: they can carry
   sensitive user input, which belongs in neither access logs nor audit trails.
3. Convert unexpected exceptions into the uniform error envelope, logging the
   traceback once with the request id attached.

Implemented as pure ASGI middleware rather than ``BaseHTTPMiddleware`` to avoid
its extra task per request and its contextvar/streaming caveats.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_engine.api.errors import error_payload
from rag_engine.core.context import reset_request_id, set_request_id

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# Conservative on purpose: the value is echoed into a response header and into
# logs, so it must not be able to inject a second header or a fake log line.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _resolve_request_id(scope: Scope) -> str:
    candidate = Headers(scope=scope).get(REQUEST_ID_HEADER)
    if candidate and _VALID_REQUEST_ID.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


class RequestContextMiddleware:
    """Attach a request id to every HTTP request and log its outcome."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _resolve_request_id(scope)
        # ``scope["state"]`` is shared with the Request object, so exception
        # handlers can still see the id after this middleware returns.
        scope.setdefault("state", {})["request_id"] = request_id
        token = set_request_id(request_id)

        started = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_with_context(message: Message) -> None:
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_context)
        except Exception:
            logger.exception("request.unhandled_exception")
            if not response_started:
                response = JSONResponse(
                    status_code=500,
                    content=error_payload(
                        "internal_error", "Internal server error", request_id=request_id
                    ),
                )
                try:
                    await response(scope, receive, send_with_context)
                except Exception:  # pragma: no cover - client gone mid-response
                    logger.debug("request.error_response_not_delivered", exc_info=True)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            extra: dict[str, Any] = {
                "method": scope.get("method"),
                "path": scope.get("path"),
                "status_code": status_code,
                "duration_ms": duration_ms,
            }
            logger.info("request.completed", extra=extra)
            reset_request_id(token)
