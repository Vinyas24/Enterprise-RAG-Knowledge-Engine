"""Uniform error responses.

Every error the API produces has the same shape, which makes client handling
and log correlation trivial::

    {"error": {"code": "not_found", "message": "...", "request_id": "...",
               "details": {...}}}

``details`` is optional and only present when it carries actionable data (the
existing document for a duplicate, the supported extensions, ...).
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_engine.core.context import get_request_id
from rag_engine.ingestion.errors import (
    DocumentTooLarge,
    DuplicateDocument,
    EmptyDocument,
    IngestionError,
    InvalidDocumentContent,
    InvalidFilename,
    ParseError,
    UnsupportedDocumentType,
)

_CODE_BY_STATUS: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "unprocessable_entity",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}

_INGESTION_ERRORS: dict[type[IngestionError], tuple[int, str]] = {
    UnsupportedDocumentType: (415, "unsupported_document_type"),
    InvalidDocumentContent: (415, "invalid_document_content"),
    InvalidFilename: (422, "invalid_filename"),
    EmptyDocument: (422, "empty_document"),
    DocumentTooLarge: (413, "file_too_large"),
    DuplicateDocument: (409, "duplicate_document"),
    ParseError: (422, "document_unparseable"),
}


class ErrorDetail(BaseModel):
    """Structured machine- and human-readable error payload."""

    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error explanation")
    request_id: str | None = Field(default=None, description="Request correlation identifier")
    details: dict[str, Any] | None = Field(default=None, description="Contextual error details")


class ErrorResponse(BaseModel):
    """Uniform error envelope for all non-2xx responses."""

    error: ErrorDetail


COMMON_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Bad Request"},
    401: {"model": ErrorResponse, "description": "Unauthorized"},
    403: {"model": ErrorResponse, "description": "Forbidden"},
    404: {"model": ErrorResponse, "description": "Not Found"},
    409: {"model": ErrorResponse, "description": "Conflict"},
    413: {"model": ErrorResponse, "description": "Payload Too Large"},
    415: {"model": ErrorResponse, "description": "Unsupported Media Type"},
    422: {"model": ErrorResponse, "description": "Unprocessable Entity"},
    429: {"model": ErrorResponse, "description": "Too Many Requests"},
    500: {"model": ErrorResponse, "description": "Internal Server Error"},
    503: {"model": ErrorResponse, "description": "Service Unavailable"},
}


class ApiError(Exception):
    """An error the transport layer raises with a fully formed HTTP meaning."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


def error_payload(
    code: str,
    message: str,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "request_id": request_id}
    if details:
        error["details"] = details
    return {"error": error}


def _request_id(request: Request) -> str | None:
    value: str | None = getattr(request.state, "request_id", None) or get_request_id()
    return value


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    http_exc = cast(StarletteHTTPException, exc)
    detail = http_exc.detail if isinstance(http_exc.detail, str) else "request failed"
    return JSONResponse(
        status_code=http_exc.status_code,
        content=error_payload(
            code=_CODE_BY_STATUS.get(http_exc.status_code, "http_error"),
            message=detail,
            request_id=_request_id(request),
        ),
        headers=http_exc.headers,
    )


async def _api_error_handler(request: Request, exc: Exception) -> JSONResponse:
    api_exc = cast(ApiError, exc)
    return JSONResponse(
        status_code=api_exc.status_code,
        content=error_payload(
            api_exc.code, api_exc.message, _request_id(request), details=api_exc.details
        ),
    )


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    validation_exc = cast(RequestValidationError, exc)
    errors = [
        {
            "location": ".".join(str(part) for part in error.get("loc", ())),
            "message": error.get("msg", "invalid value"),
            "type": error.get("type", "value_error"),
        }
        for error in validation_exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=error_payload(
            "invalid_request",
            "request validation failed",
            _request_id(request),
            details={"errors": errors},
        ),
    )


async def _ingestion_error_handler(request: Request, exc: Exception) -> JSONResponse:
    status_code, code = 422, "ingestion_error"
    for error_class in type(exc).__mro__:
        if error_class in _INGESTION_ERRORS:
            status_code, code = _INGESTION_ERRORS[error_class]
            break

    details: dict[str, Any] = {}
    if isinstance(exc, DuplicateDocument):
        details["document_id"] = str(exc.document_id)
    elif isinstance(exc, DocumentTooLarge):
        details["max_bytes"] = exc.max_bytes
    elif isinstance(exc, UnsupportedDocumentType):
        details["supported_extensions"] = list(exc.supported)

    return JSONResponse(
        status_code=status_code,
        content=error_payload(code, str(exc), _request_id(request), details=details or None),
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Register handlers that preserve the error envelope.

    Unexpected exceptions are handled by ``RequestContextMiddleware`` so they
    are logged exactly once, with the request id attached.
    """
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(IngestionError, _ingestion_error_handler)
