"""API authentication dependencies and token verification.

Provides API key and Bearer token authentication with constant-time verification.
Permissive by default (auth_enabled=False, api_key=None) to preserve backward
compatibility, and strictly enforced when RAG_AUTH_ENABLED=true or RAG_API_KEY is set.
"""

from __future__ import annotations

import logging
import secrets
from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from rag_engine.api.deps import get_app_settings
from rag_engine.api.errors import ApiError
from rag_engine.core.config import Settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"

api_key_header_scheme = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="API Key authentication header (e.g. 'X-API-Key: secret-key')",
)
bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Bearer token authentication header (e.g. 'Authorization: Bearer secret-key')",
)


def _extract_token(
    api_key_header: str | None,
    bearer: HTTPAuthorizationCredentials | None,
) -> str | None:
    if api_key_header and api_key_header.strip():
        return api_key_header.strip()
    if bearer and bearer.credentials and bearer.credentials.strip():
        return bearer.credentials.strip()
    return None


async def require_api_key(
    request: Request,
    settings: Annotated[Settings, Depends(get_app_settings)],
    api_key_header: Annotated[str | None, Security(api_key_header_scheme)] = None,
    bearer: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)] = None,
) -> str | None:
    """Validate incoming API credentials against configured secrets.

    Raises:
        ApiError: 401 Unauthorized if auth is enabled and the key is missing or invalid.
    """
    token = _extract_token(api_key_header, bearer)

    # If auth is not enabled and no key is configured, pass through (permissive mode)
    if not settings.auth_enabled and settings.api_key is None:
        if token:
            request.state.authenticated_subject = token
        return token

    # Auth is required
    if not token:
        logger.warning("auth.missing_token", extra={"path": request.url.path})
        raise ApiError(
            status_code=401,
            code="unauthorized",
            message="API key is required in Authorization or X-API-Key header",
        )

    # Constant-time comparison against configured keys
    valid_keys: list[str] = []
    if settings.api_key:
        valid_keys.append(settings.api_key)
    if settings.admin_api_key:
        valid_keys.append(settings.admin_api_key)

    is_valid = any(secrets.compare_digest(token, expected) for expected in valid_keys)
    if not is_valid:
        logger.warning("auth.invalid_token", extra={"path": request.url.path})
        raise ApiError(
            status_code=401,
            code="unauthorized",
            message="Invalid API key",
        )

    request.state.authenticated_subject = token
    return token
