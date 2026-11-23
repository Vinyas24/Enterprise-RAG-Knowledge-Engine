"""Health endpoints.

Two probes with different meanings, which orchestrators treat differently:

* ``/health/live``  process is running and able to answer HTTP. Failing it
  means "restart me". It must not check dependencies, otherwise a database
  outage would cause a restart loop that cannot fix anything.
* ``/health/ready`` dependencies are usable. Failing it means "stop sending
  traffic" without killing the process.

The readiness probe is bounded by a timeout so a hung database cannot hang the
probe, and it never echoes driver errors (which can contain DSN fragments).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, Field

from rag_engine import __version__
from rag_engine.api.deps import get_app_settings, get_database
from rag_engine.core.config import Settings
from rag_engine.core.db import DatabaseProtocol

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_NO_STORE = {"Cache-Control": "no-store"}


class ComponentHealth(BaseModel):
    status: Literal["ok", "error"]
    latency_ms: float | None = Field(default=None, description="Probe round-trip time.")
    detail: str | None = Field(default=None, description="Sanitised failure reason.")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    checks: dict[str, ComponentHealth]


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Cheap process check with no dependency calls.",
)
async def liveness(
    response: Response,
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    response.headers.update(_NO_STORE)
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment.value,
        checks={},
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
    summary="Readiness probe",
    description="Verifies that the API can reach its database within the configured timeout.",
)
async def readiness(
    response: Response,
    settings: Settings = Depends(get_app_settings),
    database: DatabaseProtocol = Depends(get_database),
) -> HealthResponse:
    response.headers.update(_NO_STORE)

    try:
        latency_ms = await asyncio.wait_for(
            database.ping(), timeout=settings.health_check_timeout_seconds
        )
        database_health = ComponentHealth(status="ok", latency_ms=round(latency_ms, 2))
    except Exception:
        # Logged with the traceback, returned without it.
        logger.warning("health.database_unavailable", exc_info=True)
        database_health = ComponentHealth(status="error", detail="database unreachable")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if database_health.status == "ok" else "degraded",
        version=__version__,
        environment=settings.environment.value,
        checks={"database": database_health},
    )
