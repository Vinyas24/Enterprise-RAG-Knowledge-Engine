"""Service discovery endpoint for humans and smoke tests."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from rag_engine import __version__
from rag_engine.api.deps import get_app_settings
from rag_engine.core.config import Settings

router = APIRouter(tags=["meta"])


class ServiceInfo(BaseModel):
    name: str
    version: str
    environment: str
    docs_url: str | None
    health: dict[str, str]


@router.get("/", response_model=ServiceInfo, summary="Service metadata")
async def service_info(settings: Settings = Depends(get_app_settings)) -> ServiceInfo:
    return ServiceInfo(
        name=settings.app_name,
        version=__version__,
        environment=settings.environment.value,
        docs_url="/docs" if settings.docs_enabled else None,
        health={"liveness": "/health/live", "readiness": "/health/ready"},
    )
