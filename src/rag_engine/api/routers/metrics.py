"""Prometheus metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Response

from rag_engine.observability import get_metrics_registry

router = APIRouter(tags=["observability"])


@router.get(
    "/metrics",
    summary="Prometheus metrics endpoint",
    description="Exposes in-memory metrics formatted for Prometheus scraping.",
    response_class=Response,
)
async def prometheus_metrics() -> Response:
    content = get_metrics_registry().generate_prometheus_exposition()
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
