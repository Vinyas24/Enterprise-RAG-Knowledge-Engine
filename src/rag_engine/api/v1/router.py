"""API version 1 consolidated router."""

from __future__ import annotations

from fastapi import APIRouter

from rag_engine.api.routers import documents, jobs, metrics, query

router = APIRouter(prefix="/v1")

router.include_router(documents.router)
router.include_router(jobs.router)
router.include_router(query.router)
router.include_router(metrics.router)
