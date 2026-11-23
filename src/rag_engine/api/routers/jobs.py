"""Ingestion job status endpoint."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.api.deps import get_session, get_tenant
from rag_engine.api.errors import ApiError
from rag_engine.api.schemas import JobRead, job_read
from rag_engine.core.models import Tenant
from rag_engine.ingestion import repository

router = APIRouter(prefix="/ingestion-jobs", tags=["ingestion"])


@router.get(
    "/{job_id}",
    response_model=JobRead,
    summary="Ingestion job status",
    description="Status, attempt count and last error for one ingestion job.",
)
async def get_ingestion_job(
    job_id: UUID,
    tenant: Annotated[Tenant, Depends(get_tenant)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JobRead:
    job = await repository.get_job(session, tenant_id=tenant.id, job_id=job_id)
    if job is None:
        raise ApiError(404, "job_not_found", "ingestion job not found")
    return job_read(job)
