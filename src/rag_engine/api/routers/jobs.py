"""Ingestion job status and listing endpoints."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.api.auth import require_api_key
from rag_engine.api.deps import get_session, get_tenant
from rag_engine.api.errors import COMMON_ERROR_RESPONSES, ApiError
from rag_engine.api.schemas import JobListResponse, JobRead, job_read
from rag_engine.core.enums import JobStatus
from rag_engine.core.models import Tenant
from rag_engine.ingestion import repository

router = APIRouter(
    prefix="/ingestion-jobs",
    tags=["ingestion"],
    dependencies=[Depends(require_api_key)],
    responses=COMMON_ERROR_RESPONSES,
)


@router.get(
    "",
    response_model=JobListResponse,
    summary="List ingestion jobs for the tenant",
    description="Paged list of ingestion jobs ordered by most recent.",
)
async def list_ingestion_jobs(
    tenant: Annotated[Tenant, Depends(get_tenant)],
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=100, description="Page size")] = 20,
    offset: Annotated[int, Query(ge=0, description="Items to skip")] = 0,
    status_filter: Annotated[
        JobStatus | None, Query(alias="status", description="Filter by job status")
    ] = None,
) -> JobListResponse:
    jobs = await repository.list_jobs(
        session,
        tenant_id=tenant.id,
        limit=limit,
        offset=offset,
        status=status_filter,
    )
    total = await repository.count_jobs(session, tenant_id=tenant.id, status=status_filter)
    return JobListResponse(
        items=[job_read(job) for job in jobs],
        total=total,
        limit=limit,
        offset=offset,
    )


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
