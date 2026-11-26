"""Document upload, status, and deletion endpoints."""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.api.auth import require_api_key
from rag_engine.api.deps import (
    get_app_settings,
    get_ingestion_service,
    get_session,
    get_tenant,
    get_vector_store,
)
from rag_engine.api.errors import COMMON_ERROR_RESPONSES, ApiError
from rag_engine.api.schemas import (
    ChunkListResponse,
    DocumentAccepted,
    DocumentListResponse,
    DocumentRead,
    chunk_read,
    document_read,
    document_summary,
)
from rag_engine.api.uploads import read_upload_limited
from rag_engine.core.config import Settings
from rag_engine.core.enums import AccessLevel, DocumentStatus, JobStatus
from rag_engine.core.models import Tenant
from rag_engine.ingestion import repository
from rag_engine.ingestion.service import IngestionService
from rag_engine.vectorstore import VectorStore
from rag_engine.vectorstore.errors import VectorStoreError
from rag_engine.vectorstore.maintenance import delete_document_vectors

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
    responses=COMMON_ERROR_RESPONSES,
)


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentAccepted,
    summary="Upload a document for ingestion",
    description=(
        "Accepts PDF, DOCX, Markdown and TXT. The response is returned as soon "
        "as the document and its job are committed; parsing happens in the "
        "background worker. Identical content for the same tenant is rejected "
        "with 409 and the existing document id."
    ),
)
async def upload_document(
    file: Annotated[UploadFile, File(description="Document to ingest")],
    tenant: Annotated[Tenant, Depends(get_tenant)],
    session: Annotated[AsyncSession, Depends(get_session)],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    title: Annotated[str | None, Form(max_length=512)] = None,
    access_level: Annotated[AccessLevel, Form()] = AccessLevel.INTERNAL,
) -> DocumentAccepted:
    # Validate the extension before buffering the body: an unsupported 25 MB
    # upload should be rejected without reading it.
    safe_name = service.ensure_supported(file.filename or "")
    data = await read_upload_limited(file, max_bytes=settings.upload_max_bytes)

    result = await service.create_upload(
        session,
        tenant=tenant,
        filename=safe_name,
        content_type=file.content_type,
        data=data,
        title=title,
        access_level=access_level,
    )
    return DocumentAccepted(
        document_id=result.document.id,
        job_id=result.job.id,
        status=result.job.status,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents for the caller's tenant",
)
async def list_documents(
    tenant: Annotated[Tenant, Depends(get_tenant)],
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=100, description="Page size")] = 20,
    offset: Annotated[int, Query(ge=0, description="Items to skip")] = 0,
    status_filter: Annotated[
        DocumentStatus | None, Query(alias="status", description="Filter by status")
    ] = None,
    access_level_filter: Annotated[
        AccessLevel | None, Query(alias="access_level", description="Filter by clearance")
    ] = None,
) -> DocumentListResponse:
    documents = await repository.list_documents(
        session,
        tenant_id=tenant.id,
        limit=limit,
        offset=offset,
        status=status_filter,
        access_level=access_level_filter,
    )
    total = await repository.count_documents(
        session,
        tenant_id=tenant.id,
        status=status_filter,
        access_level=access_level_filter,
    )
    return DocumentListResponse(
        items=[document_summary(document) for document in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Document status and metadata",
)
async def get_document(
    document_id: UUID,
    tenant: Annotated[Tenant, Depends(get_tenant)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DocumentRead:
    document = await repository.get_document(session, tenant_id=tenant.id, document_id=document_id)
    if document is None:
        raise ApiError(404, "document_not_found", "document not found")
    job = await repository.latest_job_for_document(session, document_id=document.id)
    return document_read(document, job)


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkListResponse,
    summary="Chunks of a document, in reading order",
    description=(
        "Returns the stored chunks with the metadata needed to verify a "
        "citation: character span, page or heading path, tenant, and access "
        "level. Another tenant's document is reported as 404, not 403."
    ),
)
async def list_document_chunks(
    document_id: UUID,
    tenant: Annotated[Tenant, Depends(get_tenant)],
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChunkListResponse:
    document = await repository.get_document(session, tenant_id=tenant.id, document_id=document_id)
    if document is None:
        raise ApiError(404, "document_not_found", "document not found")

    chunks = await repository.list_chunks(
        session,
        tenant_id=tenant.id,
        document_id=document.id,
        limit=limit,
        offset=offset,
    )
    total = await repository.count_chunks(session, tenant_id=tenant.id, document_id=document.id)
    return ChunkListResponse(
        items=[chunk_read(chunk) for chunk in chunks],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/{document_id}/reindex",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentAccepted,
    summary="Re-run ingestion for an existing document",
    description=(
        "Queues a fresh parse → chunk → embed → upsert pass for the stored "
        "blob. Use it after changing the embedding model, chunking settings or "
        "access level, or to repair a failed job. Existing chunks stay "
        "searchable until the new vectors are written. Returns 409 while "
        "another job is queued or processing."
    ),
)
async def reindex_document(
    document_id: UUID,
    tenant: Annotated[Tenant, Depends(get_tenant)],
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> DocumentAccepted:
    # The row lock serialises concurrent reindex requests: both would otherwise
    # observe "no active job" and enqueue duplicate work.
    document = await repository.get_document(
        session, tenant_id=tenant.id, document_id=document_id, for_update=True
    )
    if document is None:
        raise ApiError(404, "document_not_found", "document not found")

    latest = await repository.latest_job_for_document(session, document_id=document.id)
    if latest is not None and latest.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
        raise ApiError(
            409,
            "job_in_progress",
            "document already has a queued or processing job",
            details={"job_id": str(latest.id), "status": str(latest.status)},
        )

    job = await repository.enqueue_job(
        session,
        document_id=document.id,
        tenant_id=tenant.id,
        max_attempts=settings.worker_max_attempts,
    )
    await session.commit()
    logger.info(
        "document.reindex_queued",
        extra={"document_id": str(document.id), "job_id": str(job.id)},
    )
    return DocumentAccepted(document_id=document.id, job_id=job.id, status=job.status)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its vectors",
    description=(
        "Purges the document's vectors first, then deletes the document, its "
        "chunks and ingestion jobs. If the vector store is unavailable the "
        "request fails with 503 and nothing is deleted, so a retry converges "
        "instead of leaving orphan vectors. Returns 409 while an ingestion job "
        "is queued or processing."
    ),
)
async def delete_document(
    document_id: UUID,
    request: Request,
    tenant: Annotated[Tenant, Depends(get_tenant)],
    session: Annotated[AsyncSession, Depends(get_session)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> JSONResponse:
    # Lock the row so a concurrent reindex cannot enqueue between the check and
    # the delete. Reindex takes the same lock.
    document = await repository.get_document(
        session, tenant_id=tenant.id, document_id=document_id, for_update=True
    )
    if document is None:
        raise ApiError(404, "document_not_found", "document not found")

    latest = await repository.latest_job_for_document(session, document_id=document.id)
    if latest is not None and latest.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
        raise ApiError(
            409,
            "job_in_progress",
            "document has an active ingestion job; retry after it finishes",
            details={"job_id": str(latest.id), "status": str(latest.status)},
        )

    # Read vector ids before deleting the PostgreSQL rows; the prefix sweep in
    # the helper also catches ids the rows do not know about.
    vector_ids = await repository.get_chunk_vector_ids(session, document_id=document.id)
    namespace = str(tenant.id)

    try:
        await delete_document_vectors(vector_store, namespace, document.id, known_ids=vector_ids)
    except VectorStoreError as exc:
        logger.warning(
            "document.vector_cleanup_failed",
            extra={"document_id": str(document.id)},
            exc_info=True,
        )
        raise ApiError(
            503,
            "vector_store_unavailable",
            "could not delete vectors; the document was not deleted",
        ) from exc

    # Conditional delete closes the race with a job that became active after
    # the check above.
    deleted = await repository.delete_document_if_idle(
        session, tenant_id=tenant.id, document_id=document.id
    )
    if not deleted:
        raise ApiError(
            409,
            "job_in_progress",
            "document has an active ingestion job; retry after it finishes",
        )
    await session.commit()

    if hasattr(request.app.state, "retriever"):
        rtr = request.app.state.retriever
        if hasattr(rtr, "invalidate_tenant"):
            rtr.invalidate_tenant(tenant.id)

    logger.info("document.deleted", extra={"document_id": str(document.id)})
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)
