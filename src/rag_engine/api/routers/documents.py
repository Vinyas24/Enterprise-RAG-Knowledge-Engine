"""Document upload, status, and deletion endpoints."""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.api.deps import (
    get_app_settings,
    get_ingestion_service,
    get_session,
    get_tenant,
    get_vector_store,
)
from rag_engine.api.errors import ApiError
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
from rag_engine.core.enums import AccessLevel, DocumentStatus
from rag_engine.core.models import Tenant
from rag_engine.ingestion import repository
from rag_engine.ingestion.service import IngestionService
from rag_engine.vectorstore import VectorStore
from rag_engine.vectorstore.maintenance import delete_document_vectors

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


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
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
) -> DocumentListResponse:
    documents = await repository.list_documents(
        session,
        tenant_id=tenant.id,
        limit=limit,
        offset=offset,
        status=status_filter,
    )
    total = await repository.count_documents(session, tenant_id=tenant.id, status=status_filter)
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


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its vectors",
    description=(
        "Removes the document, its chunks and ingestion jobs from PostgreSQL, "
        "then purges the corresponding vectors from the vector store. The "
        "response is returned after the database delete commits; the vector "
        "cleanup is best-effort (logged if it fails)."
    ),
)
async def delete_document(
    document_id: UUID,
    tenant: Annotated[Tenant, Depends(get_tenant)],
    session: Annotated[AsyncSession, Depends(get_session)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> JSONResponse:
    document = await repository.get_document(session, tenant_id=tenant.id, document_id=document_id)
    if document is None:
        raise ApiError(404, "document_not_found", "document not found")

    # Read vector ids before deleting the PG rows.
    vector_ids = await repository.get_chunk_vector_ids(session, document_id=document_id)
    namespace = str(tenant.id)

    await repository.delete_document(session, document_id=document_id)
    await session.commit()

    # Best-effort vector cleanup: the PG rows are already gone, so even if the
    # vector store is unreachable the data is logically deleted.
    try:
        await delete_document_vectors(vector_store, namespace, document_id, known_ids=vector_ids)
    except Exception:
        logger.warning(
            "document.vector_cleanup_failed",
            extra={"document_id": str(document_id)},
            exc_info=True,
        )

    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)
