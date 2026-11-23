"""FastAPI dependencies resolved from application state."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.api.errors import ApiError
from rag_engine.core.config import Settings
from rag_engine.core.db import DatabaseProtocol
from rag_engine.core.models import Tenant
from rag_engine.generation.service import QueryEngine
from rag_engine.ingestion import repository
from rag_engine.ingestion.service import IngestionService
from rag_engine.retrieval.base import Retriever
from rag_engine.vectorstore import VectorStore

TENANT_HEADER = "X-Tenant-ID"


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_database(request: Request) -> DatabaseProtocol:
    database: DatabaseProtocol = request.app.state.db
    return database


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Request-scoped session. Routes commit explicitly; closing rolls back."""
    database = get_database(request)
    async with database.session() as session:
        yield session


async def get_tenant(
    request: Request,
    session: AsyncSession = Depends(get_session),
    x_tenant_id: Annotated[
        str | None,
        Header(
            alias=TENANT_HEADER,
            description="Tenant slug (e.g. 'acme')",
        ),
    ] = None,
) -> Tenant:
    """Resolve the caller's tenant from the header (or query parameter).

    This placeholder stands in for authentication. It is intentionally explicit
    and required: every data query is tenant-scoped from day one, so adding
    real identity later replaces one dependency instead of rewriting queries.
    """
    raw_slug = x_tenant_id or request.headers.get(TENANT_HEADER, "")
    if not raw_slug:
        raw_slug = request.query_params.get("tenant") or request.query_params.get("tenant_id", "")
    slug = raw_slug.strip().lower()
    if not slug:
        raise ApiError(
            400,
            "missing_tenant",
            f"{TENANT_HEADER} header is required",
        )
    if len(slug) > 63 or not slug.replace("-", "").isalnum():
        raise ApiError(400, "invalid_tenant", "tenant identifier is malformed")

    tenant = await repository.get_tenant_by_slug(session, slug)
    if tenant is None:
        raise ApiError(404, "tenant_not_found", f"unknown tenant {slug!r}")
    return tenant


def get_ingestion_service(request: Request) -> IngestionService:
    service: IngestionService = request.app.state.ingestion_service
    return service


def get_vector_store(request: Request) -> VectorStore:
    store: VectorStore = request.app.state.vector_store
    return store


def get_retriever(request: Request) -> Retriever:
    retriever: Retriever = request.app.state.retriever
    return retriever


def get_query_engine(request: Request) -> QueryEngine:
    engine: QueryEngine = request.app.state.query_engine
    return engine
