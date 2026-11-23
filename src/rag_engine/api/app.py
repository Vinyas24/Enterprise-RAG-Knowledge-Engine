"""Application factory.

Constructing the app is separate from module import so tests can build an app
with injected settings and a fake database, and so the dependency graph is
explicit rather than hidden in import side effects.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_engine import __version__
from rag_engine.api.errors import install_exception_handlers
from rag_engine.api.middleware import RequestContextMiddleware
from rag_engine.api.routers import documents, health, jobs, meta, query
from rag_engine.core.config import Settings, get_settings
from rag_engine.core.db import DatabaseProtocol, PostgresDatabase
from rag_engine.core.logging import configure_logging
from rag_engine.embedding import Embedder, build_embedder
from rag_engine.generation import (
    LLMProvider,
    QueryEngine,
    build_llm,
    build_query_engine,
)
from rag_engine.ingestion.blobstore import BlobStore, LocalBlobStore
from rag_engine.ingestion.parsers import ParserRegistry
from rag_engine.ingestion.service import IngestionService
from rag_engine.retrieval import Retriever, build_retriever
from rag_engine.vectorstore import VectorStore, build_vector_store

logger = logging.getLogger(__name__)

_DESCRIPTION = (
    "Enterprise RAG knowledge engine: asynchronous document ingestion, hybrid "
    "retrieval with access-control filters, re-ranking and grounded answers "
    "with source citations."
)


def create_app(
    settings: Settings | None = None,
    *,
    database: DatabaseProtocol | None = None,
    blob_store: BlobStore | None = None,
    vector_store: VectorStore | None = None,
    embedder: Embedder | None = None,
    retriever: Retriever | None = None,
    llm: LLMProvider | None = None,
    query_engine: QueryEngine | None = None,
) -> FastAPI:
    """Build the API application.

    Args:
        settings: Configuration override; defaults to the cached environment
            settings.
        database: Injected database handle (used by tests). When omitted, a
            PostgreSQL handle is created from settings and disposed on shutdown.
        blob_store: Injected blob store (used by tests). Defaults to local
            filesystem storage under ``settings.blob_store_path``.
        vector_store: Injected vector store (used by tests). Defaults to the
            store selected by ``settings.vector_store``.
        embedder: Injected embedder (used by tests). Defaults to the embedder
            built from ``settings.embedding_provider``.
        retriever: Injected retriever (used by tests).
        llm: Injected language model provider (used by tests).
        query_engine: Injected query coordinator (used by tests).
    """
    resolved = settings or get_settings()
    configure_logging(level=resolved.log_level, log_format=resolved.log_format.value)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db = database if database is not None else PostgresDatabase(resolved)
        store = blob_store if blob_store is not None else LocalBlobStore(resolved.blob_store_path)
        vs = vector_store if vector_store is not None else build_vector_store(resolved)
        emb = embedder if embedder is not None else build_embedder(resolved)
        rtr = (
            retriever
            if retriever is not None
            else build_retriever(resolved, embedder=emb, vector_store=vs, database=db)
        )
        model = llm if llm is not None else build_llm(resolved)
        engine = (
            query_engine
            if query_engine is not None
            else build_query_engine(resolved, retriever=rtr, llm=model)
        )

        app.state.db = db
        app.state.blob_store = store
        app.state.vector_store = vs
        app.state.embedder = emb
        app.state.retriever = rtr
        app.state.llm = model
        app.state.query_engine = engine
        app.state.ingestion_service = IngestionService(
            parsers=ParserRegistry.default(), blob_store=store, settings=resolved
        )
        logger.info(
            "application.starting",
            extra={
                "version": __version__,
                "environment": resolved.environment.value,
                "database": resolved.redacted_database_url,
                "blob_store_path": str(resolved.blob_store_path),
                "embedding_provider": resolved.embedding_provider,
                "vector_store": resolved.vector_store,
                "llm_provider": resolved.llm_provider,
            },
        )
        try:
            yield
        finally:
            await model.close()
            await emb.close()
            await vs.close()
            await db.dispose()
            logger.info("application.stopped")

    app = FastAPI(
        title="RAG Knowledge Engine",
        version=__version__,
        description=_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs" if resolved.docs_enabled else None,
        redoc_url="/redoc" if resolved.docs_enabled else None,
        openapi_url="/openapi.json" if resolved.docs_enabled else None,
    )
    app.state.settings = resolved

    app.add_middleware(RequestContextMiddleware)
    install_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(meta.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(query.router)

    return app
