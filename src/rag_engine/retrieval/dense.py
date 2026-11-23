"""Dense vector retrieval implementation.

Combines embedding-based vector search against the vector store with
PostgreSQL chunk hydration, applying tenant isolation, clearance filtering,
and similarity thresholding.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import and_, or_, select

from rag_engine.core.db import DatabaseProtocol
from rag_engine.core.enums import AccessLevel, visible_access_levels
from rag_engine.core.models import Chunk, Document
from rag_engine.embedding import Embedder
from rag_engine.vectorstore import VectorStore

from .base import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    """Dense semantic retrieval powered by an Embedder and VectorStore."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        database: DatabaseProtocol,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._database = database

    async def retrieve(
        self,
        *,
        tenant_id: UUID,
        query: str,
        top_k: int = 5,
        clearance: AccessLevel = AccessLevel.INTERNAL,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Embed query
        vectors = await self._embedder.embed([clean_query])
        if not vectors:
            return []
        query_vector = vectors[0]

        # 2. Query vector store across tenant namespace
        # Query for extra candidates to allow filtering by clearance / min_score
        candidate_count = max(top_k * 3, 15)
        matches = await self._vector_store.query(
            namespace=str(tenant_id),
            vector=query_vector,
            top_k=candidate_count,
        )
        if not matches:
            return []

        # 3. Filter by min_score and visible clearance
        allowed_levels = visible_access_levels(clearance)
        allowed_strings = {level.value for level in allowed_levels}

        scores: dict[tuple[UUID, int], float] = {}
        for match in matches:
            if match.score < min_score:
                continue
            meta_level = match.metadata.get("access_level")
            if meta_level is not None and str(meta_level) not in allowed_strings:
                continue

            try:
                parts = match.id.split(":", 1)
                doc_id = UUID(parts[0])
                chunk_index = int(parts[1])
                scores[(doc_id, chunk_index)] = match.score
            except (ValueError, IndexError):
                logger.warning("retrieval.malformed_vector_id", extra={"vector_id": match.id})
                continue

        if not scores:
            return []

        # 4. Hydrate chunk content and document metadata from PostgreSQL
        or_clauses = [
            and_(Chunk.document_id == doc_id, Chunk.chunk_index == chunk_idx)
            for doc_id, chunk_idx in scores
        ]

        statement = (
            select(Chunk, Document.title)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Chunk.tenant_id == tenant_id,
                Chunk.access_level.in_(allowed_levels),
                or_(*or_clauses),
            )
        )

        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()

        retrieved: list[RetrievedChunk] = []
        for chunk, doc_title in rows:
            pair = (chunk.document_id, chunk.chunk_index)
            score = scores.get(pair, 0.0)
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=doc_title,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=score,
                    access_level=chunk.access_level,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_path=chunk.section_path,
                )
            )

        # Sort descending by score
        retrieved.sort(key=lambda item: (-item.score, item.chunk_index))
        return retrieved[:top_k]
