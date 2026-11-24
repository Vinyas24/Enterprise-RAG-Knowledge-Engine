"""Unit tests for dense retrieval, clearance filtering, and hydration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from rag_engine.core.config import Settings
from rag_engine.core.enums import AccessLevel
from rag_engine.core.ids import chunk_vector_id
from rag_engine.core.models import Chunk
from rag_engine.retrieval.dense import DenseRetriever
from rag_engine.retrieval.factory import build_retriever
from rag_engine.vectorstore.base import VectorMatch


class FakeSession:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows

    async def execute(self, statement: Any) -> Any:
        mock_result = MagicMock()
        mock_result.all.return_value = self.rows
        return mock_result


class MockDatabase:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows

    @asynccontextmanager
    async def session(self) -> Any:
        yield FakeSession(self.rows)

    async def ping(self) -> float:
        return 1.0

    async def dispose(self) -> None:
        pass


async def test_dense_retriever_empty_query() -> None:
    embedder = AsyncMock()
    vector_store = AsyncMock()
    db = MockDatabase([])
    retriever = DenseRetriever(embedder=embedder, vector_store=vector_store, database=db)

    result = await retriever.retrieve(tenant_id=uuid4(), query="   ")
    assert result == []
    assert embedder.embed.call_count == 0


async def test_dense_retriever_no_vector_matches() -> None:
    embedder = AsyncMock()
    embedder.embed.return_value = [[0.1, 0.2]]
    vector_store = AsyncMock()
    vector_store.query.return_value = []
    db = MockDatabase([])
    retriever = DenseRetriever(embedder=embedder, vector_store=vector_store, database=db)

    result = await retriever.retrieve(tenant_id=uuid4(), query="sample question")
    assert result == []
    vector_store.query.assert_called_once()


async def test_dense_retriever_min_score_and_clearance_filtering() -> None:
    tenant_id = uuid4()
    doc_id = uuid4()

    embedder = AsyncMock()
    embedder.embed.return_value = [[0.1, 0.2]]

    # Match 1: score 0.9, access_level public (pass)
    # Match 2: score 0.4, access_level internal (fail min_score=0.5)
    # Match 3: score 0.95, access_level restricted (fail clearance=internal)
    # Match 4: malformed id (fail parse)
    matches = [
        VectorMatch(
            id=chunk_vector_id(doc_id, 0),
            score=0.9,
            metadata={"access_level": "public", "document_id": str(doc_id)},
        ),
        VectorMatch(
            id=chunk_vector_id(doc_id, 1),
            score=0.4,
            metadata={"access_level": "internal", "document_id": str(doc_id)},
        ),
        VectorMatch(
            id=chunk_vector_id(doc_id, 2),
            score=0.95,
            metadata={"access_level": "restricted", "document_id": str(doc_id)},
        ),
        VectorMatch(
            id="bad-format",
            score=0.99,
            metadata={"access_level": "public"},
        ),
    ]

    vector_store = AsyncMock()
    vector_store.query.return_value = matches

    mock_chunk = Chunk(
        id=uuid4(),
        document_id=doc_id,
        tenant_id=tenant_id,
        chunk_index=0,
        content="Public chunk content",
        char_count=20,
        access_level=AccessLevel.PUBLIC,
        page_start=1,
        page_end=1,
        section_path="Intro",
    )
    db = MockDatabase([(mock_chunk, "Test Document Title")])

    retriever = DenseRetriever(embedder=embedder, vector_store=vector_store, database=db)

    results = await retriever.retrieve(
        tenant_id=tenant_id,
        query="search query",
        top_k=5,
        clearance=AccessLevel.INTERNAL,
        min_score=0.5,
    )

    assert len(results) == 1
    chunk = results[0]
    assert chunk.document_id == doc_id
    assert chunk.document_title == "Test Document Title"
    assert chunk.chunk_index == 0
    assert chunk.content == "Public chunk content"
    assert chunk.score == 0.9
    assert chunk.access_level == AccessLevel.PUBLIC
    assert chunk.page_start == 1
    assert chunk.section_path == "Intro"


async def test_equal_scores_are_ordered_by_stable_document_fields() -> None:
    tenant_id = uuid4()
    first_doc = uuid4()
    second_doc = uuid4()

    embedder = AsyncMock()
    embedder.embed.return_value = [[0.1, 0.2]]
    matches = [
        VectorMatch(id=chunk_vector_id(first_doc, 0), score=0.5),
        VectorMatch(id=chunk_vector_id(second_doc, 0), score=0.5),
    ]
    vector_store = AsyncMock()
    vector_store.query.return_value = matches

    zulu = Chunk(
        id=uuid4(),
        document_id=first_doc,
        tenant_id=tenant_id,
        chunk_index=0,
        content="Zulu content",
        char_count=12,
        access_level=AccessLevel.INTERNAL,
    )
    alpha = Chunk(
        id=uuid4(),
        document_id=second_doc,
        tenant_id=tenant_id,
        chunk_index=0,
        content="Alpha content",
        char_count=13,
        access_level=AccessLevel.INTERNAL,
    )
    # PostgreSQL row order is not guaranteed, so return the losing order.
    db = MockDatabase([(zulu, "Zulu Handbook"), (alpha, "Alpha Handbook")])

    results = await DenseRetriever(
        embedder=embedder, vector_store=vector_store, database=db
    ).retrieve(tenant_id=tenant_id, query="query", top_k=2)

    assert [chunk.document_title for chunk in results] == [
        "Alpha Handbook",
        "Zulu Handbook",
    ]


def test_build_retriever() -> None:
    settings = Settings()
    retriever = build_retriever(
        settings,
        embedder=AsyncMock(),
        vector_store=AsyncMock(),
        database=MockDatabase([]),
    )
    assert isinstance(retriever, DenseRetriever)
