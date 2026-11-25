"""Keyword retrieval against real PostgreSQL.

These tests cover what a unit test with a scripted session cannot: that the
generated ``content_tsv`` column is maintained by the database, that the GIN
index supports the candidate query, and that tenant and clearance filtering
happen in SQL rather than being trusted from a client.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select, text

from rag_engine.core.db import PostgresDatabase
from rag_engine.core.enums import AccessLevel, DocumentStatus
from rag_engine.core.models import Chunk, Document, Tenant
from rag_engine.ingestion import repository
from rag_engine.ingestion.chunking import ChunkDraft
from rag_engine.retrieval.keyword import KeywordRetriever

pytestmark = pytest.mark.integration


def _draft(content: str) -> ChunkDraft:
    return ChunkDraft(
        content=content,
        char_count=len(content),
        page_start=None,
        page_end=None,
        section_path=None,
    )


async def _index_document(
    database: PostgresDatabase,
    tenant: Tenant,
    *,
    title: str,
    access_level: AccessLevel,
    contents: list[str],
) -> Document:
    document = Document(
        tenant_id=tenant.id,
        title=title,
        filename="probe.txt",
        content_type="text/plain",
        size_bytes=sum(len(content) for content in contents),
        content_hash=uuid4().hex * 2,
        storage_key=f"probe/{uuid4().hex}",
        access_level=access_level,
        status=DocumentStatus.INDEXED,
    )
    async with database.session() as session:
        session.add(document)
        await session.flush()
        await repository.replace_chunks(
            session,
            document_id=document.id,
            tenant_id=tenant.id,
            drafts=[_draft(content) for content in contents],
            access_level=access_level,
        )
        await session.commit()
    return document


async def test_generated_tsvector_column_is_populated(
    database: PostgresDatabase, create_tenant: Any
) -> None:
    tenant = await create_tenant("keyword-tsv", "Keyword TSV")
    await _index_document(
        database,
        tenant,
        title="Runbook",
        access_level=AccessLevel.INTERNAL,
        contents=["Roll back a failed Kubernetes deployment."],
    )

    async with database.session() as session:
        row = (await session.execute(select(Chunk.content_tsv, Chunk.content))).one()

    assert "deploy" in str(row.content_tsv)
    assert "kubernetes" in str(row.content_tsv)
    # The generated column is derived, not a second source of truth.
    assert row.content == "Roll back a failed Kubernetes deployment."


async def test_full_text_candidate_query_uses_the_gin_index(
    database: PostgresDatabase, create_tenant: Any
) -> None:
    tenant = await create_tenant("keyword-index", "Keyword Index")
    await _index_document(
        database,
        tenant,
        title="Runbook",
        access_level=AccessLevel.INTERNAL,
        contents=["Roll back a failed Kubernetes deployment."],
    )

    async with database.session() as session:
        indexes = (
            (
                await session.execute(
                    text(
                        "SELECT indexdef FROM pg_indexes "
                        "WHERE tablename = 'chunks' AND indexname = 'ix_chunks_content_tsv'"
                    )
                )
            )
            .scalars()
            .all()
        )

    assert len(indexes) == 1
    assert "gin" in indexes[0].lower()
    assert "content_tsv" in indexes[0]


async def test_keyword_retrieval_respects_tenant_and_clearance(
    database: PostgresDatabase, create_tenant: Any
) -> None:
    tenant = await create_tenant("keyword-acme", "Acme")
    other_tenant = await create_tenant("keyword-other", "Other")

    await _index_document(
        database,
        tenant,
        title="Public Runbook",
        access_level=AccessLevel.PUBLIC,
        contents=["Roll back a failed deployment with the public runbook."],
    )
    await _index_document(
        database,
        tenant,
        title="Internal Runbook",
        access_level=AccessLevel.INTERNAL,
        contents=["Roll back a failed deployment using the internal change process."],
    )
    await _index_document(
        database,
        tenant,
        title="Restricted Runbook",
        access_level=AccessLevel.RESTRICTED,
        contents=["Roll back a failed deployment deployment deployment in the restricted cluster."],
    )
    # A different tenant with the strongest lexical match of all. It must never
    # appear, no matter which clearance is used.
    await _index_document(
        database,
        other_tenant,
        title="Other Runbook",
        access_level=AccessLevel.PUBLIC,
        contents=["deployment deployment deployment deployment roll back failed"],
    )

    retriever = KeywordRetriever(database=database)

    internal = await retriever.retrieve(
        tenant_id=tenant.id,
        query="roll back a failed deployment",
        top_k=10,
        clearance=AccessLevel.INTERNAL,
    )
    restricted = await retriever.retrieve(
        tenant_id=tenant.id,
        query="roll back a failed deployment",
        top_k=10,
        clearance=AccessLevel.RESTRICTED,
    )

    assert {item.document_title for item in internal} == {"Public Runbook", "Internal Runbook"}
    assert {item.document_title for item in restricted} == {
        "Public Runbook",
        "Internal Runbook",
        "Restricted Runbook",
    }
    assert all(item.access_level is not AccessLevel.RESTRICTED for item in internal)


async def test_keyword_retrieval_is_scoped_to_the_requested_tenant(
    database: PostgresDatabase, create_tenant: Any
) -> None:
    tenant = await create_tenant("keyword-one", "One")
    other_tenant = await create_tenant("keyword-two", "Two")
    await _index_document(
        database,
        tenant,
        title="Mine",
        access_level=AccessLevel.PUBLIC,
        contents=["deployment rollback runbook"],
    )
    await _index_document(
        database,
        other_tenant,
        title="Theirs",
        access_level=AccessLevel.PUBLIC,
        contents=["deployment rollback runbook"],
    )

    results = await KeywordRetriever(database=database).retrieve(
        tenant_id=tenant.id, query="deployment rollback", top_k=10
    )

    assert [item.document_title for item in results] == ["Mine"]


async def test_keyword_retrieval_ranks_repeated_terms_higher(
    database: PostgresDatabase, create_tenant: Any
) -> None:
    tenant = await create_tenant("keyword-rank", "Rank")
    await _index_document(
        database,
        tenant,
        title="Mentions Once",
        access_level=AccessLevel.INTERNAL,
        contents=["The deployment checklist lives in the wiki."],
    )
    await _index_document(
        database,
        tenant,
        title="Mentions Often",
        access_level=AccessLevel.INTERNAL,
        contents=["deployment deployment deployment rollback deployment checklist"],
    )

    results = await KeywordRetriever(database=database).retrieve(
        tenant_id=tenant.id, query="deployment", top_k=2
    )

    assert [item.document_title for item in results] == ["Mentions Often", "Mentions Once"]
    assert results[0].score > results[1].score


async def test_keyword_retrieval_returns_nothing_for_an_unknown_tenant(
    database: PostgresDatabase, create_tenant: Any
) -> None:
    tenant = await create_tenant("keyword-known", "Known")
    await _index_document(
        database,
        tenant,
        title="Runbook",
        access_level=AccessLevel.PUBLIC,
        contents=["deployment rollback"],
    )

    results = await KeywordRetriever(database=database).retrieve(
        tenant_id=uuid4(), query="deployment rollback"
    )

    assert results == []
