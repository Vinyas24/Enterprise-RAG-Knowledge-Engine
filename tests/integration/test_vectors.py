"""Integration tests for Phase 3: Embedding and vector storage.

Acceptance criteria:
- Chunks are searchable by meaning (query returns relevant vectors).
- Ingested documents appear in vector storage with correct metadata.
- Namespace strategy ensures multi-tenancy isolation.
- Idempotent re-ingestion: re-uploading / updating a document replaces its vectors cleanly.
- Deleting a document removes its rows in PostgreSQL and its vectors in the vector store.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx2 import ASGITransport, AsyncClient

from rag_engine.api.app import create_app
from rag_engine.core.config import Settings
from rag_engine.core.db import PostgresDatabase
from rag_engine.core.enums import AccessLevel, DocumentStatus, JobStatus
from rag_engine.core.ids import chunk_vector_id
from rag_engine.core.models import Chunk, Document, IngestionJob, Tenant
from rag_engine.embedding import Embedder
from rag_engine.ingestion.blobstore import LocalBlobStore
from rag_engine.ingestion.chunking import ChunkDraft, build_chunker
from rag_engine.ingestion.parsers import ParserRegistry
from rag_engine.ingestion.processor import IngestionProcessor
from rag_engine.vectorstore.base import VectorRecord
from rag_engine.vectorstore.errors import VectorStoreUnavailable
from rag_engine.vectorstore.memory import MemoryVectorStore
from rag_engine.vectorstore.pinecone import PineconeVectorStore

pytestmark = pytest.mark.integration

SAMPLE_TEXT = (
    "Artificial intelligence and machine learning models require high quality embeddings. "
    "Dense retrieval allows semantic similarity search across enterprise knowledge bases. "
    "Multi-tenancy guarantees that each organization's data remains isolated. "
) * 10


async def _upload(
    client: AsyncClient,
    headers: dict[str, str],
    *,
    filename: str = "notes.txt",
    data: bytes = SAMPLE_TEXT.encode(),
    access_level: str = "internal",
    title: str = "Vector Test Document",
) -> Any:
    return await client.post(
        "/documents",
        headers=headers,
        files={"file": (filename, data, "text/plain")},
        data={"access_level": access_level, "title": title},
    )


async def test_ingested_chunks_appear_in_vector_store(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    tenant: Tenant,
    processor: IngestionProcessor,
    vector_store: MemoryVectorStore,
    database: PostgresDatabase,
) -> None:
    resp = await _upload(api_client, auth_headers)
    assert resp.status_code == 202
    doc_id = UUID(resp.json()["document_id"])

    outcome = await processor.process_next()
    assert outcome.succeeded is True
    assert outcome.chunk_count > 0

    namespace = str(tenant.id)
    stored_ids = await vector_store.list_ids(namespace)
    assert len(stored_ids) == outcome.chunk_count

    expected_ids = [chunk_vector_id(doc_id, i) for i in range(outcome.chunk_count)]
    assert sorted(stored_ids) == sorted(expected_ids)

    # Verify each vector record and its metadata
    async with database.session() as session:
        doc = await session.get(Document, doc_id)
        assert doc is not None
        assert doc.status == DocumentStatus.INDEXED

    # Query vector store
    # Query with a dummy vector of the correct dimension
    query_vec = [0.1] * 256
    matches = await vector_store.query(namespace, query_vec, top_k=5)
    assert len(matches) > 0
    top_match = matches[0]
    assert top_match.metadata["document_id"] == str(doc_id)
    assert top_match.metadata["tenant_id"] == str(tenant.id)
    assert top_match.metadata["access_level"] == "internal"
    assert "chunk_index" in top_match.metadata


async def test_multitenancy_vector_namespace_isolation(
    api_client: AsyncClient,
    create_tenant: Any,
    processor: IngestionProcessor,
    vector_store: MemoryVectorStore,
) -> None:
    tenant_a: Tenant = await create_tenant("tenant-alpha", "Tenant Alpha")
    tenant_b: Tenant = await create_tenant("tenant-beta", "Tenant Beta")

    headers_a = {"X-Tenant-ID": tenant_a.slug}
    headers_b = {"X-Tenant-ID": tenant_b.slug}

    resp_a = await _upload(api_client, headers_a, filename="alpha.txt", data=b"Alpha content " * 40)
    resp_b = await _upload(api_client, headers_b, filename="beta.txt", data=b"Beta content " * 40)
    doc_a_id = UUID(resp_a.json()["document_id"])
    doc_b_id = UUID(resp_b.json()["document_id"])

    # Process both jobs
    assert (await processor.process_next()).succeeded is True
    assert (await processor.process_next()).succeeded is True

    ns_a = str(tenant_a.id)
    ns_b = str(tenant_b.id)

    ids_a = await vector_store.list_ids(ns_a)
    ids_b = await vector_store.list_ids(ns_b)

    assert len(ids_a) > 0
    assert len(ids_b) > 0

    # Ensure all vectors in ns_a belong to doc_a
    assert all(vid.startswith(f"{doc_a_id}:") for vid in ids_a)
    # Ensure all vectors in ns_b belong to doc_b
    assert all(vid.startswith(f"{doc_b_id}:") for vid in ids_b)

    # Cross-namespace search is isolated
    matches_in_a = await vector_store.query(ns_a, [0.1] * 256, top_k=10)
    assert all(m.metadata["tenant_id"] == ns_a for m in matches_in_a)
    assert not any(m.metadata["document_id"] == str(doc_b_id) for m in matches_in_a)


async def test_document_deletion_removes_from_postgres_and_vector_store(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    tenant: Tenant,
    processor: IngestionProcessor,
    vector_store: MemoryVectorStore,
    database: PostgresDatabase,
) -> None:
    resp = await _upload(api_client, auth_headers)
    doc_id = UUID(resp.json()["document_id"])
    assert (await processor.process_next()).succeeded is True

    namespace = str(tenant.id)
    assert len(await vector_store.list_ids(namespace)) > 0

    # Delete document via API
    del_resp = await api_client.delete(f"/documents/{doc_id}", headers=auth_headers)
    assert del_resp.status_code == 204

    # Verify document and chunks are gone from PostgreSQL
    async with database.session() as session:
        doc = await session.get(Document, doc_id)
        assert doc is None

        chunks = (
            await session.execute(Chunk.__table__.select().where(Chunk.document_id == doc_id))
        ).all()
        assert len(chunks) == 0

    # Verify vectors are gone from vector store
    vectors_after = await vector_store.list_ids(namespace)
    assert len(vectors_after) == 0


async def test_delete_nonexistent_or_other_tenant_document(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    create_tenant: Any,
) -> None:
    # 404 for unknown document
    resp = await api_client.delete(f"/documents/{uuid4()}", headers=auth_headers)
    assert resp.status_code == 404

    # 404 when trying to delete another tenant's document
    other_tenant: Tenant = await create_tenant("other-tenant", "Other")
    other_headers = {"X-Tenant-ID": other_tenant.slug}
    upload_resp = await _upload(api_client, other_headers)
    other_doc_id = upload_resp.json()["document_id"]

    cross_del = await api_client.delete(f"/documents/{other_doc_id}", headers=auth_headers)
    assert cross_del.status_code == 404


async def test_idempotent_reingestion_cleans_stale_vectors(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    tenant: Tenant,
    processor: IngestionProcessor,
    vector_store: MemoryVectorStore,
    database: PostgresDatabase,
) -> None:
    # Ingest document first time
    resp = await _upload(api_client, auth_headers)
    doc_id = UUID(resp.json()["document_id"])
    outcome = await processor.process_next()
    assert outcome.succeeded is True
    initial_chunks = outcome.chunk_count

    namespace = str(tenant.id)
    assert len(await vector_store.list_ids(namespace)) == initial_chunks

    # Simulate re-processing with fewer drafts
    from rag_engine.ingestion import repository

    async with database.session() as session:
        shorter_drafts = [
            ChunkDraft(
                content="Single replacement chunk",
                char_count=24,
                page_start=1,
                page_end=1,
                section_path=None,
            )
        ]
        job = await repository.latest_job_for_document(session, document_id=doc_id)
        assert job is not None

        # Call complete_job directly with fewer drafts
        await repository.complete_job(
            session,
            job_id=job.id,
            document_id=doc_id,
            tenant_id=tenant.id,
            drafts=shorter_drafts,
            access_level=AccessLevel.INTERNAL,
        )
        await session.commit()

    # Old vector IDs before update
    old_vector_ids = [chunk_vector_id(doc_id, i) for i in range(initial_chunks)]
    new_vector_ids = [chunk_vector_id(doc_id, 0)]

    # Upsert new vector
    await vector_store.upsert(
        namespace,
        [
            VectorRecord(
                id=new_vector_ids[0],
                values=[0.2] * 256,
                metadata={"document_id": str(doc_id), "chunk_index": 0},
            )
        ],
    )

    # Stale vector cleanup
    from rag_engine.vectorstore.maintenance import delete_stale_vectors

    deleted_count = await delete_stale_vectors(
        vector_store,
        namespace,
        doc_id,
        keep_ids=new_vector_ids,
        previous_ids=old_vector_ids,
    )

    assert deleted_count == initial_chunks - 1
    current_ids = await vector_store.list_ids(namespace)
    assert current_ids == new_vector_ids


async def test_pinecone_local_sync_and_query() -> None:
    """Exercise PineconeVectorStore against the running Pinecone Local container."""
    try:
        store = PineconeVectorStore(
            api_key="pcsk-local-test",
            index_name="phase3-test",
            dimension=4,
            control_host="http://pinecone-phase3-test:5080",
        )
        await store.ensure_ready()
    except Exception as exc:
        pytest.skip(f"Pinecone Local is not reachable: {exc}")

    test_ns = f"test-ns-{uuid4()}"
    test_doc = uuid4()
    v0 = chunk_vector_id(test_doc, 0)
    v1 = chunk_vector_id(test_doc, 1)

    records = [
        VectorRecord(
            id=v0,
            values=[1.0, 0.0, 0.0, 0.0],
            metadata={"doc": str(test_doc), "chunk": 0},
        ),
        VectorRecord(
            id=v1,
            values=[0.0, 1.0, 0.0, 0.0],
            metadata={"doc": str(test_doc), "chunk": 1},
        ),
    ]

    await store.upsert(test_ns, records)
    ids = await store.list_ids(test_ns)
    assert sorted(ids) == sorted([v0, v1])

    # Query
    matches = await store.query(test_ns, [1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(matches) == 2
    assert matches[0].id == v0
    assert pytest.approx(matches[0].score, abs=1e-4) == 1.0

    # Delete
    await store.delete(test_ns, [v0])
    ids_after = await store.list_ids(test_ns)
    assert ids_after == [v1]

    # Cleanup
    await store.delete(test_ns, [v1])
    await store.close()


# ---------------------------------------------------------------------------
# Cross-store consistency: failure paths and reindex
# ---------------------------------------------------------------------------


class _FailingUpsertStore(MemoryVectorStore):
    """Vector store that accepts the index but rejects every write."""

    async def upsert(self, namespace: str, records: Sequence[VectorRecord]) -> None:
        raise VectorStoreUnavailable("simulated vector-store outage")


class _FailingDeleteStore(MemoryVectorStore):
    """Vector store that can be read but not written during deletion."""

    async def delete(self, namespace: str, ids: Sequence[str]) -> None:
        raise VectorStoreUnavailable("simulated vector-store outage")


def _processor_with(
    settings: Settings,
    *,
    database: PostgresDatabase,
    storage: LocalBlobStore,
    embedder: Embedder,
    vector_store: MemoryVectorStore,
) -> IngestionProcessor:
    return IngestionProcessor(
        database=database,
        blob_store=storage,
        parsers=ParserRegistry.default(),
        chunker=build_chunker(settings),
        embedder=embedder,
        vector_store=vector_store,
        settings=settings,
    )


@asynccontextmanager
async def _client_for_store(
    settings: Settings,
    *,
    database: PostgresDatabase,
    storage: LocalBlobStore,
    embedder: Embedder,
    vector_store: MemoryVectorStore,
) -> AsyncIterator[AsyncClient]:
    application = create_app(
        settings,
        database=database,
        blob_store=storage,
        vector_store=vector_store,
        embedder=embedder,
    )
    async with application.router.lifespan_context(application):
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


async def test_vector_upsert_failure_keeps_job_retryable_and_old_chunks(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    tenant: Tenant,
    test_settings: Settings,
    database: PostgresDatabase,
    storage: LocalBlobStore,
    embedder: Embedder,
) -> None:
    """A vector failure must not mark the document indexed."""
    resp = await _upload(api_client, auth_headers)
    doc_id = UUID(resp.json()["document_id"])

    processor = _processor_with(
        test_settings,
        database=database,
        storage=storage,
        embedder=embedder,
        vector_store=_FailingUpsertStore(dimension=test_settings.embedding_dimensions),
    )
    outcome = await processor.process_next()
    assert outcome.succeeded is False

    async with database.session() as session:
        document = await session.get(Document, doc_id)
        assert document is not None
        assert document.status is DocumentStatus.PENDING
        assert document.chunk_count == 0

        job = await session.get(IngestionJob, outcome.job_id)
        assert job is not None
        assert job.status is JobStatus.QUEUED
        assert job.attempts == 1
        assert job.error is not None and "VectorStoreUnavailable" in job.error


async def test_delete_returns_503_and_keeps_document_when_vectors_fail(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    tenant: Tenant,
    processor: IngestionProcessor,
    vector_store: MemoryVectorStore,
    test_settings: Settings,
    database: PostgresDatabase,
    storage: LocalBlobStore,
    embedder: Embedder,
) -> None:
    resp = await _upload(api_client, auth_headers)
    doc_id = UUID(resp.json()["document_id"])
    assert (await processor.process_next()).succeeded is True
    namespace = str(tenant.id)
    vectors_before = await vector_store.list_ids(namespace)
    assert vectors_before

    failing_store = _FailingDeleteStore(dimension=test_settings.embedding_dimensions)
    await failing_store.upsert(
        namespace,
        [
            VectorRecord(
                id=vector_id,
                values=[0.0] * test_settings.embedding_dimensions,
                metadata={"document_id": str(doc_id)},
            )
            for vector_id in vectors_before
        ],
    )

    async with _client_for_store(
        test_settings,
        database=database,
        storage=storage,
        embedder=embedder,
        vector_store=failing_store,
    ) as client:
        deleted = await client.delete(f"/documents/{doc_id}", headers=auth_headers)
        assert deleted.status_code == 503
        assert deleted.json()["error"]["code"] == "vector_store_unavailable"

        # The document is still readable and its vectors are untouched.
        still_there = await client.get(f"/documents/{doc_id}", headers=auth_headers)
        assert still_there.status_code == 200

    assert await failing_store.list_ids(namespace) == vectors_before


async def test_reindex_replaces_vectors_and_removes_stale_tail(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    tenant: Tenant,
    processor: IngestionProcessor,
    vector_store: MemoryVectorStore,
    database: PostgresDatabase,
) -> None:
    resp = await _upload(api_client, auth_headers)
    doc_id = UUID(resp.json()["document_id"])
    first = await processor.process_next()
    assert first.succeeded is True

    namespace = str(tenant.id)
    stale_id = chunk_vector_id(doc_id, first.chunk_count + 5)
    await vector_store.upsert(
        namespace,
        [VectorRecord(id=stale_id, values=[0.0] * 256, metadata={"document_id": str(doc_id)})],
    )

    reindexed = await api_client.post(f"/documents/{doc_id}/reindex", headers=auth_headers)
    assert reindexed.status_code == 202
    assert reindexed.json()["document_id"] == str(doc_id)

    second = await processor.process_next()
    assert second.succeeded is True
    assert second.chunk_count == first.chunk_count

    expected = [chunk_vector_id(doc_id, i) for i in range(second.chunk_count)]
    assert await vector_store.list_ids(namespace) == sorted(expected)

    async with database.session() as session:
        document = await session.get(Document, doc_id)
        assert document is not None
        assert document.status is DocumentStatus.INDEXED


async def test_reindex_conflicts_with_active_job(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    tenant: Tenant,
) -> None:
    resp = await _upload(api_client, auth_headers)
    doc_id = UUID(resp.json()["document_id"])

    conflict = await api_client.post(f"/documents/{doc_id}/reindex", headers=auth_headers)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "job_in_progress"


async def test_delete_conflicts_with_active_job(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    resp = await _upload(api_client, auth_headers)
    doc_id = UUID(resp.json()["document_id"])

    conflict = await api_client.delete(f"/documents/{doc_id}", headers=auth_headers)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "job_in_progress"

    still_there = await api_client.get(f"/documents/{doc_id}", headers=auth_headers)
    assert still_there.status_code == 200


async def test_delete_reports_409_when_a_job_becomes_active(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final conditional delete is the last line of defence."""
    resp = await _upload(api_client, auth_headers)
    doc_id = UUID(resp.json()["document_id"])
    assert (await processor.process_next()).succeeded is True

    from rag_engine.ingestion import repository

    async def _busy(*_args: Any, **_kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(repository, "delete_document_if_idle", _busy)

    conflict = await api_client.delete(f"/documents/{doc_id}", headers=auth_headers)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "job_in_progress"

    still_there = await api_client.get(f"/documents/{doc_id}", headers=auth_headers)
    assert still_there.status_code == 200


async def test_reindex_of_other_tenants_document_is_404(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    create_tenant: Any,
) -> None:
    other: Tenant = await create_tenant("reindex-other", "Other")
    uploaded = await _upload(api_client, {"X-Tenant-ID": other.slug})
    other_doc_id = uploaded.json()["document_id"]

    resp = await api_client.post(f"/documents/{other_doc_id}/reindex", headers=auth_headers)
    assert resp.status_code == 404
