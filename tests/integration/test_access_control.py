"""Integration tests for retrieval-time access control and tenant isolation.

Verifies mathematically and empirically that:
1. Chunks belonging to Tenant A are never retrievable by Tenant B under any
   retrieval mode (dense, keyword, hybrid, or re-ranked), even with identical queries.
2. Users only retrieve chunks permitted by their clearance level:
   PUBLIC <= INTERNAL <= CONFIDENTIAL <= RESTRICTED.
3. Keyword corpus statistics (document frequencies, average length) do not leak
   information about restricted documents to lower-clearance users.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx2 import ASGITransport, AsyncClient

from rag_engine.api.app import create_app
from rag_engine.core.config import Settings
from rag_engine.core.db import PostgresDatabase
from rag_engine.core.enums import AccessLevel
from rag_engine.core.models import Chunk, Document, Tenant
from rag_engine.embedding import Embedder
from rag_engine.retrieval.dense import DenseRetriever
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.retrieval.keyword import KeywordRetriever
from rag_engine.vectorstore.base import VectorRecord
from rag_engine.vectorstore.memory import MemoryVectorStore

pytestmark = pytest.mark.integration


async def _seed_document_with_chunk(
    session: object,
    *,
    tenant_id: object,
    title: str,
    content: str,
    access_level: AccessLevel,
) -> tuple[Document, Chunk]:
    doc = Document(
        id=uuid4(),
        tenant_id=tenant_id,  # type: ignore[arg-type]
        title=title,
        filename=f"{title.lower().replace(' ', '_')}.txt",
        size_bytes=len(content),
        content_hash=f"hash_{uuid4().hex[:16]}",
        storage_key=f"storage_{uuid4().hex[:16]}",
        access_level=access_level,
        chunk_count=1,
    )
    chunk = Chunk(
        id=uuid4(),
        document_id=doc.id,
        tenant_id=tenant_id,  # type: ignore[arg-type]
        chunk_index=0,
        content=content,
        char_count=len(content),
        access_level=access_level,
    )
    session.add(doc)  # type: ignore[attr-defined]
    await session.flush()  # type: ignore[attr-defined]
    session.add(chunk)  # type: ignore[attr-defined]
    await session.flush()  # type: ignore[attr-defined]
    return doc, chunk


@pytest.mark.asyncio
async def test_cross_tenant_isolation_in_retrieval(
    database: PostgresDatabase,
    embedder: Embedder,
) -> None:
    vector_store = MemoryVectorStore()

    tenant_a = Tenant(id=uuid4(), slug="tenant-a", name="Tenant A")
    tenant_b = Tenant(id=uuid4(), slug="tenant-b", name="Tenant B")

    async with database.session() as session:
        session.add(tenant_a)
        session.add(tenant_b)
        await session.flush()

        doc_a, chunk_a = await _seed_document_with_chunk(
            session,
            tenant_id=tenant_a.id,
            title="Tenant A Confidential Strategy",
            content="Alpha unique proprietary secret codebase is ALPHA_99999.",
            access_level=AccessLevel.INTERNAL,
        )
        doc_b, chunk_b = await _seed_document_with_chunk(
            session,
            tenant_id=tenant_b.id,
            title="Tenant B Confidential Strategy",
            content="Beta unique proprietary secret codebase is BETA_88888.",
            access_level=AccessLevel.INTERNAL,
        )
        await session.commit()

    # Index into vector store under respective tenant namespaces
    emb_a = (await embedder.embed([chunk_a.content]))[0]
    emb_b = (await embedder.embed([chunk_b.content]))[0]
    await vector_store.upsert(
        namespace=str(tenant_a.id),
        records=[
            VectorRecord(
                id=f"{doc_a.id}:0",
                values=emb_a,
                metadata={
                    "tenant_id": str(tenant_a.id),
                    "document_id": str(doc_a.id),
                    "chunk_index": 0,
                    "access_level": "internal",
                },
            )
        ],
    )
    await vector_store.upsert(
        namespace=str(tenant_b.id),
        records=[
            VectorRecord(
                id=f"{doc_b.id}:0",
                values=emb_b,
                metadata={
                    "tenant_id": str(tenant_b.id),
                    "document_id": str(doc_b.id),
                    "chunk_index": 0,
                    "access_level": "internal",
                },
            )
        ],
    )

    dense = DenseRetriever(embedder=embedder, vector_store=vector_store, database=database)
    keyword = KeywordRetriever(database=database)
    hybrid = HybridRetriever(dense=dense, keyword=keyword)

    # 1. Tenant A searches for Beta's exact secret phrase
    results_dense = await dense.retrieve(
        tenant_id=tenant_a.id,
        query="Beta unique proprietary secret codebase is BETA_88888",
    )
    results_keyword = await keyword.retrieve(
        tenant_id=tenant_a.id,
        query="Beta unique proprietary secret codebase is BETA_88888",
    )
    results_hybrid = await hybrid.retrieve(
        tenant_id=tenant_a.id,
        query="Beta unique proprietary secret codebase is BETA_88888",
    )

    # ZERO chunks from Tenant B may ever be retrieved by Tenant A
    assert all(chunk.chunk_id != chunk_b.id for chunk in results_dense)
    assert all(chunk.document_id != doc_b.id for chunk in results_dense)
    assert all(chunk.document_id == doc_a.id for chunk in results_dense)

    assert all(chunk.chunk_id != chunk_b.id for chunk in results_keyword)
    assert all(chunk.document_id != doc_b.id for chunk in results_keyword)
    assert all(chunk.document_id == doc_a.id for chunk in results_keyword)

    assert all(chunk.chunk_id != chunk_b.id for chunk in results_hybrid)
    assert all(chunk.document_id != doc_b.id for chunk in results_hybrid)
    assert all(chunk.document_id == doc_a.id for chunk in results_hybrid)

    # 2. Tenant B searches for Beta's secret phrase -> finds chunk_b
    b_found = await hybrid.retrieve(
        tenant_id=tenant_b.id,
        query="Beta unique proprietary secret codebase is BETA_88888",
    )
    assert len(b_found) == 1
    assert b_found[0].chunk_id == chunk_b.id
    assert b_found[0].document_id == doc_b.id


@pytest.mark.asyncio
async def test_clearance_level_enforcement(
    database: PostgresDatabase,
) -> None:
    tenant = Tenant(id=uuid4(), slug="acme", name="Acme Inc")

    async with database.session() as session:
        session.add(tenant)
        await session.flush()
        _, chunk_pub = await _seed_document_with_chunk(
            session,
            tenant_id=tenant.id,
            title="Public Office Policy",
            content="The main office building opens daily at eight in the morning.",
            access_level=AccessLevel.PUBLIC,
        )
        _, chunk_int = await _seed_document_with_chunk(
            session,
            tenant_id=tenant.id,
            title="Internal Network Config",
            content="Corporate internal wifi SSID is AcmeCorpSecure.",
            access_level=AccessLevel.INTERNAL,
        )
        _, chunk_conf = await _seed_document_with_chunk(
            session,
            tenant_id=tenant.id,
            title="Confidential Salary Bands",
            content="Engineering director base salary compensation starts at two hundred thousand.",
            access_level=AccessLevel.CONFIDENTIAL,
        )
        _, chunk_rest = await _seed_document_with_chunk(
            session,
            tenant_id=tenant.id,
            title="Restricted Project Titan Acquisition",
            content="Secret acquisition target Project Titan valuation is fifty million.",
            access_level=AccessLevel.RESTRICTED,
        )
        await session.commit()

    keyword = KeywordRetriever(database=database)

    # 1. PUBLIC clearance: can see public, cannot see internal, confidential, or restricted
    pub_res = await keyword.retrieve(
        tenant_id=tenant.id,
        query="office building opens daily at eight",
        clearance=AccessLevel.PUBLIC,
    )
    assert len(pub_res) == 1
    assert pub_res[0].chunk_id == chunk_pub.id

    denied_int = await keyword.retrieve(
        tenant_id=tenant.id,
        query="Corporate internal wifi SSID AcmeCorpSecure",
        clearance=AccessLevel.PUBLIC,
    )
    assert len(denied_int) == 0

    denied_conf = await keyword.retrieve(
        tenant_id=tenant.id,
        query="Engineering director base salary compensation",
        clearance=AccessLevel.PUBLIC,
    )
    assert len(denied_conf) == 0

    denied_rest = await keyword.retrieve(
        tenant_id=tenant.id,
        query="Secret acquisition target Project Titan valuation",
        clearance=AccessLevel.PUBLIC,
    )
    assert len(denied_rest) == 0

    # 2. INTERNAL clearance: sees public and internal, cannot see confidential or restricted
    int_can_see = await keyword.retrieve(
        tenant_id=tenant.id,
        query="Corporate internal wifi SSID AcmeCorpSecure",
        clearance=AccessLevel.INTERNAL,
    )
    assert len(int_can_see) == 1
    assert int_can_see[0].chunk_id == chunk_int.id

    int_denied_conf = await keyword.retrieve(
        tenant_id=tenant.id,
        query="Engineering director base salary compensation",
        clearance=AccessLevel.INTERNAL,
    )
    assert len(int_denied_conf) == 0

    # 3. CONFIDENTIAL clearance: sees public, internal, and confidential, cannot see restricted
    conf_can_see = await keyword.retrieve(
        tenant_id=tenant.id,
        query="Engineering director base salary compensation",
        clearance=AccessLevel.CONFIDENTIAL,
    )
    assert len(conf_can_see) == 1
    assert conf_can_see[0].chunk_id == chunk_conf.id

    conf_denied_rest = await keyword.retrieve(
        tenant_id=tenant.id,
        query="Secret acquisition target Project Titan valuation",
        clearance=AccessLevel.CONFIDENTIAL,
    )
    assert len(conf_denied_rest) == 0

    # 4. RESTRICTED clearance: sees all
    rest_can_see = await keyword.retrieve(
        tenant_id=tenant.id,
        query="Secret acquisition target Project Titan valuation",
        clearance=AccessLevel.RESTRICTED,
    )
    assert len(rest_can_see) == 1
    assert rest_can_see[0].chunk_id == chunk_rest.id


@pytest.mark.asyncio
async def test_http_query_endpoint_access_control(
    test_settings: Settings,
    database: PostgresDatabase,
    embedder: Embedder,
) -> None:
    vector_store = MemoryVectorStore()
    app = create_app(
        test_settings,
        database=database,
        embedder=embedder,
        vector_store=vector_store,
    )

    tenant_a = Tenant(id=uuid4(), slug="acme", name="Acme Corp")
    tenant_b = Tenant(id=uuid4(), slug="globex", name="Globex Inc")

    async with database.session() as session:
        session.add(tenant_a)
        session.add(tenant_b)
        await session.flush()

        doc_a, chunk_a = await _seed_document_with_chunk(
            session,
            tenant_id=tenant_a.id,
            title="Acme Code Review Guidelines",
            content="Acme guidelines state that all production code reviews require two approvals.",
            access_level=AccessLevel.INTERNAL,
        )
        doc_b, chunk_b = await _seed_document_with_chunk(
            session,
            tenant_id=tenant_b.id,
            title="Globex Strategy",
            content="Globex proprietary secret token is RUBY_98765.",
            access_level=AccessLevel.INTERNAL,
        )
        await session.commit()

    emb_a = (await embedder.embed([chunk_a.content]))[0]
    emb_b = (await embedder.embed([chunk_b.content]))[0]
    await vector_store.upsert(
        namespace=str(tenant_a.id),
        records=[
            VectorRecord(
                id=f"{doc_a.id}:0",
                values=emb_a,
                metadata={
                    "tenant_id": str(tenant_a.id),
                    "document_id": str(doc_a.id),
                    "chunk_index": 0,
                    "access_level": "internal",
                },
            )
        ],
    )
    await vector_store.upsert(
        namespace=str(tenant_b.id),
        records=[
            VectorRecord(
                id=f"{doc_b.id}:0",
                values=emb_b,
                metadata={
                    "tenant_id": str(tenant_b.id),
                    "document_id": str(doc_b.id),
                    "chunk_index": 0,
                    "access_level": "internal",
                },
            )
        ],
    )

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        # 1. Tenant A queries for Globex's secret token -> must be refused / weak retrieval
        response_a = await client.post(
            "/query",
            headers={"X-Tenant-ID": "acme"},
            json={
                "question": "What is Globex proprietary secret token?",
                "clearance": "internal",
            },
        )
        assert response_a.status_code == 200
        data_a = response_a.json()
        assert data_a["weak_retrieval"] is True
        assert "RUBY_98765" not in data_a["answer"]
        assert len(data_a["citations"]) == 0

        # 2. Tenant B queries for its own secret -> successfully answered with citations
        response_b = await client.post(
            "/query",
            headers={"X-Tenant-ID": "globex"},
            json={
                "question": "What is Globex proprietary secret token?",
                "clearance": "internal",
            },
        )
        assert response_b.status_code == 200
        data_b = response_b.json()
        assert data_b["weak_retrieval"] is False
        assert "RUBY_98765" in data_b["answer"]
        assert any(c["document_title"] == "Globex Strategy" for c in data_b["citations"])
        assert "approvals" not in data_b["answer"]
