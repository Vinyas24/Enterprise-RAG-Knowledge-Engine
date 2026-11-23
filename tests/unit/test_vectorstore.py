"""Unit tests for vector store implementations, factory, and maintenance helpers."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import SecretStr

from rag_engine.core.config import Settings
from rag_engine.core.ids import chunk_vector_id
from rag_engine.vectorstore.base import VectorRecord
from rag_engine.vectorstore.errors import VectorStoreConfigurationError
from rag_engine.vectorstore.factory import build_vector_store
from rag_engine.vectorstore.maintenance import (
    delete_document_vectors,
    delete_stale_vectors,
    delete_vectors,
)
from rag_engine.vectorstore.memory import MemoryVectorStore
from rag_engine.vectorstore.pinecone import (
    PineconeVectorStore,
    metadata_for_vector,
)

# ---------------------------------------------------------------------------
# MemoryVectorStore
# ---------------------------------------------------------------------------


async def test_memory_vector_store_basic() -> None:
    store = MemoryVectorStore(dimension=3)
    await store.ensure_ready()

    r1 = VectorRecord(id="v1", values=[1.0, 0.0, 0.0], metadata={"category": "A"})
    r2 = VectorRecord(id="v2", values=[0.0, 1.0, 0.0], metadata={"category": "B"})
    r3 = VectorRecord(id="v3", values=[0.7071, 0.7071, 0.0], metadata={"category": "A"})

    await store.upsert("ns1", [r1, r2, r3])
    assert store.count("ns1") == 3
    assert store.count("ns2") == 0

    # Idempotent upsert (replacement)
    r1_updated = VectorRecord(id="v1", values=[0.0, 0.0, 1.0], metadata={"category": "C"})
    await store.upsert("ns1", [r1_updated])
    assert store.count("ns1") == 3

    # Dimension validation
    with pytest.raises(ValueError, match="dimensions"):
        await store.upsert("ns1", [VectorRecord(id="v_bad", values=[1.0, 0.0])])

    # Listing IDs with and without prefix
    all_ids = await store.list_ids("ns1")
    assert all_ids == ["v1", "v2", "v3"]
    assert await store.list_ids("ns1", prefix="v1") == ["v1"]
    assert await store.list_ids("ns1", prefix="x") == []

    # Query with cosine similarity
    query_vec = [0.0, 1.0, 0.0]
    matches = await store.query("ns1", query_vec, top_k=2)
    assert len(matches) == 2
    assert matches[0].id == "v2"
    assert pytest.approx(matches[0].score, abs=1e-4) == 1.0

    # Query with metadata filter
    matches_filtered = await store.query(
        "ns1", query_vec, top_k=5, metadata_filter={"category": "A"}
    )
    assert len(matches_filtered) == 1
    assert matches_filtered[0].id == "v3"

    # Deletion
    await store.delete("ns1", ["v1", "non_existent"])
    assert store.count("ns1") == 2
    assert await store.list_ids("ns1") == ["v2", "v3"]

    await store.close()
    assert store.closed is True


# ---------------------------------------------------------------------------
# PineconeVectorStore
# ---------------------------------------------------------------------------


def test_pinecone_validation() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        PineconeVectorStore(api_key="", index_name="test", dimension=4)
    with pytest.raises(ValueError, match="batch sizes must be positive"):
        PineconeVectorStore(api_key="key", index_name="test", dimension=4, upsert_batch_size=0)


async def test_pinecone_ensure_ready_creates_index(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client_class = MagicMock()
    mock_client = mock_client_class.return_value
    mock_client.has_index.return_value = False
    mock_client.describe_index.return_value = {
        "dimension": 4,
        "metric": "cosine",
        "host": "test-host.pinecone.io",
    }
    monkeypatch.setattr("rag_engine.vectorstore.pinecone.Pinecone", mock_client_class)

    store = PineconeVectorStore(
        api_key="pcsk-test",
        index_name="my-index",
        dimension=4,
        control_host="https://control.pinecone.io",
    )
    await store.ensure_ready()

    mock_client.create_index.assert_called_once()
    assert mock_client.create_index.call_args[1]["name"] == "my-index"
    assert mock_client.create_index.call_args[1]["dimension"] == 4
    mock_client.Index.assert_called_once_with(host="https://test-host.pinecone.io")

    # Second call should be a no-op
    await store.ensure_ready()
    assert mock_client.create_index.call_count == 1

    await store.close()


async def test_pinecone_dimension_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client_class = MagicMock()
    mock_client = mock_client_class.return_value
    mock_client.has_index.return_value = True
    mock_client.describe_index.return_value = {"dimension": 8, "metric": "cosine"}
    monkeypatch.setattr("rag_engine.vectorstore.pinecone.Pinecone", mock_client_class)

    store = PineconeVectorStore(api_key="pcsk-test", index_name="my-index", dimension=4)
    with pytest.raises(VectorStoreConfigurationError, match="dimension 8"):
        await store.ensure_ready()


async def test_pinecone_metric_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client_class = MagicMock()
    mock_client = mock_client_class.return_value
    mock_client.has_index.return_value = True
    mock_client.describe_index.return_value = {"dimension": 4, "metric": "euclidean"}
    monkeypatch.setattr("rag_engine.vectorstore.pinecone.Pinecone", mock_client_class)

    store = PineconeVectorStore(
        api_key="pcsk-test", index_name="my-index", dimension=4, metric="cosine"
    )
    with pytest.raises(VectorStoreConfigurationError, match="metric 'euclidean'"):
        await store.ensure_ready()


async def test_pinecone_upsert_and_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client_class = MagicMock()
    mock_client = mock_client_class.return_value
    mock_client.has_index.return_value = True
    mock_client.describe_index.return_value = {
        "dimension": 4,
        "metric": "cosine",
        "host": "localhost:5081",
    }
    mock_index = MagicMock()
    mock_client.Index.return_value = mock_index
    monkeypatch.setattr("rag_engine.vectorstore.pinecone.Pinecone", mock_client_class)

    store = PineconeVectorStore(
        api_key="pcsk-test",
        index_name="my-index",
        dimension=4,
        upsert_batch_size=2,
        delete_batch_size=2,
        control_host="http://localhost:5080",
    )

    # Empty upsert
    await store.upsert("ns1", [])
    assert mock_index.upsert.call_count == 0

    # Upsert with dimension mismatch
    with pytest.raises(VectorStoreConfigurationError, match="dimensions"):
        await store.upsert("ns1", [VectorRecord(id="v1", values=[1.0, 2.0])])

    # 3 records with batch_size=2
    records = [
        VectorRecord(id="v1", values=[1.0, 0.0, 0.0, 0.0], metadata={"tag": "1"}),
        VectorRecord(id="v2", values=[0.0, 1.0, 0.0, 0.0], metadata={"tag": "2"}),
        VectorRecord(id="v3", values=[0.0, 0.0, 1.0, 0.0], metadata={"tag": "3"}),
    ]
    await store.upsert("ns1", records)
    assert mock_index.upsert.call_count == 2

    # Delete
    await store.delete("ns1", [])
    assert mock_index.delete.call_count == 0

    await store.delete("ns1", ["v1", "v2", "v3"])
    assert mock_index.delete.call_count == 2

    await store.close()


async def test_pinecone_list_ids_and_query(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client_class = MagicMock()
    mock_client = mock_client_class.return_value
    mock_client.has_index.return_value = True
    mock_client.describe_index.return_value = {
        "dimension": 2,
        "metric": "cosine",
        "host": "http://localhost:5081",
    }
    mock_index = MagicMock()
    mock_client.Index.return_value = mock_index
    monkeypatch.setattr("rag_engine.vectorstore.pinecone.Pinecone", mock_client_class)

    mock_index.list_paginated.side_effect = [
        {"vectors": [{"id": "doc1:0"}, {"id": "doc1:1"}], "pagination": {"next": "token2"}},
        {"vectors": [{"id": "doc1:2"}], "pagination": {"next": None}},
    ]
    mock_index.query.return_value = {
        "matches": [
            {"id": "doc1:0", "score": 0.95, "metadata": {"chunk_index": 0}},
            {"id": "doc1:1", "score": 0.85, "metadata": {"chunk_index": 1}},
        ]
    }

    store = PineconeVectorStore(api_key="pcsk-test", index_name="my-index", dimension=2)

    # list_ids
    ids = await store.list_ids("ns1", prefix="doc1:")
    assert ids == ["doc1:0", "doc1:1", "doc1:2"]
    assert mock_index.list_paginated.call_count == 2

    # query
    with pytest.raises(VectorStoreConfigurationError, match="dimensions"):
        await store.query("ns1", [1.0, 2.0, 3.0], top_k=2)

    matches = await store.query("ns1", [1.0, 0.0], top_k=2, metadata_filter={"document_id": "doc1"})
    assert len(matches) == 2
    assert matches[0].id == "doc1:0"
    assert matches[0].score == 0.95
    assert matches[0].metadata == {"chunk_index": 0}
    assert mock_index.query.call_args[1]["filter"] == {"document_id": {"$eq": "doc1"}}

    await store.close()


def test_metadata_for_vector() -> None:
    input_meta = {"key1": "val1", "key2": None, "key3": 42}
    result = metadata_for_vector(input_meta)
    assert result == {"key1": "val1", "key3": 42}


# ---------------------------------------------------------------------------
# Maintenance helpers
# ---------------------------------------------------------------------------


async def test_delete_vectors() -> None:
    store = MemoryVectorStore()
    await store.upsert(
        "ns",
        [
            VectorRecord(id="v1", values=[]),
            VectorRecord(id="v2", values=[]),
            VectorRecord(id="v3", values=[]),
        ],
    )
    deleted = await delete_vectors(store, "ns", ["v1", "v2", "v3"], chunk_size=2)
    assert deleted == 3
    assert store.count("ns") == 0


async def test_delete_document_vectors() -> None:
    store = MemoryVectorStore()
    doc_id = uuid4()
    other_id = uuid4()

    v1 = chunk_vector_id(doc_id, 0)
    v2 = chunk_vector_id(doc_id, 1)
    other_v = chunk_vector_id(other_id, 0)

    await store.upsert(
        "ns",
        [
            VectorRecord(id=v1, values=[]),
            VectorRecord(id=v2, values=[]),
            VectorRecord(id=other_v, values=[]),
        ],
    )

    deleted = await delete_document_vectors(store, "ns", doc_id, known_ids=[v1])
    assert deleted >= 2
    assert store.count("ns") == 1
    assert await store.list_ids("ns") == [other_v]


async def test_delete_stale_vectors() -> None:
    store = MemoryVectorStore()
    doc_id = uuid4()

    v0 = chunk_vector_id(doc_id, 0)
    v1 = chunk_vector_id(doc_id, 1)
    v2 = chunk_vector_id(doc_id, 2)

    await store.upsert(
        "ns",
        [
            VectorRecord(id=v0, values=[]),
            VectorRecord(id=v1, values=[]),
            VectorRecord(id=v2, values=[]),
        ],
    )

    # Re-ingestion produced only v0 and v1; v2 is stale
    deleted = await delete_stale_vectors(
        store,
        "ns",
        doc_id,
        keep_ids=[v0, v1],
        previous_ids=[v0, v1, v2],
    )
    assert deleted == 1
    remaining = await store.list_ids("ns")
    assert remaining == [v0, v1]

    # No stale vectors to delete
    deleted_noop = await delete_stale_vectors(
        store,
        "ns",
        doc_id,
        keep_ids=[v0, v1],
        previous_ids=[v0, v1],
    )
    assert deleted_noop == 0


# ---------------------------------------------------------------------------
# Factory: build_vector_store
# ---------------------------------------------------------------------------


def test_build_vector_store_memory() -> None:
    settings = Settings(vector_store="memory")
    store = build_vector_store(settings)
    assert isinstance(store, MemoryVectorStore)


def test_build_vector_store_pinecone() -> None:
    settings = Settings(
        vector_store="pinecone",
        pinecone_api_key=SecretStr("pcsk-key"),
        embedding_dimensions=128,
    )
    store = build_vector_store(settings)
    assert isinstance(store, PineconeVectorStore)


def test_build_vector_store_unknown() -> None:
    settings = Settings()
    object.__setattr__(settings, "vector_store", "invalid")
    with pytest.raises(ValueError, match="unknown vector store"):
        build_vector_store(settings)
