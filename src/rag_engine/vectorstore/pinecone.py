"""Pinecone adapter.

The official SDK is synchronous (urllib3 under the hood), so every call is
offloaded with ``asyncio.to_thread``. Blocking the event loop would stall the
worker's shutdown handling and every concurrent request in the API.

Tenancy uses one namespace per tenant. Vector ids are ``<document_id>:<index>``
(see :mod:`rag_engine.core.ids`), which makes re-ingestion an upsert and lets
deletion enumerate a document's vectors by prefix instead of a metadata-filtered
delete. That matters because Pinecone Local rejects filtered deletes, and not
every production tier supports them.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from pinecone import Pinecone, ServerlessSpec  # type: ignore[import-untyped]

from .base import Metadata, VectorMatch, VectorRecord
from .errors import VectorStoreConfigurationError, VectorStoreError, VectorStoreUnavailable

logger = logging.getLogger(__name__)


class PineconeVectorStore:
    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        control_host: str | None = None,
        index_host: str | None = None,
        upsert_batch_size: int = 100,
        delete_batch_size: int = 1000,
        list_page_size: int = 100,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("Pinecone API key is required")
        if upsert_batch_size < 1 or delete_batch_size < 1 or list_page_size < 1:
            raise ValueError("Pinecone batch sizes must be positive")
        self._api_key = api_key
        self._index_name = index_name
        self._dimension = dimension
        self._metric = metric
        self._cloud = cloud
        self._region = region
        self._control_host = control_host
        self._index_host = index_host
        self._upsert_batch_size = upsert_batch_size
        self._delete_batch_size = delete_batch_size
        self._list_page_size = list_page_size
        self._timeout_seconds = timeout_seconds
        # Local deployments speak plain HTTP. Deriving the scheme from the
        # control-plane URL avoids the classic Pinecone Local SSL failure.
        self._scheme = "http" if (control_host or "").startswith("http://") else "https"
        self._client: Any | None = None
        self._index: Any | None = None
        self._connect_lock = asyncio.Lock()

    async def ensure_ready(self) -> None:
        if self._index is not None:
            return
        async with self._connect_lock:
            index = getattr(self, "_index", None)
            if index is not None:
                return
            try:
                await asyncio.to_thread(self._connect_blocking)
            except VectorStoreError:
                raise
            except Exception as exc:  # pragma: no cover - SDK error surface
                raise VectorStoreUnavailable(f"could not connect to Pinecone: {exc}") from exc

    def _connect_blocking(self) -> None:
        kwargs: dict[str, Any] = {
            "api_key": self._api_key,
            "timeout": self._timeout_seconds,
        }
        if self._control_host is not None:
            kwargs["host"] = self._control_host
        client = Pinecone(**kwargs)

        if not client.has_index(self._index_name):
            logger.info(
                "vectorstore.index_creating",
                extra={"index": self._index_name, "dimension": self._dimension},
            )
            client.create_index(
                name=self._index_name,
                dimension=self._dimension,
                metric=self._metric,
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )

        description = client.describe_index(self._index_name)
        actual_dimension = _field(description, "dimension")
        if actual_dimension is not None and int(actual_dimension) != self._dimension:
            raise VectorStoreConfigurationError(
                f"index {self._index_name!r} has dimension {actual_dimension}, "
                f"but embeddings produce {self._dimension}"
            )
        actual_metric = _field(description, "metric")
        if actual_metric is not None and str(actual_metric) != self._metric:
            raise VectorStoreConfigurationError(
                f"index {self._index_name!r} uses metric {actual_metric!r}, "
                f"expected {self._metric!r}"
            )

        host = self._index_host or _field(description, "host")
        if host:
            if self._control_host:
                ctrl_hostname = urlsplit(self._control_host).hostname
                if ctrl_hostname and (
                    host.startswith("localhost:") or host.startswith("127.0.0.1:")
                ):
                    port = host.split(":", 1)[1]
                    host = f"{ctrl_hostname}:{port}"
            if "://" not in host:
                host = f"{self._scheme}://{host}"
            index = client.Index(host=host)
        else:  # pragma: no cover - cloud responses always include a host
            index = client.Index(self._index_name)

        self._client = client
        self._index = index

    async def upsert(self, namespace: str, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        for record in records:
            if len(record.values) != self._dimension:
                raise VectorStoreConfigurationError(
                    f"vector {record.id!r} has {len(record.values)} dimensions, "
                    f"expected {self._dimension}"
                )
        await self.ensure_ready()
        index = self._require_index()
        for batch in _batched(records, self._upsert_batch_size):
            vectors = [
                {"id": record.id, "values": list(record.values), "metadata": dict(record.metadata)}
                for record in batch
            ]
            await self._run(index.upsert, vectors=vectors, namespace=namespace)

    async def delete(self, namespace: str, ids: Sequence[str]) -> None:
        if not ids:
            return
        await self.ensure_ready()
        index = self._require_index()
        for batch in _batched(list(ids), self._delete_batch_size):
            await self._run(index.delete, ids=batch, namespace=namespace)

    async def list_ids(self, namespace: str, *, prefix: str = "") -> list[str]:
        await self.ensure_ready()
        index = self._require_index()
        ids: list[str] = []
        token: str | None = None
        while True:
            response = await self._run(
                index.list_paginated,
                namespace=namespace,
                prefix=prefix,
                limit=self._list_page_size,
                pagination_token=token,
            )
            for vector in _field(response, "vectors") or []:
                vector_id = _field(vector, "id")
                if vector_id is not None:
                    ids.append(str(vector_id))
            pagination = _field(response, "pagination")
            token = _field(pagination, "next") if pagination is not None else None
            if not token:
                return ids

    async def query(
        self,
        namespace: str,
        vector: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Mapping[str, str | int | float | bool] | None = None,
    ) -> list[VectorMatch]:
        if len(vector) != self._dimension:
            raise VectorStoreConfigurationError(
                f"query vector has {len(vector)} dimensions, expected {self._dimension}"
            )
        await self.ensure_ready()
        index = self._require_index()
        pinecone_filter = (
            {key: {"$eq": value} for key, value in metadata_filter.items()}
            if metadata_filter
            else None
        )
        kwargs: dict[str, Any] = {
            "vector": list(vector),
            "top_k": top_k,
            "namespace": namespace,
            "include_metadata": True,
            "include_values": False,
        }
        if pinecone_filter is not None:
            kwargs["filter"] = pinecone_filter
        response = await self._run(index.query, **kwargs)
        matches: list[VectorMatch] = []
        for match in _field(response, "matches") or []:
            matches.append(
                VectorMatch(
                    id=str(_field(match, "id")),
                    score=float(_field(match, "score") or 0.0),
                    metadata=dict(_field(match, "metadata") or {}),
                )
            )
        return matches

    async def close(self) -> None:
        index, self._index = self._index, None
        self._client = None
        if index is None:
            return
        close = getattr(index, "close", None)
        if close is not None:
            try:
                await asyncio.to_thread(close)
            except Exception:  # pragma: no cover - close is best effort
                logger.debug("vectorstore.close_failed", exc_info=True)

    def _require_index(self) -> Any:
        if self._index is None:  # pragma: no cover - ensure_ready precedes this
            raise VectorStoreUnavailable("Pinecone index is not initialised")
        return self._index

    async def _run(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.to_thread(function, *args, **kwargs)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreUnavailable(f"Pinecone request failed: {exc}") from exc


def _field(value: Any, name: str) -> Any:
    """Read a field from SDK models (attribute access) or plain dicts."""
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _batched(items: Sequence[VectorRecord] | list[str], size: int) -> list[Sequence[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def metadata_for_vector(metadata: Metadata) -> dict[str, str | int | float | bool]:
    """Drop ``None`` values, which Pinecone metadata does not accept."""
    return {key: value for key, value in metadata.items() if value is not None}
