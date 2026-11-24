"""Processor lifecycle behaviour that does not need a database."""

from __future__ import annotations

from typing import Any, cast

from rag_engine.ingestion.processor import IngestionProcessor


class _RecordingCloser:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FailingCloser:
    async def close(self) -> None:
        raise RuntimeError("client already closed")


def _processor(*, embedder: Any, vector_store: Any) -> IngestionProcessor:
    # Only aclose() is exercised here; the pipeline dependencies are unused.
    return IngestionProcessor(
        database=cast(Any, None),
        blob_store=cast(Any, None),
        parsers=cast(Any, None),
        chunker=cast(Any, None),
        embedder=embedder,
        vector_store=vector_store,
        settings=cast(Any, None),
    )


async def test_aclose_closes_every_client() -> None:
    embedder = _RecordingCloser()
    store = _RecordingCloser()

    await _processor(embedder=embedder, vector_store=store).aclose()

    assert embedder.closed is True
    assert store.closed is True


async def test_aclose_still_closes_store_when_embedder_fails() -> None:
    store = _RecordingCloser()

    await _processor(embedder=_FailingCloser(), vector_store=store).aclose()

    assert store.closed is True
