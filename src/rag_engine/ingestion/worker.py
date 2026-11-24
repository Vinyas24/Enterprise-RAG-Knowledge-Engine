"""Ingestion worker process.

Runs as its own service in Docker Compose so the API never blocks on parsing.
Shutdown is graceful: SIGTERM stops the loop *after* the current job finishes,
which is why in-flight work is not lost on redeploy.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
from collections.abc import Sequence

from rag_engine.core.config import Settings, get_settings
from rag_engine.core.db import PostgresDatabase
from rag_engine.core.logging import configure_logging
from rag_engine.embedding import build_embedder
from rag_engine.ingestion.blobstore import LocalBlobStore
from rag_engine.ingestion.chunking import build_chunker
from rag_engine.ingestion.parsers import ParserRegistry
from rag_engine.ingestion.processor import IngestionProcessor
from rag_engine.vectorstore import build_vector_store

logger = logging.getLogger(__name__)


def build_processor(settings: Settings, *, database: PostgresDatabase) -> IngestionProcessor:
    return IngestionProcessor(
        database=database,
        blob_store=LocalBlobStore(settings.blob_store_path),
        parsers=ParserRegistry.default(),
        chunker=build_chunker(settings),
        embedder=build_embedder(settings),
        vector_store=build_vector_store(settings),
        settings=settings,
    )


async def run_forever(processor: IngestionProcessor, *, poll_interval: float) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    logger.info("worker.started", extra={"poll_interval_seconds": poll_interval})
    while not stop.is_set():
        outcome = await processor.process_next()
        if outcome.processed:
            continue
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=poll_interval)
    logger.info("worker.stopped")


async def _main(settings: Settings, *, once: bool) -> int:
    database = PostgresDatabase(settings)
    processor = build_processor(settings, database=database)
    try:
        if once:
            outcome = await processor.process_next()
            logger.info(
                "worker.single_run",
                extra={"processed": outcome.processed, "succeeded": outcome.succeeded},
            )
            return 0
        await run_forever(processor, poll_interval=settings.worker_poll_interval_seconds)
        return 0
    finally:
        # Close HTTP/SDK clients before the database so shutdown is ordered:
        # no new request can start once the processor is closed.
        await processor.aclose()
        await database.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-engine-worker")
    parser.add_argument("--once", action="store_true", help="process at most one job and exit")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level=settings.log_level, log_format=settings.log_format.value)
    try:
        return asyncio.run(_main(settings, once=args.once))
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        return 130


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
