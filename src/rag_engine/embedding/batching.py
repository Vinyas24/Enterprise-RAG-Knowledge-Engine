"""Batching, validation and retry policy around any provider.

The wrapper exists so provider implementations stay small and the retry rules
are consistent:

* a provider's ``Retry-After`` hint wins over computed backoff,
* transient errors (timeouts, 5xx) use capped exponential backoff,
* permanent errors (bad request, wrong vector count) fail immediately,
* the output order always matches the input order.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence

from .base import Embedder, EmbeddingProvider
from .errors import (
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbeddingResponseError,
    EmbeddingTransientError,
)

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


class BatchedEmbedder(Embedder):
    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        batch_size: int = 64,
        max_retries: int = 4,
        retry_base_seconds: float = 0.5,
        retry_max_seconds: float = 20.0,
        sleep: Sleep | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._provider = provider
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds
        self._sleep: Sleep = sleep or asyncio.sleep
        self.dimensions = provider.dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(self._max_retries + 1):
            try:
                vectors = await self._provider.embed(batch)
                return self._validate(vectors, expected=len(batch))
            except EmbeddingRateLimitError as exc:
                if attempt >= self._max_retries:
                    raise
                delay = exc.retry_after
                if delay is None:
                    delay = self._backoff(attempt)
            except EmbeddingTransientError:
                if attempt >= self._max_retries:
                    raise
                delay = self._backoff(attempt)

            logger.warning(
                "embedding.retry",
                extra={
                    "provider": self._provider.name,
                    "attempt": attempt + 1,
                    "max_retries": self._max_retries,
                    "delay_seconds": delay,
                },
            )
            await self._sleep(delay)

        # The loop either returns or raises; this keeps the type checker honest.
        raise EmbeddingError("embedding retry loop exited unexpectedly")

    def _validate(self, vectors: list[list[float]], *, expected: int) -> list[list[float]]:
        if len(vectors) != expected:
            raise EmbeddingResponseError(
                f"provider returned {len(vectors)} vectors for {expected} inputs"
            )
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise EmbeddingResponseError(
                    f"provider returned a {len(vector)}-dimensional vector, "
                    f"expected {self.dimensions}"
                )
        return vectors

    def _backoff(self, attempt: int) -> float:
        return float(
            min(
                self._retry_max_seconds,
                self._retry_base_seconds * (2**attempt),
            )
        )

    async def close(self) -> None:
        await self._provider.close()
