"""OpenAI-compatible embeddings over HTTP.

The wire format is the de-facto standard: ``POST {base}/embeddings`` with
``{"model": ..., "input": [...]}`` and a ``data`` list of ``{index, embedding}``
objects. Any compatible gateway (OpenAI, Azure-style proxies, vLLM, Ollama)
works by changing the base URL and model.

The provider classifies failures; retries live in :mod:`rag_engine.embedding.batching`.
That keeps the retry policy identical for every provider.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from .base import EmbeddingProvider
from .errors import (
    EmbeddingRateLimitError,
    EmbeddingResponseError,
    EmbeddingTransientError,
)

_MAX_ERROR_BODY_CHARS = 300


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("embedding API key is required")
        self.dimensions = dimensions
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0,
            ),
            transport=transport,
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                "embeddings",
                json={"model": self._model, "input": list(texts)},
            )
        except httpx.TimeoutException as exc:
            raise EmbeddingTransientError("embedding request timed out") from exc
        except httpx.RequestError as exc:
            raise EmbeddingTransientError(f"embedding request failed: {exc}") from exc

        if response.status_code == 429:
            raise EmbeddingRateLimitError(
                "embedding provider rate limit reached",
                retry_after=_retry_after_seconds(response),
            )
        if response.status_code >= 500:
            raise EmbeddingTransientError(f"embedding provider returned {response.status_code}")
        if response.status_code >= 400:
            raise EmbeddingResponseError(
                f"embedding provider rejected the request ({response.status_code}): "
                f"{response.text[:_MAX_ERROR_BODY_CHARS]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise EmbeddingResponseError("embedding provider returned invalid JSON") from exc

        return self._parse(payload, expected=len(texts))

    def _parse(self, payload: Any, *, expected: int) -> list[list[float]]:
        if not isinstance(payload, dict):
            raise EmbeddingResponseError("embedding response is not an object")
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise EmbeddingResponseError(
                f"embedding response contained {len(data) if isinstance(data, list) else 0} "
                f"vectors for {expected} inputs"
            )

        ordered: list[tuple[int, list[float]]] = []
        for position, item in enumerate(data):
            if not isinstance(item, dict):
                raise EmbeddingResponseError("embedding response item is not an object")
            index = item.get("index", position)
            embedding = item.get("embedding")
            if not isinstance(index, int) or not isinstance(embedding, list):
                raise EmbeddingResponseError("embedding response item is malformed")
            vector: list[float] = []
            for value in embedding:
                if not isinstance(value, int | float):
                    raise EmbeddingResponseError("embedding vector contains a non-number")
                vector.append(float(value))
            if len(vector) != self.dimensions:
                raise EmbeddingResponseError(
                    f"embedding has {len(vector)} dimensions, expected {self.dimensions}"
                )
            ordered.append((index, vector))

        ordered.sort(key=lambda pair: pair[0])
        if [index for index, _ in ordered] != list(range(expected)):
            raise EmbeddingResponseError("embedding response indexes are not contiguous")
        return [vector for _, vector in ordered]

    async def close(self) -> None:
        await self._client.aclose()


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        # HTTP-date Retry-After values are legal but uncommon for embeddings;
        # falling back to exponential backoff is safer than guessing a date.
        return None
