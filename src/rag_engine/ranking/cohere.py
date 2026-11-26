"""Cohere Rerank over HTTP.

The request is the documented v2 shape:

    POST {base}/rerank
    {"model": ..., "query": ..., "documents": [...], "top_n": ...}

and the response is a ``results`` list of ``{index, relevance_score}`` objects.
Indexes point back into the candidate list, so the caller keeps ownership of
chunk metadata; the provider only supplies an order and a score.

Failures are classified, not retried. The retrieval wrapper's deadline plus
fallback is the recovery policy: retrying a slow hosted re-ranker spends the
same budget twice and still risks a late answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import httpx

from rag_engine.retrieval.base import RetrievedChunk

from .errors import RerankResponseError, RerankTransientError

DEFAULT_COHERE_MODEL = "rerank-v3.5"
_MAX_ERROR_BODY_CHARS = 300


class CohereReranker:
    """Hosted re-ranking through Cohere's ``/v2/rerank`` endpoint."""

    name = "cohere"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_COHERE_MODEL,
        base_url: str = "https://api.cohere.com/v2",
        timeout_seconds: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Cohere re-ranking requires an API key")
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

    async def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        pool = list(candidates)
        if top_n < 1 or not pool:
            return []

        try:
            response = await self._client.post(
                "rerank",
                json={
                    "model": self._model,
                    "query": query,
                    "documents": [chunk.content for chunk in pool],
                    "top_n": min(top_n, len(pool)),
                },
            )
        except httpx.TimeoutException as exc:
            raise RerankTransientError("re-rank request timed out") from exc
        except httpx.RequestError as exc:
            raise RerankTransientError(f"re-rank request failed: {exc}") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise RerankTransientError(f"re-rank provider returned {response.status_code}")
        if response.status_code >= 400:
            raise RerankResponseError(
                f"re-rank provider rejected the request ({response.status_code}): "
                f"{response.text[:_MAX_ERROR_BODY_CHARS]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RerankResponseError("re-rank provider returned invalid JSON") from exc

        results = self._parse(payload, pool_size=len(pool))
        ranked = sorted(results, key=lambda item: (-item[1], item[0]))
        return [replace(pool[index], score=score) for index, score in ranked[:top_n]]

    def _parse(self, payload: Any, *, pool_size: int) -> list[tuple[int, float]]:
        if not isinstance(payload, dict):
            raise RerankResponseError("re-rank response is not an object")
        raw_results = payload.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            raise RerankResponseError("re-rank response contains no results")

        parsed: list[tuple[int, float]] = []
        seen: set[int] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                raise RerankResponseError("re-rank result is not an object")
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < pool_size:
                raise RerankResponseError("re-rank result has an out-of-range index")
            if index in seen:
                raise RerankResponseError("re-rank result repeats an index")
            if not isinstance(score, int | float) or isinstance(score, bool):
                raise RerankResponseError("re-rank result has a non-numeric score")
            seen.add(index)
            parsed.append((index, float(score)))
        return parsed

    async def warmup(self) -> None:
        """The client is created eagerly; nothing else to load."""

    async def close(self) -> None:
        await self._client.aclose()
