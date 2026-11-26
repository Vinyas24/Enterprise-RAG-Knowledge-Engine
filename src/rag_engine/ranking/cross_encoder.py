"""Local cross-encoder re-ranking.

A cross-encoder reads the query and a chunk together and emits one relevance
score, which is what makes it sharper than the bi-encoder used for retrieval.
The model lives behind the ``rerank`` optional extra so the default install
stays small:

    pip install "rag-engine[rerank]"

The heavy import happens in :meth:`warmup`, on a worker thread, so the event
loop is never blocked by model loading. Inference also runs in a thread for the
same reason. ``scorer`` is an injection seam: tests exercise ordering and error
handling without downloading a model.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from rag_engine.retrieval.base import RetrievedChunk

from .errors import RerankResponseError, RerankUnavailableError

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

Scorer = Callable[[Sequence[tuple[str, str]]], Sequence[float]]


class CrossEncoderReranker:
    """Re-rank with a local sentence-transformers cross-encoder."""

    name = "cross-encoder"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        batch_size: int = 32,
        device: str | None = None,
        scorer: Scorer | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._scorer = scorer
        self._model: Any | None = None

    async def warmup(self) -> None:
        """Load the model now; a missing extra or model fails startup."""
        if self._scorer is None and self._model is None:
            await asyncio.to_thread(self._load_model)

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

        pairs = [(query, chunk.content) for chunk in pool]
        scores = await asyncio.to_thread(self._score, pairs)
        if len(scores) != len(pool):
            raise RerankResponseError(
                f"cross-encoder returned {len(scores)} scores for {len(pool)} candidates"
            )

        ranked = sorted(
            (
                (score, position, chunk)
                for position, (score, chunk) in enumerate(zip(scores, pool, strict=True))
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return [replace(chunk, score=score) for score, _, chunk in ranked[:top_n]]

    async def close(self) -> None:
        """Drop the model reference so a shutdown can reclaim it."""
        self._model = None

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RerankUnavailableError(
                "cross-encoder re-ranking needs the optional 'rerank' extra; "
                "install it with: pip install 'rag-engine[rerank]'"
            ) from exc
        self._model = CrossEncoder(self._model_name, device=self._device)

    def _score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if self._scorer is not None:
            return [float(score) for score in self._scorer(pairs)]
        if self._model is None:
            self._load_model()
        model = self._model
        if model is None:  # pragma: no cover - _load_model raises instead
            raise RerankUnavailableError("cross-encoder model failed to load")
        raw = model.predict(list(pairs), batch_size=self._batch_size)
        return [float(score) for score in raw]
