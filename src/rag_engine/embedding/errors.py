"""Embedding errors.

Providers raise these instead of leaking HTTP or SDK exceptions, so the
ingestion pipeline can decide what is retryable without knowing which provider
is configured. The batching wrapper owns retry policy; providers only classify
the failure.
"""

from __future__ import annotations


class EmbeddingError(Exception):
    """Base class for expected embedding failures."""


class EmbeddingTransientError(EmbeddingError):
    """A failure worth retrying: timeout, connection reset, 5xx."""


class EmbeddingRateLimitError(EmbeddingTransientError):
    """The provider asked us to slow down.

    ``retry_after`` is the provider's explicit hint in seconds (HTTP
    ``Retry-After``) when it supplied one.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(message)


class EmbeddingResponseError(EmbeddingError):
    """The provider answered, but not with usable embeddings.

    Wrong vector count, wrong dimensions or an auth/validation rejection. A
    retry cannot fix this, so the job should surface it rather than spin.
    """
