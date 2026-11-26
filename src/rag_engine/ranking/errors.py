"""Re-ranking errors.

Providers translate transport, SDK and parse failures into these types so the
retrieval wrapper can apply one fallback policy without importing httpx or
sentence-transformers. The wrapper catches ``RerankError`` and timeouts; a
programming error still propagates instead of being hidden behind a fallback.
"""

from __future__ import annotations


class RerankError(Exception):
    """Base class for expected re-ranking failures."""


class RerankUnavailableError(RerankError):
    """The configured provider cannot run here (missing extra, model, or key).

    Raised during warm-up so the failure surfaces at startup. Falling back on
    every query because a dependency is missing would disguise a configuration
    error as a quality regression.
    """


class RerankTransientError(RerankError):
    """A failure that might succeed on a later request: timeout, 5xx, 429.

    The wrapper does not retry: a retry spends the caller's deadline twice.
    The fallback order is the safer answer for a single query.
    """


class RerankResponseError(RerankError):
    """The provider answered, but not with a usable ranking."""

    ...
