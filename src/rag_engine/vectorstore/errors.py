"""Vector-store errors.

The ingestion worker treats every store failure as retryable; the API maps the
same errors to ``503`` instead of leaking SDK messages. Keeping the taxonomy
small is intentional: callers cannot do anything useful with a finer
distinction.
"""

from __future__ import annotations


class VectorStoreError(Exception):
    """Base class for expected vector-store failures."""


class VectorStoreUnavailable(VectorStoreError):
    """A transient transport or service failure; retrying may succeed."""


class VectorStoreConfigurationError(VectorStoreError):
    """The index cannot be used as configured (dimension or metric mismatch)."""
