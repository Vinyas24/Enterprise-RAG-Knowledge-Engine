"""Retry policy for ingestion jobs.

Deliberately simple and pure so it is trivially testable: exponential backoff
based on the number of attempts already made, capped. No jitter yet; with a
single worker it buys nothing, and it would make tests non-deterministic.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from rag_engine.core.models import utcnow


def retry_delay_seconds(attempts: int, *, base_seconds: float, max_seconds: float) -> float:
    """Delay before the next attempt after ``attempts`` failed attempts."""
    if attempts < 1:
        return 0.0
    delay: float = base_seconds * float(2 ** (attempts - 1))
    return min(delay, max_seconds)


def next_attempt_at(
    attempts: int, *, base_seconds: float, max_seconds: float, now: datetime | None = None
) -> datetime:
    return (now or utcnow()) + timedelta(
        seconds=retry_delay_seconds(attempts, base_seconds=base_seconds, max_seconds=max_seconds)
    )
