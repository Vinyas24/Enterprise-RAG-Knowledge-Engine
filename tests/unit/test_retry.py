"""Retry backoff: deterministic, capped, and easy to reason about."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rag_engine.ingestion.retry import next_attempt_at, retry_delay_seconds


@pytest.mark.parametrize("attempts", [0, -1])
def test_no_delay_before_the_first_attempt(attempts: int) -> None:
    assert retry_delay_seconds(attempts, base_seconds=2, max_seconds=60) == 0.0


def test_delay_grows_exponentially() -> None:
    delays = [
        retry_delay_seconds(attempt, base_seconds=2, max_seconds=60) for attempt in range(1, 5)
    ]

    assert delays == [2.0, 4.0, 8.0, 16.0]


def test_delay_is_capped() -> None:
    assert retry_delay_seconds(10, base_seconds=10, max_seconds=60) == 60.0


def test_next_attempt_at_uses_the_supplied_clock() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    result = next_attempt_at(2, base_seconds=5, max_seconds=60, now=now)

    assert result == now + timedelta(seconds=10)
    assert result.tzinfo is UTC


def test_next_attempt_at_defaults_to_now() -> None:
    before = datetime.now(UTC)

    result = next_attempt_at(1, base_seconds=3, max_seconds=60)

    assert before <= result <= datetime.now(UTC) + timedelta(seconds=3)
