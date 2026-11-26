"""Unit tests for sliding-window rate limiter."""

from __future__ import annotations

import pytest

from rag_engine.guardrails.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_requests_under_limit() -> None:
    limiter = RateLimiter()
    key = "tenant:acme"

    for i in range(5):
        allowed, remaining, retry_after = await limiter.is_allowed(
            key, limit=5, window_seconds=10.0
        )
        assert allowed
        assert remaining == 5 - (i + 1)
        assert retry_after == 0.0


@pytest.mark.asyncio
async def test_rate_limiter_blocks_requests_exceeding_limit() -> None:
    limiter = RateLimiter()
    key = "tenant:acme"

    # Exhaust limit of 3
    for _ in range(3):
        allowed, _, _ = await limiter.is_allowed(key, limit=3, window_seconds=10.0)
        assert allowed

    # 4th request must be blocked
    allowed, remaining, retry_after = await limiter.is_allowed(key, limit=3, window_seconds=10.0)
    assert not allowed
    assert remaining == 0
    assert retry_after > 0.0


@pytest.mark.asyncio
async def test_rate_limiter_isolates_different_keys() -> None:
    limiter = RateLimiter()
    key_a = "tenant:alpha"
    key_b = "tenant:beta"

    # Exhaust key_a
    for _ in range(2):
        await limiter.is_allowed(key_a, limit=2, window_seconds=10.0)

    # key_a is blocked
    allowed_a, _, _ = await limiter.is_allowed(key_a, limit=2, window_seconds=10.0)
    assert not allowed_a

    # key_b is still allowed
    allowed_b, remaining_b, _ = await limiter.is_allowed(key_b, limit=2, window_seconds=10.0)
    assert allowed_b
    assert remaining_b == 1


@pytest.mark.asyncio
async def test_rate_limiter_reset() -> None:
    limiter = RateLimiter()
    key = "tenant:acme"

    for _ in range(2):
        await limiter.is_allowed(key, limit=2, window_seconds=10.0)

    allowed, _, _ = await limiter.is_allowed(key, limit=2, window_seconds=10.0)
    assert not allowed

    await limiter.reset()

    allowed_after_reset, _, _ = await limiter.is_allowed(key, limit=2, window_seconds=10.0)
    assert allowed_after_reset
