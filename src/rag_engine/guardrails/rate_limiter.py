"""In-memory sliding window rate limiter for multi-tenant query operations."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque


class RateLimiter:
    """Sliding-window request rate limiter per tenant/key."""

    def __init__(self) -> None:
        self._history: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: float = 60.0,
    ) -> tuple[bool, int, float]:
        """Check whether a request under ``key`` is permitted.

        Returns:
            Tuple of:
            - is_allowed (bool): True if under limit, False if rate-limited.
            - remaining (int): Number of requests remaining in current window.
            - retry_after (float): Seconds to wait before another request is allowed.
        """
        if limit <= 0:
            return True, limit, 0.0

        async with self._lock:
            now = time.perf_counter()
            cutoff = now - window_seconds
            timestamps = self._history[key]

            # Evict timestamps outside the window
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            count = len(timestamps)
            if count >= limit:
                # Oldest timestamp defines when the next slot frees up
                oldest = timestamps[0]
                retry_after = max(0.1, round((oldest + window_seconds) - now, 2))
                return False, 0, retry_after

            timestamps.append(now)
            remaining = limit - (count + 1)
            return True, remaining, 0.0

    async def reset(self) -> None:
        """Clear all rate limiting history (for test isolation)."""
        async with self._lock:
            self._history.clear()
