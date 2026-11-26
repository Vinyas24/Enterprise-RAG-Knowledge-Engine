"""Thread-safe, bounded LRU cache with optional TTL expiration."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True)
class CacheStats:
    """Snapshot of cache performance statistics."""

    hits: int
    misses: int
    evictions: int
    expirations: int
    size: int
    max_size: int

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LruCache(Generic[K, V]):
    """Thread-safe, bounded Least-Recently-Used (LRU) cache supporting TTL expiration."""

    def __init__(self, *, max_size: int = 1000, default_ttl_seconds: float | None = None) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        if default_ttl_seconds is not None and default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive when provided")
        self._max_size = max_size
        self._default_ttl = default_ttl_seconds
        self._items: OrderedDict[K, tuple[V, float | None]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def get(self, key: K) -> V | None:
        """Retrieve an item from the cache. Returns None on miss or expiration."""
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                self._misses += 1
                return None

            value, expires_at = entry
            if expires_at is not None and time.monotonic() > expires_at:
                del self._items[key]
                self._expirations += 1
                self._misses += 1
                return None

            self._items.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: K, value: V, *, ttl_seconds: float | None = None) -> None:
        """Store an item in the cache, evicting the least recently used item if full."""
        with self._lock:
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            expires_at = (time.monotonic() + ttl) if ttl is not None and ttl > 0 else None

            if key in self._items:
                self._items[key] = (value, expires_at)
                self._items.move_to_end(key)
                return

            if len(self._items) >= self._max_size:
                # Evict oldest entry (first item)
                self._items.popitem(last=False)
                self._evictions += 1

            self._items[key] = (value, expires_at)

    def delete(self, key: K) -> bool:
        """Remove a specific key from the cache. Returns True if deleted."""
        with self._lock:
            if key in self._items:
                del self._items[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries in the cache."""
        with self._lock:
            self._items.clear()

    def stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                expirations=self._expirations,
                size=len(self._items),
                max_size=self._max_size,
            )

    def cleanup_expired(self) -> int:
        """Evict all currently expired items and return the count evicted."""
        with self._lock:
            now = time.monotonic()
            expired_keys = [
                k
                for k, (_, expires_at) in self._items.items()
                if expires_at is not None and now > expires_at
            ]
            for k in expired_keys:
                del self._items[k]
                self._expirations += 1
            return len(expired_keys)

    def invalidate_where(self, predicate: Callable[[K], bool]) -> int:
        """Remove all entries where predicate(key) is True. Returns count removed."""
        with self._lock:
            matching_keys = [k for k in self._items if predicate(k)]
            for k in matching_keys:
                del self._items[k]
            return len(matching_keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __contains__(self, key: K) -> bool:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return False
            _, expires_at = entry
            if expires_at is not None and time.monotonic() > expires_at:
                del self._items[key]
                self._expirations += 1
                return False
            return True
