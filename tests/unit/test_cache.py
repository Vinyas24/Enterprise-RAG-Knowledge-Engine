"""Unit tests for the thread-safe LruCache implementation."""

from __future__ import annotations

import time

import pytest

from rag_engine.cache.base import LruCache


def test_lru_cache_basic_get_set() -> None:
    cache: LruCache[str, str] = LruCache(max_size=3)
    assert cache.get("k1") is None
    stats = cache.stats()
    assert stats.misses == 1
    assert stats.hits == 0

    cache.set("k1", "v1")
    assert cache.get("k1") == "v1"
    stats = cache.stats()
    assert stats.hits == 1
    assert stats.size == 1
    assert "k1" in cache
    assert "k2" not in cache


def test_lru_cache_eviction() -> None:
    cache: LruCache[str, int] = LruCache(max_size=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert len(cache) == 3

    # Access 'a' to make it most recently used; 'b' is now least recently used
    assert cache.get("a") == 1

    # Insert 'd' -> 'b' should be evicted
    cache.set("d", 4)
    assert len(cache) == 3
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4

    stats = cache.stats()
    assert stats.evictions == 1


def test_lru_cache_ttl_expiration() -> None:
    cache: LruCache[str, str] = LruCache(max_size=10, default_ttl_seconds=0.05)
    cache.set("quick", "expires_soon")
    cache.set("never", "persists", ttl_seconds=10.0)

    assert cache.get("quick") == "expires_soon"
    time.sleep(0.06)

    # 'quick' should be expired and return None on get
    assert cache.get("quick") is None
    assert "quick" not in cache
    assert cache.get("never") == "persists"

    stats = cache.stats()
    assert stats.expirations >= 1


def test_lru_cache_cleanup_expired() -> None:
    cache: LruCache[str, str] = LruCache(max_size=10, default_ttl_seconds=0.05)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    time.sleep(0.06)

    assert len(cache) == 2  # Before cleanup
    evicted = cache.cleanup_expired()
    assert evicted == 2
    assert len(cache) == 0


def test_lru_cache_delete_and_clear() -> None:
    cache: LruCache[str, int] = LruCache(max_size=5)
    cache.set("x", 10)
    cache.set("y", 20)

    assert cache.delete("x") is True
    assert cache.delete("x") is False
    assert cache.get("x") is None
    assert len(cache) == 1

    cache.clear()
    assert len(cache) == 0
    assert cache.get("y") is None


def test_lru_cache_invalidate_where() -> None:
    cache: LruCache[str, int] = LruCache(max_size=10)
    cache.set("tenant1:q1", 1)
    cache.set("tenant1:q2", 2)
    cache.set("tenant2:q1", 3)

    removed = cache.invalidate_where(lambda k: k.startswith("tenant1:"))
    assert removed == 2
    assert len(cache) == 1
    assert cache.get("tenant2:q1") == 3


def test_lru_cache_validation() -> None:
    with pytest.raises(ValueError, match="max_size must be at least 1"):
        LruCache(max_size=0)
    with pytest.raises(ValueError, match="default_ttl_seconds must be positive"):
        LruCache(max_size=10, default_ttl_seconds=-1.0)
