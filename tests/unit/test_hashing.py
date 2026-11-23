"""Content hashing: the deduplication key must be stable and unforgiving."""

from __future__ import annotations

import hashlib

import pytest

from rag_engine.ingestion.hashing import hash_content


def test_digest_is_sha256_hex() -> None:
    digest = hash_content(b"hello")

    assert len(digest) == 64
    assert digest == hashlib.sha256(b"hello").hexdigest()
    int(digest, 16)  # every character is valid hex


def test_digest_is_deterministic() -> None:
    assert hash_content(b"same bytes") == hash_content(b"same bytes")


@pytest.mark.parametrize("data", [b"", b"a", b"\x00\x01\x02"])
def test_empty_and_binary_content_are_hashable(data: bytes) -> None:
    assert hash_content(data) == hashlib.sha256(data).hexdigest()


def test_identical_bytes_in_different_upload_names_hash_identically() -> None:
    # The upload path hashes bytes only; the filename plays no part.
    assert hash_content(b"policy") == hash_content(b"policy")


def test_large_input_is_hashed_across_block_boundaries() -> None:
    # Larger than the streamed block size, so the loop runs more than once.
    data = b"x" * (3 * 1024 * 1024 + 17)

    assert hash_content(data) == hashlib.sha256(data).hexdigest()


def test_different_bytes_produce_different_digests() -> None:
    assert hash_content(b"policy v1") != hash_content(b"policy v2")
    assert hash_content(b"a") != hash_content(b"a ")
