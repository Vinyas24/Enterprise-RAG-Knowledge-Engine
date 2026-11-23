"""Content hashing used for deduplication and integrity checks."""

from __future__ import annotations

import hashlib

# Streaming-friendly block size; uploaded files are fully in memory today
# (bounded by upload_max_bytes), but the API mirrors hashlib's contract.
_BLOCK_SIZE = 1024 * 1024


def hash_content(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``.

    Hashing the raw bytes (not extracted text) means an identical re-upload is
    detected before any parsing work happens, and a corrupted blob can be
    detected when the worker re-reads it.
    """
    digest = hashlib.sha256()
    for offset in range(0, len(data), _BLOCK_SIZE):
        digest.update(data[offset : offset + _BLOCK_SIZE])
    return digest.hexdigest()
