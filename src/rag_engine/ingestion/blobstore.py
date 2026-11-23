"""Binary blob storage behind an interface.

The API writes uploads, the worker reads them. Content-addressed keys make
writes idempotent and duplicates harmless: two uploads of the same bytes under
the same tenant resolve to the same key, so a race cannot corrupt either copy.

Swap-in note: an S3/MinIO implementation only has to satisfy ``BlobStore``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Protocol

from rag_engine.ingestion.hashing import hash_content


class BlobStore(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...

    async def get(self, key: str) -> bytes: ...


class LocalBlobStore:
    """Filesystem store rooted at a configured directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    @staticmethod
    def build_key(*, tenant_slug: str, content_hash: str, suffix: str) -> str:
        """Content-addressed, sharded key: ``<tenant>/<ab>/<hash><suffix>``.

        Sharding by the first two hash characters keeps directory sizes
        manageable on a plain filesystem.
        """
        return f"{tenant_slug}/{content_hash[:2]}/{content_hash}{suffix}"

    def _path_for(self, key: str) -> Path:
        candidate = (self._root / key).resolve()
        root = self._root.resolve()
        if not candidate.is_relative_to(root):
            # Keys are generated internally, but defence in depth is cheap.
            raise ValueError(f"blob key escapes the storage root: {key!r}")
        return candidate

    async def put(self, key: str, data: bytes) -> None:
        path = self._path_for(key)
        await asyncio.to_thread(self._write_atomic, path, data)

    async def get(self, key: str) -> bytes:
        path = self._path_for(key)
        return await asyncio.to_thread(path.read_bytes)

    def _write_atomic(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(data)
        # Atomic replace: a crash mid-write never leaves a truncated blob.
        os.replace(temporary, path)


def verify_content(data: bytes, expected_hash: str) -> None:
    """Raise ``ValueError`` if stored bytes do not match the recorded hash."""
    if hash_content(data) != expected_hash:
        raise ValueError("stored content does not match the recorded content hash")
