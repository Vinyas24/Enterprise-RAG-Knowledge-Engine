"""Upload reading with a hard size ceiling.

``UploadFile.read()`` with no argument would buffer the whole request into
memory before the limit is checked. Reading in blocks and aborting at the limit
bounds memory per request to roughly ``max_bytes`` plus one block, even for a
hostile client.
"""

from __future__ import annotations

from fastapi import UploadFile

from rag_engine.api.errors import ApiError

_READ_BLOCK_BYTES = 1024 * 1024


async def read_upload_limited(upload: UploadFile, *, max_bytes: int) -> bytes:
    buffer = bytearray()
    while block := await upload.read(_READ_BLOCK_BYTES):
        buffer.extend(block)
        if len(buffer) > max_bytes:
            raise ApiError(
                413,
                "file_too_large",
                f"uploaded file exceeds the {max_bytes} byte limit",
                details={"max_bytes": max_bytes},
            )
    if not buffer:
        raise ApiError(422, "empty_document", "uploaded file is empty")
    return bytes(buffer)
