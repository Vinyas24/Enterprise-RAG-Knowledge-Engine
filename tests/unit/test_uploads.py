"""Upload size limiting: bounded memory, correct error semantics."""

from __future__ import annotations

import io

import pytest
from starlette.datastructures import UploadFile

from rag_engine.api.errors import ApiError
from rag_engine.api.uploads import read_upload_limited


def _upload(data: bytes, filename: str = "notes.txt") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename, size=len(data))


async def test_reads_the_whole_file_below_the_limit() -> None:
    assert await read_upload_limited(_upload(b"hello"), max_bytes=10) == b"hello"


async def test_exact_limit_is_allowed() -> None:
    assert await read_upload_limited(_upload(b"12345"), max_bytes=5) == b"12345"


async def test_oversized_upload_raises_413_with_the_limit() -> None:
    with pytest.raises(ApiError) as excinfo:
        await read_upload_limited(_upload(b"123456"), max_bytes=5)

    error = excinfo.value
    assert error.status_code == 413
    assert error.code == "file_too_large"
    assert error.details == {"max_bytes": 5}


async def test_empty_upload_raises_422() -> None:
    with pytest.raises(ApiError) as excinfo:
        await read_upload_limited(_upload(b""), max_bytes=10)

    assert excinfo.value.status_code == 422
    assert excinfo.value.code == "empty_document"


async def test_streaming_input_larger_than_one_block_is_bounded(tmp_path: object) -> None:
    # >1 MiB exercises the read loop more than once; the limit still applies.
    payload = b"a" * (1024 * 1024 + 10)

    with pytest.raises(ApiError):
        await read_upload_limited(_upload(payload), max_bytes=1024 * 1024)
