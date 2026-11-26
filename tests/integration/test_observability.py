"""Integration tests for Phase 10: Observability, Tracing, Metrics, and API v1 Polish."""

from __future__ import annotations

from typing import Any

import pytest
from httpx2 import AsyncClient

from rag_engine.core.config import Environment, Settings
from rag_engine.ingestion.processor import IngestionProcessor

pytestmark = pytest.mark.integration

DOC_CONTENT = (
    "Observability in RAG Knowledge Engine is powered by stage tracing. "
    "W3C Server-Timing headers provide millisecond breakdown. "
    "Prometheus metrics expose query counts, error rates, and cache hits."
)


async def _upload(
    client: AsyncClient,
    headers: dict[str, str],
    *,
    filename: str = "observability.txt",
    data: bytes = DOC_CONTENT.encode(),
    access_level: str = "internal",
    title: str = "Observability Spec",
) -> Any:
    return await client.post(
        "/v1/documents",
        headers=headers,
        files={"file": (filename, data, "text/plain")},
        data={"access_level": access_level, "title": title},
    )


async def test_pipeline_tracing_and_server_timing(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
) -> None:
    # 1. Ingest document
    resp = await _upload(api_client, auth_headers)
    assert resp.status_code == 202
    await processor.process_next()

    # 2. Query /v1/query
    query_resp = await api_client.post(
        "/v1/query",
        headers=auth_headers,
        json={"question": "How is observability powered in RAG Engine?"},
    )
    assert query_resp.status_code == 200

    # 3. Verify W3C Server-Timing header
    server_timing = query_resp.headers.get("Server-Timing")
    assert server_timing is not None
    assert "total;dur=" in server_timing
    assert "retrieve;dur=" in server_timing

    # 4. Verify timings payload in response body
    data = query_resp.json()
    assert "timings" in data
    timings = data["timings"]
    assert timings is not None
    assert timings["total_ms"] > 0
    assert timings["retrieve_ms"] >= 0


async def test_prometheus_metrics_endpoint(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    # Run a query to generate telemetry
    await api_client.post(
        "/v1/query",
        headers=auth_headers,
        json={"question": "What is the metrics format?"},
    )

    # Scrape /metrics
    metrics_resp = await api_client.get("/v1/metrics")
    assert metrics_resp.status_code == 200
    assert "text/plain" in metrics_resp.headers["content-type"]
    text = metrics_resp.text
    assert "rag_http_requests_total" in text
    assert "rag_pipeline_stage_duration_seconds" in text


async def test_v1_documents_pagination_and_access_filter(
    api_client: AsyncClient,
    auth_headers: dict[str, str],
    processor: IngestionProcessor,
) -> None:
    resp = await _upload(api_client, auth_headers, access_level="internal")
    assert resp.status_code == 202
    await processor.process_next()

    # Query matching access level
    list_resp = await api_client.get(
        "/v1/documents",
        headers=auth_headers,
        params={"access_level": "internal", "limit": 10, "offset": 0},
    )
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["total"] >= 1
    assert any(item["access_level"] == "internal" for item in data["items"])

    # Query non-matching access level
    restricted_resp = await api_client.get(
        "/v1/documents",
        headers=auth_headers,
        params={"access_level": "restricted", "limit": 10, "offset": 0},
    )
    assert restricted_resp.status_code == 200
    r_data = restricted_resp.json()
    assert r_data["total"] == 0


async def test_auth_enforcement_in_integration(
    api_client_factory: Any,
    auth_headers: dict[str, str],
) -> None:
    custom_settings = Settings(
        environment=Environment.TEST,
        auth_enabled=True,
        api_key="integration-key-secret",
        _env_file=None,
    )

    async with api_client_factory(custom_settings) as client:
        # Request without API key should be rejected
        unauth_resp = await client.post(
            "/v1/query",
            headers=auth_headers,
            json={"question": "Will this fail?"},
        )
        assert unauth_resp.status_code == 401
        assert unauth_resp.json()["error"]["code"] == "unauthorized"

        # Request with valid API key should proceed
        authed_headers = {**auth_headers, "X-API-Key": "integration-key-secret"}
        auth_resp = await client.post(
            "/v1/query",
            headers=authed_headers,
            json={"question": "Will this succeed?"},
        )
        assert auth_resp.status_code == 200
