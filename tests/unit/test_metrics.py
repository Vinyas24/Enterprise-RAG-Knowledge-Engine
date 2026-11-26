"""Unit tests for the in-memory Prometheus metrics registry."""

from __future__ import annotations

from rag_engine.observability.metrics import MetricsRegistry


def test_metrics_counter_increment_and_query() -> None:
    registry = MetricsRegistry()
    assert registry.get_counter_value("rag_http_requests_total") == 0.0

    registry.increment_counter(
        "rag_http_requests_total",
        labels={"method": "POST", "path": "/query", "status": "200"},
    )
    registry.increment_counter(
        "rag_http_requests_total",
        labels={"method": "POST", "path": "/query", "status": "200"},
    )
    registry.increment_counter(
        "rag_http_requests_total",
        labels={"method": "GET", "path": "/health/live", "status": "200"},
    )

    assert (
        registry.get_counter_value(
            "rag_http_requests_total",
            labels={"method": "POST", "path": "/query", "status": "200"},
        )
        == 2.0
    )
    assert (
        registry.get_counter_value(
            "rag_http_requests_total",
            labels={"method": "GET", "path": "/health/live", "status": "200"},
        )
        == 1.0
    )


def test_metrics_timing_and_stats() -> None:
    registry = MetricsRegistry()
    registry.record_timing("rag_pipeline_stage_duration_seconds", 0.05, labels={"stage": "dense"})
    registry.record_timing("rag_pipeline_stage_duration_seconds", 0.07, labels={"stage": "dense"})

    count, total = registry.get_timing_stats(
        "rag_pipeline_stage_duration_seconds", labels={"stage": "dense"}
    )
    assert count == 2
    assert round(total, 4) == 0.12


def test_metrics_gauge() -> None:
    registry = MetricsRegistry()
    registry.set_gauge("rag_active_connections", 5.0)
    exposition = registry.generate_prometheus_exposition()
    assert "rag_active_connections 5.0" in exposition


def test_metrics_exposition_output_format() -> None:
    registry = MetricsRegistry()
    registry.increment_counter(
        "rag_cache_hits_total", labels={"cache": "embedding"}, help_text="Cache hits"
    )
    registry.record_timing("rag_pipeline_stage_duration_seconds", 0.015, labels={"stage": "fuse"})

    exposition = registry.generate_prometheus_exposition()
    assert "# HELP rag_cache_hits_total Cache hits" in exposition
    assert "# TYPE rag_cache_hits_total counter" in exposition
    assert 'rag_cache_hits_total{cache="embedding"} 1.0' in exposition

    assert "# HELP rag_pipeline_stage_duration_seconds" in exposition
    assert "# TYPE rag_pipeline_stage_duration_seconds summary" in exposition
    assert 'rag_pipeline_stage_duration_seconds_count{stage="fuse"} 1' in exposition
    assert 'rag_pipeline_stage_duration_seconds_sum{stage="fuse"} 0.015000' in exposition
    assert exposition.endswith("\n")


def test_metrics_reset() -> None:
    registry = MetricsRegistry()
    registry.increment_counter("test_counter", 1.0)
    registry.reset()
    assert registry.get_counter_value("test_counter") == 0.0
