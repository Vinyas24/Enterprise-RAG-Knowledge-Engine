"""Prometheus-compatible in-memory metrics registry.

Tracks HTTP request counts, request durations, pipeline stage timings, cache
utilization, and guardrail events without external agent dependencies.
"""

from __future__ import annotations

import threading


class MetricsRegistry:
    """Thread-safe in-memory metrics registry formatted for Prometheus scraping."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._timings: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._help_texts: dict[str, str] = {
            "rag_http_requests_total": "Total number of HTTP requests received.",
            "rag_http_request_duration_seconds": "Duration of HTTP requests in seconds.",
            "rag_pipeline_stage_duration_seconds": "Duration of pipeline stages in seconds.",
            "rag_cache_hits_total": "Total cache hits.",
            "rag_cache_misses_total": "Total cache misses.",
            "rag_guardrail_actions_total": "Total guardrail actions evaluated.",
        }

    def _format_labels(self, labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
        help_text: str | None = None,
    ) -> None:
        """Increment a Prometheus counter metric."""
        label_key = self._format_labels(labels)
        with self._lock:
            key = (name, label_key)
            self._counters[key] = self._counters.get(key, 0.0) + value
            if help_text:
                self._help_texts[name] = help_text

    def record_timing(
        self,
        name: str,
        seconds: float,
        labels: dict[str, str] | None = None,
        help_text: str | None = None,
    ) -> None:
        """Record a latency observation in seconds."""
        label_key = self._format_labels(labels)
        with self._lock:
            key = (name, label_key)
            if key not in self._timings:
                self._timings[key] = []
            self._timings[key].append(seconds)
            if help_text:
                self._help_texts[name] = help_text

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        help_text: str | None = None,
    ) -> None:
        """Set the value of a Prometheus gauge."""
        label_key = self._format_labels(labels)
        with self._lock:
            key = (name, label_key)
            self._gauges[key] = value
            if help_text:
                self._help_texts[name] = help_text

    def get_counter_value(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get the current value of a counter."""
        label_key = self._format_labels(labels)
        with self._lock:
            return self._counters.get((name, label_key), 0.0)

    def get_timing_stats(
        self, name: str, labels: dict[str, str] | None = None
    ) -> tuple[int, float]:
        """Get count and sum of recorded timings."""
        label_key = self._format_labels(labels)
        with self._lock:
            values = self._timings.get((name, label_key), [])
            return len(values), sum(values)

    def reset(self) -> None:
        """Reset all metrics (primarily for test isolation)."""
        with self._lock:
            self._counters.clear()
            self._timings.clear()
            self._gauges.clear()

    def generate_prometheus_exposition(self) -> str:
        """Generate text formatted for Prometheus scraping."""
        lines: list[str] = []

        with self._lock:
            # 1. Output counters
            seen_counter_names: set[str] = set()
            for (name, label_tuple), val in sorted(self._counters.items()):
                if name not in seen_counter_names:
                    seen_counter_names.add(name)
                    help_str = self._help_texts.get(name, f"Total count of {name}")
                    lines.append(f"# HELP {name} {help_str}")
                    lines.append(f"# TYPE {name} counter")
                label_str = self._render_labels(label_tuple)
                lines.append(f"{name}{label_str} {val}")

            # 2. Output summaries for timings
            seen_timing_names: set[str] = set()
            for (name, label_tuple), values in sorted(self._timings.items()):
                if name not in seen_timing_names:
                    seen_timing_names.add(name)
                    help_str = self._help_texts.get(name, f"Summary of {name}")
                    lines.append(f"# HELP {name} {help_str}")
                    lines.append(f"# TYPE {name} summary")
                count = len(values)
                total_sum = sum(values)
                label_str_count = self._render_labels(label_tuple)
                label_str_sum = self._render_labels(label_tuple)
                lines.append(f"{name}_count{label_str_count} {count}")
                lines.append(f"{name}_sum{label_str_sum} {total_sum:.6f}")

            # 3. Output gauges
            seen_gauge_names: set[str] = set()
            for (name, label_tuple), val in sorted(self._gauges.items()):
                if name not in seen_gauge_names:
                    seen_gauge_names.add(name)
                    help_str = self._help_texts.get(name, f"Current value of {name}")
                    lines.append(f"# HELP {name} {help_str}")
                    lines.append(f"# TYPE {name} gauge")
                label_str = self._render_labels(label_tuple)
                lines.append(f"{name}{label_str} {val}")

        lines.append("")  # trailing newline required by prometheus
        return "\n".join(lines)

    @staticmethod
    def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return ""
        rendered = ",".join(f'{k}="{v}"' for k, v in labels)
        return f"{{{rendered}}}"


_REGISTRY = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """Return the global process-wide MetricsRegistry."""
    return _REGISTRY
