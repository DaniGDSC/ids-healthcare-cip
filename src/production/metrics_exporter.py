"""Prometheus metrics exporter for observability.

Exposes inference latency, throughput, alert rates, and drift events
as Prometheus-compatible metrics. Designed for Grafana dashboards
used by the hospital ops team (separate from clinical Streamlit dashboard).

If prometheus_client is not installed, operates in dry_run mode
with in-memory counters only.

Endpoint: GET /metrics (integrated into FastAPI app)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Try to import prometheus_client; fall back to in-memory counters
try:
    from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsExporter:
    """Prometheus metrics for the IoMT IDS inference pipeline.

    Metrics:
      iomt_inference_total          — total inferences run (counter)
      iomt_inference_latency_ms     — inference latency histogram
      iomt_alerts_emitted_total     — alerts emitted by severity (counter)
      iomt_alerts_suppressed_total  — alerts suppressed by fatigue (counter)
      iomt_drift_events_total       — concept drift events (counter)
      iomt_escalations_total        — CRITICAL escalations (counter)
      iomt_buffer_flows             — current buffer flow count (gauge)
      iomt_model_info               — model metadata (info)

    Args:
        prefix: Metric name prefix.
    """

    def __init__(self, prefix: str = "iomt") -> None:
        self._prefix = prefix
        self._counters: Dict[str, int] = {
            "inference_total": 0,
            "alerts_emitted": 0,
            "alerts_suppressed": 0,
            "drift_events": 0,
            "escalations": 0,
        }
        self._latencies: list = []
        self._buffer_flows: int = 0

        if PROMETHEUS_AVAILABLE:
            self._prom_inference_total = Counter(
                f"{prefix}_inference_total",
                "Total model inferences",
            )
            self._prom_latency = Histogram(
                f"{prefix}_inference_latency_ms",
                "Inference latency in milliseconds",
                buckets=[10, 25, 50, 75, 100, 150, 200, 500, 1000],
            )
            self._prom_alerts_emitted = Counter(
                f"{prefix}_alerts_emitted_total",
                "Alerts emitted to channels",
                ["severity"],
            )
            self._prom_alerts_suppressed = Counter(
                f"{prefix}_alerts_suppressed_total",
                "Alerts suppressed by fatigue mitigation",
            )
            self._prom_drift = Counter(
                f"{prefix}_drift_events_total",
                "Concept drift detection events",
            )
            self._prom_escalations = Counter(
                f"{prefix}_escalations_total",
                "CRITICAL escalation events",
            )
            self._prom_buffer = Gauge(
                f"{prefix}_buffer_flows",
                "Current flows in sliding window buffer",
            )

    def record_inference(self, latency_ms: float) -> None:
        """Record a completed inference."""
        self._counters["inference_total"] += 1
        self._latencies.append(latency_ms)
        if PROMETHEUS_AVAILABLE:
            self._prom_inference_total.inc()
            self._prom_latency.observe(latency_ms)

    def record_alert(self, severity: int, emitted: bool) -> None:
        """Record an alert (emitted or suppressed)."""
        if emitted:
            self._counters["alerts_emitted"] += 1
            if PROMETHEUS_AVAILABLE:
                self._prom_alerts_emitted.labels(severity=str(severity)).inc()
        else:
            self._counters["alerts_suppressed"] += 1
            if PROMETHEUS_AVAILABLE:
                self._prom_alerts_suppressed.inc()

    def record_drift(self) -> None:
        """Record a concept drift event."""
        self._counters["drift_events"] += 1
        if PROMETHEUS_AVAILABLE:
            self._prom_drift.inc()

    def record_escalation(self) -> None:
        """Record a CRITICAL escalation."""
        self._counters["escalations"] += 1
        if PROMETHEUS_AVAILABLE:
            self._prom_escalations.inc()

    def update_buffer_size(self, flow_count: int) -> None:
        """Update the buffer flow gauge."""
        self._buffer_flows = flow_count
        if PROMETHEUS_AVAILABLE:
            self._prom_buffer.set(flow_count)

    def get_prometheus_output(self) -> str:
        """Generate Prometheus exposition format output."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest().decode()

        # Fallback: manual format
        lines = []
        for name, val in self._counters.items():
            lines.append(f"# TYPE {self._prefix}_{name} counter")
            lines.append(f"{self._prefix}_{name} {val}")
        lines.append(f"# TYPE {self._prefix}_buffer_flows gauge")
        lines.append(f"{self._prefix}_buffer_flows {self._buffer_flows}")
        return "\n".join(lines) + "\n"

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary as dict."""
        latencies = self._latencies[-1000:]  # last 1000
        p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0
        p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1) if latencies else 0
        p95 = sorted(latencies)[p95_idx] if latencies else 0

        return {
            **self._counters,
            "buffer_flows": self._buffer_flows,
            "latency_p50_ms": round(p50, 1),
            "latency_p95_ms": round(p95, 1),
            "latency_samples": len(latencies),
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }
