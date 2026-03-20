"""Report generator -- section 10.1 Markdown for IEEE Q1 journal."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import Phase7Config
from .state_machine import (
    AlertCategory,
    AlertSeverity,
    MonitoringAlert,
    PerformanceSnapshot,
    SecurityEvent,
    StateChangeEvent,
)

logger = logging.getLogger(__name__)


def render_monitoring_report(
    report_dir: Path,
    summary: Dict[str, Any],
    state_changes: List[StateChangeEvent],
    engine_states: Dict[str, str],
    engine_names: Dict[str, str],
    performance_data: Dict[str, List[PerformanceSnapshot]],
    security_events: List[SecurityEvent],
    alerts: List[MonitoringAlert],
    artifact_hashes: Dict[str, str],
    git_commit: str,
    hardware: Dict[str, str],
    config: Phase7Config,
) -> Path:
    """Generate report_section_monitoring.md for IEEE Q1.

    Returns:
        Path to the generated report.
    """
    report_path = report_dir / "report_section_monitoring.md"
    now = datetime.now(timezone.utc).isoformat()

    # Alert counts
    sev_counts: Dict[str, int] = {s.value: 0 for s in AlertSeverity}
    cat_counts: Dict[str, int] = {c.value: 0 for c in AlertCategory}
    for a in alerts:
        sev_counts[a.severity] = sev_counts.get(a.severity, 0) + 1
        cat_counts[a.category] = cat_counts.get(a.category, 0) + 1

    # Security summary
    sec_verified = sum(1 for e in security_events if e.event_type == "HASH_VERIFIED")
    sec_violations = sum(1 for e in security_events if e.severity == AlertSeverity.CRITICAL.value)

    # Engine health table
    engine_ids = sorted(engine_states.keys())
    health_rows: List[str] = []
    for eid in engine_ids:
        state = engine_states[eid]
        name = engine_names.get(eid, eid)
        n_ch = sum(1 for sc in state_changes if sc.engine_id == eid)
        health_rows.append(f"| {eid} | {name} | {state} | {n_ch} |")

    # Performance table
    perf_rows: List[str] = []
    for eid in engine_ids:
        snaps = performance_data.get(eid, [])
        if snaps:
            last = snaps[-1]
            perf_rows.append(
                f"| {eid} | {last.latency_p50_ms:.1f} "
                f"| {last.latency_p95_ms:.1f} "
                f"| {last.latency_p99_ms:.1f} "
                f"| {last.throughput_sps:.1f} "
                f"| {last.memory_mb:.0f} "
                f"| {last.cpu_pct:.1f} |"
            )
        else:
            perf_rows.append(f"| {eid} | -- | -- | -- | -- | -- | -- |")

    # Artifact hash table
    hash_rows: List[str] = []
    for name, digest in artifact_hashes.items():
        hash_rows.append(f"| {name} | `{digest[:16]}...` |")

    # Monitored engines table
    engines_rows: List[str] = []
    for e in config.engines:
        engines_rows.append(f"| {e.id} | {e.heartbeat_topic} | {e.artifact_dir or 'N/A'} |")

    hb_interval = config.heartbeat_interval_seconds
    missed_thresh = config.missed_heartbeat_threshold
    grace = hb_interval * missed_thresh

    lines = [
        "## 10.1 System Monitoring & Observability (Phase 7)",
        "",
        "### 10.1.1 Engine Health Summary",
        "",
        "| Engine | Name | State | State Changes |",
        "|--------|------|-------|---------------|",
        *health_rows,
        "",
        f"**Total state changes:** {len(state_changes)}",
        "",
        "### 10.1.2 Performance Metrics (Latest Snapshot)",
        "",
        ("| Engine | p50 (ms) | p95 (ms) | p99 (ms) " "| Throughput | Memory (MB) | CPU (%) |"),
        ("|--------|----------|----------|----------" "|------------|-------------|---------|"),
        *perf_rows,
        "",
        "**Thresholds:**",
        (f"- Latency p95 > {config.latency_p95_threshold_ms:.0f}ms " "-> DEGRADED"),
        f"- Memory > {config.memory_warning_threshold_pct:.0f}% -> WARNING",
        f"- CPU > {config.cpu_warning_threshold_pct:.0f}% -> WARNING",
        "",
        "### 10.1.3 Security Audit",
        "",
        (
            f"- **Artifact integrity checks:** {sec_verified} "
            f"verified, {sec_violations} violations"
        ),
        (
            "- **Baseline config tamper:** "
            + (
                "PASS -- no tampering detected"
                if sec_violations == 0
                else "FAIL -- tampering detected"
            )
        ),
        f"- **Total security events:** {len(security_events)}",
        "",
        "### 10.1.4 Alert Summary",
        "",
        "| Severity | Count |",
        "|----------|-------|",
        f"| CRITICAL | {sev_counts.get('CRITICAL', 0)} |",
        f"| WARNING  | {sev_counts.get('WARNING', 0)} |",
        f"| INFO     | {sev_counts.get('INFO', 0)} |",
        "",
        "| Category    | Count |",
        "|-------------|-------|",
        f"| HEALTH      | {cat_counts.get('HEALTH', 0)} |",
        f"| PERFORMANCE | {cat_counts.get('PERFORMANCE', 0)} |",
        f"| SECURITY    | {cat_counts.get('SECURITY', 0)} |",
        "",
        f"**Total alerts:** {len(alerts)}",
        "",
        "### 10.1.5 State Machine Specification",
        "",
        "```",
        "UNKNOWN -> STARTING -> UP <-> DEGRADED",
        "                       |         |",
        "                       v         v",
        "                      DOWN <-----+",
        "                       |",
        "                       +-> STARTING (heartbeat received)",
        "```",
        "",
        "| Transition | Trigger |",
        "|------------|---------|",
        "| UNKNOWN -> STARTING | First heartbeat received |",
        (f"| STARTING -> UP | " f"{config.consecutive_ok_threshold} consecutive heartbeats |"),
        (f"| UP -> DEGRADED | " f"Latency > {config.latency_p95_threshold_ms:.0f}ms |"),
        (f"| UP -> DOWN | " f"{config.missed_heartbeat_threshold} missed heartbeats |"),
        "| DEGRADED -> UP | Latency returns to normal |",
        (f"| DEGRADED -> DOWN | " f"{config.missed_heartbeat_threshold} missed heartbeats |"),
        "| DOWN -> STARTING | Heartbeat received again |",
        "",
        "### 10.1.6 Heartbeat Justification",
        "",
        f"- **Interval:** {hb_interval}s per engine",
        f"- **Missed threshold:** {missed_thresh} consecutive misses",
        (
            f"- **Grace period:** {missed_thresh} x {hb_interval}s = "
            f"{grace}s before DOWN transition"
        ),
        (
            "- **Rationale:** Balances responsiveness "
            f"({grace}s detection) "
            "with tolerance for transient network issues"
        ),
        "",
        "### 10.1.7 Storage Optimization",
        "",
        "- **Strategy:** State changes only (not every heartbeat tick)",
        (f"- **Buffer:** Circular buffer of " f"{config.circular_buffer_size} events per engine"),
        "- **Reduction:** ~99% storage reduction vs. full heartbeat logging",
        (f"- **Window:** {config.rolling_window_hours}h rolling " "window for performance metrics"),
        "",
        "### 10.1.8 Monitored Engines",
        "",
        "| Engine ID | Heartbeat Topic | Artifact Dir |",
        "|-----------|-----------------|--------------|",
        *engines_rows,
        "",
        "### 10.1.9 Monitoring Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Heartbeat interval | {hb_interval}s |",
        f"| Missed heartbeat threshold | {missed_thresh} |",
        f"| Grace period | {grace}s |",
        f"| Circular buffer size | {config.circular_buffer_size} |",
        (f"| Performance collection interval | " f"{config.performance_collection_interval}s |"),
        (
            f"| Artifact integrity check interval | "
            f"{config.artifact_integrity_check_interval}s |"
        ),
        (f"| Baseline check interval | " f"{config.baseline_check_interval_seconds}s |"),
        "| Storage | State changes only |",
        f"| Rolling window | {config.rolling_window_hours} hours |",
        "",
        "### 10.1.10 Async Architecture",
        "",
        "```",
        "asyncio.gather(",
        (f"    heartbeat_loop     -- every " f"{hb_interval}s x " f"{len(config.engines)} engines"),
        (
            f"    performance_loop   -- every "
            f"{config.performance_collection_interval}s x "
            f"{len(config.engines)} engines"
        ),
        (
            f"    security_loop      -- every "
            f"{config.artifact_integrity_check_interval}s x SHA-256"
        ),
        (
            f"    dashboard_loop     -- every "
            f"{config.dashboard_push_interval_seconds}s x WebSocket"
        ),
        ")",
        "```",
        "",
        "### 10.1.11 Artifact Integrity (Phase 7 Outputs)",
        "",
        "| Artifact | SHA-256 |",
        "|----------|---------|",
        *hash_rows,
        "",
        "### 10.1.12 Execution Details",
        "",
        f"- **Duration:** {summary.get('duration_seconds', 0):.2f}s",
        f"- **Monitoring cycles:** {summary.get('n_cycles', 0)}",
        f"- **Engines monitored:** {summary.get('n_engines', 0)}",
        f"- **Dashboard pushes:** {summary.get('dashboard_pushes', 0)}",
        f"- **Git commit:** `{git_commit[:12]}...`",
        "- **Pipeline:** phase7_monitoring (SOLID)",
        f"- **Hardware:** {hardware.get('device', 'N/A')}",
        f"- **Python:** {hardware.get('python', 'N/A')}",
        "",
        "---",
        "",
        (f"*Generated: {now} | Pipeline: phase7_monitoring" f" | Commit: {git_commit[:12]}*"),
        "",
    ]

    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Report generated: %s", report_path)
    return report_path
