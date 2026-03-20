"""Monitoring artifact exporter -- 4 JSON files with SHA-256 hashes."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .integrity_checker import ArtifactIntegrityChecker
from .state_machine import (
    MonitoringAlert,
    PerformanceSnapshot,
    SecurityEvent,
    StateChangeEvent,
)

logger = logging.getLogger(__name__)

MONITORING_LOG: str = "monitoring_log.json"
HEALTH_REPORT: str = "health_report.json"
PERFORMANCE_REPORT: str = "performance_report.json"
SECURITY_AUDIT_LOG: str = "security_audit_log.json"


class MonitoringExporter:
    """Exports Phase 7 monitoring artifacts with SHA-256 hashes.

    Produces 4 JSON files:
    - monitoring_log.json: state change events only
    - health_report.json: current engine states + alerts
    - performance_report.json: rolling metrics per engine
    - security_audit_log.json: security events
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def export(
        self,
        state_changes: List[StateChangeEvent],
        engine_states: Dict[str, str],
        engine_names: Dict[str, str],
        performance_data: Dict[str, List[PerformanceSnapshot]],
        security_events: List[SecurityEvent],
        alerts: List[MonitoringAlert],
        summary: Dict[str, Any],
    ) -> Dict[str, str]:
        """Export all monitoring artifacts.

        Returns:
            Mapping of artifact name to SHA-256 hash.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # 1. monitoring_log.json -- state changes only
        monitoring_log = [asdict(sc) for sc in state_changes]
        ml_path = self._output_dir / MONITORING_LOG
        with open(ml_path, "w") as f:
            json.dump(monitoring_log, f, indent=2)

        # 2. health_report.json -- current state per engine + alerts
        health_report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": "phase7_monitoring",
            "engine_health": {
                eid: {
                    "state": engine_states[eid],
                    "name": engine_names.get(eid, eid),
                }
                for eid in engine_states
            },
            "alerts": [asdict(a) for a in alerts],
            "summary": summary,
        }
        hr_path = self._output_dir / HEALTH_REPORT
        with open(hr_path, "w") as f:
            json.dump(health_report, f, indent=2)

        # 3. performance_report.json -- rolling metrics per engine
        perf_report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": "phase7_monitoring",
            "metrics": {
                eid: [asdict(s) for s in snapshots] for eid, snapshots in performance_data.items()
            },
        }
        pr_path = self._output_dir / PERFORMANCE_REPORT
        with open(pr_path, "w") as f:
            json.dump(perf_report, f, indent=2)

        # 4. security_audit_log.json -- security events only
        sec_log = [asdict(e) for e in security_events]
        sal_path = self._output_dir / SECURITY_AUDIT_LOG
        with open(sal_path, "w") as f:
            json.dump(sec_log, f, indent=2)

        # Compute SHA-256 hashes
        artifact_hashes: Dict[str, str] = {}
        for name, path in [
            (MONITORING_LOG, ml_path),
            (HEALTH_REPORT, hr_path),
            (PERFORMANCE_REPORT, pr_path),
            (SECURITY_AUDIT_LOG, sal_path),
        ]:
            digest = ArtifactIntegrityChecker.compute_sha256(path)
            artifact_hashes[name] = digest
            logger.info("Exported %s -- SHA-256: %s...", name, digest[:16])

        return artifact_hashes
