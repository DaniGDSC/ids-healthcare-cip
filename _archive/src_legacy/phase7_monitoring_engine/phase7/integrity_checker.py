"""Artifact integrity checker -- SHA-256 verification for phase artifacts.

Reads metadata files from Phases 2-5 and recomputes hashes for
all artifacts listed in artifact_hashes sections.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .alert_dispatcher import AlertDispatcher
from .base import BaseMonitor, MonitorStatus
from .config import Phase7Config
from .state_machine import AlertSeverity, SecurityEvent

logger = logging.getLogger(__name__)

# Default metadata paths (used when engine config omits metadata_path)
DEFAULT_METADATA_MAP: Dict[str, str] = {
    "phase2_detection": "data/phase2/detection_metadata.json",
    "phase3_classification": "data/phase3/classification_metadata.json",
    "phase4_risk_adaptive": "data/phase4/risk_metadata.json",
    "phase5_explanation": "data/phase5/explanation_metadata.json",
}

DEFAULT_ARTIFACT_DIRS: Dict[str, str] = {
    "phase2_detection": "data/phase2",
    "phase3_classification": "data/phase3",
    "phase4_risk_adaptive": "data/phase4",
    "phase5_explanation": "data/phase5",
}


class ArtifactIntegrityChecker(BaseMonitor):
    """Verifies artifact integrity via SHA-256 hash comparison.

    Reads metadata files from Phases 2-5 and recomputes hashes for
    all artifacts listed in artifact_hashes sections.
    """

    def __init__(
        self,
        config: Phase7Config,
        alert_dispatcher: AlertDispatcher,
        project_root: Path,
    ) -> None:
        self._config = config
        self._dispatcher = alert_dispatcher
        self._root = project_root
        self._running = False
        self._events: List[SecurityEvent] = []
        self._baseline_hash: Optional[str] = None

    async def start(self) -> None:
        """Mark integrity checker as running."""
        self._running = True

    def stop(self) -> None:
        """Mark integrity checker as stopped."""
        self._running = False

    def get_status(self) -> MonitorStatus:
        """Return current integrity checker status."""
        verified = sum(1 for e in self._events if e.event_type == "HASH_VERIFIED")
        violations = sum(1 for e in self._events if e.severity == AlertSeverity.CRITICAL.value)
        return MonitorStatus(
            name="ArtifactIntegrityChecker",
            running=self._running,
            details={"verified": verified, "violations": violations},
        )

    def get_config(self) -> Dict[str, Any]:
        """Return integrity checker configuration."""
        return {
            "check_interval": self._config.artifact_integrity_check_interval,
            "algorithm": "SHA-256",
            "baseline_config_path": self._config.baseline_config_path,
        }

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify_all_artifacts(self) -> List[SecurityEvent]:
        """Verify SHA-256 hashes for all phase artifacts.

        Returns:
            List of SecurityEvent records for this verification cycle.
        """
        events: List[SecurityEvent] = []
        now = datetime.now(timezone.utc).isoformat()

        # Build metadata/artifact maps from engine config
        metadata_map: Dict[str, Optional[str]] = {}
        artifact_dirs: Dict[str, str] = {}

        for engine in self._config.engines:
            if engine.metadata_path:
                metadata_map[engine.id] = engine.metadata_path
            elif engine.id in DEFAULT_METADATA_MAP:
                metadata_map[engine.id] = DEFAULT_METADATA_MAP[engine.id]

            if engine.artifact_dir:
                artifact_dirs[engine.id] = engine.artifact_dir
            elif engine.id in DEFAULT_ARTIFACT_DIRS:
                artifact_dirs[engine.id] = DEFAULT_ARTIFACT_DIRS[engine.id]

        for engine_id, meta_rel in metadata_map.items():
            if meta_rel is None:
                continue
            meta_path = self._root / meta_rel
            if not meta_path.exists():
                events.append(
                    SecurityEvent(
                        event_type="METADATA_MISSING",
                        engine_id=engine_id,
                        timestamp=now,
                        detail=f"Not found: {meta_rel}",
                        severity=AlertSeverity.WARNING.value,
                    )
                )
                continue

            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                events.append(
                    SecurityEvent(
                        event_type="METADATA_READ_ERROR",
                        engine_id=engine_id,
                        timestamp=now,
                        detail=f"Cannot read {meta_rel}: {exc}",
                        severity=AlertSeverity.CRITICAL.value,
                    )
                )
                continue

            artifact_hashes = metadata.get("artifact_hashes", {})
            artifact_dir_str = artifact_dirs.get(engine_id, "")
            artifact_dir = self._root / artifact_dir_str if artifact_dir_str else self._root

            for artifact_name, hash_info in artifact_hashes.items():
                expected = hash_info.get("sha256", "")
                artifact_path = artifact_dir / artifact_name
                if not artifact_path.exists():
                    events.append(
                        SecurityEvent(
                            event_type="ARTIFACT_MISSING",
                            engine_id=engine_id,
                            timestamp=now,
                            detail=f"Missing: {artifact_name}",
                            severity=AlertSeverity.CRITICAL.value,
                        )
                    )
                    continue

                actual = self.compute_sha256(artifact_path)
                if actual != expected:
                    events.append(
                        SecurityEvent(
                            event_type="HASH_MISMATCH",
                            engine_id=engine_id,
                            timestamp=now,
                            detail=(
                                f"{artifact_name}: " f"{expected[:16]}... != " f"{actual[:16]}..."
                            ),
                            severity=AlertSeverity.CRITICAL.value,
                        )
                    )
                else:
                    events.append(
                        SecurityEvent(
                            event_type="HASH_VERIFIED",
                            engine_id=engine_id,
                            timestamp=now,
                            detail=f"{artifact_name}: {actual[:16]}... OK",
                            severity=AlertSeverity.INFO.value,
                        )
                    )

        self._events.extend(events)
        return events

    def verify_baseline_config(self) -> Optional[SecurityEvent]:
        """Verify Phase 4 baseline_config.json for tampering.

        Returns:
            SecurityEvent if status changed, None otherwise.
        """
        now = datetime.now(timezone.utc).isoformat()
        bl_path = self._root / self._config.baseline_config_path
        if not bl_path.exists():
            return None

        current = self.compute_sha256(bl_path)
        if self._baseline_hash is None:
            self._baseline_hash = current
            event = SecurityEvent(
                event_type="BASELINE_INITIALIZED",
                engine_id="phase4_risk_adaptive",
                timestamp=now,
                detail=f"Hash: {current[:16]}...",
                severity=AlertSeverity.INFO.value,
            )
            self._events.append(event)
            return event

        if current != self._baseline_hash:
            event = SecurityEvent(
                event_type="BASELINE_TAMPER_DETECTED",
                engine_id="phase4_risk_adaptive",
                timestamp=now,
                detail=(f"Changed: {self._baseline_hash[:16]}... " f"-> {current[:16]}..."),
                severity=AlertSeverity.CRITICAL.value,
            )
            self._events.append(event)
            return event

        return None

    @property
    def all_events(self) -> List[SecurityEvent]:
        """Return all security events."""
        return list(self._events)
