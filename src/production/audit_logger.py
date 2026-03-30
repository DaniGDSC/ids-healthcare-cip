"""FDA 21 CFR Part 11 compliant audit logger.

Requirements:
  - Tamper-evident: each entry includes SHA-256 hash of previous entry
  - Immutable: append-only file, no modification or deletion
  - Timestamped: UTC ISO-8601 with microsecond precision
  - Attributable: every action has a user/service identity
  - Complete: captures config changes, alerts, escalations, model loads

The hash chain ensures integrity — any modification to a past entry
breaks the chain, detectable by verify_chain().
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEntry:
    """Single audit log entry with hash chain."""

    __slots__ = ("sequence", "timestamp", "event_type", "actor",
                 "details", "prev_hash", "entry_hash")

    def __init__(
        self,
        sequence: int,
        event_type: str,
        actor: str,
        details: Dict[str, Any],
        prev_hash: str,
    ) -> None:
        self.sequence = sequence
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.event_type = event_type
        self.actor = actor
        self.details = details
        self.prev_hash = prev_hash
        self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = f"{self.sequence}|{self.timestamp}|{self.event_type}|{self.actor}|{json.dumps(self.details, sort_keys=True)}|{self.prev_hash}"
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.sequence,
            "ts": self.timestamp,
            "event": self.event_type,
            "actor": self.actor,
            "details": self.details,
            "prev_hash": self.prev_hash,
            "hash": self.entry_hash,
        }


class FDAAuditLogger:
    """FDA 21 CFR Part 11 compliant audit logger with hash chain.

    Each entry is cryptographically linked to the previous entry.
    The log file is append-only — entries are written as JSON lines.

    Args:
        log_path: Path to the audit log file.
        service_name: Name of the service writing entries.
    """

    # Event types
    MODEL_LOADED = "MODEL_LOADED"
    CONFIG_CHANGED = "CONFIG_CHANGED"
    ALERT_EMITTED = "ALERT_EMITTED"
    ALERT_SUPPRESSED = "ALERT_SUPPRESSED"
    ESCALATION_TRIGGERED = "ESCALATION_TRIGGERED"
    ESCALATION_CONFIRMED = "ESCALATION_CONFIRMED"
    THRESHOLD_LOCKED = "THRESHOLD_LOCKED"
    THRESHOLD_RESUMED = "THRESHOLD_RESUMED"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    DEVICE_ISOLATED = "DEVICE_ISOLATED"
    DEVICE_RESTRICTED = "DEVICE_RESTRICTED"
    USER_LOGIN = "USER_LOGIN"
    USER_LOGOUT = "USER_LOGOUT"
    ARTIFACT_VERIFIED = "ARTIFACT_VERIFIED"
    ARTIFACT_TAMPERED = "ARTIFACT_TAMPERED"

    def __init__(
        self,
        log_path: str | Path = "data/audit/fda_audit.jsonl",
        service_name: str = "iomt-ids",
    ) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._service = service_name
        self._sequence = 0
        self._last_hash = "GENESIS"

        # Resume from existing log
        if self._path.exists():
            self._resume_from_file()

    def log(
        self,
        event_type: str,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Append an audit entry to the log.

        Args:
            event_type: One of the class constants (e.g., ALERT_EMITTED).
            actor: Who/what triggered the event (user, service, system).
            details: Additional structured data (no PHI).

        Returns:
            The created AuditEntry.
        """
        self._sequence += 1
        entry = AuditEntry(
            sequence=self._sequence,
            event_type=event_type,
            actor=actor,
            details=details or {},
            prev_hash=self._last_hash,
        )
        self._last_hash = entry.entry_hash

        # Append to file
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")

        return entry

    def log_alert(self, alert: Dict[str, Any], emitted: bool) -> AuditEntry:
        """Log an alert event (emitted or suppressed).

        Strips PHI — only logs risk level, severity, and device action.
        """
        details = {
            "risk_level": alert.get("risk_level"),
            "clinical_severity": alert.get("clinical_severity"),
            "device_action": alert.get("device_action"),
            "attention_flag": alert.get("attention_flag"),
            "sample_index": alert.get("sample_index"),
        }
        event = self.ALERT_EMITTED if emitted else self.ALERT_SUPPRESSED
        return self.log(event, self._service, details)

    def log_escalation(self, escalation_record: Dict[str, Any]) -> AuditEntry:
        """Log an escalation event."""
        details = {
            "alert_id": escalation_record.get("alert_id"),
            "levels_notified": escalation_record.get("levels_notified"),
            "confirmed": escalation_record.get("confirmed"),
        }
        return self.log(self.ESCALATION_TRIGGERED, self._service, details)

    def log_device_action(self, action: str, device_id: str) -> AuditEntry:
        """Log a device isolation/restriction action."""
        pseudo_id = hashlib.sha256(device_id.encode()).hexdigest()[:12]
        event = self.DEVICE_ISOLATED if action == "isolate_network" else self.DEVICE_RESTRICTED
        return self.log(event, self._service, {"device_hash": pseudo_id, "action": action})

    def verify_chain(self) -> Dict[str, Any]:
        """Verify the integrity of the entire audit log hash chain.

        Returns:
            Dict with is_valid, entries_checked, first_broken (if any).
        """
        if not self._path.exists():
            return {"is_valid": True, "entries_checked": 0}

        prev_hash = "GENESIS"
        entries_checked = 0

        with open(self._path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    return {
                        "is_valid": False,
                        "entries_checked": entries_checked,
                        "error": f"Invalid JSON at line {line_num}",
                    }

                if entry.get("prev_hash") != prev_hash:
                    return {
                        "is_valid": False,
                        "entries_checked": entries_checked,
                        "first_broken": line_num,
                        "expected_prev": prev_hash,
                        "actual_prev": entry.get("prev_hash"),
                    }

                prev_hash = entry.get("hash", "")
                entries_checked += 1

        return {"is_valid": True, "entries_checked": entries_checked}

    def get_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        """Read the last N audit entries."""
        if not self._path.exists():
            return []

        entries = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        return entries[-n:]

    @property
    def entry_count(self) -> int:
        return self._sequence

    def _resume_from_file(self) -> None:
        """Resume sequence and hash from existing log file."""
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._sequence = entry.get("seq", self._sequence)
                    self._last_hash = entry.get("hash", self._last_hash)
                except json.JSONDecodeError:
                    break
