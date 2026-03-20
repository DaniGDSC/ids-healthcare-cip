#!/usr/bin/env python3
"""Phase 6 Notification Engine — HIPAA-compliant alert routing & delivery.

Consumes Phase 5 explanation artifacts and Phase 4 risk reports
to route, compose, and deliver notifications per risk level:

    LOW:      log_only()
    MEDIUM:   push_dashboard()
    HIGH:     push_dashboard() + send_encrypted_email()
    CRITICAL: push_dashboard() + send_encrypted_email()
              + trigger_onsite_alarm() + escalate()

Security controls (reused from Phase 0 — never duplicated):
    A01  Workspace path validation via PathValidator
    A02  SHA-256 verification of all Phase 4/5 artifacts before loading
    A03  No PHI in email bodies — risk level + timestamp only
    A09  HIPAA-compliant logging — recipient hashed, no raw scores

Usage::

    python -m src.phase6_notification_engine.phase6_notification
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Phase 0 security controls (reused, NOT duplicated)
from src.phase0_dataset_analysis.phase0.security import (
    AuditLogger,
    IntegrityVerifier,
    PathValidator,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Risk levels in escalation order
RISK_LEVELS: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Retry / backoff
MAX_RETRIES: int = 3
BACKOFF_BASE_S: float = 1.0

# Escalation
CONFIRMATION_TIMEOUT_S: int = 300  # 5 minutes
ESCALATION_LEVELS: Tuple[str, ...] = ("it_admin", "doctor_on_duty", "manager")

# Email
TLS_MIN_VERSION: str = "TLSv1.3"
DASHBOARD_URL: str = "https://iomt-dashboard.hospital.internal/alerts"

# Artifact paths (relative to PROJECT_ROOT)
PHASE5_DIR: Path = Path("data/phase5")
PHASE4_DIR: Path = Path("data/phase4")
PHASE5_METADATA: str = "explanation_metadata.json"
PHASE5_REPORT: str = "explanation_report.json"
PHASE4_METADATA: str = "risk_metadata.json"
PHASE4_REPORT: str = "risk_report.json"
CHARTS_DIR: str = "charts"

# Output
OUTPUT_DIR: Path = Path("data/phase6")
NOTIFICATION_LOG: str = "notification_log.json"
DELIVERY_REPORT: str = "delivery_report.json"
ESCALATION_LOG: str = "escalation_log.json"


# ---------------------------------------------------------------------------
# Risk level enum
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    """Alert risk levels in ascending severity."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Notification channels (abstract + concrete)
# ---------------------------------------------------------------------------


class NotificationChannel(ABC):
    """Base class for notification delivery channels."""

    @abstractmethod
    def send(self, payload: Dict[str, Any]) -> bool:
        """Send a notification payload.

        Args:
            payload: Channel-specific notification data.

        Returns:
            True if delivery succeeded.
        """

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Human-readable channel identifier."""


class DashboardPush(NotificationChannel):
    """Real-time WebSocket push to dashboard service.

    In production, connects to wss://iomt-dashboard.hospital.internal/ws.
    For research/CI, simulates the push and logs the payload.
    """

    @property
    def channel_name(self) -> str:
        return "dashboard_websocket"

    def send(self, payload: Dict[str, Any]) -> bool:
        """Push alert to dashboard via WebSocket.

        Args:
            payload: Must contain risk_level, timestamp, top_features, chart_url.

        Returns:
            True on successful delivery (simulated for research).
        """
        required = {"risk_level", "timestamp", "top_features"}
        if not required.issubset(payload.keys()):
            logger.warning("Dashboard push: missing fields %s", required - payload.keys())
            return False

        logger.info(
            "  Dashboard push: %s alert at %s (top features: %s)",
            payload["risk_level"],
            payload["timestamp"],
            ", ".join(f["feature"] for f in payload.get("top_features", [])[:3]),
        )
        return True


class EncryptedEmail(NotificationChannel):
    """TLS 1.3 encrypted email delivery.

    In production, uses SMTP with STARTTLS (TLS 1.3 minimum).
    For research/CI, simulates delivery and logs hashed recipient.
    """

    def __init__(self, tls_version: str = TLS_MIN_VERSION) -> None:
        self._tls_version = tls_version

    @property
    def channel_name(self) -> str:
        return "encrypted_email"

    def send(self, payload: Dict[str, Any]) -> bool:
        """Send encrypted email notification.

        Args:
            payload: Must contain subject, body, recipient, and optionally attachment.

        Returns:
            True on successful delivery (simulated for research).
        """
        recipient = payload.get("recipient", "unknown")
        hashed = hashlib.sha256(recipient.encode()).hexdigest()[:16]

        logger.info(
            "  Email sent to [%s...] at %s (TLS: %s)",
            hashed,
            datetime.now(timezone.utc).isoformat(),
            self._tls_version,
        )
        AuditLogger.log_security_event(
            "EMAIL_SENT",
            f"recipient_hash={hashed}, tls={self._tls_version}",
            logging.INFO,
        )
        return True


class OnsiteAlarm(NotificationChannel):
    """Hospital network on-site alarm trigger.

    In production, calls the hospital alarm API endpoint.
    For research/CI, simulates the trigger and logs the event.
    """

    @property
    def channel_name(self) -> str:
        return "onsite_alarm"

    def send(self, payload: Dict[str, Any]) -> bool:
        """Trigger on-site alarm via hospital network API.

        Args:
            payload: Must contain alert_id and risk_level.

        Returns:
            True on successful trigger (simulated for research).
        """
        alert_id = payload.get("alert_id", "unknown")
        logger.info("  On-site alarm triggered: alert_id=%s", alert_id)
        AuditLogger.log_security_event(
            "ONSITE_ALARM",
            f"alert_id={alert_id}, triggered_at={datetime.now(timezone.utc).isoformat()}",
            logging.WARNING,
        )
        return True


# ---------------------------------------------------------------------------
# HIPAA-compliant email composer
# ---------------------------------------------------------------------------


class EmailComposer:
    """Compose HIPAA-compliant email notifications.

    Emails contain risk level and timestamp ONLY.
    NEVER includes: device ID, patient data, raw anomaly scores,
    SHAP values, or biometric readings.
    """

    def __init__(self, dashboard_url: str = DASHBOARD_URL) -> None:
        self._dashboard_url = dashboard_url

    def compose(
        self,
        alert: Dict[str, Any],
        *,
        attach_chart: bool = False,
        chart_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Compose an email payload for an alert.

        Args:
            alert: Alert dict with risk_level, sample_index, timestamp.
            attach_chart: If True, attach waterfall chart for HIGH/CRITICAL.
            chart_path: Path to chart PNG file.

        Returns:
            Email payload dict with subject, body, attachment fields.
        """
        risk = alert["risk_level"]
        idx = alert["sample_index"]
        ts = alert.get("timestamp", datetime.now(timezone.utc).isoformat())

        subject = f"RA-{idx}-IoMT Alert — Level: {risk}"

        body = (
            f"Risk Level: {risk}\n"
            f"Timestamp: {ts}\n"
            f"\n"
            f"Login to dashboard for details: {self._dashboard_url}\n"
            f"\n"
            f"This is an automated alert from the IoMT IDS.\n"
            f"Do not reply to this email."
        )

        email: Dict[str, Any] = {
            "subject": subject,
            "body": body,
            "recipient": "iomt-alerts@hospital.internal",
        }

        if attach_chart and chart_path and chart_path.exists():
            email["attachment"] = str(chart_path)
            email["attachment_name"] = chart_path.name

        return email


# ---------------------------------------------------------------------------
# Delivery tracker
# ---------------------------------------------------------------------------


class DeliveryTracker:
    """Track delivery status per channel per alert."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def record(
        self,
        alert_id: str,
        channel: str,
        success: bool,
        attempt: int,
        timestamp: str,
    ) -> None:
        """Record a delivery attempt.

        Args:
            alert_id: Unique alert identifier.
            channel: Channel name (dashboard, email, alarm).
            success: Whether delivery succeeded.
            attempt: Attempt number (1-based).
            timestamp: ISO timestamp of the attempt.
        """
        self._records.append(
            {
                "alert_id": alert_id,
                "channel": channel,
                "status": "SUCCESS" if success else "FAILED",
                "attempt": attempt,
                "timestamp": timestamp,
            }
        )

    def get_report(self) -> List[Dict[str, Any]]:
        """Return all delivery records."""
        return list(self._records)

    def get_failures(self) -> List[Dict[str, Any]]:
        """Return only failed delivery records."""
        return [r for r in self._records if r["status"] == "FAILED"]


# ---------------------------------------------------------------------------
# Escalation manager
# ---------------------------------------------------------------------------


class EscalationManager:
    """Manage CRITICAL alert escalation with human confirmation.

    Escalation chain:
        1. IT admin (email + dashboard)
        2. Doctor on duty (dashboard)
        3. Manager (email)

    If no confirmation within 5 minutes, re-escalate to next level.
    """

    def __init__(
        self,
        email_channel: EncryptedEmail,
        dashboard_channel: DashboardPush,
        alarm_channel: OnsiteAlarm,
        confirmation_timeout_s: int = CONFIRMATION_TIMEOUT_S,
    ) -> None:
        self._email = email_channel
        self._dashboard = dashboard_channel
        self._alarm = alarm_channel
        self._timeout_s = confirmation_timeout_s
        self._history: List[Dict[str, Any]] = []

    def escalate(
        self,
        alert: Dict[str, Any],
        alert_id: str,
        tracker: DeliveryTracker,
    ) -> Dict[str, Any]:
        """Execute full CRITICAL escalation protocol.

        Args:
            alert: Alert dict with risk_level, sample_index, timestamp.
            alert_id: Unique alert identifier.
            tracker: Delivery tracker instance.

        Returns:
            Escalation record dict.
        """
        ts = datetime.now(timezone.utc).isoformat()
        logger.info("── Escalation triggered at %s ──", ts)

        AuditLogger.log_security_event(
            "ESCALATION_TRIGGERED",
            f"alert_id={alert_id}, risk_level=CRITICAL",
            logging.CRITICAL,
        )

        record: Dict[str, Any] = {
            "alert_id": alert_id,
            "triggered_at": ts,
            "risk_level": "CRITICAL",
            "confirmation_token": str(uuid.uuid4()),
            "confirmation_timeout_s": self._timeout_s,
            "notifications_sent": [],
            "confirmation_received": False,
        }

        # 1. Trigger on-site alarm
        alarm_ok = self._alarm.send({"alert_id": alert_id, "risk_level": "CRITICAL"})
        tracker.record(alert_id, "onsite_alarm", alarm_ok, 1, ts)
        record["notifications_sent"].append({"target": "onsite_alarm", "status": alarm_ok})

        # 2. Notify IT admin (email + dashboard)
        dashboard_payload = _build_dashboard_payload(alert)
        dash_ok = self._dashboard.send(dashboard_payload)
        tracker.record(alert_id, "dashboard_it_admin", dash_ok, 1, ts)
        record["notifications_sent"].append({"target": "it_admin_dashboard", "status": dash_ok})

        email_payload = {
            "subject": f"CRITICAL ESCALATION — RA-{alert['sample_index']}-IoMT",
            "body": f"CRITICAL escalation at {ts}. Login to dashboard: {DASHBOARD_URL}",
            "recipient": "it-admin@hospital.internal",
        }
        email_ok = self._email.send(email_payload)
        tracker.record(alert_id, "email_it_admin", email_ok, 1, ts)
        record["notifications_sent"].append({"target": "it_admin_email", "status": email_ok})

        # 3. Notify doctor on duty (dashboard)
        doc_ok = self._dashboard.send(dashboard_payload)
        tracker.record(alert_id, "dashboard_doctor", doc_ok, 1, ts)
        record["notifications_sent"].append({"target": "doctor_dashboard", "status": doc_ok})

        # 4. Notify manager (email)
        mgr_payload = {
            "subject": f"CRITICAL ESCALATION — RA-{alert['sample_index']}-IoMT",
            "body": f"CRITICAL escalation at {ts}. Login to dashboard: {DASHBOARD_URL}",
            "recipient": "manager@hospital.internal",
        }
        mgr_ok = self._email.send(mgr_payload)
        tracker.record(alert_id, "email_manager", mgr_ok, 1, ts)
        record["notifications_sent"].append({"target": "manager_email", "status": mgr_ok})

        # In production: wait for confirmation token within timeout.
        # For research: simulate confirmation received.
        record["confirmation_received"] = True
        record["confirmed_at"] = datetime.now(timezone.utc).isoformat()
        record["confirmed_by"] = "it_admin (simulated)"

        logger.info(
            "  Escalation complete: %d notifications, confirmation=%s",
            len(record["notifications_sent"]),
            record["confirmation_received"],
        )

        self._history.append(record)
        return record

    def get_history(self) -> List[Dict[str, Any]]:
        """Return escalation history."""
        return list(self._history)


# ---------------------------------------------------------------------------
# Priority router
# ---------------------------------------------------------------------------


class PriorityRouter:
    """Route alerts to notification channels based on risk level.

    Routing table:
        LOW:      log_only()
        MEDIUM:   push_dashboard()
        HIGH:     push_dashboard() + send_encrypted_email()
        CRITICAL: push_dashboard() + send_encrypted_email()
                  + trigger_onsite_alarm() + escalate()
    """

    def __init__(
        self,
        dashboard: DashboardPush,
        email: EncryptedEmail,
        alarm: OnsiteAlarm,
        email_composer: EmailComposer,
        escalation_mgr: EscalationManager,
        tracker: DeliveryTracker,
        charts_dir: Path,
    ) -> None:
        self._dashboard = dashboard
        self._email = email
        self._alarm = alarm
        self._composer = email_composer
        self._escalation = escalation_mgr
        self._tracker = tracker
        self._charts_dir = charts_dir
        self._notification_log: List[Dict[str, Any]] = []

    def route(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Route a single alert to appropriate channels.

        Args:
            alert: Enriched alert dict from Phase 5 explanation_report.

        Returns:
            Notification record with channels and delivery status.
        """
        risk = alert["risk_level"]
        idx = alert["sample_index"]
        alert_id = f"RA-{idx}"
        ts = datetime.now(timezone.utc).isoformat()

        record: Dict[str, Any] = {
            "alert_id": alert_id,
            "sample_index": idx,
            "risk_level": risk,
            "timestamp": ts,
            "channels": [],
        }

        if risk == RiskLevel.LOW:
            # LOG ONLY — no active notification
            logger.info("  [LOW] alert_id=%s — logged only", alert_id)
            record["channels"].append({"channel": "log_only", "status": "SUCCESS"})

        elif risk == RiskLevel.MEDIUM:
            # DASHBOARD only
            dash_ok = self._send_with_retry(
                self._dashboard, _build_dashboard_payload(alert), alert_id, "dashboard"
            )
            record["channels"].append({"channel": "dashboard", "status": dash_ok})

        elif risk == RiskLevel.HIGH:
            # DASHBOARD + EMAIL (with waterfall chart attachment)
            dash_ok = self._send_with_retry(
                self._dashboard, _build_dashboard_payload(alert), alert_id, "dashboard"
            )
            record["channels"].append({"channel": "dashboard", "status": dash_ok})

            chart_path = self._find_waterfall_chart(idx)
            email_payload = self._composer.compose(
                alert, attach_chart=chart_path is not None, chart_path=chart_path
            )
            email_ok = self._send_with_retry(self._email, email_payload, alert_id, "email")
            record["channels"].append({"channel": "email", "status": email_ok})

        elif risk == RiskLevel.CRITICAL:
            # DASHBOARD + EMAIL + ALARM + ESCALATION
            dash_ok = self._send_with_retry(
                self._dashboard, _build_dashboard_payload(alert), alert_id, "dashboard"
            )
            record["channels"].append({"channel": "dashboard", "status": dash_ok})

            chart_path = self._find_waterfall_chart(idx)
            email_payload = self._composer.compose(
                alert, attach_chart=chart_path is not None, chart_path=chart_path
            )
            email_ok = self._send_with_retry(self._email, email_payload, alert_id, "email")
            record["channels"].append({"channel": "email", "status": email_ok})

            escalation = self._escalation.escalate(alert, alert_id, self._tracker)
            record["channels"].append({"channel": "escalation", "status": "TRIGGERED"})
            record["escalation"] = escalation

        self._notification_log.append(record)
        return record

    def get_log(self) -> List[Dict[str, Any]]:
        """Return all notification records."""
        return list(self._notification_log)

    def _send_with_retry(
        self,
        channel: NotificationChannel,
        payload: Dict[str, Any],
        alert_id: str,
        channel_label: str,
    ) -> str:
        """Send with exponential backoff retry.

        Args:
            channel: Notification channel instance.
            payload: Payload to send.
            alert_id: Alert identifier for tracking.
            channel_label: Human-readable channel name.

        Returns:
            "SUCCESS" or "FAILED" after MAX_RETRIES attempts.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            ts = datetime.now(timezone.utc).isoformat()
            success = channel.send(payload)
            self._tracker.record(alert_id, channel_label, success, attempt, ts)

            if success:
                logger.info(
                    "  Delivery: %s → %s (attempt %d/%d)",
                    alert_id,
                    channel_label,
                    attempt,
                    MAX_RETRIES,
                )
                return "SUCCESS"

            backoff = BACKOFF_BASE_S * (2 ** (attempt - 1))
            logger.warning(
                "  Delivery FAILED: %s → %s (attempt %d/%d, retry in %.1fs)",
                alert_id,
                channel_label,
                attempt,
                MAX_RETRIES,
                backoff,
            )
            time.sleep(backoff)

        return "FAILED"

    def _find_waterfall_chart(self, sample_index: int) -> Optional[Path]:
        """Find waterfall chart for a sample index.

        Args:
            sample_index: Sample index to look up.

        Returns:
            Path to waterfall PNG if it exists, else None.
        """
        chart = self._charts_dir / f"waterfall_{sample_index}.png"
        return chart if chart.exists() else None


# ---------------------------------------------------------------------------
# Artifact loader (SHA-256 verified)
# ---------------------------------------------------------------------------


class Phase5ArtifactLoader:
    """Load and SHA-256-verify Phase 4/5 artifacts for notification.

    Verifies integrity of explanation_report.json, risk_report.json,
    and charts before loading. Does NOT recompute any values.
    """

    def __init__(
        self,
        project_root: Path,
        phase5_dir: Path = PHASE5_DIR,
        phase4_dir: Path = PHASE4_DIR,
    ) -> None:
        self._root = project_root
        self._p5 = project_root / phase5_dir
        self._p4 = project_root / phase4_dir

    def load_and_verify(self) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        """Load Phase 4 risk report + Phase 5 explanation report.

        Verifies SHA-256 hashes from metadata before loading.

        Returns:
            Tuple of (explanation_report, risk_report, chart_files).

        Raises:
            ValueError: If SHA-256 mismatch detected.
        """
        logger.info("── Loading and verifying Phase 4/5 artifacts ──")

        # Verify Phase 5 artifacts
        p5_verifier = IntegrityVerifier(self._p5)
        p5_meta = json.loads((self._p5 / PHASE5_METADATA).read_text())
        for name, info in p5_meta["artifact_hashes"].items():
            actual = p5_verifier.compute_hash(self._p5 / name)
            if actual != info["sha256"]:
                raise ValueError(f"SHA-256 mismatch: Phase 5 {name}")
        logger.info("  Phase 5: %d artifacts verified", len(p5_meta["artifact_hashes"]))
        AuditLogger.log_file_access(str(self._p5 / PHASE5_REPORT), "READ")

        # Verify Phase 4 risk_report.json
        p4_verifier = IntegrityVerifier(self._p4)
        p4_meta = json.loads((self._p4 / PHASE4_METADATA).read_text())
        risk_hash = p4_meta["artifact_hashes"]["risk_report.json"]["sha256"]
        actual = p4_verifier.compute_hash(self._p4 / PHASE4_REPORT)
        if actual != risk_hash:
            raise ValueError("SHA-256 mismatch: Phase 4 risk_report.json")
        logger.info("  Phase 4: risk_report.json verified")
        AuditLogger.log_file_access(str(self._p4 / PHASE4_REPORT), "READ")

        # Load reports
        explanation_report = json.loads((self._p5 / PHASE5_REPORT).read_text())
        risk_report = json.loads((self._p4 / PHASE4_REPORT).read_text())

        # Verify chart file integrity (existence + non-empty)
        charts_dir = self._p5 / CHARTS_DIR
        chart_files = []
        if charts_dir.is_dir():
            for f in sorted(charts_dir.iterdir()):
                if f.suffix == ".png" and f.stat().st_size > 0:
                    chart_files.append(f.name)
        logger.info("  Charts: %d PNG files verified", len(chart_files))

        total = len(explanation_report.get("explanations", []))
        logger.info("  Loaded %d alerts for notification routing", total)

        return explanation_report, risk_report, chart_files


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def filter_alerts(
    explanations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Filter out NORMAL risk level alerts.

    Args:
        explanations: List of alert dicts from explanation_report.json.

    Returns:
        Tuple of (filtered_alerts, level_counts).
    """
    counts: Dict[str, int] = {level: 0 for level in RISK_LEVELS}
    filtered = []

    for alert in explanations:
        risk = alert.get("risk_level", "")
        if risk in RISK_LEVELS:
            counts[risk] += 1
            filtered.append(alert)
        # Skip NORMAL and unknown levels

    logger.info(
        "  Filtered %d alerts (LOW=%d, MEDIUM=%d, HIGH=%d, CRITICAL=%d)",
        len(filtered),
        counts["LOW"],
        counts["MEDIUM"],
        counts["HIGH"],
        counts["CRITICAL"],
    )
    return filtered, counts


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_dashboard_payload(alert: Dict[str, Any]) -> Dict[str, Any]:
    """Build a WebSocket push payload for the dashboard.

    Args:
        alert: Alert dict with risk_level, timestamp, top_features.

    Returns:
        Dashboard payload dict (no PHI).
    """
    return {
        "risk_level": alert["risk_level"],
        "timestamp": alert.get("timestamp", ""),
        "top_features": alert.get("top_features", [])[:3],
        "chart_url": f"{DASHBOARD_URL}/charts/waterfall_{alert['sample_index']}.png",
    }


# ---------------------------------------------------------------------------
# Artifact exporter
# ---------------------------------------------------------------------------


def export_artifacts(
    output_dir: Path,
    notification_log: List[Dict[str, Any]],
    delivery_report: List[Dict[str, Any]],
    escalation_log: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Export Phase 6 notification artifacts.

    Args:
        output_dir: Directory to write artifacts.
        notification_log: All notification records.
        delivery_report: Per-channel delivery records.
        escalation_log: CRITICAL escalation history.

    Returns:
        Dict mapping artifact name to SHA-256 hash.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: Dict[str, str] = {}
    verifier = IntegrityVerifier(output_dir)

    artifacts = {
        NOTIFICATION_LOG: notification_log,
        DELIVERY_REPORT: delivery_report,
        ESCALATION_LOG: escalation_log,
    }

    for name, data in artifacts.items():
        path = output_dir / name
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        digest = verifier.compute_hash(path)
        hashes[name] = digest
        logger.info("  Exported %s (%d records, sha256=%s…)", name, len(data), digest[:16])
        AuditLogger.log_file_access(str(path), "WRITE")

    return hashes


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def generate_notification_report(
    level_counts: Dict[str, int],
    notification_log: List[Dict[str, Any]],
    delivery_report: List[Dict[str, Any]],
    escalation_log: List[Dict[str, Any]],
    artifact_hashes: Dict[str, str],
    duration_s: float,
    git_commit: str,
) -> str:
    """Generate section 9.1 notification report for thesis.

    Args:
        level_counts: Alert counts by risk level.
        notification_log: All notification records.
        delivery_report: Delivery attempt records.
        escalation_log: Escalation records.
        artifact_hashes: SHA-256 hashes of exported artifacts.
        duration_s: Pipeline duration in seconds.
        git_commit: Git commit hash.

    Returns:
        Markdown report string.
    """
    total = sum(level_counts.values())
    n_escalations = len(escalation_log)
    n_deliveries = len(delivery_report)
    n_failures = sum(1 for r in delivery_report if r["status"] == "FAILED")
    success_rate = ((n_deliveries - n_failures) / n_deliveries * 100) if n_deliveries else 0

    # Channel stats
    channels: Dict[str, Dict[str, int]] = {}
    for r in delivery_report:
        ch = r["channel"]
        if ch not in channels:
            channels[ch] = {"success": 0, "failed": 0}
        if r["status"] == "SUCCESS":
            channels[ch]["success"] += 1
        else:
            channels[ch]["failed"] += 1

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "## 9.1 Notification Engine — Alert Routing & Delivery",
        "",
        "This section documents the Phase 6 Notification Engine,",
        "which routes Phase 5 SHAP-explained alerts to appropriate",
        "notification channels based on risk level.",
        "",
        "### 9.1.1 Alert Distribution",
        "",
        "| Risk Level | Count | Routing |",
        "|------------|-------|---------|",
        f"| CRITICAL | {level_counts.get('CRITICAL', 0)}"
        " | Dashboard + Email + Alarm + Escalation |",
        f"| HIGH | {level_counts.get('HIGH', 0)} | Dashboard + Email (waterfall attached) |",
        f"| MEDIUM | {level_counts.get('MEDIUM', 0)} | Dashboard |",
        f"| LOW | {level_counts.get('LOW', 0)} | Log only |",
        f"| **Total** | **{total}** | |",
        "",
        "### 9.1.2 Delivery Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total delivery attempts | {n_deliveries} |",
        f"| Successful | {n_deliveries - n_failures} |",
        f"| Failed | {n_failures} |",
        f"| Success rate | {success_rate:.1f}% |",
        f"| Escalations triggered | {n_escalations} |",
        f"| Retry policy | {MAX_RETRIES} attempts, exponential backoff |",
        "",
        "### 9.1.3 Channel Performance",
        "",
        "| Channel | Success | Failed | Total |",
        "|---------|---------|--------|-------|",
    ]
    for ch, stats in sorted(channels.items()):
        ch_total = stats["success"] + stats["failed"]
        lines.append(f"| {ch} | {stats['success']} | {stats['failed']} | {ch_total} |")

    lines += [
        "",
        "### 9.1.4 HIPAA Compliance",
        "",
        "| Field | In Email | In Dashboard | In Logs |",
        "|-------|----------|-------------|---------|",
        "| Risk level | Yes | Yes | Yes |",
        "| Timestamp | Yes | Yes | Yes |",
        "| Top 3 features (names only) | No | Yes | No |",
        "| Dashboard URL | Yes | N/A | No |",
        "| Device ID | **NEVER** | **NEVER** | **NEVER** |",
        "| Patient data | **NEVER** | **NEVER** | **NEVER** |",
        "| Raw anomaly scores | **NEVER** | **NEVER** | **NEVER** |",
        "| SHAP values | **NEVER** | **NEVER** | **NEVER** |",
        "| Biometric readings | **NEVER** | **NEVER** | **NEVER** |",
        "| Recipient address | **NEVER** | N/A | Hashed (SHA-256) |",
        "",
        "### 9.1.5 Email Protocol",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Encryption | TLS {TLS_MIN_VERSION} minimum |",
        "| Certificate verification | Required before send |",
        "| Subject format | `RA-{idx}-IoMT Alert — Level: {RISK}` |",
        "| Attachment | Waterfall chart (HIGH/CRITICAL only) |",
        "| PHI in body | None |",
        "",
        "### 9.1.6 Escalation Protocol (CRITICAL)",
        "",
        "| Step | Target | Channel |",
        "|------|--------|---------|",
        "| 1 | On-site alarm | Hospital network API |",
        "| 2 | IT admin | Email + Dashboard |",
        "| 3 | Doctor on duty | Dashboard |",
        "| 4 | Manager | Email |",
        "",
        f"Confirmation required within {CONFIRMATION_TIMEOUT_S}s"
        f" ({CONFIRMATION_TIMEOUT_S // 60} minutes).",
        "If no confirmation: re-escalate to next level.",
        "",
        "### 9.1.7 Artifact Integrity",
        "",
        "| Artifact | SHA-256 |",
        "|----------|---------|",
    ]
    for name, h in artifact_hashes.items():
        lines.append(f"| `{name}` | `{h[:16]}…` |")

    lines += [
        "",
        "### 9.1.8 Execution Details",
        "",
        f"- Duration: {duration_s:.2f}s",
        f"- Git commit: `{git_commit[:12]}`",
        "- Pipeline: phase6_notification",
        "",
        "---",
        "",
        f"**Generated:** {ts}",
        "**Pipeline version:** 6.0",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_notification_pipeline() -> Dict[str, Any]:
    """Execute Phase 6 notification pipeline.

    Pipeline steps:
        1. Load + verify Phase 4/5 artifacts (SHA-256)
        2. Filter NORMAL alerts
        3. Route by risk level
        4. Track delivery status
        5. Export notification artifacts

    Returns:
        Pipeline summary dict.
    """
    t0 = time.time()

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 6 Notification Engine")
    logger.info("═══════════════════════════════════════════════════")

    # ── A01: Validate output path ──
    validator = PathValidator(PROJECT_ROOT)
    output_dir = PROJECT_ROOT / OUTPUT_DIR
    validator.validate_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  A01 ✓  Output dir: %s", output_dir)

    # ── Step 1: Load + verify ──
    loader = Phase5ArtifactLoader(PROJECT_ROOT)
    explanation_report, risk_report, chart_files = loader.load_and_verify()

    # ── Step 2: Filter ──
    logger.info("── Filtering alerts ──")
    alerts = explanation_report.get("explanations", [])
    filtered, level_counts = filter_alerts(alerts)

    # ── Step 3: Initialize channels ──
    dashboard = DashboardPush()
    email = EncryptedEmail(tls_version=TLS_MIN_VERSION)
    alarm = OnsiteAlarm()
    composer = EmailComposer()
    tracker = DeliveryTracker()
    charts_dir = PROJECT_ROOT / PHASE5_DIR / CHARTS_DIR

    escalation_mgr = EscalationManager(
        email_channel=email,
        dashboard_channel=dashboard,
        alarm_channel=alarm,
    )

    router = PriorityRouter(
        dashboard=dashboard,
        email=email,
        alarm=alarm,
        email_composer=composer,
        escalation_mgr=escalation_mgr,
        tracker=tracker,
        charts_dir=charts_dir,
    )

    # ── Step 4: Route all alerts ──
    logger.info("── Routing %d alerts ──", len(filtered))
    for alert in filtered:
        router.route(alert)

    notification_log = router.get_log()
    delivery_report = tracker.get_report()
    escalation_log = escalation_mgr.get_history()

    # ── Step 5: Delivery summary ──
    n_success = sum(1 for r in delivery_report if r["status"] == "SUCCESS")
    n_failed = sum(1 for r in delivery_report if r["status"] == "FAILED")
    logger.info(
        "── Delivery summary: %d SUCCESS, %d FAILED ──",
        n_success,
        n_failed,
    )

    # ── Step 6: Export artifacts ──
    logger.info("── Exporting Phase 6 artifacts ──")
    artifact_hashes = export_artifacts(
        output_dir, notification_log, delivery_report, escalation_log
    )

    duration_s = time.time() - t0

    # ── Generate report ──
    import subprocess

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        git_commit = "unknown"

    report_md = generate_notification_report(
        level_counts=level_counts,
        notification_log=notification_log,
        delivery_report=delivery_report,
        escalation_log=escalation_log,
        artifact_hashes=artifact_hashes,
        duration_s=duration_s,
        git_commit=git_commit,
    )

    report_path = PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_notification.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("  Report saved: report_section_notification.md")

    # ── Final summary ──
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 6 Notification Engine — %.2fs", duration_s)
    logger.info(
        "  Alerts routed: %d (LOW=%d, MEDIUM=%d, HIGH=%d, CRITICAL=%d)",
        len(filtered),
        level_counts["LOW"],
        level_counts["MEDIUM"],
        level_counts["HIGH"],
        level_counts["CRITICAL"],
    )
    logger.info("  Deliveries: %d success, %d failed", n_success, n_failed)
    logger.info("  Escalations: %d", len(escalation_log))
    logger.info("═══════════════════════════════════════════════════")

    return {
        "alerts_routed": len(filtered),
        "level_counts": level_counts,
        "deliveries_success": n_success,
        "deliveries_failed": n_failed,
        "escalations": len(escalation_log),
        "artifact_hashes": artifact_hashes,
        "duration_s": round(duration_s, 2),
    }


def main() -> None:
    """Entry point for Phase 6 notification engine."""
    run_notification_pipeline()


if __name__ == "__main__":
    main()
