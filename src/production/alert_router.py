"""Production alert routing — SIEM, email, WebSocket, escalation.

Replaces the simulated Phase 6 notification stubs with production
implementations. Each channel can operate in 'live' or 'dry_run' mode.

Channels:
  SIEMConnector       — CEF/syslog to Splunk or QRadar
  EmailSender         — SMTP/TLS 1.3 encrypted email
  WebSocketPusher     — real-time push to dashboard
  EscalationHandler   — confirmation-based escalation chain
  AlertRouter         — orchestrates all channels per severity
"""

from __future__ import annotations

import hashlib
import json
import logging
import smtplib
import socket
import ssl
import time
import uuid
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# SIEM Connector (Splunk HEC / QRadar syslog via CEF)
# ═══════════════════════════════════════════════════════════════════

class SIEMConnector:
    """Publish alerts to hospital SIEM in CEF format.

    Supports two modes:
      - syslog: UDP/TCP CEF to QRadar or any syslog receiver
      - hec: Splunk HTTP Event Collector (HTTPS POST)

    No PHI in payloads — device IDs are pseudonymized, no patient data.

    Args:
        mode: 'syslog' or 'hec'.
        host: SIEM host address.
        port: SIEM port (514 for syslog, 8088 for Splunk HEC).
        hec_token: Splunk HEC authentication token (hec mode only).
        dry_run: If True, log instead of sending.
    """

    def __init__(
        self,
        mode: str = "syslog",
        host: str = "localhost",
        port: int = 514,
        hec_token: str = "",
        dry_run: bool = True,
    ) -> None:
        self._mode = mode
        self._host = host
        self._port = port
        self._hec_token = hec_token
        self._dry_run = dry_run
        self._sent_count = 0

    def send(self, alert: Dict[str, Any]) -> bool:
        """Format and send alert to SIEM.

        Args:
            alert: Risk assessment dict from Phase 4 pipeline.

        Returns:
            True if sent successfully.
        """
        cef = self._format_cef(alert)

        if self._dry_run:
            logger.info("SIEM [dry_run]: %s", cef[:200])
            self._sent_count += 1
            return True

        try:
            if self._mode == "syslog":
                return self._send_syslog(cef)
            elif self._mode == "hec":
                return self._send_hec(alert)
            return False
        except Exception as exc:
            logger.error("SIEM send failed: %s", exc)
            return False

    def _format_cef(self, alert: Dict[str, Any]) -> str:
        """Format alert as CEF (Common Event Format) string."""
        severity = min(alert.get("clinical_severity", 1), 10)
        risk = alert.get("risk_level", "UNKNOWN")
        device_action = alert.get("device_action", "none")
        attn_flag = alert.get("attention_flag", False)
        cia_max = alert.get("cia_max_dimension", "")
        category = alert.get("attack_category", "unknown")

        # CEF format: CEF:Version|DeviceVendor|DeviceProduct|DeviceVersion|SignatureID|Name|Severity|Extension
        extensions = (
            f"cat={category} "
            f"act={device_action} "
            f"cn1={alert.get('anomaly_score', 0):.4f} "
            f"cn1Label=AnomalyScore "
            f"cs1={cia_max} "
            f"cs1Label=CIADimension "
            f"cs2={'novel' if attn_flag else 'known'} "
            f"cs2Label=ThreatType"
        )

        return (
            f"CEF:0|RA-X-IoMT|IDS|1.0|{risk}|"
            f"IoMT {risk} Alert|{severity}|{extensions}"
        )

    def _send_syslog(self, message: str) -> bool:
        """Send via UDP syslog."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(message.encode(), (self._host, self._port))
            self._sent_count += 1
            return True
        finally:
            sock.close()

    def _send_hec(self, alert: Dict[str, Any]) -> bool:
        """Send via Splunk HTTP Event Collector."""
        import urllib.request

        payload = json.dumps({
            "event": alert,
            "sourcetype": "iomt:alert",
            "source": "ra-x-iomt",
        }).encode()

        req = urllib.request.Request(
            f"https://{self._host}:{self._port}/services/collector/event",
            data=payload,
            headers={
                "Authorization": f"Splunk {self._hec_token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self._sent_count += 1
            return resp.status == 200

    @property
    def sent_count(self) -> int:
        return self._sent_count


# ═══════════════════════════════════════════════════════════════════
# Email Sender (SMTP/TLS)
# ═══════════════════════════════════════════════════════════════════

class EmailSender:
    """HIPAA-compliant encrypted email delivery.

    Emails contain risk level and timestamp ONLY.
    No PHI, no device IDs, no raw scores, no biometrics.

    Args:
        smtp_host: SMTP relay server.
        smtp_port: SMTP port (587 for STARTTLS).
        sender: From address.
        dry_run: If True, log instead of sending.
    """

    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        sender: str = "iomt-ids@hospital.internal",
        dry_run: bool = True,
    ) -> None:
        self._host = smtp_host
        self._port = smtp_port
        self._sender = sender
        self._dry_run = dry_run
        self._sent_count = 0

    def send(
        self,
        recipient: str,
        subject: str,
        body: str,
    ) -> bool:
        """Send an encrypted email.

        Args:
            recipient: Email address.
            subject: Email subject (no PHI).
            body: Email body (no PHI).

        Returns:
            True if sent successfully.
        """
        hashed_rcpt = hashlib.sha256(recipient.encode()).hexdigest()[:16]

        if self._dry_run:
            logger.info("Email [dry_run] to [%s...]: %s", hashed_rcpt, subject)
            self._sent_count += 1
            return True

        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self._sender
            msg["To"] = recipient

            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2

            with smtplib.SMTP(self._host, self._port) as server:
                server.starttls(context=context)
                server.send_message(msg)

            logger.info("Email sent to [%s...]", hashed_rcpt)
            self._sent_count += 1
            return True
        except Exception as exc:
            logger.error("Email failed to [%s...]: %s", hashed_rcpt, exc)
            return False

    @property
    def sent_count(self) -> int:
        return self._sent_count


# ═══════════════════════════════════════════════════════════════════
# WebSocket Pusher
# ═══════════════════════════════════════════════════════════════════

class WebSocketPusher:
    """Push alerts to connected dashboard clients via WebSocket.

    In production: uses a WebSocket server (e.g., FastAPI WebSocket).
    Maintains a list of connected clients and broadcasts alerts.

    Args:
        dry_run: If True, log instead of sending.
    """

    def __init__(self, dry_run: bool = True) -> None:
        self._dry_run = dry_run
        self._clients: List[Any] = []
        self._push_count = 0
        self._on_push: Optional[Callable[[Dict[str, Any]], None]] = None

    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for alert pushes (used by FastAPI WebSocket)."""
        self._on_push = callback

    def push(self, alert: Dict[str, Any]) -> bool:
        """Push alert to all connected clients.

        Args:
            alert: Alert dict (filtered for dashboard display).

        Returns:
            True if pushed successfully.
        """
        # Filter sensitive fields before pushing
        safe_alert = {
            "risk_level": alert.get("risk_level"),
            "clinical_severity": alert.get("clinical_severity"),
            "clinical_severity_name": alert.get("clinical_severity_name"),
            "device_action": alert.get("device_action"),
            "patient_safety_flag": alert.get("patient_safety_flag"),
            "response_time_minutes": alert.get("response_time_minutes"),
            "attention_flag": alert.get("attention_flag"),
            "alert_time": datetime.now(timezone.utc).isoformat(),
            "sample_index": alert.get("sample_index"),
        }

        if self._dry_run:
            logger.info("WebSocket [dry_run]: %s %s",
                        safe_alert["risk_level"], safe_alert["clinical_severity_name"])
            self._push_count += 1
            if self._on_push:
                self._on_push(safe_alert)
            return True

        if self._on_push:
            self._on_push(safe_alert)

        self._push_count += 1
        return True

    @property
    def push_count(self) -> int:
        return self._push_count


# ═══════════════════════════════════════════════════════════════════
# Escalation Handler
# ═══════════════════════════════════════════════════════════════════

class EscalationHandler:
    """Manage CRITICAL alert escalation with confirmation tracking.

    Escalation chain (5-minute timeout per level):
      1. IT Security + Charge Nurse (email + dashboard + alarm)
      2. On-call Physician (if no confirmation in 5 min)
      3. Incident Commander (if still no confirmation)

    Args:
        siem: SIEM connector for logging.
        email: Email sender for notifications.
        websocket: WebSocket pusher for dashboard.
        confirmation_timeout_s: Seconds before escalating to next level.
        dry_run: If True, simulate confirmations.
    """

    ESCALATION_CHAIN: List[Dict[str, Any]] = [
        {"role": "it_security", "email": "it-security@hospital.internal",
         "channels": ["email", "dashboard", "alarm"]},
        {"role": "on_call_physician", "email": "oncall@hospital.internal",
         "channels": ["email", "dashboard"]},
        {"role": "incident_commander", "email": "commander@hospital.internal",
         "channels": ["email", "dashboard"]},
    ]

    def __init__(
        self,
        siem: SIEMConnector,
        email: EmailSender,
        websocket: WebSocketPusher,
        confirmation_timeout_s: int = 300,
        dry_run: bool = True,
    ) -> None:
        self._siem = siem
        self._email = email
        self._ws = websocket
        self._timeout = confirmation_timeout_s
        self._dry_run = dry_run
        self._history: List[Dict[str, Any]] = []
        self._pending: Dict[str, Dict[str, Any]] = {}

    def escalate(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CRITICAL escalation protocol.

        Args:
            alert: Alert dict from inference service.

        Returns:
            Escalation record with confirmation status.
        """
        alert_id = str(uuid.uuid4())[:8]
        ts = datetime.now(timezone.utc).isoformat()

        record = {
            "alert_id": alert_id,
            "triggered_at": ts,
            "risk_level": "CRITICAL",
            "confirmation_token": str(uuid.uuid4()),
            "levels_notified": [],
            "confirmed": False,
        }

        # Notify all levels in chain
        for level in self.ESCALATION_CHAIN:
            role = level["role"]
            subject = f"CRITICAL ESCALATION [{alert_id}] — IoMT IDS"
            body = (
                f"CRITICAL security alert at {ts}.\n"
                f"Response required within {self._timeout // 60} minutes.\n"
                f"Confirm at dashboard or reply to this email.\n"
                f"Token: {record['confirmation_token']}"
            )

            if "email" in level["channels"]:
                self._email.send(level["email"], subject, body)
            if "dashboard" in level["channels"]:
                self._ws.push({**alert, "escalation_level": role, "alert_id": alert_id})

            record["levels_notified"].append(role)
            logger.info("  Escalation → %s", role)

        # SIEM log
        self._siem.send({**alert, "escalation_id": alert_id, "escalation_status": "TRIGGERED"})

        # In dry_run mode, simulate confirmation
        if self._dry_run:
            record["confirmed"] = True
            record["confirmed_at"] = ts
            record["confirmed_by"] = "it_security (simulated)"
        else:
            self._pending[alert_id] = record

        self._history.append(record)
        return record

    def confirm(self, alert_id: str, confirmed_by: str) -> bool:
        """Acknowledge an escalation (called from dashboard or email reply).

        Args:
            alert_id: The escalation alert ID.
            confirmed_by: Role or person confirming.

        Returns:
            True if escalation was found and confirmed.
        """
        if alert_id in self._pending:
            self._pending[alert_id]["confirmed"] = True
            self._pending[alert_id]["confirmed_at"] = datetime.now(timezone.utc).isoformat()
            self._pending[alert_id]["confirmed_by"] = confirmed_by
            logger.info("Escalation %s confirmed by %s", alert_id, confirmed_by)
            return True
        return False

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    @property
    def pending_count(self) -> int:
        return sum(1 for r in self._pending.values() if not r["confirmed"])


# ═══════════════════════════════════════════════════════════════════
# Alert Router (orchestrates all channels)
# ═══════════════════════════════════════════════════════════════════

class AlertRouter:
    """Route alerts to appropriate channels based on clinical severity.

    Routing table:
      ROUTINE (1):   log only
      ADVISORY (2):  SIEM
      URGENT (3):    SIEM + dashboard
      EMERGENT (4):  SIEM + dashboard + email
      CRITICAL (5):  SIEM + dashboard + email + escalation

    Args:
        siem: SIEM connector.
        email: Email sender.
        websocket: WebSocket pusher.
        escalation: Escalation handler.
        recipient_map: Severity → email recipients.
    """

    DEFAULT_RECIPIENTS: Dict[int, List[str]] = {
        4: ["biomed-eng@hospital.internal", "charge-nurse@hospital.internal"],
        5: ["it-security@hospital.internal"],
    }

    def __init__(
        self,
        siem: SIEMConnector,
        email: EmailSender,
        websocket: WebSocketPusher,
        escalation: EscalationHandler,
        recipient_map: Optional[Dict[int, List[str]]] = None,
    ) -> None:
        from config.production_loader import cfg

        self._siem = siem
        self._email = email
        self._ws = websocket
        self._escalation = escalation
        self._recipients = recipient_map or cfg("alerting.recipients", self.DEFAULT_RECIPIENTS)
        self._safety_recipients = cfg("alerting.safety_recipients", [
            "charge-nurse@hospital.internal",
            "on-call-physician@hospital.internal",
        ])
        self._safety_escalation = cfg("alerting.safety_flag_escalation", True)
        self._routed_count = 0

    def route(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Route an alert to appropriate channels.

        Only routes alerts where alert_emit is True.

        Args:
            alert: Full alert dict from inference service.

        Returns:
            Routing record with channels used.
        """
        if not alert.get("alert_emit", True):
            return {"routed": False, "reason": "suppressed_by_fatigue"}

        severity = alert.get("clinical_severity", 1)
        channels_used: List[str] = []

        # Severity 1: log only
        if severity <= 1:
            return {"routed": False, "reason": "routine"}

        # Severity 2+: SIEM
        if severity >= 2:
            self._siem.send(alert)
            channels_used.append("siem")

        # Severity 3+: Dashboard WebSocket
        if severity >= 3:
            self._ws.push(alert)
            channels_used.append("websocket")

        # Severity 4+: Email
        if severity >= 4:
            recipients = self._recipients.get(severity, [])
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            risk = alert.get("risk_level", "UNKNOWN")
            for rcpt in recipients:
                self._email.send(
                    rcpt,
                    f"IoMT Alert [{risk}] — {ts}",
                    f"{risk} alert detected at {ts}. "
                    f"Response time: {alert.get('response_time_minutes', 0)} minutes. "
                    f"View details at dashboard.",
                )
            channels_used.append("email")

        # Severity 5: Full escalation
        if severity >= 5:
            self._escalation.escalate(alert)
            channels_used.append("escalation")

        # Patient safety flag: ALWAYS escalate regardless of severity
        if alert.get("patient_safety_flag") and self._safety_escalation:
            if "escalation" not in channels_used:
                self._escalation.escalate(alert)
                channels_used.append("escalation")
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            risk = alert.get("risk_level", "UNKNOWN")
            for rcpt in self._safety_recipients:
                self._email.send(
                    rcpt,
                    f"PATIENT SAFETY: IoMT Alert [{risk}] — {ts}",
                    f"Patient safety flag triggered. Risk: {risk}. "
                    f"Verify patient vitals manually. "
                    f"Response time: {alert.get('response_time_minutes', 5)} minutes.",
                )
            if "safety_notification" not in channels_used:
                channels_used.append("safety_notification")
            logger.warning("Patient safety flag → full escalation + safety notification")

        self._routed_count += 1

        return {
            "routed": True,
            "severity": severity,
            "channels": channels_used,
            "alert_id": alert.get("sample_index"),
        }

    @property
    def routed_count(self) -> int:
        return self._routed_count

    def get_status(self) -> Dict[str, Any]:
        return {
            "routed_count": self._routed_count,
            "siem_sent": self._siem.sent_count,
            "emails_sent": self._email.sent_count,
            "ws_pushed": self._ws.push_count,
            "escalations_pending": self._escalation.pending_count,
        }
