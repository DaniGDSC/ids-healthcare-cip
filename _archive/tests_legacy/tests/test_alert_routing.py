"""Tests for production alert routing — SIEM, email, WebSocket, escalation."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.production.alert_router import (
    AlertRouter,
    EmailSender,
    EscalationHandler,
    SIEMConnector,
    WebSocketPusher,
)


def _make_alert(severity: int = 4, risk: str = "HIGH", emit: bool = True) -> Dict[str, Any]:
    return {
        "sample_index": 0,
        "anomaly_score": 0.75,
        "risk_level": risk,
        "clinical_severity": severity,
        "clinical_severity_name": {1: "ROUTINE", 2: "ADVISORY", 3: "URGENT", 4: "EMERGENT", 5: "CRITICAL"}[severity],
        "device_action": "isolate_network" if severity >= 4 else "none",
        "patient_safety_flag": severity >= 4,
        "attention_flag": False,
        "response_time_minutes": {1: 0, 2: 480, 3: 60, 4: 15, 5: 5}[severity],
        "attack_category": "Spoofing",
        "cia_max_dimension": "I",
        "alert_emit": emit,
    }


@pytest.fixture
def router() -> AlertRouter:
    siem = SIEMConnector(dry_run=True)
    email = EmailSender(dry_run=True)
    ws = WebSocketPusher(dry_run=True)
    escalation = EscalationHandler(siem, email, ws, dry_run=True)
    return AlertRouter(siem, email, ws, escalation)


class TestSIEMConnector:

    def test_cef_format_contains_risk_level(self) -> None:
        siem = SIEMConnector(dry_run=True)
        alert = _make_alert(severity=4, risk="HIGH")
        cef = siem._format_cef(alert)
        assert "HIGH" in cef
        assert "CEF:0" in cef
        assert "RA-X-IoMT" in cef

    def test_dry_run_increments_count(self) -> None:
        siem = SIEMConnector(dry_run=True)
        siem.send(_make_alert())
        assert siem.sent_count == 1

    def test_cef_includes_cia_dimension(self) -> None:
        siem = SIEMConnector(dry_run=True)
        alert = _make_alert()
        alert["cia_max_dimension"] = "A"
        cef = siem._format_cef(alert)
        assert "cs1=A" in cef

    def test_cef_marks_novel_threats(self) -> None:
        siem = SIEMConnector(dry_run=True)
        alert = _make_alert()
        alert["attention_flag"] = True
        cef = siem._format_cef(alert)
        assert "novel" in cef


class TestEmailSender:

    def test_dry_run_succeeds(self) -> None:
        sender = EmailSender(dry_run=True)
        ok = sender.send("nurse@hospital.internal", "Test", "Body")
        assert ok is True
        assert sender.sent_count == 1

    def test_recipient_is_hashed_in_log(self, caplog) -> None:
        sender = EmailSender(dry_run=True)
        with caplog.at_level("INFO"):
            sender.send("secret@hospital.internal", "Test", "Body")
        # Raw email should NOT appear in logs
        assert "secret@hospital.internal" not in caplog.text


class TestWebSocketPusher:

    def test_dry_run_push(self) -> None:
        ws = WebSocketPusher(dry_run=True)
        ok = ws.push(_make_alert())
        assert ok is True
        assert ws.push_count == 1

    def test_callback_receives_alert(self) -> None:
        received: List[Dict] = []
        ws = WebSocketPusher(dry_run=True)
        ws.register_callback(lambda a: received.append(a))
        ws.push(_make_alert())
        assert len(received) == 1
        assert "risk_level" in received[0]

    def test_push_strips_sensitive_fields(self) -> None:
        received: List[Dict] = []
        ws = WebSocketPusher(dry_run=True)
        ws.register_callback(lambda a: received.append(a))
        alert = _make_alert()
        alert["anomaly_score"] = 0.999
        ws.push(alert)
        # anomaly_score should NOT be in pushed alert
        assert "anomaly_score" not in received[0]


class TestEscalationHandler:

    def test_escalation_notifies_all_levels(self) -> None:
        siem = SIEMConnector(dry_run=True)
        email = EmailSender(dry_run=True)
        ws = WebSocketPusher(dry_run=True)
        handler = EscalationHandler(siem, email, ws, dry_run=True)

        record = handler.escalate(_make_alert(severity=5, risk="CRITICAL"))
        assert len(record["levels_notified"]) == 3
        assert "it_security" in record["levels_notified"]
        assert "on_call_physician" in record["levels_notified"]
        assert "incident_commander" in record["levels_notified"]

    def test_dry_run_auto_confirms(self) -> None:
        siem = SIEMConnector(dry_run=True)
        email = EmailSender(dry_run=True)
        ws = WebSocketPusher(dry_run=True)
        handler = EscalationHandler(siem, email, ws, dry_run=True)

        record = handler.escalate(_make_alert(severity=5))
        assert record["confirmed"] is True

    def test_confirmation_mechanism(self) -> None:
        siem = SIEMConnector(dry_run=True)
        email = EmailSender(dry_run=True)
        ws = WebSocketPusher(dry_run=True)
        handler = EscalationHandler(siem, email, ws, dry_run=False)

        record = handler.escalate(_make_alert(severity=5))
        alert_id = record["alert_id"]

        # Not confirmed yet
        assert record["confirmed"] is False
        assert handler.pending_count == 1

        # Confirm
        ok = handler.confirm(alert_id, "charge_nurse")
        assert ok is True
        assert handler.pending_count == 0


class TestAlertRouter:

    def test_routine_not_routed(self, router: AlertRouter) -> None:
        result = router.route(_make_alert(severity=1, risk="NORMAL"))
        assert result["routed"] is False

    def test_advisory_goes_to_siem(self, router: AlertRouter) -> None:
        result = router.route(_make_alert(severity=2, risk="LOW"))
        assert result["routed"] is True
        assert "siem" in result["channels"]
        assert "websocket" not in result["channels"]

    def test_urgent_goes_to_siem_and_dashboard(self, router: AlertRouter) -> None:
        result = router.route(_make_alert(severity=3, risk="MEDIUM"))
        assert "siem" in result["channels"]
        assert "websocket" in result["channels"]
        assert "email" not in result["channels"]

    def test_emergent_adds_email(self, router: AlertRouter) -> None:
        result = router.route(_make_alert(severity=4, risk="HIGH"))
        assert "siem" in result["channels"]
        assert "websocket" in result["channels"]
        assert "email" in result["channels"]

    def test_critical_triggers_escalation(self, router: AlertRouter) -> None:
        result = router.route(_make_alert(severity=5, risk="CRITICAL"))
        assert "escalation" in result["channels"]

    def test_suppressed_alert_not_routed(self, router: AlertRouter) -> None:
        result = router.route(_make_alert(severity=5, emit=False))
        assert result["routed"] is False
        assert result["reason"] == "suppressed_by_fatigue"

    def test_status_tracks_counts(self, router: AlertRouter) -> None:
        router.route(_make_alert(severity=3))
        router.route(_make_alert(severity=4))
        status = router.get_status()
        assert status["routed_count"] == 2
        assert status["siem_sent"] >= 2
