"""Tests for AlertDispatcher."""

from __future__ import annotations

from src.phase7_monitoring_engine.phase7.alert_dispatcher import AlertDispatcher
from src.phase7_monitoring_engine.phase7.state_machine import MonitoringAlert


def _make_alert(severity="WARNING", category="HEALTH", resolved=False):
    return MonitoringAlert(
        alert_id="test-1",
        category=category,
        severity=severity,
        engine_id="eng1",
        timestamp="ts",
        message="test",
        resolved=resolved,
    )


class TestAlertDispatcher:
    """Test alert dispatch and tracking."""

    def test_dispatch_and_count(self):
        d = AlertDispatcher()
        d.dispatch(_make_alert(severity="CRITICAL"))
        d.dispatch(_make_alert(severity="WARNING"))
        assert len(d.all_alerts) == 2

    def test_get_active_excludes_resolved(self):
        d = AlertDispatcher()
        d.dispatch(_make_alert(resolved=False))
        d.dispatch(_make_alert(resolved=True))
        assert len(d.get_active()) == 1

    def test_count_by_severity(self):
        d = AlertDispatcher()
        d.dispatch(_make_alert(severity="CRITICAL"))
        d.dispatch(_make_alert(severity="CRITICAL"))
        d.dispatch(_make_alert(severity="WARNING"))
        counts = d.count_by_severity()
        assert counts["CRITICAL"] == 2
        assert counts["WARNING"] == 1
        assert counts["INFO"] == 0

    def test_count_by_category(self):
        d = AlertDispatcher()
        d.dispatch(_make_alert(category="HEALTH"))
        d.dispatch(_make_alert(category="SECURITY"))
        counts = d.count_by_category()
        assert counts["HEALTH"] == 1
        assert counts["SECURITY"] == 1
        assert counts["PERFORMANCE"] == 0

    def test_empty_dispatcher(self):
        d = AlertDispatcher()
        assert d.all_alerts == []
        assert d.get_active() == []
        assert d.count_by_severity()["CRITICAL"] == 0
