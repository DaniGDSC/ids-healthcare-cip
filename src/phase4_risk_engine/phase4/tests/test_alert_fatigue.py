"""Unit tests for AlertFatigueManager — B1 blocker fix."""

from __future__ import annotations

import pytest

from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager


def _alert(severity: int = 4, risk: str = "HIGH") -> dict:
    return {"clinical_severity": severity, "risk_level": risk}


class TestRoutineSuppression:
    def test_severity_1_never_emitted(self) -> None:
        mgr = AlertFatigueManager()
        results = [_alert(1, "NORMAL") for _ in range(20)]
        mgr.process(results)
        assert all(not r["alert_emit"] for r in results)

    def test_severity_1_reason_is_routine(self) -> None:
        mgr = AlertFatigueManager()
        results = [_alert(1, "NORMAL")]
        mgr.process(results)
        assert results[0]["alert_reason"] == "routine_suppressed"


class TestEscalationBypass:
    def test_escalation_always_emits(self) -> None:
        mgr = AlertFatigueManager()
        results = [_alert(2, "LOW"), _alert(4, "HIGH")]
        mgr.process(results)
        assert results[1]["alert_emit"] is True
        assert results[1]["alert_reason"] == "escalation"

    def test_escalation_after_rate_limit(self) -> None:
        mgr = AlertFatigueManager(alerts_per_window=2, rate_window_size=10)
        results = [_alert(2, "LOW") for _ in range(10)]
        results.append(_alert(4, "HIGH"))
        mgr.process(results)
        assert results[-1]["alert_emit"] is True
        assert results[-1]["alert_reason"] == "escalation"

    def test_de_escalation_emits(self) -> None:
        mgr = AlertFatigueManager()
        results = [_alert(4, "HIGH"), _alert(2, "LOW")]
        mgr.process(results)
        assert results[1]["alert_emit"] is True
        assert results[1]["alert_reason"] == "de_escalation"


class TestAggregation:
    def test_first_at_level_emits(self) -> None:
        mgr = AlertFatigueManager(aggregation_window=5)
        results = [_alert(3, "MEDIUM")]
        mgr.process(results)
        assert results[0]["alert_emit"] is True
        assert results[0]["alert_reason"] == "first_at_level"

    def test_intermediate_suppressed(self) -> None:
        mgr = AlertFatigueManager(aggregation_window=5)
        results = [_alert(3, "MEDIUM") for _ in range(5)]
        mgr.process(results)
        assert results[0]["alert_emit"] is True
        assert results[1]["alert_emit"] is False
        assert results[1]["alert_reason"] == "aggregation_suppressed"

    def test_nth_window_emits_summary(self) -> None:
        mgr = AlertFatigueManager(aggregation_window=5)
        results = [_alert(3, "MEDIUM") for _ in range(5)]
        mgr.process(results)
        assert results[4]["alert_emit"] is True
        assert results[4]["alert_reason"] == "aggregation_summary"
        assert results[4]["alert_aggregated_count"] > 0

    def test_50pct_suppression_on_20_consecutive(self) -> None:
        mgr = AlertFatigueManager(aggregation_window=5)
        results = [_alert(4, "HIGH") for _ in range(20)]
        mgr.process(results)
        emitted = sum(1 for r in results if r["alert_emit"])
        assert emitted < 20
        assert emitted <= 10


class TestRateLimit:
    def test_rate_limit_caps_per_window(self) -> None:
        mgr = AlertFatigueManager(aggregation_window=100, alerts_per_window=3, rate_window_size=20)
        results = [_alert(4, "HIGH") for _ in range(20)]
        mgr.process(results)
        emitted = sum(1 for r in results if r["alert_emit"])
        assert emitted <= 4  # first_at_level + up to 3

    def test_rate_limited_reason(self) -> None:
        mgr = AlertFatigueManager(aggregation_window=100, alerts_per_window=1, rate_window_size=10)
        results = [_alert(3, "MEDIUM") for _ in range(5)]
        mgr.process(results)
        limited = [r for r in results if r.get("alert_reason") == "rate_limited"]
        assert len(limited) > 0


class TestSummaryStats:
    def test_summary_totals_correct(self) -> None:
        mgr = AlertFatigueManager()
        results = [_alert(1, "NORMAL")] * 5 + [_alert(4, "HIGH")] * 5
        mgr.process(results)
        s = mgr.get_summary()
        assert s["total_samples"] == 10
        assert s["alerts_emitted"] + s["alerts_suppressed"] == 10

    def test_suppression_rate(self) -> None:
        mgr = AlertFatigueManager()
        results = [_alert(1, "NORMAL")] * 10
        mgr.process(results)
        assert mgr.get_summary()["suppression_rate"] == 1.0
