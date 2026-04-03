"""Tests for SQLite production database."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.production.database import Database


@pytest.fixture
def db(tmp_path):
    """Fresh database for each test."""
    return Database(tmp_path / "test.db")


class TestInserts:
    def test_insert_prediction(self, db):
        row_id = db.insert_prediction({
            "sample_index": 42,
            "anomaly_score": 0.95,
            "risk_level": "HIGH",
            "clinical_severity": 4,
            "attention_flag": True,
            "patient_safety_flag": False,
            "device_id": "infusion_pump",
            "device_action": "restrict_network",
            "latency_ms": 15.2,
            "ground_truth": 1,
        })
        assert row_id == 1

    def test_insert_alert(self, db):
        row_id = db.insert_alert({
            "sample_index": 42,
            "risk_level": "CRITICAL",
            "clinical_severity": 5,
            "patient_safety_flag": True,
            "device_id": "ecg_monitor",
            "cia_scores": {"C": 0.5, "I": 1.0, "A": 1.0},
            "alert_emit": True,
            "alert_reason": "escalation",
        })
        assert row_id == 1

    def test_insert_access(self, db):
        db.insert_access("analyst", "IT Security Analyst", "view_Live Monitor")
        logs = db.query_access_log(limit=10)
        assert len(logs) == 1
        assert logs[0]["username"] == "analyst"

    def test_insert_feedback(self, db):
        db.insert_feedback(alert_id=1, analyst="analyst", ground_truth=0, confidence=0.9)
        # No crash = success (no query method for feedback yet)

    def test_insert_calibration(self, db):
        db.insert_calibration({"mode": "benign_only", "n_calibration": 200})


class TestQueries:
    def test_query_alerts_empty(self, db):
        assert db.query_alerts() == []

    def test_query_alerts_with_data(self, db):
        db.insert_alert({"sample_index": 1, "risk_level": "HIGH", "clinical_severity": 4})
        db.insert_alert({"sample_index": 2, "risk_level": "CRITICAL", "clinical_severity": 5})
        alerts = db.query_alerts()
        assert len(alerts) == 2

    def test_query_alerts_filter_risk(self, db):
        db.insert_alert({"sample_index": 1, "risk_level": "HIGH", "clinical_severity": 4})
        db.insert_alert({"sample_index": 2, "risk_level": "CRITICAL", "clinical_severity": 5})
        critical = db.query_alerts(risk_level="CRITICAL")
        assert len(critical) == 1
        assert critical[0]["risk_level"] == "CRITICAL"

    def test_query_predictions(self, db):
        for i in range(5):
            db.insert_prediction({"sample_index": i, "risk_level": "NORMAL", "anomaly_score": 0.5})
        preds = db.query_predictions(limit=3)
        assert len(preds) == 3

    def test_risk_distribution(self, db):
        for _ in range(3):
            db.insert_prediction({"risk_level": "NORMAL"})
        for _ in range(2):
            db.insert_prediction({"risk_level": "HIGH"})
        dist = db.get_risk_distribution()
        assert dist["NORMAL"] == 3
        assert dist["HIGH"] == 2

    def test_alert_count(self, db):
        db.insert_alert({"sample_index": 1, "risk_level": "HIGH", "clinical_severity": 4})
        db.insert_alert({"sample_index": 2, "risk_level": "HIGH", "clinical_severity": 4})
        assert db.get_alert_count() == 2

    def test_prediction_count(self, db):
        db.insert_prediction({"risk_level": "NORMAL"})
        db.insert_prediction({"risk_level": "LOW"})
        assert db.get_prediction_count() == 2


class TestAcknowledge:
    def test_acknowledge_alert(self, db):
        db.insert_alert({"sample_index": 1, "risk_level": "HIGH", "clinical_severity": 4})
        result = db.acknowledge_alert(1, "analyst")
        assert result is True
        alerts = db.query_alerts()
        assert alerts[0]["acknowledged"] == 1
        assert alerts[0]["acknowledged_by"] == "analyst"

    def test_acknowledge_nonexistent(self, db):
        result = db.acknowledge_alert(999, "analyst")
        assert result is False


class TestPersistence:
    def test_data_survives_reconnect(self, tmp_path):
        db_path = tmp_path / "persist.db"
        db1 = Database(db_path)
        db1.insert_prediction({"sample_index": 1, "risk_level": "HIGH", "anomaly_score": 0.9})
        db1.close()

        db2 = Database(db_path)
        preds = db2.query_predictions()
        assert len(preds) == 1
        assert preds[0]["risk_level"] == "HIGH"
        db2.close()
