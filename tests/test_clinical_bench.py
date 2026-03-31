"""Clinical Test Bench — 8 scenarios validating clinical safety logic.

Tests the Phase 4 risk pipeline (scorer → CIA → clinical → fatigue)
against clinical scenarios without requiring the TF model. Uses
pre-computed anomaly scores to isolate clinical decision logic.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from dashboard.streaming.window_buffer import WindowBuffer, SystemState
from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager
from src.phase4_risk_engine.phase4.cia_risk_modifier import CIARiskModifier
from src.phase4_risk_engine.phase4.cia_threat_mapper import CIAThreatMapper
from src.phase4_risk_engine.phase4.clinical_impact import ClinicalImpactAssessor
from src.phase4_risk_engine.phase4.cross_modal import CrossModalFusionDetector
from src.phase4_risk_engine.phase4.device_registry import DeviceRegistry
from src.phase4_risk_engine.phase4.risk_level import RiskLevel
from src.phase4_risk_engine.phase4.risk_scorer import RiskScorer

BIO_COLS = ["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"]
MODEL_FEATURES = [
    "SrcBytes", "DstBytes", "SrcLoad", "DstLoad",
    "SIntPkt", "DIntPkt", "SIntPktAct",
    "sMaxPktSz", "dMaxPktSz", "sMinPktSz",
    "Dur", "TotBytes", "Load", "pSrcLoss", "pDstLoss", "Packet_num",
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST",
]
N_FEATURES = len(MODEL_FEATURES)
MAD = 0.025


def _make_features(
    network_anomaly: float = 0.0,
    bio_anomaly: float = 0.0,
) -> np.ndarray:
    """Create a 24-dim feature vector with controlled anomaly levels.

    Args:
        network_anomaly: Z-score for network features (0 = normal).
        bio_anomaly: Z-score for biometric features (0 = normal).
    """
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    vec[:16] = network_anomaly  # Network features
    vec[16:] = bio_anomaly      # Biometric features
    return vec


def _run_pipeline(
    distance: float,
    features: np.ndarray,
    device_id: str = "generic_iomt_sensor",
    attack_category: str = "unknown",
    attention_flag: bool = False,
) -> Dict[str, Any]:
    """Run a single flow through the full Phase 4 pipeline."""
    scorer = RiskScorer(cross_modal=CrossModalFusionDetector(biometric_columns=BIO_COLS))
    cia = CIARiskModifier(threat_mapper=CIAThreatMapper(), device_registry=DeviceRegistry())
    clinical = ClinicalImpactAssessor(biometric_columns=BIO_COLS)

    risk = scorer.classify_single(distance, MAD, features, MODEL_FEATURES)

    if attention_flag and risk in (RiskLevel.NORMAL, RiskLevel.LOW):
        risk = RiskLevel.MEDIUM

    cia_result = cia.modify(risk, attack_category, device_id, attention_flag)
    clin_result = clinical.assess(
        cia_result.adjusted_risk_level, device_id, features, MODEL_FEATURES, attention_flag,
    )

    return {
        "base_risk": risk.value,
        "adjusted_risk": cia_result.adjusted_risk_level.value,
        "clinical_severity": clin_result.clinical_severity.value,
        "patient_safety_flag": clin_result.patient_safety_flag,
        "device_action": clin_result.protocol.device_action,
        "response_time_minutes": clin_result.protocol.response_time_minutes,
        "biometric_status": clin_result.biometric_status,
        "rationale": clin_result.rationale,
    }


# ═══════════════════════════════════════════════════════════════════
# Scenario A: Normal Operation (FPR Characterization)
# ═══════════════════════════════════════════════════════════════════

class TestScenarioA_NormalOperation:
    """500 benign flows — verify no false alarms."""

    def test_no_critical_alerts(self):
        results = [
            _run_pipeline(distance=-0.05, features=_make_features())
            for _ in range(100)
        ]
        critical = [r for r in results if r["adjusted_risk"] == "CRITICAL"]
        assert len(critical) == 0

    def test_no_safety_flags(self):
        results = [
            _run_pipeline(distance=-0.05, features=_make_features())
            for _ in range(100)
        ]
        flags = [r for r in results if r["patient_safety_flag"]]
        assert len(flags) == 0

    def test_all_normal_or_low(self):
        results = [
            _run_pipeline(distance=-0.02, features=_make_features())
            for _ in range(100)
        ]
        abnormal = [r for r in results if r["adjusted_risk"] not in ("NORMAL", "LOW")]
        assert len(abnormal) == 0


# ═══════════════════════════════════════════════════════════════════
# Scenario B: Gradual Attack Escalation
# ═══════════════════════════════════════════════════════════════════

class TestScenarioB_GradualEscalation:
    """Distance increases over time — verify detection triggers."""

    def test_escalation_detected(self):
        distances = (
            [-0.05] * 50         # benign
            + [0.005] * 30       # LOW
            + [0.015] * 30       # MEDIUM
            + [0.030] * 30       # HIGH
        )
        results = [
            _run_pipeline(distance=d, features=_make_features(network_anomaly=max(d * 10, 0)))
            for d in distances
        ]
        risk_levels = [r["adjusted_risk"] for r in results]
        assert "MEDIUM" in risk_levels or "HIGH" in risk_levels

    def test_severity_increases(self):
        low = _run_pipeline(distance=0.005, features=_make_features())
        high = _run_pipeline(distance=0.030, features=_make_features())
        assert high["clinical_severity"] > low["clinical_severity"]


# ═══════════════════════════════════════════════════════════════════
# Scenario C: Device-Specific Response
# ═══════════════════════════════════════════════════════════════════

class TestScenarioC_DeviceSpecific:
    """Same anomaly, different devices — verify device-appropriate response."""

    def test_pump_gets_safety_flag(self):
        result = _run_pipeline(
            distance=0.030,  # HIGH risk
            features=_make_features(network_anomaly=2.0, bio_anomaly=2.0),
            device_id="infusion_pump",
        )
        assert result["patient_safety_flag"] is True

    def test_sensor_no_safety_flag(self):
        result = _run_pipeline(
            distance=0.030,  # HIGH risk
            features=_make_features(network_anomaly=2.0, bio_anomaly=0.5),
            device_id="temperature_sensor",
        )
        assert result["patient_safety_flag"] is False

    def test_pump_higher_severity_than_sensor(self):
        pump = _run_pipeline(
            distance=0.030,
            features=_make_features(network_anomaly=2.0, bio_anomaly=2.0),
            device_id="infusion_pump",
        )
        sensor = _run_pipeline(
            distance=0.030,
            features=_make_features(network_anomaly=2.0, bio_anomaly=0.5),
            device_id="temperature_sensor",
        )
        assert pump["clinical_severity"] >= sensor["clinical_severity"]


# ═══════════════════════════════════════════════════════════════════
# Scenario D: Circuit Breaker & Recovery
# ═══════════════════════════════════════════════════════════════════

class TestScenarioD_CircuitBreaker:
    """Verify DEGRADED state and recovery."""

    def test_degraded_on_inference_failure(self):
        buf = WindowBuffer(window_size=2, calibration_threshold=5)
        # Fill past calibration
        for _ in range(10):
            buf.append(np.zeros(N_FEATURES))
            buf.record_prediction({"risk_level": "NORMAL", "ground_truth": 0})

        assert buf.state == SystemState.OPERATIONAL

        # Inference failure
        buf.record_prediction({"risk_level": "LOW", "inference_failed": True})
        assert buf.state == SystemState.DEGRADED

    def test_recovery_from_degraded(self):
        buf = WindowBuffer(window_size=2, calibration_threshold=5)
        for _ in range(10):
            buf.append(np.zeros(N_FEATURES))
            buf.record_prediction({"risk_level": "NORMAL", "ground_truth": 0})

        buf.record_prediction({"risk_level": "LOW", "inference_failed": True})
        assert buf.state == SystemState.DEGRADED

        # Successful prediction → recovery
        buf.record_prediction({"risk_level": "NORMAL", "ground_truth": 0})
        assert buf.state == SystemState.OPERATIONAL

    def test_alert_state_recovery(self):
        buf = WindowBuffer(window_size=2, calibration_threshold=5)
        for _ in range(10):
            buf.append(np.zeros(N_FEATURES))
            buf.record_prediction({"risk_level": "NORMAL", "ground_truth": 0})

        # Trigger ALERT
        buf.record_prediction({"risk_level": "CRITICAL", "ground_truth": 1})
        assert buf.state == SystemState.ALERT

        # 50 non-CRITICAL → recovery
        for _ in range(50):
            buf.record_prediction({"risk_level": "NORMAL", "ground_truth": 0})
        assert buf.state == SystemState.OPERATIONAL


# ═══════════════════════════════════════════════════════════════════
# Scenario E: Alert Fatigue Under Load
# ═══════════════════════════════════════════════════════════════════

class TestScenarioE_AlertFatigue:
    """500 consecutive HIGH alerts — verify suppression."""

    def test_suppression_rate(self):
        fatigue = AlertFatigueManager()
        results = []
        for i in range(500):
            r = {"clinical_severity": 4, "risk_level": "HIGH", "sample_index": i}
            fatigue.process([r], device_id="pump_01")
            results.append(r)

        emitted = sum(1 for r in results if r.get("alert_emit", True))
        suppressed = sum(1 for r in results if not r.get("alert_emit", True))
        rate = suppressed / max(len(results), 1)
        assert 0.80 <= rate <= 0.98, f"Suppression rate {rate:.1%} outside 80-98%"

    def test_escalation_always_emits(self):
        fatigue = AlertFatigueManager()
        # 10 HIGH alerts
        for i in range(10):
            r = {"clinical_severity": 4, "risk_level": "HIGH", "sample_index": i}
            fatigue.process([r], device_id="pump_01")

        # Escalation to CRITICAL
        r = {"clinical_severity": 5, "risk_level": "CRITICAL", "sample_index": 10}
        fatigue.process([r], device_id="pump_01")
        assert r.get("alert_emit", False) is True


# ═══════════════════════════════════════════════════════════════════
# Scenario F: Cross-Modal Fusion
# ═══════════════════════════════════════════════════════════════════

class TestScenarioF_CrossModal:
    """Verify CRITICAL requires both biometric AND network anomaly."""

    def test_bio_only_not_critical(self):
        result = _run_pipeline(
            distance=0.060,  # Above HIGH threshold
            features=_make_features(network_anomaly=0.5, bio_anomaly=3.0),
        )
        # Network below sigma threshold (2.0), bio above → NOT cross-modal
        assert result["adjusted_risk"] != "CRITICAL"

    def test_network_only_not_critical(self):
        result = _run_pipeline(
            distance=0.060,
            features=_make_features(network_anomaly=3.0, bio_anomaly=0.5),
        )
        assert result["adjusted_risk"] != "CRITICAL"

    def test_both_triggers_critical(self):
        result = _run_pipeline(
            distance=0.060,
            features=_make_features(network_anomaly=3.0, bio_anomaly=3.0),
        )
        assert result["adjusted_risk"] == "CRITICAL"


# ═══════════════════════════════════════════════════════════════════
# Scenario G: Calibration Phase (state machine)
# ═══════════════════════════════════════════════════════════════════

class TestScenarioG_CalibrationPhase:
    """Verify alerts suppressed during calibration."""

    def test_alerts_suppressed_during_calibration(self):
        buf = WindowBuffer(window_size=2, calibration_threshold=10)
        for _ in range(5):
            buf.append(np.zeros(N_FEATURES))
            buf.record_prediction({"risk_level": "CRITICAL", "ground_truth": 1})

        # Still calibrating → no alerts
        assert buf.state == SystemState.CALIBRATING
        assert len(buf.get_alerts()) == 0

    def test_alerts_emit_after_calibration(self):
        buf = WindowBuffer(window_size=2, calibration_threshold=5)
        # Fill past calibration
        for _ in range(10):
            buf.append(np.zeros(N_FEATURES))
            buf.record_prediction({"risk_level": "NORMAL", "ground_truth": 0})

        assert buf.state == SystemState.OPERATIONAL

        # Now CRITICAL should emit
        buf.record_prediction({"risk_level": "CRITICAL", "ground_truth": 1})
        alerts = buf.get_alerts()
        assert len(alerts) >= 1
        assert alerts[0]["risk_level"] == "CRITICAL"


# ═══════════════════════════════════════════════════════════════════
# Scenario H: Patient Safety Escalation Chain
# ═══════════════════════════════════════════════════════════════════

class TestScenarioH_PatientSafetyChain:
    """Infusion pump + abnormal vitals + HIGH risk → full escalation."""

    def test_full_escalation(self):
        result = _run_pipeline(
            distance=0.030,  # HIGH risk
            features=_make_features(network_anomaly=2.0, bio_anomaly=3.0),
            device_id="infusion_pump",
            attack_category="Spoofing",
        )
        assert result["patient_safety_flag"] is True
        assert result["clinical_severity"] >= 4  # EMERGENT or CRITICAL
        assert result["device_action"] in ("restrict_network", "isolate_network")
        assert result["response_time_minutes"] <= 15

    def test_biometric_clinical_notes_populated(self):
        result = _run_pipeline(
            distance=0.030,
            features=_make_features(network_anomaly=2.0, bio_anomaly=3.0),
            device_id="infusion_pump",
        )
        bio = result["biometric_status"]
        assert bio.any_abnormal is True
        assert len(bio.abnormal_features) > 0
        assert len(bio.clinical_notes) > 0

    def test_attention_flag_on_safety_device(self):
        result = _run_pipeline(
            distance=-0.01,  # NORMAL risk
            features=_make_features(),
            device_id="ecg_monitor",
            attention_flag=True,  # Novel threat
        )
        # Attention escalates NORMAL → MEDIUM, safety device → flag
        assert result["patient_safety_flag"] is True
