"""Integration tests — cross-phase artifact flow validation.

Tests that artifacts produced by one phase are consumable by
downstream phases. Uses mock data to avoid real model training.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest


class TestReshaperLabelStrategy:
    """Test that the reshaper any_attack label strategy works correctly."""

    def test_any_attack_flags_mixed_windows(self) -> None:
        from src.phase2_detection_engine.phase2.reshaper import DataReshaper

        X = np.random.randn(30, 24).astype(np.float32)
        y = np.zeros(30, dtype=np.int32)
        y[5] = 1  # Single attack at position 5

        reshaper = DataReshaper(timesteps=10, stride=1)
        _, y_w = reshaper.reshape(X, y)

        # Windows containing position 5 should be labeled 1
        # Windows 0-5 contain index 5 (window 0=[0:10], window 5=[5:15])
        for i in range(6):
            assert y_w[i] == 1, f"Window {i} should be attack (contains index 5)"

    def test_any_attack_pure_normal_windows(self) -> None:
        from src.phase2_detection_engine.phase2.reshaper import DataReshaper

        X = np.random.randn(30, 24).astype(np.float32)
        y = np.zeros(30, dtype=np.int32)
        y[0] = 1  # Attack only at first position

        reshaper = DataReshaper(timesteps=10, stride=1)
        _, y_w = reshaper.reshape(X, y)

        # Window starting at index 10+ should be pure normal
        assert y_w[10] == 0

    def test_last_strategy_backward_compatible(self) -> None:
        from src.phase2_detection_engine.phase2.reshaper import DataReshaper

        X = np.random.randn(30, 24).astype(np.float32)
        y = np.zeros(30, dtype=np.int32)
        y[5] = 1

        reshaper = DataReshaper(timesteps=10, stride=1, label_strategy="last")
        _, y_w = reshaper.reshape(X, y)

        # Only window where last sample is index 5: window index would be
        # start=5-9=-4 which is invalid, so no window has last=5 for stride=1
        # Window 0: last=9, window 1: last=10, etc.
        # Actually window i has last = i+9. So last=5 means i=-4 (invalid)
        # All windows should be 0 since y[9]=0, y[10]=0, etc.
        assert y_w[0] == 0  # last sample is y[9]=0


class TestVarianceFilter:
    """Test Phase 1 variance filter removes zero-variance features."""

    def test_drops_constant_columns(self) -> None:
        import pandas as pd
        from src.phase1_preprocessing.phase1.variance import VarianceFilter

        df = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "B": [0.0, 0.0, 0.0],  # constant
            "C": [1.0, 1.0, 1.0],  # constant
            "Label": [0, 1, 0],
        })

        vf = VarianceFilter(max_unique=1, label_column="Label")
        result = vf.transform(df)

        assert "A" in result.columns
        assert "B" not in result.columns
        assert "C" not in result.columns
        assert "Label" in result.columns

    def test_preserves_label_column(self) -> None:
        import pandas as pd
        from src.phase1_preprocessing.phase1.variance import VarianceFilter

        df = pd.DataFrame({"X": [1.0, 2.0], "Label": [0, 0]})
        vf = VarianceFilter(max_unique=1, label_column="Label")
        result = vf.transform(df)
        assert "Label" in result.columns

    def test_report_lists_dropped(self) -> None:
        import pandas as pd
        from src.phase1_preprocessing.phase1.variance import VarianceFilter

        df = pd.DataFrame({"A": [1, 2], "B": [0, 0], "Label": [0, 1]})
        vf = VarianceFilter(max_unique=1, label_column="Label")
        vf.transform(df)
        report = vf.get_report()
        assert report["n_dropped"] == 1
        assert "B" in report["columns_dropped"]


class TestPhase4CIAThreatMapper:
    """Test CIA threat mapping."""

    def test_known_attack_mapping(self) -> None:
        from src.phase4_risk_engine.phase4.cia_threat_mapper import CIAThreatMapper

        mapper = CIAThreatMapper()
        vec = mapper.map("Spoofing")
        assert vec.integrity == 0.9
        assert vec.confidentiality == 0.6

    def test_unknown_falls_back(self) -> None:
        from src.phase4_risk_engine.phase4.cia_threat_mapper import CIAThreatMapper

        mapper = CIAThreatMapper()
        vec = mapper.map("ransomware_never_seen")
        assert vec.confidentiality == 0.5
        assert vec.integrity == 0.5
        assert vec.availability == 0.5

    def test_normal_is_zero(self) -> None:
        from src.phase4_risk_engine.phase4.cia_threat_mapper import CIAThreatMapper

        mapper = CIAThreatMapper()
        vec = mapper.map("normal")
        assert vec.confidentiality == 0.0
        assert vec.integrity == 0.0

    def test_from_config(self) -> None:
        from src.phase4_risk_engine.phase4.cia_threat_mapper import CIAThreatMapper

        entries = [
            {"attack_category": "custom_attack",
             "cia_weights": {"confidentiality": 0.1, "integrity": 0.2, "availability": 0.3}},
        ]
        mapper = CIAThreatMapper.from_config(entries)
        vec = mapper.map("custom_attack")
        assert vec.availability == 0.3
        # Fallbacks still exist
        assert mapper.map("unknown").confidentiality == 0.5


class TestPhase4DeviceRegistry:
    """Test device registry lookups."""

    def test_known_device(self) -> None:
        from src.phase4_risk_engine.phase4.device_registry import DeviceRegistry

        reg = DeviceRegistry()
        profile = reg.lookup("infusion_pump")
        assert profile.fda_class == "III"
        assert profile.cia_priority.availability == 1.0

    def test_unknown_returns_generic(self) -> None:
        from src.phase4_risk_engine.phase4.device_registry import DeviceRegistry

        reg = DeviceRegistry()
        profile = reg.lookup("unknown_device_xyz")
        assert profile.device_type == "generic_iomt_sensor"


class TestPhase4CIARiskModifier:
    """Test CIA risk modification with adaptive scenario shifting."""

    def test_high_risk_uses_high_threat_scenario(self) -> None:
        from src.phase4_risk_engine.phase4.cia_risk_modifier import (
            CIARiskModifier, Scenario, _determine_scenario,
        )
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        scenario = _determine_scenario(RiskLevel.HIGH, attention_flag=False)
        assert scenario == Scenario.HIGH_THREAT

    def test_medium_risk_uses_clinical_emergency(self) -> None:
        from src.phase4_risk_engine.phase4.cia_risk_modifier import (
            _determine_scenario, Scenario,
        )
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        scenario = _determine_scenario(RiskLevel.MEDIUM, attention_flag=False)
        assert scenario == Scenario.CLINICAL_EMERGENCY

    def test_normal_uses_normal_monitoring(self) -> None:
        from src.phase4_risk_engine.phase4.cia_risk_modifier import (
            _determine_scenario, Scenario,
        )
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        scenario = _determine_scenario(RiskLevel.NORMAL, attention_flag=False)
        assert scenario == Scenario.NORMAL_MONITORING

    def test_attention_flag_forces_high_threat(self) -> None:
        from src.phase4_risk_engine.phase4.cia_risk_modifier import (
            _determine_scenario, Scenario,
        )
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        scenario = _determine_scenario(RiskLevel.LOW, attention_flag=True)
        assert scenario == Scenario.HIGH_THREAT

    def test_escalation_on_high_cia_score(self) -> None:
        from src.phase4_risk_engine.phase4.cia_risk_modifier import CIARiskModifier
        from src.phase4_risk_engine.phase4.cia_threat_mapper import CIAThreatMapper
        from src.phase4_risk_engine.phase4.device_registry import DeviceRegistry
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        modifier = CIARiskModifier(
            threat_mapper=CIAThreatMapper(),
            device_registry=DeviceRegistry(),
            escalation_threshold=0.7,
        )
        # Spoofing on infusion_pump at HIGH risk → High Threat scenario
        # I = 0.9 * 1.0 * 1.0 = 0.9 >= 0.7 → escalate HIGH → CRITICAL
        result = modifier.modify(RiskLevel.HIGH, "Spoofing", "infusion_pump")
        assert result.adjusted_risk_level == RiskLevel.CRITICAL
        assert result.cia_modifier >= 0.7

    def test_no_escalation_on_low_cia(self) -> None:
        from src.phase4_risk_engine.phase4.cia_risk_modifier import CIARiskModifier
        from src.phase4_risk_engine.phase4.cia_threat_mapper import CIAThreatMapper
        from src.phase4_risk_engine.phase4.device_registry import DeviceRegistry
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        modifier = CIARiskModifier(
            threat_mapper=CIAThreatMapper(),
            device_registry=DeviceRegistry(),
            escalation_threshold=0.7,
        )
        # normal on temperature_sensor: all zeros → no escalation
        result = modifier.modify(RiskLevel.LOW, "normal", "temperature_sensor")
        assert result.adjusted_risk_level == RiskLevel.LOW


class TestPhase4ClinicalImpact:
    """Test clinical impact assessor."""

    def test_severity_mapping(self) -> None:
        from src.phase4_risk_engine.phase4.clinical_impact import ClinicalImpactAssessor, ClinicalSeverity
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        assessor = ClinicalImpactAssessor(biometric_columns=["SpO2", "Temp"])
        result = assessor.assess(RiskLevel.HIGH, "generic_iomt_sensor", None, None)
        assert result.clinical_severity == ClinicalSeverity.EMERGENT

    def test_safety_flag_on_critical_device(self) -> None:
        from src.phase4_risk_engine.phase4.clinical_impact import ClinicalImpactAssessor
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        assessor = ClinicalImpactAssessor(biometric_columns=["SpO2"])
        result = assessor.assess(RiskLevel.HIGH, "infusion_pump", None, None)
        assert result.patient_safety_flag is True

    def test_response_protocol_time(self) -> None:
        from src.phase4_risk_engine.phase4.clinical_impact import ClinicalImpactAssessor, ClinicalSeverity
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        assessor = ClinicalImpactAssessor(biometric_columns=[])
        result = assessor.assess(RiskLevel.CRITICAL, "ecg_monitor", None, None)
        assert result.protocol.response_time_minutes == 5


class TestPhase4AlertFatigue:
    """Test alert fatigue mitigation."""

    def test_routine_always_suppressed(self) -> None:
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager

        mgr = AlertFatigueManager()
        results = [{"clinical_severity": 1, "risk_level": "NORMAL"} for _ in range(10)]
        mgr.process(results)
        assert all(not r["alert_emit"] for r in results)

    def test_escalation_always_emits(self) -> None:
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager

        mgr = AlertFatigueManager()
        results = [
            {"clinical_severity": 2, "risk_level": "LOW"},
            {"clinical_severity": 4, "risk_level": "HIGH"},  # escalation
        ]
        mgr.process(results)
        assert results[1]["alert_emit"] is True
        assert results[1]["alert_reason"] == "escalation"

    def test_aggregation_suppresses_consecutive(self) -> None:
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager

        mgr = AlertFatigueManager(aggregation_window=5)
        # 10 consecutive HIGH alerts
        results = [{"clinical_severity": 4, "risk_level": "HIGH"} for _ in range(10)]
        mgr.process(results)

        emitted = [r for r in results if r["alert_emit"]]
        suppressed = [r for r in results if not r["alert_emit"]]
        # First emits, then suppressed until window boundary
        assert len(emitted) < 10
        assert len(suppressed) > 0

    def test_suppression_summary(self) -> None:
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager

        mgr = AlertFatigueManager()
        results = [{"clinical_severity": 1, "risk_level": "NORMAL"} for _ in range(5)]
        mgr.process(results)
        summary = mgr.get_summary()
        assert summary["alerts_suppressed"] == 5
        assert summary["suppression_rate"] == 1.0


class TestPhase4CognitiveTranslator:
    """Test cognitive translation for stakeholders."""

    def test_routine_gets_no_views(self) -> None:
        from dashboard.utils.cognitive_translator import CognitiveTranslator

        translator = CognitiveTranslator()
        results = [{"clinical_severity": 1, "risk_level": "NORMAL"}]
        translator.translate(results)
        assert results[0]["stakeholder_views"] is None

    def test_high_severity_gets_all_views(self) -> None:
        from dashboard.utils.cognitive_translator import CognitiveTranslator

        translator = CognitiveTranslator()
        results = [{
            "clinical_severity": 4, "risk_level": "HIGH",
            "attack_category": "Spoofing", "attention_flag": False,
            "cia_scores": {"C": 0.2, "I": 0.9, "A": 0.3},
            "cia_max_dimension": "I", "scenario": "high_threat",
            "device_action": "isolate_network", "patient_safety_flag": True,
            "response_time_minutes": 15, "explanation": {},
            "device_type": "pulse_oximeter", "clinical_rationale": "test",
        }]
        translator.translate(results)
        views = results[0]["stakeholder_views"]
        assert views is not None
        assert "soc_analyst" in views
        assert "clinician" in views
        assert "ciso" in views
        assert "biomed_engineer" in views

    def test_clinician_view_has_plain_language(self) -> None:
        from dashboard.utils.cognitive_translator import CognitiveTranslator

        translator = CognitiveTranslator()
        results = [{
            "clinical_severity": 5, "risk_level": "CRITICAL",
            "attack_category": "unknown", "attention_flag": True,
            "cia_scores": {}, "cia_max_dimension": "",
            "scenario": "", "device_action": "isolate_network",
            "patient_safety_flag": True, "response_time_minutes": 5,
            "explanation": {}, "clinical_rationale": "",
        }]
        translator.translate(results)
        clinician = results[0]["stakeholder_views"]["clinician"]
        assert "safety" in clinician.get("message", "").lower() or "concern" in clinician.get("message", "").lower()
        assert clinician.get("patient_safety_concern") is True
