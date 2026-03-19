"""Unit tests for CrossModalFusionDetector (biometric + network)."""

from __future__ import annotations

import numpy as np

from src.phase4_risk_engine.phase4.base import BaseDetector
from src.phase4_risk_engine.phase4.cross_modal import CrossModalFusionDetector


class TestCrossModalFusionDetector:
    """Test cross-modal anomaly detection."""

    def test_implements_base(self) -> None:
        assert issubclass(CrossModalFusionDetector, BaseDetector)

    def test_both_modalities_anomalous(self) -> None:
        detector = CrossModalFusionDetector(
            biometric_columns=["Temp", "SpO2"],
            sigma_threshold=2.0,
        )
        # Bio > 2σ AND net > 2σ
        values = np.array([3.0, 3.0, 3.0, 3.0])
        names = ["Temp", "SpO2", "net1", "net2"]
        assert detector.detect(values, names) is True

    def test_only_bio_anomalous(self) -> None:
        detector = CrossModalFusionDetector(
            biometric_columns=["Temp", "SpO2"],
            sigma_threshold=2.0,
        )
        # Bio > 2σ, net normal
        values = np.array([3.0, 3.0, 0.5, 0.5])
        names = ["Temp", "SpO2", "net1", "net2"]
        assert detector.detect(values, names) is False

    def test_only_net_anomalous(self) -> None:
        detector = CrossModalFusionDetector(
            biometric_columns=["Temp", "SpO2"],
            sigma_threshold=2.0,
        )
        # Bio normal, net > 2σ
        values = np.array([0.5, 0.5, 3.0, 3.0])
        names = ["Temp", "SpO2", "net1", "net2"]
        assert detector.detect(values, names) is False

    def test_neither_anomalous(self) -> None:
        detector = CrossModalFusionDetector(
            biometric_columns=["Temp", "SpO2"],
            sigma_threshold=2.0,
        )
        values = np.array([0.5, 0.5, 0.5, 0.5])
        names = ["Temp", "SpO2", "net1", "net2"]
        assert detector.detect(values, names) is False

    def test_get_config(self) -> None:
        detector = CrossModalFusionDetector(
            biometric_columns=["Temp", "SpO2"],
            sigma_threshold=2.0,
        )
        config = detector.get_config()
        assert config["biometric_columns"] == ["Temp", "SpO2"]
        assert config["sigma_threshold"] == 2.0

    def test_custom_sigma_threshold(self) -> None:
        detector = CrossModalFusionDetector(
            biometric_columns=["Temp"],
            sigma_threshold=1.0,
        )
        # Values > 1σ but < 2σ
        values = np.array([1.5, 1.5])
        names = ["Temp", "net1"]
        assert detector.detect(values, names) is True
