"""Unit tests for ConceptDriftDetector (baseline-relative ratio)."""

from __future__ import annotations

from src.phase4_risk_engine.phase4.base import BaseDetector
from src.phase4_risk_engine.phase4.drift_detector import ConceptDriftDetector


class TestConceptDriftDetector:
    """Test concept drift detection logic."""

    def test_implements_base(self) -> None:
        assert issubclass(ConceptDriftDetector, BaseDetector)

    def test_drift_detected(self) -> None:
        detector = ConceptDriftDetector(drift_threshold=0.20)
        # dynamic=0.30, baseline=0.20 → ratio = 0.50 > 0.20
        assert detector.detect(0.30, 0.20) is True

    def test_no_drift(self) -> None:
        detector = ConceptDriftDetector(drift_threshold=0.20)
        # dynamic=0.22, baseline=0.20 → ratio = 0.10 < 0.20
        assert detector.detect(0.22, 0.20) is False

    def test_compute_drift_ratio(self) -> None:
        detector = ConceptDriftDetector(drift_threshold=0.20)
        ratio = detector.compute_drift_ratio(0.30, 0.20)
        assert abs(ratio - 0.5) < 1e-6

    def test_compute_drift_ratio_symmetric(self) -> None:
        detector = ConceptDriftDetector(drift_threshold=0.20)
        ratio = detector.compute_drift_ratio(0.10, 0.20)
        assert abs(ratio - 0.5) < 1e-6

    def test_boundary_exact(self) -> None:
        detector = ConceptDriftDetector(drift_threshold=0.20)
        # dynamic=0.24, baseline=0.20 → ratio = 0.20 (NOT > 0.20)
        assert detector.detect(0.24, 0.20) is False

    def test_get_config(self) -> None:
        detector = ConceptDriftDetector(drift_threshold=0.15)
        assert detector.get_config() == {"drift_threshold": 0.15}
