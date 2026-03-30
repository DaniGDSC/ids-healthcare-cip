"""QA/QC remediation tests — fixes for audit findings F1-F4.

F1: WindowBuffer unit tests
F2: Cross-modal fusion edge cases
F3: Risk level transition chains
F4: API integration test (ingest → model → alert)
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

import numpy as np
import pytest

from dashboard.streaming.feature_aligner import MODEL_FEATURES, N_FEATURES
from dashboard.streaming.window_buffer import WindowBuffer, SystemState


# ═══════════════════════════════════════════════════════════════════
# F1: WindowBuffer Unit Tests
# ═══════════════════════════════════════════════════════════════════

class TestWindowBufferStateMachine:

    def test_initial_state_is_initializing(self) -> None:
        buf = WindowBuffer(window_size=5, calibration_threshold=10)
        assert buf.state == SystemState.INITIALIZING

    def test_transitions_to_calibrating(self) -> None:
        buf = WindowBuffer(window_size=3, calibration_threshold=10)
        for _ in range(5):
            buf.append(np.zeros(N_FEATURES, dtype=np.float32))
        assert buf.state == SystemState.CALIBRATING

    def test_transitions_to_operational(self) -> None:
        buf = WindowBuffer(window_size=3, calibration_threshold=5)
        for _ in range(6):
            buf.append(np.zeros(N_FEATURES, dtype=np.float32))
        assert buf.state == SystemState.OPERATIONAL

    def test_alert_state_on_critical(self) -> None:
        buf = WindowBuffer(window_size=3, calibration_threshold=3)
        for _ in range(5):
            buf.append(np.zeros(N_FEATURES, dtype=np.float32))
        buf.record_prediction({"risk_level": "CRITICAL"})
        assert buf.state == SystemState.ALERT

    def test_suppress_alerts_during_calibration(self) -> None:
        buf = WindowBuffer(window_size=3, calibration_threshold=10)
        for _ in range(5):
            buf.append(np.zeros(N_FEATURES, dtype=np.float32))
        assert buf.suppress_alerts is True

    def test_no_suppress_after_calibration(self) -> None:
        buf = WindowBuffer(window_size=3, calibration_threshold=5)
        for _ in range(6):
            buf.append(np.zeros(N_FEATURES, dtype=np.float32))
        assert buf.suppress_alerts is False


class TestWindowBufferWindowing:

    def test_get_window_returns_none_when_empty(self) -> None:
        buf = WindowBuffer(window_size=5)
        assert buf.get_window() is None

    def test_get_window_returns_none_insufficient(self) -> None:
        buf = WindowBuffer(window_size=5)
        for _ in range(3):
            buf.append(np.ones(N_FEATURES, dtype=np.float32))
        assert buf.get_window() is None

    def test_get_window_shape(self) -> None:
        buf = WindowBuffer(window_size=5)
        for _ in range(10):
            buf.append(np.random.randn(N_FEATURES).astype(np.float32))
        w = buf.get_window()
        assert w is not None
        assert w.shape == (1, 5, N_FEATURES)

    def test_get_window_uses_latest(self) -> None:
        buf = WindowBuffer(window_size=3)
        for i in range(5):
            buf.append(np.full(N_FEATURES, float(i), dtype=np.float32))
        w = buf.get_window()
        # Last 3 vectors: [2, 3, 4]
        assert w[0, 0, 0] == 2.0
        assert w[0, 2, 0] == 4.0

    def test_get_all_windows_shape(self) -> None:
        buf = WindowBuffer(window_size=3)
        for _ in range(10):
            buf.append(np.random.randn(N_FEATURES).astype(np.float32))
        ws = buf.get_all_windows(n_latest=5)
        assert ws is not None
        assert ws.shape == (5, 3, N_FEATURES)

    def test_maxlen_drops_oldest(self) -> None:
        buf = WindowBuffer(window_size=3)
        buf._buffer = buf._buffer.__class__(maxlen=5)
        for i in range(10):
            buf.append(np.full(N_FEATURES, float(i), dtype=np.float32))
        assert len(buf._buffer) == 5


class TestWindowBufferThreadSafety:

    def test_concurrent_append_and_get(self) -> None:
        buf = WindowBuffer(window_size=5, calibration_threshold=5)
        errors: List[str] = []

        def writer():
            for _ in range(100):
                buf.append(np.random.randn(N_FEATURES).astype(np.float32))
                time.sleep(0.001)

        def reader():
            for _ in range(50):
                w = buf.get_window()
                if w is not None and w.shape != (1, 5, N_FEATURES):
                    errors.append(f"Bad shape: {w.shape}")
                time.sleep(0.002)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == [], f"Thread safety errors: {errors}"


class TestWindowBufferReset:

    def test_reset_clears_state(self) -> None:
        buf = WindowBuffer(window_size=3, calibration_threshold=3)
        for _ in range(5):
            buf.append(np.zeros(N_FEATURES, dtype=np.float32))
        buf.record_prediction({"risk_level": "HIGH"})
        buf.reset()
        assert buf.flow_count == 0
        assert buf.state == SystemState.INITIALIZING
        assert buf.get_window() is None
        assert buf.get_alerts() == []


# ═══════════════════════════════════════════════════════════════════
# F2: Cross-Modal Fusion Edge Cases
# ═══════════════════════════════════════════════════════════════════

class TestCrossModalEdgeCases:

    @pytest.fixture
    def detector(self):
        from src.phase4_risk_engine.phase4.cross_modal import CrossModalFusionDetector
        return CrossModalFusionDetector(
            biometric_columns=["Temp", "SpO2", "Heart_rate"],
            sigma_threshold=2.0,
        )

    def test_both_anomalous_returns_true(self, detector) -> None:
        features = np.array([0.0] * N_FEATURES, dtype=np.float32)
        # Set biometric and network features beyond 2σ
        bio_idx = MODEL_FEATURES.index("Temp")
        net_idx = MODEL_FEATURES.index("SrcBytes")
        features[bio_idx] = 3.0
        features[net_idx] = 3.0
        assert detector.detect(features, MODEL_FEATURES) is True

    def test_only_bio_anomalous_returns_false(self, detector) -> None:
        features = np.zeros(N_FEATURES, dtype=np.float32)
        features[MODEL_FEATURES.index("SpO2")] = 5.0
        assert detector.detect(features, MODEL_FEATURES) is False

    def test_only_network_anomalous_returns_false(self, detector) -> None:
        features = np.zeros(N_FEATURES, dtype=np.float32)
        features[MODEL_FEATURES.index("SrcBytes")] = 5.0
        assert detector.detect(features, MODEL_FEATURES) is False

    def test_neither_anomalous_returns_false(self, detector) -> None:
        features = np.zeros(N_FEATURES, dtype=np.float32)
        assert detector.detect(features, MODEL_FEATURES) is False

    def test_missing_biometric_columns_no_crash(self) -> None:
        from src.phase4_risk_engine.phase4.cross_modal import CrossModalFusionDetector
        detector = CrossModalFusionDetector(
            biometric_columns=["NonExistentFeature"],
        )
        features = np.ones(N_FEATURES, dtype=np.float32) * 5
        # Should not crash, but bio_anomaly=False (no matching columns)
        result = detector.detect(features, MODEL_FEATURES)
        assert isinstance(result, bool)

    def test_empty_features_no_crash(self, detector) -> None:
        features = np.array([], dtype=np.float32)
        # Should handle gracefully
        result = detector.detect(features, [])
        assert result is False


# ═══════════════════════════════════════════════════════════════════
# F3: Risk Level Transition Chains
# ═══════════════════════════════════════════════════════════════════

class TestRiskLevelTransitions:

    @pytest.fixture
    def scorer(self):
        from src.phase4_risk_engine.phase4.risk_scorer import RiskScorer
        return RiskScorer(low_upper=0.5, medium_upper=1.0, high_upper=2.0)

    def test_normal_to_critical_chain(self, scorer) -> None:
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel
        mad = 0.025
        levels = []
        for distance in [-0.1, 0.005, 0.015, 0.03, 0.06]:
            level = scorer.classify_single(distance=distance, mad=mad)
            levels.append(level)
        assert levels == [
            RiskLevel.NORMAL,
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.HIGH,  # no cross-modal → stays HIGH
        ]

    def test_monotonic_escalation(self, scorer) -> None:
        """Increasing distance should never decrease risk level."""
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel
        mad = 0.025
        order = [RiskLevel.NORMAL, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        prev_idx = 0
        for d in np.linspace(-0.1, 0.1, 50):
            level = scorer.classify_single(distance=d, mad=mad)
            curr_idx = order.index(level)
            assert curr_idx >= prev_idx, f"Risk decreased at distance={d}: {order[prev_idx]} → {level}"
            prev_idx = curr_idx

    def test_attention_escalation_from_normal(self) -> None:
        """Attention anomaly should escalate NORMAL → MEDIUM."""
        from src.phase4_risk_engine.phase4.risk_scorer import RiskScorer
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        scorer = RiskScorer()
        scores = np.array([0.1])
        thresholds = np.array([0.5])
        features = np.zeros((1, N_FEATURES), dtype=np.float32)
        attn = np.array([True])

        results = scorer.score(scores, thresholds, 0.025, features, MODEL_FEATURES, attn)
        assert results[0]["risk_level"] == RiskLevel.MEDIUM.value


# ═══════════════════════════════════════════════════════════════════
# F4: API Integration Test
# ═══════════════════════════════════════════════════════════════════

class TestAPIIntegration:

    def test_ingest_to_prediction_flow(self) -> None:
        """Full flow: ingest 25 flows → buffer fills → model predicts."""
        try:
            from fastapi.testclient import TestClient
            from src.production.api import app
        except ImportError:
            pytest.skip("FastAPI not installed")

        with TestClient(app, raise_server_exceptions=False) as client:
            health = client.get("/health").json()
            if not health.get("model_loaded"):
                pytest.skip("Model not loaded (startup event may not fire)")

            # Ingest enough flows to fill buffer
            last_response = None
            for i in range(25):
                r = client.post("/ingest", json={
                    "network_features": {
                        "SrcBytes": 310, "DstBytes": 246, "SrcLoad": 6891,
                        "DstLoad": 5576, "SIntPkt": 39969, "DIntPkt": 39969,
                        "SIntPktAct": 22912, "sMaxPktSz": 60, "dMaxPktSz": 54,
                        "sMinPktSz": 40, "Dur": 0.079, "TotBytes": 556,
                        "Load": 12468, "pSrcLoss": 0, "pDstLoss": 0,
                        "Packet_num": 10,
                    },
                    "biometrics": {
                        "Temp": 36.8, "SpO2": 97, "Pulse_Rate": 75,
                        "SYS": 120, "DIA": 80, "Heart_rate": 72,
                        "Resp_Rate": 16, "ST": 0.1,
                    },
                    "device_id": "infusion_pump",
                })
                assert r.status_code == 200
                last_response = r.json()

            # Last response should be a real prediction (not CALIBRATING)
            assert last_response is not None
            assert last_response["risk_level"] != "CALIBRATING"
            assert "clinical_severity" in last_response

            # Status should show inferences
            status = client.get("/status").json()
            assert status.get("buffer", {}).get("flow_count", 0) == 25


# ═══════════════════════════════════════════════════════════════════
# B5: Biometric Staleness Tests
# ═══════════════════════════════════════════════════════════════════

class TestBiometricStaleness:

    def test_fresh_data_returned(self) -> None:
        from src.production.biometric_bridge import BiometricBridge
        bridge = BiometricBridge(publish_fn=lambda t, p: None, stale_threshold_s=60)
        bridge.update("pump-1", {"SpO2": 95.0})
        latest = bridge.get_latest("pump-1")
        assert latest["SpO2"] == 95.0

    def test_stale_data_returns_empty(self) -> None:
        from src.production.biometric_bridge import BiometricBridge
        bridge = BiometricBridge(publish_fn=lambda t, p: None, stale_threshold_s=0.01)
        bridge.update("pump-1", {"SpO2": 95.0})
        time.sleep(0.02)  # exceed threshold
        latest = bridge.get_latest("pump-1")
        assert latest == {}
        assert bridge.stale_count == 1

    def test_cache_age_tracking(self) -> None:
        from src.production.biometric_bridge import BiometricBridge
        bridge = BiometricBridge(publish_fn=lambda t, p: None)
        bridge.update("pump-1", {"Temp": 37.0})
        age = bridge.get_cache_age("pump-1")
        assert age is not None
        assert age < 1.0  # should be near-zero

    def test_unknown_device_age_is_none(self) -> None:
        from src.production.biometric_bridge import BiometricBridge
        bridge = BiometricBridge(publish_fn=lambda t, p: None)
        assert bridge.get_cache_age("nonexistent") is None
