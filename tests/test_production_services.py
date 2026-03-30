"""Tests for production deployment services.

Tests the ingestion pipeline without requiring Kafka, HL7, or
a running model — uses inject/mock patterns for deterministic testing.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from dashboard.streaming.feature_aligner import MODEL_FEATURES, N_FEATURES
from dashboard.streaming.window_buffer import WindowBuffer, SystemState


# ── Helpers ────────────────────────────────────────────────────────

def _make_network_features() -> Dict[str, float]:
    """Generate realistic network feature values."""
    return {
        "SrcBytes": 310.0, "DstBytes": 246.0, "SrcLoad": 6891.0,
        "DstLoad": 5576.0, "SIntPkt": 39969.0, "DIntPkt": 39969.0,
        "SIntPktAct": 22912.0, "sMaxPktSz": 60.0, "dMaxPktSz": 54.0,
        "sMinPktSz": 40.0, "Dur": 0.079, "TotBytes": 556.0,
        "Load": 12468.0, "pSrcLoss": 0.0, "pDstLoss": 0.0,
        "Packet_num": 10.0,
    }


def _make_biometrics() -> Dict[str, float]:
    """Generate realistic biometric values."""
    return {
        "Temp": 36.8, "SpO2": 97.0, "Pulse_Rate": 75.0,
        "SYS": 120.0, "DIA": 80.0, "Heart_rate": 72.0,
        "Resp_Rate": 16.0, "ST": 0.1,
    }


# ── FlowCollector Tests ───────────────────────────────────────────

class TestFlowCollector:

    def test_inject_publishes_to_topic(self) -> None:
        from src.production.flow_collector import FlowCollector

        published: List[Dict] = []
        collector = FlowCollector(
            publish_fn=lambda topic, payload: published.append({"topic": topic, **payload}),
            topic="test.flows",
        )
        collector.inject(_make_network_features())

        assert len(published) == 1
        assert published[0]["topic"] == "test.flows"
        assert "features" in published[0]
        assert len(published[0]["features"]) == 16

    def test_inject_rejects_missing_features(self) -> None:
        from src.production.flow_collector import FlowCollector

        published: List[Dict] = []
        collector = FlowCollector(publish_fn=lambda t, p: published.append(p))
        collector.inject({"SrcBytes": 310.0})  # missing 15 features

        assert len(published) == 0

    def test_flow_count_increments(self) -> None:
        from src.production.flow_collector import FlowCollector

        collector = FlowCollector(publish_fn=lambda t, p: None)
        assert collector.flow_count == 0
        collector.inject(_make_network_features())
        assert collector.flow_count == 1


# ── BiometricBridge Tests ──────────────────────────────────────────

class TestBiometricBridge:

    def test_update_stores_latest_vitals(self) -> None:
        from src.production.biometric_bridge import BiometricBridge

        bridge = BiometricBridge(publish_fn=lambda t, p: None)
        bridge.update("pump-001", {"SpO2": 95.0, "Heart_rate": 88.0})

        latest = bridge.get_latest("pump-001")
        assert latest["SpO2"] == 95.0
        assert latest["Heart_rate"] == 88.0

    def test_update_overwrites_previous(self) -> None:
        from src.production.biometric_bridge import BiometricBridge

        bridge = BiometricBridge(publish_fn=lambda t, p: None)
        bridge.update("pump-001", {"SpO2": 97.0})
        bridge.update("pump-001", {"SpO2": 91.0})

        assert bridge.get_latest("pump-001")["SpO2"] == 91.0

    def test_pseudonymizes_device_id(self) -> None:
        from src.production.biometric_bridge import BiometricBridge

        bridge = BiometricBridge(publish_fn=lambda t, p: None)
        bridge.update("pump-001", {"Temp": 37.0})

        # Raw ID should not appear in internal storage
        assert "pump-001" not in bridge._latest
        # Pseudonymized ID should
        pseudo = bridge._pseudonymize("pump-001")
        assert pseudo in bridge._latest

    def test_unknown_device_returns_empty(self) -> None:
        from src.production.biometric_bridge import BiometricBridge

        bridge = BiometricBridge(publish_fn=lambda t, p: None)
        assert bridge.get_latest("nonexistent") == {}

    def test_active_devices_count(self) -> None:
        from src.production.biometric_bridge import BiometricBridge

        bridge = BiometricBridge(publish_fn=lambda t, p: None)
        bridge.update("dev-1", {"Temp": 36.5})
        bridge.update("dev-2", {"Temp": 37.0})
        assert bridge.active_devices == 2


# ── KafkaConsumer Tests ────────────────────────────────────────────

class TestKafkaConsumer:

    def test_inject_feeds_buffer(self) -> None:
        from src.production.kafka_consumer import KafkaFlowConsumer

        buffer = WindowBuffer(window_size=5, calibration_threshold=3)
        consumer = KafkaFlowConsumer(buffer)

        features = np.random.randn(N_FEATURES).tolist()
        consumer.inject({"features": features, "timestamp": 1.0})

        assert buffer.flow_count == 1
        assert consumer.consumed_count == 1

    def test_inject_dict_format(self) -> None:
        from src.production.kafka_consumer import KafkaFlowConsumer

        buffer = WindowBuffer(window_size=5)
        consumer = KafkaFlowConsumer(buffer)

        features = {feat: float(i) for i, feat in enumerate(MODEL_FEATURES)}
        consumer.inject({"features": features})

        assert buffer.flow_count == 1

    def test_inject_rejects_wrong_size(self) -> None:
        from src.production.kafka_consumer import KafkaFlowConsumer

        buffer = WindowBuffer(window_size=5)
        consumer = KafkaFlowConsumer(buffer)

        consumer.inject({"features": [1.0, 2.0, 3.0]})  # wrong size
        assert buffer.flow_count == 0

    def test_window_ready_callback(self) -> None:
        from src.production.kafka_consumer import KafkaFlowConsumer

        buffer = WindowBuffer(window_size=3, calibration_threshold=3)
        windows_received: List[np.ndarray] = []

        consumer = KafkaFlowConsumer(
            buffer,
            on_window_ready=lambda w: windows_received.append(w),
        )

        for _ in range(5):
            features = np.random.randn(N_FEATURES).tolist()
            consumer.inject({"features": features})

        assert len(windows_received) > 0
        assert windows_received[0].shape == (1, 3, N_FEATURES)

    def test_batch_inject(self) -> None:
        from src.production.kafka_consumer import KafkaFlowConsumer

        buffer = WindowBuffer(window_size=5)
        consumer = KafkaFlowConsumer(buffer)

        messages = [
            {"features": np.random.randn(N_FEATURES).tolist()}
            for _ in range(10)
        ]
        count = consumer.inject_batch(messages)

        assert count == 10
        assert buffer.flow_count == 10


# ── FeatureFuser Tests ─────────────────────────────────────────────

class TestFeatureFuser:

    @pytest.fixture
    def fuser(self, tmp_path):
        """Create a fuser with a mock scaler."""
        import joblib
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
        # Fit on dummy data matching 24 features
        dummy = np.random.randn(50, N_FEATURES)
        scaler.fit(dummy)
        scaler_path = tmp_path / "scaler.pkl"
        joblib.dump(scaler, scaler_path)

        from src.production.feature_fuser import FeatureFuser
        buffer = WindowBuffer(window_size=5)
        return FeatureFuser(scaler_path, buffer)

    def test_fuse_produces_24_features(self, fuser) -> None:
        result = fuser.fuse(_make_network_features(), _make_biometrics())
        assert result.shape == (N_FEATURES,)
        assert result.dtype == np.float32

    def test_fuse_increments_count(self, fuser) -> None:
        fuser.fuse(_make_network_features(), _make_biometrics())
        fuser.fuse(_make_network_features(), _make_biometrics())
        assert fuser.fused_count == 2

    def test_fuse_feeds_buffer(self, fuser) -> None:
        fuser.fuse(_make_network_features(), _make_biometrics())
        assert fuser._buffer.flow_count == 1

    def test_missing_biometrics_uses_defaults(self, fuser) -> None:
        result = fuser.fuse(_make_network_features(), {})
        # Should still produce 24 features (biometrics defaulted)
        assert result.shape == (N_FEATURES,)
        assert fuser.imputed_count == 1


# ── Feature Aligner Tests ─────────────────────────────────────────

class TestFeatureAligner:

    def test_model_features_count(self) -> None:
        assert N_FEATURES == 24
        assert len(MODEL_FEATURES) == 24

    def test_dropped_features_not_in_model(self) -> None:
        from dashboard.streaming.feature_aligner import DROPPED_FEATURES
        for feat in DROPPED_FEATURES:
            assert feat not in MODEL_FEATURES

    def test_wustl_native_alignment(self) -> None:
        import pandas as pd
        from dashboard.streaming.feature_aligner import align_wustl_native

        # DataFrame with all 24 features
        data = {feat: [float(i)] for i, feat in enumerate(MODEL_FEATURES)}
        df = pd.DataFrame(data)
        result = align_wustl_native(df)

        assert list(result.columns) == MODEL_FEATURES
        assert len(result) == 1

    def test_wustl_native_drops_extra_features(self) -> None:
        import pandas as pd
        from dashboard.streaming.feature_aligner import align_wustl_native, DROPPED_FEATURES

        # DataFrame with 29 features (includes 5 dropped)
        data = {feat: [0.0] for feat in MODEL_FEATURES}
        for feat in DROPPED_FEATURES:
            data[feat] = [0.0]
        df = pd.DataFrame(data)
        result = align_wustl_native(df)

        assert len(result.columns) == 24
        for feat in DROPPED_FEATURES:
            assert feat not in result.columns

    def test_wustl_native_raises_on_missing(self) -> None:
        import pandas as pd
        from dashboard.streaming.feature_aligner import align_wustl_native

        df = pd.DataFrame({"SrcBytes": [1.0]})  # only 1 feature
        with pytest.raises(ValueError, match="Missing"):
            align_wustl_native(df)

    def test_validate_schema_24_features(self) -> None:
        import pandas as pd
        from dashboard.streaming.feature_aligner import validate_schema

        data = {feat: [0.0] for feat in MODEL_FEATURES}
        df = pd.DataFrame(data)
        valid, msg = validate_schema(df)
        assert valid
        assert "24/24" in msg
