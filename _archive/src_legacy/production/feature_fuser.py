"""Feature fuser — joins network flows with biometric data.

Produces the complete 24-feature vector by joining:
  - 16 network features (from FlowCollector)
  - 8 biometric features (from BiometricBridge, latest per device)

Applies RobustScaler (fitted during Phase 1 training, never refit)
and feeds the result to the WindowBuffer.

For WUSTL-compatible networks: all 24 features are real (no imputation).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from dashboard.streaming.feature_aligner import MODEL_FEATURES, N_FEATURES
from dashboard.streaming.window_buffer import WindowBuffer

logger = logging.getLogger(__name__)


class FeatureFuser:
    """Joins network + biometric features into scaled model input.

    Args:
        scaler_path: Path to fitted RobustScaler pickle.
        buffer: WindowBuffer to feed fused vectors into.
        default_biometrics: Default biometric values when device has
            no recent vitals. In production this should rarely trigger
            (biometric bridge provides real values).
    """

    NETWORK_FEATURES: List[str] = [
        "SrcBytes", "DstBytes", "SrcLoad", "DstLoad",
        "SIntPkt", "DIntPkt", "SIntPktAct",
        "sMaxPktSz", "dMaxPktSz", "sMinPktSz",
        "Dur", "TotBytes", "Load", "pSrcLoss", "pDstLoss", "Packet_num",
    ]

    BIOMETRIC_FEATURES: List[str] = [
        "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
        "Heart_rate", "Resp_Rate", "ST",
    ]

    DEFAULT_BIOMETRICS: Dict[str, float] = {
        "Temp": 36.8, "SpO2": 97.0, "Pulse_Rate": 75.0,
        "SYS": 120.0, "DIA": 80.0, "Heart_rate": 72.0,
        "Resp_Rate": 16.0, "ST": 0.1,
    }

    def __init__(
        self,
        scaler_path: str | Path,
        buffer: WindowBuffer,
        default_biometrics: Optional[Dict[str, float]] = None,
    ) -> None:
        self._scaler = joblib.load(scaler_path)
        self._buffer = buffer
        self._defaults = default_biometrics or self.DEFAULT_BIOMETRICS
        self._fused_count = 0
        self._imputed_count = 0

        # Verify scaler feature count matches model
        if hasattr(self._scaler, "n_features_in_"):
            expected = self._scaler.n_features_in_
            if expected != N_FEATURES:
                logger.warning(
                    "Scaler expects %d features but model uses %d",
                    expected, N_FEATURES,
                )

    def fuse(
        self,
        network_features: Dict[str, float],
        biometrics: Dict[str, float],
    ) -> np.ndarray:
        """Join network + biometric features, scale, and feed buffer.

        Args:
            network_features: 16 network feature values from FlowCollector.
            biometrics: 8 biometric values from BiometricBridge (may be partial).

        Returns:
            Scaled 24-feature vector.
        """
        # Build raw feature vector in canonical order
        raw = np.zeros(N_FEATURES, dtype=np.float32)
        imputed_any = False

        for i, feat in enumerate(MODEL_FEATURES):
            if feat in network_features:
                raw[i] = float(network_features[feat])
            elif feat in biometrics:
                raw[i] = float(biometrics[feat])
            elif feat in self._defaults:
                raw[i] = self._defaults[feat]
                imputed_any = True
            # else: remains 0.0

        if imputed_any:
            self._imputed_count += 1

        # Scale (transform only, never fit)
        scaled = self._scaler.transform(raw.reshape(1, -1)).ravel().astype(np.float32)

        # Feed into window buffer
        self._buffer.append(scaled)
        self._fused_count += 1

        return scaled

    def fuse_from_messages(
        self,
        flow_msg: Dict[str, Any],
        bridge: Any,
        device_id: str = "unknown",
    ) -> np.ndarray:
        """Fuse from raw Kafka message payloads.

        Args:
            flow_msg: FlowCollector message with "features" dict.
            bridge: BiometricBridge instance for latest vitals lookup.
            device_id: Device ID for biometric lookup.

        Returns:
            Scaled 24-feature vector.
        """
        network = flow_msg.get("features", {})
        biometrics = bridge.get_latest(device_id) if bridge else {}
        return self.fuse(network, biometrics)

    @property
    def fused_count(self) -> int:
        return self._fused_count

    @property
    def imputed_count(self) -> int:
        return self._imputed_count

    def get_status(self) -> Dict[str, Any]:
        return {
            "fused_count": self._fused_count,
            "imputed_count": self._imputed_count,
            "imputation_rate": round(
                self._imputed_count / max(self._fused_count, 1), 4,
            ),
            "n_features": N_FEATURES,
            "buffer_flow_count": self._buffer.flow_count,
        }
