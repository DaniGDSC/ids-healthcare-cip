"""Sliding window buffer for streaming flow ingestion.

Manages a fixed-size deque of preprocessed flow vectors,
providing windowed input for the CNN-BiLSTM-Attention model.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

WINDOW_SIZE: int = 20
CALIBRATION_THRESHOLD: int = 200


class SystemState(str, Enum):
    """Pipeline operational state."""

    INITIALIZING = "INITIALIZING"
    CALIBRATING = "CALIBRATING"
    OPERATIONAL = "OPERATIONAL"
    DEGRADED = "DEGRADED"
    ALERT = "ALERT"


class WindowBuffer:
    """Thread-safe sliding window buffer for streaming flows.

    Attributes:
        window_size: Number of timesteps per model input window.
        calibration_threshold: Minimum flows before alert generation.
    """

    _DETECTED_LEVELS = frozenset({"MEDIUM", "HIGH", "CRITICAL"})

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        calibration_threshold: int = CALIBRATION_THRESHOLD,
    ) -> None:
        self._window_size = window_size
        self._calibration_threshold = calibration_threshold
        self._alert_recovery_count = 0
        self._buffer: Deque[np.ndarray] = deque(maxlen=5000)
        self._lock = threading.Lock()
        self._flow_count: int = 0
        self._state = SystemState.INITIALIZING
        self._alerts: Deque[Dict[str, Any]] = deque(maxlen=500)
        self._risk_counts: Dict[str, int] = {
            "NORMAL": 0, "LOW": 0, "MEDIUM": 0,
            "HIGH": 0, "CRITICAL": 0,
        }
        self._predictions: Deque[Dict[str, Any]] = deque(maxlen=500)
        self._started_at: Optional[datetime] = None
        # Cumulative detection counters (not limited by deque size)
        self._detection_counts: Dict[str, int] = {
            "tp": 0, "fn": 0, "fp": 0, "tn": 0, "total_with_gt": 0,
        }

    @property
    def state(self) -> SystemState:
        """Current system operational state."""
        return self._state

    @property
    def flow_count(self) -> int:
        """Total flows ingested."""
        return self._flow_count

    @property
    def suppress_alerts(self) -> bool:
        """Whether alert generation should be suppressed."""
        return self._flow_count < self._calibration_threshold

    def append(self, flow_vector: np.ndarray) -> None:
        """Append a single preprocessed flow vector to the buffer.

        Args:
            flow_vector: Scaled feature vector of shape (n_features,).
        """
        with self._lock:
            self._buffer.append(flow_vector)
            self._flow_count += 1

            if self._started_at is None:
                self._started_at = datetime.now(timezone.utc)

            if self._flow_count < self._window_size:
                self._state = SystemState.INITIALIZING
            elif self._flow_count < self._calibration_threshold:
                self._state = SystemState.CALIBRATING
            elif self._state in (SystemState.INITIALIZING,
                                 SystemState.CALIBRATING):
                self._state = SystemState.OPERATIONAL
                logger.info("System transitioned to OPERATIONAL "
                            "(%d flows)", self._flow_count)

    def get_window(self) -> Optional[np.ndarray]:
        """Get the latest sliding window for prediction.

        Returns:
            Array of shape (1, window_size, n_features) or None if insufficient data.
        """
        with self._lock:
            if len(self._buffer) < self._window_size:
                return None
            window = list(self._buffer)[-self._window_size:]
            return np.array(window).reshape(1, self._window_size, -1)

    def get_all_windows(self, n_latest: int = 10) -> Optional[np.ndarray]:
        """Get multiple recent windows for batch prediction.

        Args:
            n_latest: Number of windows to extract.

        Returns:
            Array of shape (N, window_size, n_features) or None.
        """
        with self._lock:
            buf_len = len(self._buffer)
            if buf_len < self._window_size:
                return None
            n_available = buf_len - self._window_size + 1
            n = min(n_latest, n_available)
            buf_list = list(self._buffer)
            windows = []
            for i in range(n_available - n, n_available):
                w = buf_list[i: i + self._window_size]
                windows.append(np.array(w))
            return np.array(windows)

    def record_prediction(self, prediction: Dict[str, Any]) -> None:
        """Record a prediction result."""
        with self._lock:
            self._predictions.append(prediction)
            risk = prediction.get("risk_level", "NORMAL")
            if risk in self._risk_counts:
                self._risk_counts[risk] += 1

            # Update cumulative detection counters
            gt = prediction.get("ground_truth", -1)
            if gt >= 0:
                detected = risk in self._DETECTED_LEVELS
                is_attack = gt == 1
                if is_attack and detected:
                    self._detection_counts["tp"] += 1
                elif is_attack and not detected:
                    self._detection_counts["fn"] += 1
                elif not is_attack and detected:
                    self._detection_counts["fp"] += 1
                else:
                    self._detection_counts["tn"] += 1
                self._detection_counts["total_with_gt"] += 1

            # Transition to DEGRADED if inference failed
            if prediction.get("inference_failed", False):
                if self._state in (SystemState.OPERATIONAL, SystemState.ALERT):
                    self._state = SystemState.DEGRADED
                    logger.warning("System DEGRADED — inference circuit breaker active")
            else:
                # Recovery: DEGRADED → OPERATIONAL when inference succeeds
                if self._state == SystemState.DEGRADED:
                    self._state = SystemState.OPERATIONAL
                    logger.info("System recovered: DEGRADED → OPERATIONAL")

            if risk in ("MEDIUM", "HIGH", "CRITICAL") and not self.suppress_alerts:
                self._alerts.appendleft({
                    **prediction,
                    "alert_time": datetime.now(timezone.utc).isoformat(),
                })
                if risk == "CRITICAL":
                    self._state = SystemState.ALERT
                    self._alert_recovery_count = 0
                elif self._state == SystemState.ALERT:
                    # Count consecutive non-CRITICAL for ALERT recovery
                    self._alert_recovery_count += 1
                    if self._alert_recovery_count >= 50:
                        self._state = SystemState.OPERATIONAL
                        logger.info("System recovered: ALERT → OPERATIONAL "
                                    "after %d clean predictions",
                                    self._alert_recovery_count)
            elif self._state == SystemState.ALERT:
                self._alert_recovery_count += 1
                if self._alert_recovery_count >= 50:
                    self._state = SystemState.OPERATIONAL
                    logger.info("System recovered: ALERT → OPERATIONAL")

    def get_alerts(self, n: int = 50) -> List[Dict[str, Any]]:
        """Get the latest alerts."""
        with self._lock:
            return list(self._alerts)[:n]

    def get_risk_counts(self) -> Dict[str, int]:
        """Get cumulative risk level counts."""
        with self._lock:
            return dict(self._risk_counts)

    def get_recent_predictions(self, n: int = 50) -> List[Dict[str, Any]]:
        """Get the most recent predictions."""
        with self._lock:
            return list(self._predictions)[:n]

    def get_status(self) -> Dict[str, Any]:
        """Get buffer status summary."""
        with self._lock:
            return {
                "state": self._state.value,
                "flow_count": self._flow_count,
                "buffer_size": len(self._buffer),
                "window_size": self._window_size,
                "calibration_progress": min(
                    self._flow_count / self._calibration_threshold, 1.0,
                ),
                "suppress_alerts": self.suppress_alerts,
                "alert_count": len(self._alerts),
                "risk_counts": dict(self._risk_counts),
                "started_at": self._started_at.isoformat()
                if self._started_at else None,
                "detection_counts": dict(self._detection_counts),
            }

    def get_flow_vectors(self, n: int = 50) -> List[np.ndarray]:
        """Get the last N raw flow vectors for visualization."""
        with self._lock:
            buf = list(self._buffer)
            return buf[-n:]

    def get_prediction_timeseries(self, n: int = 200) -> List[Dict[str, Any]]:
        """Get prediction timeseries for charts (score, risk, phase)."""
        with self._lock:
            return [
                {
                    "index": p.get("sample_index", i),
                    "score": p.get("anomaly_score", 0),
                    "risk": p.get("risk_level", "NORMAL"),
                    "severity": p.get("clinical_severity", 1),
                    "attention": p.get("attention_flag", False),
                    "ground_truth": p.get("ground_truth", -1),
                    "latency": p.get("latency_ms", 0),
                    "phase": p.get("phase", ""),
                }
                for i, p in enumerate(list(self._predictions)[-n:])
            ]

    def reset(self) -> None:
        """Reset the buffer to initial state."""
        with self._lock:
            self._buffer.clear()
            self._flow_count = 0
            self._state = SystemState.INITIALIZING
            self._alerts.clear()
            self._risk_counts = {
                "NORMAL": 0, "LOW": 0, "MEDIUM": 0,
                "HIGH": 0, "CRITICAL": 0,
            }
            self._predictions.clear()
            self._started_at = None
            self._detection_counts = {
                "tp": 0, "fn": 0, "fp": 0, "tn": 0, "total_with_gt": 0,
            }
            logger.info("Buffer reset to INITIALIZING")
