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
CALIBRATION_THRESHOLD: int = 100


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

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        calibration_threshold: int = CALIBRATION_THRESHOLD,
    ) -> None:
        self._window_size = window_size
        self._calibration_threshold = calibration_threshold
        self._buffer: Deque[np.ndarray] = deque(maxlen=5000)
        self._lock = threading.Lock()
        self._flow_count: int = 0
        self._state = SystemState.INITIALIZING
        self._alerts: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._risk_counts: Dict[str, int] = {
            "NORMAL": 0, "LOW": 0, "MEDIUM": 0,
            "HIGH": 0, "CRITICAL": 0,
        }
        self._predictions: Deque[Dict[str, Any]] = deque(maxlen=500)
        self._started_at: Optional[datetime] = None

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

            if risk in ("HIGH", "CRITICAL") and not self.suppress_alerts:
                self._alerts.appendleft({
                    **prediction,
                    "alert_time": datetime.now(timezone.utc).isoformat(),
                })
                if risk == "CRITICAL":
                    self._state = SystemState.ALERT

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
            }

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
            logger.info("Buffer reset to INITIALIZING")
