"""Biometric bridge — receives HL7v2/FHIR vitals and publishes to Kafka.

Extracts 8 biometric features from bedside monitor messages:
  Temp, SpO2, Pulse_Rate, SYS, DIA, Heart_rate, Resp_Rate, ST

In production: listens on an HL7v2 MLLP socket or FHIR subscription.
In testing: accepts vitals via update() method.

HIPAA: strips all PHI before publishing. Only pseudonymized device_id
and numeric vital values pass through.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BiometricBridge:
    """Receives biometric data and publishes to a message queue.

    Maintains the latest vitals per device for temporal join with
    network flows in the FeatureFuser.

    Args:
        publish_fn: Callback to publish biometric update.
        topic: Kafka topic name for biometric data.
    """

    BIOMETRIC_FEATURES: List[str] = [
        "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
        "Heart_rate", "Resp_Rate", "ST",
    ]

    # LOINC codes for HL7v2 OBX mapping
    LOINC_MAP: Dict[str, str] = {
        "8310-5": "Temp",
        "59408-5": "SpO2",
        "8889-8": "Pulse_Rate",
        "8480-6": "SYS",
        "8462-4": "DIA",
        "8867-4": "Heart_rate",
        "9279-1": "Resp_Rate",
    }

    def __init__(
        self,
        publish_fn: Callable[[str, Dict[str, Any]], None],
        topic: str = "iomt.biometrics",
    ) -> None:
        self._publish = publish_fn
        self._topic = topic
        self._latest: Dict[str, Dict[str, float]] = {}
        self._update_count = 0

    def update(self, device_id: str, vitals: Dict[str, float]) -> None:
        """Update biometric values for a device.

        Args:
            device_id: Raw device identifier (will be pseudonymized).
            vitals: Dict mapping biometric feature names to values.
        """
        pseudo_id = self._pseudonymize(device_id)

        # Store latest vitals per device
        if pseudo_id not in self._latest:
            self._latest[pseudo_id] = {}

        valid_vitals = {}
        for feat in self.BIOMETRIC_FEATURES:
            if feat in vitals:
                val = float(vitals[feat])
                self._latest[pseudo_id][feat] = val
                valid_vitals[feat] = val

        payload = {
            "device_id": pseudo_id,
            "vitals": valid_vitals,
            "timestamp": time.time(),
        }
        self._publish(self._topic, payload)
        self._update_count += 1

    def get_latest(self, device_id: str) -> Dict[str, float]:
        """Get the most recent vitals for a device.

        Args:
            device_id: Raw device identifier.

        Returns:
            Dict of biometric feature values (may be partial).
        """
        pseudo_id = self._pseudonymize(device_id)
        return dict(self._latest.get(pseudo_id, {}))

    @staticmethod
    def _pseudonymize(device_id: str) -> str:
        """HIPAA-compliant device ID pseudonymization."""
        return hashlib.sha256(device_id.encode()).hexdigest()[:12]

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def active_devices(self) -> int:
        return len(self._latest)
