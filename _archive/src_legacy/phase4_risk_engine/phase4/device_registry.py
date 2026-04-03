"""Device registry — maps device IDs to profiles with CIA priorities.

Each medical IoT device has an FDA class and CIA priority weights
reflecting its clinical role. An infusion pump prioritises Availability
(downtime = patient risk), while a records system prioritises
Confidentiality (breach = HIPAA violation).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, NamedTuple

from .base import BaseDetector

logger = logging.getLogger(__name__)


class CIAPriority(NamedTuple):
    """CIA priority weights for a device type (each 0.0–1.0)."""

    confidentiality: float
    integrity: float
    availability: float


class DeviceProfile(NamedTuple):
    """Device profile with FDA class and CIA priorities."""

    device_type: str
    fda_class: str
    cia_priority: CIAPriority
    description: str
    detection_percentile: float = 93.0  # Lower = more sensitive detection


_DEFAULT_DEVICES: Dict[str, DeviceProfile] = {
    "infusion_pump": DeviceProfile(
        device_type="infusion_pump",
        fda_class="III",
        cia_priority=CIAPriority(0.4, 1.0, 1.0),
        description="IV medication delivery — availability-critical",
        detection_percentile=30.0,  # Aggressive: missing attack = patient harm
    ),
    "ecg_monitor": DeviceProfile(
        device_type="ecg_monitor",
        fda_class="III",
        cia_priority=CIAPriority(0.5, 1.0, 1.0),
        description="Cardiac monitoring — integrity and availability critical",
        detection_percentile=30.0,  # Aggressive: life-sustaining device
    ),
    "pulse_oximeter": DeviceProfile(
        device_type="pulse_oximeter",
        fda_class="II",
        cia_priority=CIAPriority(0.3, 0.9, 0.8),
        description="SpO2 monitoring — integrity-critical",
        detection_percentile=40.0,  # Sensitive: clinical monitoring device
    ),
    "blood_pressure_monitor": DeviceProfile(
        device_type="blood_pressure_monitor",
        fda_class="II",
        cia_priority=CIAPriority(0.3, 0.9, 0.7),
        description="BP monitoring — integrity-critical",
        detection_percentile=50.0,  # Moderate: monitoring device
    ),
    "temperature_sensor": DeviceProfile(
        device_type="temperature_sensor",
        fda_class="I",
        cia_priority=CIAPriority(0.2, 0.7, 0.4),
        description="Temperature monitoring — low criticality",
        detection_percentile=70.0,  # Conservative: low clinical impact
    ),
    "generic_iomt_sensor": DeviceProfile(
        device_type="generic_iomt_sensor",
        fda_class="II",
        cia_priority=CIAPriority(0.4, 0.8, 0.7),
        description="Default IoMT sensor profile",
        detection_percentile=50.0,  # Default: moderate sensitivity
    ),
}


class DeviceRegistry(BaseDetector):
    """Registry mapping device_id to DeviceProfile.

    Args:
        devices: Dict of device_id → DeviceProfile.
            If None, uses default profiles.
    """

    def __init__(self, devices: Dict[str, DeviceProfile] | None = None) -> None:
        self._devices = devices or dict(_DEFAULT_DEVICES)

    def lookup(self, device_id: str) -> DeviceProfile:
        """Lookup device profile, returning generic default if unknown."""
        return self._devices.get(
            device_id,
            self._devices.get("generic_iomt_sensor", _DEFAULT_DEVICES["generic_iomt_sensor"]),
        )

    @classmethod
    def from_config(cls, entries: List[Dict[str, Any]]) -> DeviceRegistry:
        """Build from YAML config entries."""
        devices: Dict[str, DeviceProfile] = {}
        for e in entries:
            p = e["cia_priority"]
            devices[e["device_type"]] = DeviceProfile(
                device_type=e["device_type"],
                fda_class=e.get("fda_class", "II"),
                cia_priority=CIAPriority(
                    confidentiality=p["confidentiality"],
                    integrity=p["integrity"],
                    availability=p["availability"],
                ),
                description=e.get("description", ""),
                detection_percentile=float(e.get("detection_percentile", 93.0)),
            )
        # Ensure generic fallback exists
        if "generic_iomt_sensor" not in devices:
            devices["generic_iomt_sensor"] = _DEFAULT_DEVICES["generic_iomt_sensor"]
        return cls(devices)

    def get_config(self) -> Dict[str, Any]:
        return {
            "devices": {
                k: {
                    "fda_class": v.fda_class,
                    "C": v.cia_priority.confidentiality,
                    "I": v.cia_priority.integrity,
                    "A": v.cia_priority.availability,
                }
                for k, v in self._devices.items()
            },
        }
