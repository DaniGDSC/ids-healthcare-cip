"""CIA threat mapper — maps attack categories to CIA threat vectors.

Each attack type threatens Confidentiality, Integrity, and Availability
to different degrees. The mapper produces a normalised threat vector
that feeds into the CIA risk modifier.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, NamedTuple

from .base import BaseDetector

logger = logging.getLogger(__name__)


class CIAThreatVector(NamedTuple):
    """CIA impact weights for an attack type (each 0.0–1.0)."""

    confidentiality: float
    integrity: float
    availability: float


# Default threat mapping based on WUSTL-EHMS-2020 attack semantics
_DEFAULT_THREAT_MAP: Dict[str, CIAThreatVector] = {
    "Spoofing": CIAThreatVector(confidentiality=0.6, integrity=0.9, availability=0.3),
    "Data Alteration": CIAThreatVector(confidentiality=0.3, integrity=1.0, availability=0.2),
    "normal": CIAThreatVector(confidentiality=0.0, integrity=0.0, availability=0.0),
    "unknown": CIAThreatVector(confidentiality=0.5, integrity=0.5, availability=0.5),
}


class CIAThreatMapper(BaseDetector):
    """Map attack categories to CIA threat vectors.

    Args:
        threat_map: Dict mapping attack_category → CIAThreatVector.
            If None, uses defaults for WUSTL-EHMS-2020 attack types.
    """

    def __init__(self, threat_map: Dict[str, CIAThreatVector] | None = None) -> None:
        self._map = threat_map or dict(_DEFAULT_THREAT_MAP)

    def map(self, attack_category: str) -> CIAThreatVector:
        """Return CIA threat vector for an attack category.

        Falls back to 'unknown' vector if category not in map.
        """
        return self._map.get(attack_category, self._map.get("unknown", _DEFAULT_THREAT_MAP["unknown"]))

    @classmethod
    def from_config(cls, entries: List[Dict[str, Any]]) -> CIAThreatMapper:
        """Build from YAML config entries.

        Each entry: {"attack_category": str, "cia_weights": {"confidentiality": f, "integrity": f, "availability": f}}
        """
        threat_map: Dict[str, CIAThreatVector] = {}
        for e in entries:
            w = e["cia_weights"]
            threat_map[e["attack_category"]] = CIAThreatVector(
                confidentiality=w["confidentiality"],
                integrity=w["integrity"],
                availability=w["availability"],
            )
        # Ensure fallbacks exist
        if "normal" not in threat_map:
            threat_map["normal"] = _DEFAULT_THREAT_MAP["normal"]
        if "unknown" not in threat_map:
            threat_map["unknown"] = _DEFAULT_THREAT_MAP["unknown"]
        return cls(threat_map)

    def get_config(self) -> Dict[str, Any]:
        return {
            "threat_mappings": {
                k: {"C": v.confidentiality, "I": v.integrity, "A": v.availability}
                for k, v in self._map.items()
            },
        }
