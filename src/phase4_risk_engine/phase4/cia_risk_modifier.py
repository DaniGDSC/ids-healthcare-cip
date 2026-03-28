"""CIA risk modifier — adaptive priority shifting by operational scenario.

Implements three operational scenarios with different CIA priority rankings:

  Normal Monitoring:   I >= C > A  (data privacy + diagnostic accuracy)
  Clinical Emergency:  A > I > C   (immediate access vital; delays fatal)
  High Threat Level:   I > A > C   (prevent tampering / malicious control)

The scenario is determined automatically from the current risk level
and attack characteristics, then the CIA priorities are reweighted
accordingly before computing the final risk adjustment.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, NamedTuple

from .cia_threat_mapper import CIAThreatMapper, CIAThreatVector
from .device_registry import DeviceProfile, DeviceRegistry
from .risk_level import RiskLevel

logger = logging.getLogger(__name__)


class Scenario(str, Enum):
    """Operational scenario for adaptive CIA priority shifting."""

    NORMAL_MONITORING = "normal_monitoring"
    CLINICAL_EMERGENCY = "clinical_emergency"
    HIGH_THREAT = "high_threat"


# Scenario-specific CIA priority multipliers.
# These reweight device CIA priorities based on operational context.
_SCENARIO_WEIGHTS: Dict[Scenario, CIAThreatVector] = {
    # I >= C > A — emphasis on data privacy and diagnostic accuracy
    Scenario.NORMAL_MONITORING: CIAThreatVector(
        confidentiality=0.9,
        integrity=1.0,
        availability=0.6,
    ),
    # A > I > C — immediate access vital; delays can be fatal
    Scenario.CLINICAL_EMERGENCY: CIAThreatVector(
        confidentiality=0.3,
        integrity=0.7,
        availability=1.0,
    ),
    # I > A > C — focus on preventing data tampering or malicious device control
    Scenario.HIGH_THREAT: CIAThreatVector(
        confidentiality=0.4,
        integrity=1.0,
        availability=0.8,
    ),
}


class CIARiskAssessment(NamedTuple):
    """Per-sample CIA risk assessment result."""

    base_risk_level: RiskLevel
    adjusted_risk_level: RiskLevel
    scenario: Scenario
    cia_scores: Dict[str, float]
    cia_max_dimension: str
    cia_modifier: float
    attack_category: str
    device_type: str


def _determine_scenario(
    base_risk: RiskLevel,
    attention_flag: bool,
) -> Scenario:
    """Determine operational scenario from current risk indicators.

    Args:
        base_risk: Risk level from the classification-based scorer.
        attention_flag: True if attention anomaly detector flagged this sample.

    Returns:
        The active operational scenario.
    """
    # High Threat: attention anomaly (potential novel/zero-day) OR HIGH/CRITICAL risk
    if attention_flag or base_risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        return Scenario.HIGH_THREAT

    # Clinical Emergency: MEDIUM risk — device may be under active attack,
    # prioritise availability to avoid interrupting patient care
    if base_risk == RiskLevel.MEDIUM:
        return Scenario.CLINICAL_EMERGENCY

    # Normal Monitoring: NORMAL or LOW risk
    return Scenario.NORMAL_MONITORING


def _escalate(level: RiskLevel) -> RiskLevel:
    """Escalate risk by one level (capped at CRITICAL)."""
    order = [RiskLevel.NORMAL, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    idx = order.index(level)
    return order[min(idx + 1, len(order) - 1)]


class CIARiskModifier:
    """Apply CIA-weighted risk adjustment with adaptive priority shifting.

    Combines three inputs:
      1. Attack threat vector (which CIA properties the attack threatens)
      2. Device CIA priorities (which CIA properties matter for this device)
      3. Operational scenario (reweights priorities by clinical context)

    CIA can only ESCALATE risk, never reduce it — safety-first for
    medical devices.

    Args:
        threat_mapper: Maps attack categories to CIA threat vectors.
        device_registry: Maps device IDs to profiles with CIA priorities.
        escalation_threshold: CIA modifier above which risk is escalated.
    """

    def __init__(
        self,
        threat_mapper: CIAThreatMapper,
        device_registry: DeviceRegistry,
        escalation_threshold: float = 0.7,
    ) -> None:
        self._mapper = threat_mapper
        self._registry = device_registry
        self._threshold = escalation_threshold

    def modify(
        self,
        base_risk: RiskLevel,
        attack_category: str,
        device_id: str,
        attention_flag: bool = False,
    ) -> CIARiskAssessment:
        """Compute CIA-weighted risk for a single sample.

        Args:
            base_risk: Risk level from classification-based scorer.
            attack_category: Attack type (e.g. "Spoofing", "Data Alteration", "unknown").
            device_id: Device identifier for registry lookup.
            attention_flag: Whether attention anomaly detector flagged this sample.

        Returns:
            CIARiskAssessment with base and adjusted risk levels.
        """
        threat = self._mapper.map(attack_category)
        device = self._registry.lookup(device_id)
        scenario = _determine_scenario(base_risk, attention_flag)
        scenario_weights = _SCENARIO_WEIGHTS[scenario]

        # Compute per-dimension CIA score:
        #   score[d] = threat[d] × device_priority[d] × scenario_weight[d]
        cia_scores = {
            "C": threat.confidentiality * device.cia_priority.confidentiality * scenario_weights.confidentiality,
            "I": threat.integrity * device.cia_priority.integrity * scenario_weights.integrity,
            "A": threat.availability * device.cia_priority.availability * scenario_weights.availability,
        }

        cia_modifier = max(cia_scores.values())
        cia_max_dim = max(cia_scores, key=cia_scores.get)  # type: ignore[arg-type]

        # CIA can only escalate, never reduce
        if cia_modifier >= self._threshold and base_risk != RiskLevel.CRITICAL:
            adjusted = _escalate(base_risk)
        else:
            adjusted = base_risk

        return CIARiskAssessment(
            base_risk_level=base_risk,
            adjusted_risk_level=adjusted,
            scenario=scenario,
            cia_scores={k: round(v, 4) for k, v in cia_scores.items()},
            cia_max_dimension=cia_max_dim,
            cia_modifier=round(cia_modifier, 4),
            attack_category=attack_category,
            device_type=device.device_type,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "escalation_threshold": self._threshold,
            "threat_mapper": self._mapper.get_config(),
            "device_registry": self._registry.get_config(),
            "scenarios": {
                s.value: {"C": w.confidentiality, "I": w.integrity, "A": w.availability}
                for s, w in _SCENARIO_WEIGHTS.items()
            },
        }
