"""Shared device context and severity-derivation mappings.

Used by Module 3 (batch risk scoring) and Module 6 (evaluation).
This is the single source of truth — do not redefine these elsewhere.

Per RQ1_pipeline.md §3.1, this module unifies two formerly-duplicated
schemas:

  * Display fields (consumed by ``module6_evaluation.py`` for alert
    cards): ``affected_system``, ``patient_care_impact``,
    ``active_device``.
  * Risk-scoring fields (required by RQ1 pipeline §4 for the
    ``risk_scores.npz`` schema-v1.1 extension): ``patchable``,
    ``d_crit``.

``device_criticality`` is shared by both.
"""

from __future__ import annotations

from typing import Any, Dict

DEVICE_CONTEXT: Dict[str, Dict[str, Any]] = {
    "infusion_pump": {
        "affected_system": "Infusion pump — active drug delivery",
        "patient_care_impact": "Compromise could alter infusion parameters for active patients.",
        "device_criticality": "CRITICAL",
        "active_device": True,
        "patchable": False,
        "d_crit": 0.80,
    },
    "ventilator": {
        "affected_system": "Ventilator — active respiratory support",
        "patient_care_impact": "Device disruption directly affects patient breathing.",
        "device_criticality": "CRITICAL",
        "active_device": True,
        "patchable": False,
        "d_crit": 0.80,
    },
    "patient_monitor": {
        "affected_system": "Patient monitor — vital signs tracking",
        "patient_care_impact": "Isolation removes automated vital sign alerts for nursing staff.",
        "device_criticality": "HIGH",
        "active_device": True,
        "patchable": False,
        "d_crit": 0.72,
    },
    "ehr_workstation": {
        "affected_system": "EHR workstation — clinical documentation",
        "patient_care_impact": "Disruption affects active patient charting for floor nurses.",
        "device_criticality": "HIGH",
        "active_device": False,
        "patchable": True,
        "d_crit": 0.72,
    },
    "pacs_server": {
        "affected_system": "PACS server — diagnostic imaging",
        "patient_care_impact": "Disruption affects radiology reads and image delivery.",
        "device_criticality": "HIGH",
        "active_device": False,
        "patchable": False,
        "d_crit": 0.72,
    },
    "insulin_pump": {
        "affected_system": "Insulin pump — active drug delivery (mobile)",
        "patient_care_impact": "Compromise could alter insulin dosing. Hypo/hyperglycemia risk.",
        "device_criticality": "HIGH",
        "active_device": True,
        "patchable": False,
        "d_crit": 0.72,
    },
    "pharmacy_system": {
        "affected_system": "Pharmacy system — medication dispensing",
        "patient_care_impact": "Disruption affects automated drug dispensing for all patients.",
        "device_criticality": "HIGH",
        "active_device": False,
        "patchable": True,
        "d_crit": 0.72,
    },
    "server": {
        "affected_system": "Clinical server — infrastructure",
        "patient_care_impact": "Server disruption may cascade to dependent clinical systems.",
        "device_criticality": "MEDIUM",
        "active_device": False,
        "patchable": True,
        "d_crit": 0.40,
    },
    "other": {
        "affected_system": "Clinical network device",
        "patient_care_impact": "Impact depends on device function — verify with Biomed.",
        "device_criticality": "MEDIUM",
        "active_device": False,
        "patchable": True,
        "d_crit": 0.40,
    },
}

# Fallback for any unmapped device_class — DO NOT silently change this.
UNKNOWN_DEVICE_FALLBACK: Dict[str, Any] = DEVICE_CONTEXT["other"]

# Life-critical device classes (used by map_true_severity).
LIFE_CRITICAL_DEVICES = frozenset({"ventilator", "patient_monitor", "infusion_pump"})


def lookup_device_context(device_class: str) -> Dict[str, Any]:
    """Safe lookup with explicit fallback to 'other'."""
    return DEVICE_CONTEXT.get(device_class, UNKNOWN_DEVICE_FALLBACK)


def map_true_severity(attack_category: str, device_class: str) -> str:
    """Derive ground-truth severity from raw labels.

    Rule (matches the pre-refactor inline definition in
    ``module6_evaluation.py``):

      - ``"normal"``                                              → ``"LOW"``
      - {Data Alteration, Spoofing} on life-critical devices      → ``"CRITICAL"``
      - other attacks on life-critical devices                    → ``"HIGH"``
      - ``"Data Alteration"`` on other devices                    → ``"HIGH"``
      - ``"Spoofing"`` on other devices                           → ``"MEDIUM"``
      - everything else                                           → ``"MEDIUM"``

    Life-critical devices are: ventilator, patient_monitor, infusion_pump.
    """
    if attack_category == "normal":
        return "LOW"
    if device_class in LIFE_CRITICAL_DEVICES:
        if attack_category in {"Data Alteration", "Spoofing"}:
            return "CRITICAL"
        return "HIGH"
    if attack_category == "Data Alteration":
        return "HIGH"
    if attack_category == "Spoofing":
        return "MEDIUM"
    return "MEDIUM"
