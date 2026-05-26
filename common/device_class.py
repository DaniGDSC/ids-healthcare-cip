"""Heuristic per-row device-class derivation from biometric feature activity.

Mirrors ``module6_evaluation._derive_device_class`` so Phase 1 parquets and
Module 6 evaluation tag rows the same way. Operates on the post-scaling
feature matrix produced by Phase 1; absent biometric columns contribute zero.

Also exports the per-class clinical context map (``DEVICE_CONTEXT``) and a
single-row helper (``device_context_for_idx``) so Module 5 can derive the
same device classification + criticality the dashboard uses when building
its MVE generation inputs.
"""

from __future__ import annotations

from typing import Any, Iterable, List

import numpy as np
import pandas as pd

_BIO_FEATS = ("Temp", "SpO2", "Pulse_Rate", "Heart_rate", "Resp_Rate", "ST")
_ACTIVATION = 0.5


# Per-class clinical context — affected_system + patient_care_impact +
# device_criticality + active_device. Used by Module 5 (MVE generation)
# and Module 6 (evaluation alert curation, dashboard enrichment). The two
# pipelines MUST agree on this map; keeping it here avoids drift.
DEVICE_CONTEXT: dict[str, dict[str, Any]] = {
    "infusion_pump": {
        "affected_system": "Infusion pump — active drug delivery",
        "patient_care_impact": "Compromise could alter infusion parameters for active patients.",
        "device_criticality": "CRITICAL",
        "active_device": True,
    },
    "ventilator": {
        "affected_system": "Ventilator — active respiratory support",
        "patient_care_impact": "Device disruption directly affects patient breathing.",
        "device_criticality": "CRITICAL",
        "active_device": True,
    },
    "patient_monitor": {
        "affected_system": "Patient monitor — vital signs tracking",
        "patient_care_impact": "Isolation removes automated vital sign alerts for nursing staff.",
        "device_criticality": "HIGH",
        "active_device": True,
    },
    "ehr_workstation": {
        "affected_system": "EHR workstation — clinical documentation",
        "patient_care_impact": "Disruption affects active patient charting for floor nurses.",
        "device_criticality": "HIGH",
        "active_device": False,
    },
    "pacs_server": {
        "affected_system": "PACS server — diagnostic imaging",
        "patient_care_impact": "Disruption affects radiology reads and image delivery.",
        "device_criticality": "HIGH",
        "active_device": False,
    },
    "insulin_pump": {
        "affected_system": "Insulin pump — active drug delivery (mobile)",
        "patient_care_impact": "Compromise could alter insulin dosing. Hypo/hyperglycemia risk.",
        "device_criticality": "HIGH",
        "active_device": True,
    },
    "pharmacy_system": {
        "affected_system": "Pharmacy system — medication dispensing",
        "patient_care_impact": "Disruption affects automated drug dispensing for all patients.",
        "device_criticality": "HIGH",
        "active_device": False,
    },
    "server": {
        "affected_system": "Clinical server — infrastructure",
        "patient_care_impact": "Server disruption may cascade to dependent clinical systems.",
        "device_criticality": "MEDIUM",
        "active_device": False,
    },
    "other": {
        "affected_system": "Clinical network device",
        "patient_care_impact": "Impact depends on device function — verify with Biomed.",
        "device_criticality": "MEDIUM",
        "active_device": False,
    },
}


# Map attack category to a protocol/keyword hint so mve_generator._detect_alert_type
# can classify the alert into T1–T5. The pipeline doesn't carry real protocol
# strings per sample; this hint just steers the rule-based template chooser.
_ATTACK_TO_PROTO_HINT: dict[str, str] = {
    "Spoofing": "rogue_session",
    "Data Alteration": "modified_payload",
    "iomt_deviation": "behavioral_drift",
    "anomalous_outbound": "outbound_anomaly",
    "lateral_movement": "smb_rdp_445",
    "data_exfiltration": "large_transfer_https",
    "ehr_access": "ehr_query",
}


def derive_device_class_array(
    X: np.ndarray,
    feature_names: Iterable[str],
) -> List[str]:
    """Classify each row of *X* into a device class.

    Args:
        X: 2-D feature matrix (rows × features) after scaling.
        feature_names: Column names aligned with ``X``'s second axis.

    Returns:
        List of device-class labels, one per row. Labels are one of
        ``ventilator``, ``patient_monitor``, ``infusion_pump``,
        ``ehr_workstation``, ``other``.
    """
    names = list(feature_names)
    idx = {name: i for i, name in enumerate(names)}

    def col(name: str) -> np.ndarray:
        i = idx.get(name)
        if i is None:
            return np.zeros(len(X), dtype=float)
        return np.abs(X[:, i].astype(float))

    bio_cols = {name: col(name) > _ACTIVATION for name in _BIO_FEATS}
    bio_active = np.sum(np.stack(list(bio_cols.values()), axis=1), axis=1)

    sport = col("Sport")
    src_bytes = col("SrcBytes")

    out = np.full(len(X), "other", dtype=object)
    out[(bio_active <= 1) & ((sport > 0.1) | (src_bytes > 0.1))] = "ehr_workstation"
    out[bio_cols["Temp"] & (bio_active >= 2)] = "infusion_pump"
    out[bio_cols["Pulse_Rate"] & bio_cols["Heart_rate"] & (bio_active >= 3)] = (
        "patient_monitor"
    )
    out[bio_cols["Resp_Rate"] & bio_cols["SpO2"] & (bio_active >= 4)] = "ventilator"
    return out.tolist()


def derive_device_class_row(row: pd.Series) -> str:
    """Single-row equivalent of ``derive_device_class_array``.

    Mirrors the if/elif chain used by ``module6_evaluation._derive_device_class``
    so a Module 5 caller iterating rows produces identical labels to the
    Module 6 batch path.
    """
    vals = {f: abs(float(row.get(f, 0))) > _ACTIVATION for f in _BIO_FEATS}
    bio_active = sum(vals.values())

    if vals["Resp_Rate"] and vals["SpO2"] and bio_active >= 4:
        return "ventilator"
    if vals["Pulse_Rate"] and vals["Heart_rate"] and bio_active >= 3:
        return "patient_monitor"
    if vals["Temp"] and bio_active >= 2:
        return "infusion_pump"
    sport = abs(float(row.get("Sport", 0)))
    src = abs(float(row.get("SrcBytes", 0)))
    if bio_active <= 1 and (sport > 0.1 or src > 0.1):
        return "ehr_workstation"
    return "other"


def device_context_for_idx(idx: int, df: pd.DataFrame) -> dict[str, Any]:
    """Return DEVICE_CONTEXT entry for the device class derived from ``df.iloc[idx]``.

    The returned dict includes ``device_class`` (the derived class label)
    alongside the context fields (``affected_system``, ``patient_care_impact``,
    ``device_criticality``, ``active_device``), so downstream callers don't
    have to call ``derive_device_class_row`` separately.
    """
    cls = derive_device_class_row(df.iloc[idx])
    ctx = DEVICE_CONTEXT.get(cls, DEVICE_CONTEXT["other"])
    return {**ctx, "device_class": cls}


def synthesize_raw_alert(
    sample_index: int,
    attack_category: str,
    risk_score: float,
) -> dict[str, Any]:
    """Build a minimal raw_alert dict for ``mve_generator.generate_mve``.

    The offline pipeline does not carry per-sample IPs or timestamps. The
    rule-based MVE fallback handles missing fields; we only populate what
    ``_detect_alert_type`` needs to pick the right T1–T5 template:

    * ``alert_name`` — keyword-bearing string for T3/T4/T5 detection.
    * ``protocol``   — attack-category-derived hint for T3 (lateral movement).
    * ``severity_score`` — passed through from Module 3's composite R.

    Returns:
        Dict with alert_name, protocol, severity_score. Other fields omitted
        deliberately — fake values would mislead operators reading the
        explanation.
    """
    # mve_generator._confidence_level expects severity_score on a [0, 10]
    # scale (HIGH > 7.0, MEDIUM > 4.0). Module 3 risk_score is normalised
    # to [0, 1], so multiply by 10 to land in the same bucket the live
    # detection pipeline uses.
    return {
        "alert_name": f"{attack_category} anomaly (sample {sample_index})",
        "protocol": _ATTACK_TO_PROTO_HINT.get(attack_category, "unknown"),
        "severity_score": float(risk_score) * 10.0,
    }
