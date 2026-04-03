"""Clinical impact assessor — translates technical risk into clinical actions.

Bridges the gap between statistical risk scores (MAD-relative distances)
and clinically actionable outputs with patient safety context.

Maps (risk_level, device_type, biometric_status) → ClinicalAssessment
containing:
  - Clinical severity (1–5 scale aligned with hospital triage)
  - Required response actions (who to notify, what to do)
  - Maximum acceptable response time
  - Patient safety flags
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np

from .base import BaseDetector
from .risk_level import RiskLevel

logger = logging.getLogger(__name__)


class ClinicalSeverity(IntEnum):
    """Clinical severity aligned with hospital triage codes.

    1 = routine monitoring, 5 = immediate life threat.
    """

    ROUTINE = 1
    ADVISORY = 2
    URGENT = 3
    EMERGENT = 4
    CRITICAL = 5


class ResponseProtocol(NamedTuple):
    """Clinical response protocol for a given severity."""

    severity: ClinicalSeverity
    description: str
    response_time_minutes: int
    actions: List[str]
    notify: List[str]
    device_action: str


# Response protocols by clinical severity
_PROTOCOLS: Dict[ClinicalSeverity, ResponseProtocol] = {
    ClinicalSeverity.ROUTINE: ResponseProtocol(
        severity=ClinicalSeverity.ROUTINE,
        description="Normal operation — log for audit",
        response_time_minutes=0,
        actions=["Log event to audit trail"],
        notify=[],
        device_action="none",
    ),
    ClinicalSeverity.ADVISORY: ResponseProtocol(
        severity=ClinicalSeverity.ADVISORY,
        description="Minor anomaly — review at next shift",
        response_time_minutes=480,
        actions=["Log event", "Flag for biomedical engineering review"],
        notify=["biomed_engineer"],
        device_action="none",
    ),
    ClinicalSeverity.URGENT: ResponseProtocol(
        severity=ClinicalSeverity.URGENT,
        description="Confirmed anomaly — active investigation required",
        response_time_minutes=60,
        actions=[
            "Log event with full context",
            "Restrict device to essential traffic only",
            "Initiate network forensics",
        ],
        notify=["biomed_engineer", "it_security"],
        device_action="restrict_network",
    ),
    ClinicalSeverity.EMERGENT: ResponseProtocol(
        severity=ClinicalSeverity.EMERGENT,
        description="Active threat to device integrity — immediate response",
        response_time_minutes=15,
        actions=[
            "Log event with full context",
            "Isolate device from network",
            "Switch to manual/backup monitoring",
            "Verify patient vitals independently",
        ],
        notify=["biomed_engineer", "it_security", "charge_nurse"],
        device_action="isolate_network",
    ),
    ClinicalSeverity.CRITICAL: ResponseProtocol(
        severity=ClinicalSeverity.CRITICAL,
        description="Immediate patient safety threat — do NOT power off device",
        response_time_minutes=5,
        actions=[
            "Log event with full context",
            "Isolate device from network immediately",
            "Switch to manual/backup monitoring",
            "Verify patient vitals independently",
            "Page on-call physician",
            "Activate incident response team",
        ],
        notify=["biomed_engineer", "it_security", "charge_nurse", "on_call_physician", "incident_commander"],
        device_action="isolate_network",
    ),
}


class BiometricStatus(NamedTuple):
    """Summary of biometric feature status for a sample."""

    any_abnormal: bool
    abnormal_features: List[str]
    max_deviation: float
    clinical_notes: List[str] = []


class ClinicalAssessment(NamedTuple):
    """Complete clinical impact assessment for a sample."""

    clinical_severity: ClinicalSeverity
    protocol: ResponseProtocol
    risk_level: RiskLevel
    device_type: str
    biometric_status: BiometricStatus
    patient_safety_flag: bool
    rationale: str


# Biometric clinical thresholds (post-RobustScaler, approximate z-scores).
# The warn/critical values are z-score thresholds against scaled features.
# clinical_note provides absolute clinical context for hospital staff.
_BIOMETRIC_CLINICAL_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "SpO2": {"warn": 1.5, "critical": 2.5, "label": "Blood oxygen",
             "clinical_note": "SpO2 < 92% requires clinical assessment"},
    "Heart_rate": {"warn": 1.5, "critical": 2.5, "label": "Heart rate",
                   "clinical_note": "HR > 120 or < 50 bpm requires assessment"},
    "Pulse_Rate": {"warn": 1.5, "critical": 2.5, "label": "Pulse rate",
                   "clinical_note": "PR > 120 or < 50 bpm requires assessment"},
    "Resp_Rate": {"warn": 1.5, "critical": 2.5, "label": "Respiratory rate",
                  "clinical_note": "RR > 25 or < 10/min requires assessment"},
    "SYS": {"warn": 1.5, "critical": 2.5, "label": "Systolic BP",
            "clinical_note": "SBP > 180 or < 90 mmHg requires assessment"},
    "DIA": {"warn": 1.5, "critical": 2.5, "label": "Diastolic BP",
            "clinical_note": "DBP > 110 or < 60 mmHg requires assessment"},
    "Temp": {"warn": 1.5, "critical": 2.5, "label": "Temperature",
             "clinical_note": "Temp > 38.5C or < 35C requires assessment"},
    "ST": {"warn": 1.5, "critical": 2.5, "label": "ST segment",
           "clinical_note": "ST elevation > 1mm requires cardiology consult"},
}

# Device types where biometric abnormality + network anomaly = patient safety flag
_SAFETY_CRITICAL_DEVICES = {"infusion_pump", "ecg_monitor", "pulse_oximeter"}

# Risk level → base clinical severity mapping
_BASE_SEVERITY: Dict[RiskLevel, ClinicalSeverity] = {
    RiskLevel.NORMAL: ClinicalSeverity.ROUTINE,
    RiskLevel.LOW: ClinicalSeverity.ADVISORY,
    RiskLevel.MEDIUM: ClinicalSeverity.URGENT,
    RiskLevel.HIGH: ClinicalSeverity.EMERGENT,
    RiskLevel.CRITICAL: ClinicalSeverity.CRITICAL,
}


class ClinicalImpactAssessor(BaseDetector):
    """Translate technical risk into clinical impact with response protocols.

    Combines:
      1. Technical risk level (from RiskScorer + CIA modifier)
      2. Device clinical context (safety-critical vs routine)
      3. Biometric status (are patient vitals abnormal?)

    Produces a ClinicalAssessment with severity, response protocol,
    and patient safety flags.

    Args:
        biometric_columns: List of biometric feature column names.
        biometric_thresholds: Per-feature clinical thresholds (z-score).
        safety_critical_devices: Device types that trigger safety escalation.
    """

    def __init__(
        self,
        biometric_columns: List[str],
        biometric_thresholds: Dict[str, Dict[str, float]] | None = None,
        safety_critical_devices: set[str] | None = None,
    ) -> None:
        self._bio_cols = biometric_columns
        self._bio_thresholds = biometric_thresholds or _BIOMETRIC_CLINICAL_THRESHOLDS
        self._safety_devices = safety_critical_devices or _SAFETY_CRITICAL_DEVICES

    def assess(
        self,
        risk_level: RiskLevel,
        device_type: str,
        feature_values: np.ndarray | None,
        feature_names: List[str] | None,
        attention_flag: bool = False,
    ) -> ClinicalAssessment:
        """Produce clinical impact assessment for a single sample.

        Args:
            risk_level: Final risk level (after CIA modifier).
            device_type: Device type from registry.
            feature_values: Raw feature values, shape (F,).
            feature_names: Feature names matching feature_values.
            attention_flag: True if attention anomaly detector flagged this.

        Returns:
            ClinicalAssessment with severity, protocol, and safety flags.
        """
        # 1. Check biometric status
        bio_status = self._assess_biometrics(feature_values, feature_names)

        # 2. Start with base severity from risk level
        severity = _BASE_SEVERITY[risk_level]

        # 3. Escalate if safety-critical device + biometric abnormality
        is_safety_device = device_type in self._safety_devices
        patient_safety_flag = False
        rationale = f"Base: {risk_level.value} on {device_type}"

        if is_safety_device and bio_status.any_abnormal and risk_level != RiskLevel.NORMAL:
            # Biometric abnormality on safety-critical device during anomaly
            # → escalate by one severity level
            if severity < ClinicalSeverity.CRITICAL:
                severity = ClinicalSeverity(min(severity + 1, ClinicalSeverity.CRITICAL))
                rationale += f" | Escalated: abnormal vitals ({', '.join(bio_status.abnormal_features)}) on safety-critical device"

        if is_safety_device and risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            patient_safety_flag = True
            rationale += " | PATIENT SAFETY FLAG"

        if attention_flag and is_safety_device:
            patient_safety_flag = True
            rationale += " | Novel threat on safety-critical device"

        protocol = _PROTOCOLS[severity]

        return ClinicalAssessment(
            clinical_severity=severity,
            protocol=protocol,
            risk_level=risk_level,
            device_type=device_type,
            biometric_status=bio_status,
            patient_safety_flag=patient_safety_flag,
            rationale=rationale,
        )

    def _assess_biometrics(
        self,
        feature_values: np.ndarray | None,
        feature_names: List[str] | None,
    ) -> BiometricStatus:
        """Check biometric features against clinical thresholds."""
        if feature_values is None or feature_names is None:
            return BiometricStatus(any_abnormal=False, abnormal_features=[], max_deviation=0.0)

        abnormal: List[str] = []
        notes: List[str] = []
        max_dev = 0.0

        for col in self._bio_cols:
            if col not in feature_names:
                continue
            idx = feature_names.index(col)
            val = abs(float(feature_values[idx]))
            if not np.isfinite(val):
                continue
            thresholds = self._bio_thresholds.get(col, {"warn": 1.5})
            warn = thresholds.get("warn", 1.5)

            if val > warn:
                label = thresholds.get("label", col)
                abnormal.append(label)
                note = thresholds.get("clinical_note", "")
                if note:
                    notes.append(note)
            if val > max_dev:
                max_dev = val

        return BiometricStatus(
            any_abnormal=len(abnormal) > 0,
            abnormal_features=abnormal,
            max_deviation=round(max_dev, 4),
            clinical_notes=notes,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "biometric_columns": self._bio_cols,
            "safety_critical_devices": sorted(self._safety_devices),
            "biometric_thresholds": {
                k: v.get("warn", 1.5) for k, v in self._bio_thresholds.items()
            },
        }
