"""Module 4 configuration constants.

Single source of truth for templates, feature concepts, and model
configuration. Both the offline batch path
(``module4_explanations.py``) and the online per-alert path
(``module4_online_explainer.py``) consume the same constants from here
so they cannot drift apart.
"""

from __future__ import annotations

from common.phi import BIOMETRIC_COLUMNS

# Stable name for downstream callers. Use BIOMETRIC_COLUMNS directly
# from common.phi where possible; alias kept for backward-compat with
# pre-cleanup code that imported BIOMETRIC_FEATURES.
BIOMETRIC_FEATURES = BIOMETRIC_COLUMNS

TOP_N_WATERFALL: int = 5
TOP_K_FEATURES: int = 10


# ── Track A model registry ─────────────────────────────────────────────
#
# Runtime split: Module 4 (and downstream M5/M6) consume only XGBoost +
# DAE. RandomForest and DecisionTree are RQ1 R2 baselines whose
# predictions live on disk in ``results/models/{rf,dt}_test_predictions.npz``
# but are NOT loaded by the analyst/clinician/admin builders — only by
# ``tools/rq1_compute_metrics.compute_track_a_ablation``.

TRACK_A_MODELS: dict[str, dict[str, str]] = {
    "xgboost": {
        "pipeline": "results/models/xgboost_final_pipeline.pkl",
        "predictions": "results/models/xgboost_test_predictions.npz",
        "report": "results/models/xgboost_final_report.json",
    },
}

BASELINE_TRACK_A_MODELS: dict[str, dict[str, str]] = {
    "random_forest": {
        "pipeline": "results/models/random_forest_final_pipeline.pkl",
        "predictions": "results/models/random_forest_test_predictions.npz",
        "report": "results/models/random_forest_final_report.json",
    },
    "decision_tree": {
        "pipeline": "results/models/decision_tree_final_pipeline.pkl",
        "predictions": "results/models/decision_tree_test_predictions.npz",
        "report": "results/models/decision_tree_final_report.json",
    },
}

# Models that produce TreeSHAP outputs (xgboost-only for the thesis path).
# Offline + online both consult this single constant — previously the
# online explainer carried its own SKIP_SHAP frozenset that drifted.
SHAP_MODELS: tuple[str, ...] = ("xgboost",)

# Inverse view used by online path for predict-but-not-explain models.
# Empty after Phase 2 — RF/DT no longer participate in the runtime vote.
SKIP_SHAP_MODELS: frozenset[str] = frozenset(
    set(TRACK_A_MODELS.keys()) - set(SHAP_MODELS)
)


# ── Clinician NLG templates (single canonical version) ───────────────
#
# All templates carry the same placeholder set so the offline batch and
# online per-alert paths can fill them with the same data. ``idx`` is
# always optional — pass ``idx=""`` to suppress the sample reference.

CLINICIAN_TEMPLATES: dict[str, str] = {
    "CRITICAL": (
        "CRITICAL ALERT{idx}: The system detected a likely intrusion "
        "affecting this patient's monitoring session. The primary indicator was "
        "{narrative}. "
        "{secondary_note}"
        "Recommend immediate review of device connectivity and patient vitals."
    ),
    "HIGH": (
        "HIGH ALERT{idx}: Suspicious activity detected. "
        "Key factor: {narrative}. "
        "{secondary_note}"
        "Consider verifying device integrity."
    ),
    "MEDIUM": (
        "MODERATE ALERT{idx}: Minor anomaly detected — "
        "{narrative}. "
        "{secondary_note}"
        "No immediate clinical action required, but flagged for review."
    ),
    "LOW": (
        "LOW ALERT{idx}: Borderline detection by one model. "
        "Likely benign; logged for audit purposes."
    ),
}


def format_clinician_template(
    severity: str,
    *,
    sample_index: int | None = None,
    narrative: str = "",
    secondary_note: str = "",
) -> str:
    """Render the canonical clinician template.

    Args:
        severity: CRITICAL / HIGH / MEDIUM / LOW.
        sample_index: optional sample index. When provided, formats as
            ``" (Sample N)"``; when ``None`` it expands to an empty
            string so the template stays grammatical.
        narrative: short clinical narrative for the primary indicator.
        secondary_note: optional follow-up clause (e.g. for the
            confidence-band biometric hint).
    """
    idx_str = f" (Sample {sample_index})" if sample_index is not None else ""
    return CLINICIAN_TEMPLATES[severity].format(
        idx=idx_str,
        narrative=narrative,
        secondary_note=secondary_note,
    )


# ── 6-step NLG component library (admin / analyst / clinician) ───────

NLG_TEMPLATES: dict[str, object] = {
    "severity_header": {
        "CRITICAL": "CRITICAL SECURITY ALERT — Immediate action required.",
        "HIGH": "HIGH PRIORITY ALERT — Active investigation needed.",
        "MEDIUM": "MODERATE ALERT — Flagged for review.",
        "LOW": "LOW PRIORITY — Logged for audit.",
    },
    "detection_sentence": (
        "The intrusion detection system flagged this network flow with "
        "{consensus} using a confidence of {confidence:.0%}."
    ),
    "feature_explanation_network": (
        "{label} showed {direction} (SHAP contribution: {shap_value:+.3f})."
    ),
    "feature_explanation_biometric": (
        "Patient {label} showed {direction} "
        "(SHAP contribution: {shap_value:+.3f}). "
        "Clinical review of this vital sign is recommended."
    ),
    "risk_context": (
        "Composite risk score: {risk_score:.2f} ({risk_level}). "
        "Components — detection confidence: {c_detect:.2f}, "
        "device criticality: {d_crit:.2f}, "
        "data sensitivity: {s_data:.2f}, "
        "clinical tier: {d_clinical_tier:.2f}."
    ),
    "acuity_note_normal": "Patient vitals are within normal ranges.",
    "acuity_note_abnormal": (
        "Note: {n_abnormal} of 8 biometric readings are outside normal range. "
        "Verify patient condition independently."
    ),
    "action_recommendation": {
        "CRITICAL": (
            "Recommended: Isolate device immediately. Page on-call physician "
            "and CISO. Initiate incident response."
        ),
        "HIGH": (
            "Recommended: Isolate network segment. Notify SOC and biomedical "
            "engineering."
        ),
        "MEDIUM": (
            "Recommended: Enable enhanced monitoring. Queue for security team "
            "review."
        ),
        "LOW": "Recommended: No immediate action. Review at next security audit.",
    },
}


# ── Per-feature concept taxonomy (offline thesis / dashboard view) ──

FEATURE_CONCEPTS: dict[str, dict[str, str]] = {
    "Flgs": {
        "label": "TCP Flag Pattern",
        "category": "network",
        "direction_high": "unusual flag combination detected",
        "direction_low": "normal flag pattern",
    },
    "Sport": {
        "label": "Source Port",
        "category": "network",
        "direction_high": "abnormal source port used",
        "direction_low": "standard port range",
    },
    "SrcBytes": {
        "label": "Outbound Byte Volume",
        "category": "network",
        "direction_high": "unusually high outbound data volume",
        "direction_low": "minimal outbound traffic",
    },
    "DstBytes": {
        "label": "Inbound Byte Volume",
        "category": "network",
        "direction_high": "unusually high inbound data volume",
        "direction_low": "minimal inbound traffic",
    },
    "SrcLoad": {
        "label": "Source Load",
        "category": "network",
        "direction_high": "high source bandwidth utilization",
        "direction_low": "normal source load",
    },
    "DstLoad": {
        "label": "Destination Load",
        "category": "network",
        "direction_high": "high destination bandwidth utilization",
        "direction_low": "normal destination load",
    },
    "SIntPkt": {
        "label": "Source Inter-Packet Gap",
        "category": "network",
        "direction_high": "abnormal packet timing (slow)",
        "direction_low": "rapid packet bursts",
    },
    "DIntPkt": {
        "label": "Dest Inter-Packet Gap",
        "category": "network",
        "direction_high": "abnormal response timing",
        "direction_low": "unusually fast responses",
    },
    "SIntPktAct": {
        "label": "Active Inter-Packet Time",
        "category": "network",
        "direction_high": "extended active session timing",
        "direction_low": "brief session activity",
    },
    "sMaxPktSz": {
        "label": "Max Source Packet Size",
        "category": "network",
        "direction_high": "large packets sent",
        "direction_low": "small packet sizes",
    },
    "dMaxPktSz": {
        "label": "Max Dest Packet Size",
        "category": "network",
        "direction_high": "large packets received",
        "direction_low": "small packet sizes",
    },
    "sMinPktSz": {
        "label": "Min Source Packet Size",
        "category": "network",
        "direction_high": "varying source packet sizes",
        "direction_low": "consistent small packets",
    },
    "Dur": {
        "label": "Flow Duration",
        "category": "network",
        "direction_high": "prolonged connection duration",
        "direction_low": "unusually brief connection",
    },
    "TotBytes": {
        "label": "Total Byte Volume",
        "category": "network",
        "direction_high": "high total data transfer",
        "direction_low": "minimal data transferred",
    },
    "Load": {
        "label": "Network Load",
        "category": "network",
        "direction_high": "high network utilization",
        "direction_low": "low network activity",
    },
    "pSrcLoss": {
        "label": "Source Packet Loss",
        "category": "network",
        "direction_high": "significant packet loss from source",
        "direction_low": "no source packet loss",
    },
    "pDstLoss": {
        "label": "Dest Packet Loss",
        "category": "network",
        "direction_high": "significant packet loss at destination",
        "direction_low": "no destination packet loss",
    },
    "Temp": {
        "label": "Body Temperature",
        "category": "biometric",
        "direction_high": "elevated temperature reading",
        "direction_low": "below-normal temperature",
    },
    "SpO2": {
        "label": "Blood Oxygen Saturation",
        "category": "biometric",
        "direction_high": "normal SpO2",
        "direction_low": "dangerously low oxygen saturation",
    },
    "Pulse_Rate": {
        "label": "Pulse Rate",
        "category": "biometric",
        "direction_high": "elevated heart rate (tachycardia)",
        "direction_low": "low heart rate (bradycardia)",
    },
    "SYS": {
        "label": "Systolic Blood Pressure",
        "category": "biometric",
        "direction_high": "elevated systolic BP (hypertension)",
        "direction_low": "low systolic BP (hypotension)",
    },
    "DIA": {
        "label": "Diastolic Blood Pressure",
        "category": "biometric",
        "direction_high": "elevated diastolic BP",
        "direction_low": "low diastolic BP",
    },
    "Heart_rate": {
        "label": "Heart Rate",
        "category": "biometric",
        "direction_high": "elevated heart rate",
        "direction_low": "low heart rate",
    },
    "Resp_Rate": {
        "label": "Respiratory Rate",
        "category": "biometric",
        "direction_high": "rapid breathing (tachypnea)",
        "direction_low": "slow breathing (bradypnea)",
    },
    "ST": {
        "label": "ST Segment (ECG)",
        "category": "biometric",
        "direction_high": "ST elevation (possible cardiac event)",
        "direction_low": "ST depression",
    },
}


__all__ = [
    "BIOMETRIC_FEATURES",
    "TOP_N_WATERFALL",
    "TOP_K_FEATURES",
    "TRACK_A_MODELS",
    "BASELINE_TRACK_A_MODELS",
    "SHAP_MODELS",
    "SKIP_SHAP_MODELS",
    "CLINICIAN_TEMPLATES",
    "format_clinician_template",
    "NLG_TEMPLATES",
    "FEATURE_CONCEPTS",
]
