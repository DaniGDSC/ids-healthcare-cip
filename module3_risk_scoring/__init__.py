"""Module 3 — Composite Risk Scoring.

Public API
----------
Constants:
    WEIGHTS, RISK_THRESHOLDS, BIOMETRIC_FEATURES, SIGMA_THRESHOLD,
    CIA_THREATS, DEVICE_TIERS, DATA_SENSITIVITY, RESPONSE_MAPPING

Compute:
    compute_d_crit, compute_s_data, compute_d_clinical_tier,
    compute_composite_risk, assign_risk_levels

Feedback:
    apply_feedback, apply_weight_feedback

Analysis:
    dual_track_fusion_analysis, component_contribution_analysis,
    weight_sensitivity_analysis, generate_worked_examples

I/O:
    load_test_data, load_xgboost_proba, save_outputs, export_config_jsons

CLI entry point: ``python -m module3_risk_scoring.module3_risk_scores``
"""

from .analysis import (
    component_contribution_analysis,
    dual_track_fusion_analysis,
    generate_worked_examples,
    weight_sensitivity_analysis,
)
from .components import (
    compute_d_clinical_tier,
    compute_d_crit,
    compute_s_data,
)
from .composition import (
    assign_risk_levels,
    compute_composite_risk,
)
from .config import (
    BIOMETRIC_FEATURES,
    CIA_THREATS,
    DAE_BINARY_THRESHOLD,
    DATA_SENSITIVITY,
    DEFAULT_DEVICE_TIER,
    DEVICE_TIERS,
    RESPONSE_MAPPING,
    RISK_THRESHOLDS,
    SIGMA_THRESHOLD,
    WEIGHTS,
)
from .feedback import apply_feedback, apply_weight_feedback
from .io import (
    export_config_jsons,
    load_test_data,
    load_xgboost_proba,
    save_outputs,
)

__all__ = [
    # Constants
    "WEIGHTS", "RISK_THRESHOLDS", "BIOMETRIC_FEATURES", "SIGMA_THRESHOLD",
    "DAE_BINARY_THRESHOLD", "CIA_THREATS", "DEVICE_TIERS",
    "DEFAULT_DEVICE_TIER", "DATA_SENSITIVITY", "RESPONSE_MAPPING",
    # Compute
    "compute_d_crit", "compute_s_data", "compute_d_clinical_tier",
    "compute_composite_risk", "assign_risk_levels",
    # Feedback
    "apply_feedback", "apply_weight_feedback",
    # Analysis
    "dual_track_fusion_analysis", "component_contribution_analysis",
    "weight_sensitivity_analysis", "generate_worked_examples",
    # I/O
    "load_test_data", "load_xgboost_proba",
    "save_outputs", "export_config_jsons",
]
