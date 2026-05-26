"""Module 4 — Generate Explanations (RQ1/RO1).

Public API surface — re-exports from the modular sub-files. Both CLI
entry points (``module4_explanations.py`` for offline batch +
``module4_online_explainer.py`` for online per-alert) carry their own
back-compat re-exports too.
"""

from .compute import (
    compute_dae_feature_errors,
    compute_global_importance,
    compute_tree_shap,
)
from .config import (
    BIOMETRIC_FEATURES,
    CLINICIAN_TEMPLATES,
    FEATURE_CONCEPTS,
    NLG_TEMPLATES,
    SHAP_MODELS,
    TOP_K_FEATURES,
    TOP_N_WATERFALL,
    TRACK_A_MODELS,
)
from .example_explanations import generate_example_explanations
from .io import (
    NumpyJSONEncoder,
    export_feature_concepts,
    export_nlg_templates,
    load_predictions,
    load_test_data,
    save_dae_errors,
    save_global_importance,
    save_shap_values,
    write_json_strict,
)
from .nlg import (
    build_shap_context,
    clinician_nlg,
    generate_clinician_alert,
    route_explanation,
)
from .online_explainer import AlertExplainer
from .stakeholder import (
    build_admin_dashboard,
    build_analyst_report,
    build_clinician_summaries,
)
from .validation import (
    validate_consistency,
    validate_cross_model,
    validate_perturbation,
)

__all__ = [
    "TRACK_A_MODELS", "SHAP_MODELS", "BIOMETRIC_FEATURES",
    "CLINICIAN_TEMPLATES", "NLG_TEMPLATES", "FEATURE_CONCEPTS",
    "TOP_K_FEATURES", "TOP_N_WATERFALL",
    "compute_tree_shap", "compute_dae_feature_errors",
    "compute_global_importance",
    "build_analyst_report", "build_clinician_summaries",
    "build_admin_dashboard",
    "clinician_nlg", "build_shap_context",
    "generate_clinician_alert", "route_explanation",
    "validate_consistency", "validate_perturbation", "validate_cross_model",
    "load_test_data", "load_predictions",
    "save_shap_values", "save_global_importance", "save_dae_errors",
    "export_feature_concepts", "export_nlg_templates",
    "write_json_strict", "NumpyJSONEncoder",
    "generate_example_explanations",
    "AlertExplainer",
]
