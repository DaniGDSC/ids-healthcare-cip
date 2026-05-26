"""Module 6 — Evaluation Interface + Build (RQ2/RO2).

Public API surface — re-exports from the decomposed sub-modules.
Streamlit-dependent pieces (``module6_app``) are NOT eagerly imported so
``import module6_evaluation`` is side-effect-free and test-friendly.
"""
from __future__ import annotations

from .alerts import (
    _build_eval_alert,
    _curate_split_paths,
    _derive_device_class,
    _ground_truth_action,
    curate_evaluation_alerts,
)
from .constants import (
    ACTIONS,
    PAGE_SPLIT,
    ROLE_DISPLAY_LIST,
    ROLE_DISPLAY_NAMES,
    ROLE_INTERNAL_KEY,
    ROLE_ORDER,
    ROLE_SHORT_LABELS,
    TIER_COLORS,
    TIER_STREAMLIT_COLORS,
    resolve_suffix,
)
from .feedback import analyze_feedback
from .figures import generate_thesis_figures
from .forms import assign_ab_conditions, build_fda_record_for_alert, process_alert
from .irr import compute_inter_rater_reliability
from .loaders import (
    ENRICH_KEYS,
    EVAL_DIR,
    LoaderError,
    enrich_with_device_context,
    load_provenance_inner,
    load_responses_inner,
)
from .metrics import compute_evaluation_metrics
from .pipeline import main as run_evaluation
from .simulated_responses import generate_simulated_responses
from .statistical import statistical_analysis
from .stream_helpers import (
    draw_latency_sample,
    push_latency_sample,
    stream_simulator,
)
from .triage_helpers import (
    apply_dashboard_filters,
    build_feed_dataframe,
    compute_tier_counts,
    floor_elevated,
    primary_action,
)

__all__ = [
    # alerts
    "curate_evaluation_alerts", "_derive_device_class", "_curate_split_paths",
    "_ground_truth_action", "_build_eval_alert",
    # constants
    "ACTIONS", "PAGE_SPLIT", "TIER_COLORS", "TIER_STREAMLIT_COLORS",
    "ROLE_DISPLAY_NAMES", "ROLE_DISPLAY_LIST", "ROLE_INTERNAL_KEY",
    "ROLE_ORDER", "ROLE_SHORT_LABELS", "resolve_suffix",
    # evaluation pipeline
    "compute_evaluation_metrics", "statistical_analysis",
    "compute_inter_rater_reliability", "analyze_feedback",
    "generate_thesis_figures", "generate_simulated_responses",
    "run_evaluation",
    # forms
    "assign_ab_conditions", "build_fda_record_for_alert", "process_alert",
    # loaders
    "EVAL_DIR", "ENRICH_KEYS", "LoaderError",
    "enrich_with_device_context",
    "load_responses_inner", "load_provenance_inner",
    # triage helpers
    "floor_elevated", "apply_dashboard_filters",
    "compute_tier_counts", "build_feed_dataframe", "primary_action",
    # stream helpers
    "stream_simulator", "draw_latency_sample", "push_latency_sample",
]
