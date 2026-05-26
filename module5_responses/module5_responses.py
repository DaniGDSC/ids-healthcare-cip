"""Back-compat shim for ``module5_responses.module5_responses``.

The original 1.2k-LOC monolith was decomposed into:

* ``module5_responses.loaders``        — _paths, load_risk_scores, ...
* ``module5_responses.adaptive``       — select_adaptive_response, build_audit_record
* ``module5_responses.effectiveness``  — compute_effectiveness, compute_response_stats
* ``module5_responses.plotting``       — plot_*
* ``module5_responses.pipeline``       — build_all_records, _run_one_split
* ``module5_responses.responses_cli``  — main()

The unified taxonomy (formerly DEVICE_TIERS, MITIGATION_ACTIONS,
BASE_PROTOCOL, ESCALATION_ROUTING) now lives in ``module5_responses.config``.
Legacy aliases below preserve the names external readers may have been
importing.
"""
from __future__ import annotations

from .adaptive import build_audit_record, select_adaptive_response
from .config import (
    ACTION_CATALOGUE as MITIGATION_ACTIONS,
    ATTACK_ROUTING as ESCALATION_ROUTING,
    DEFAULT_DEVICE_TIER,
    DEFAULT_ROUTING,
    DEVICE_TIERS,
    TIER_POLICIES as BASE_PROTOCOL,
)
from .effectiveness import compute_effectiveness, compute_response_stats
from .loaders import (
    CHARTS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    _paths,
    load_attack_categories,
    load_explanations,
    load_risk_scores,
)
from .pipeline import (
    _assert_no_score_drift,
    _build_provenance,
    build_all_records,
    run_one_split as _run_one_split,
)
from .plotting import (
    plot_effectiveness_by_action,
    plot_escalation_funnel,
    plot_precision_by_level,
    plot_response_distribution,
    plot_response_sankey,
)
from .responses_cli import main


__all__ = [
    # config aliases
    "MITIGATION_ACTIONS", "DEVICE_TIERS", "BASE_PROTOCOL", "ESCALATION_ROUTING",
    "DEFAULT_DEVICE_TIER", "DEFAULT_ROUTING",
    # constants
    "PROJECT_ROOT", "OUTPUT_DIR", "CHARTS_DIR",
    # loaders
    "_paths", "load_risk_scores", "load_explanations", "load_attack_categories",
    # core engine
    "select_adaptive_response", "build_audit_record",
    "compute_effectiveness", "compute_response_stats",
    "build_all_records", "_build_provenance", "_assert_no_score_drift",
    "_run_one_split",
    # plotting
    "plot_response_distribution", "plot_precision_by_level",
    "plot_escalation_funnel", "plot_effectiveness_by_action",
    "plot_response_sankey",
    # CLI
    "main",
]


if __name__ == "__main__":
    main()
