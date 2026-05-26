"""Back-compat shim for ``module6_evaluation.module6_evaluation``.

The original 944-LOC monolith was decomposed into:

* ``module6_evaluation.alerts``               — alert curation + device class
* ``module6_evaluation.simulated_responses``  — synthetic participant data
* ``module6_evaluation.metrics``              — compute_evaluation_metrics
* ``module6_evaluation.statistical``          — Wilcoxon + Cohen's d
* ``module6_evaluation.irr``                  — Krippendorff alpha
* ``module6_evaluation.feedback``             — thematic feedback analysis
* ``module6_evaluation.figures``              — thesis figures
* ``module6_evaluation.pipeline``             — main() orchestration

External consumers continue to import via this module path.
"""
from __future__ import annotations

from .alerts import (
    ACTIONS,
    _build_eval_alert,
    _curate_split_paths,
    _derive_device_class,
    _ground_truth_action,
    curate_evaluation_alerts,
)
from .feedback import THEMES, analyze_feedback
from .figures import (
    CHARTS_DIR,
    _plot_accuracy_by_tier,
    _plot_decision_time_by_tier,
    _plot_effect_sizes,
    _plot_radar_chart,
    generate_thesis_figures,
)
from .irr import compute_inter_rater_reliability
from .metrics import compute_evaluation_metrics
from .pipeline import OUTPUT_DIR, PROJECT_ROOT, main
from .simulated_responses import generate_simulated_responses
from .statistical import statistical_analysis

__all__ = [
    "ACTIONS", "THEMES",
    "PROJECT_ROOT", "OUTPUT_DIR", "CHARTS_DIR",
    "_derive_device_class", "_curate_split_paths",
    "_ground_truth_action", "_build_eval_alert",
    "curate_evaluation_alerts",
    "generate_simulated_responses",
    "compute_evaluation_metrics",
    "statistical_analysis",
    "compute_inter_rater_reliability",
    "analyze_feedback",
    "generate_thesis_figures",
    "_plot_radar_chart", "_plot_decision_time_by_tier",
    "_plot_accuracy_by_tier", "_plot_effect_sizes",
    "main",
]


if __name__ == "__main__":
    main()
