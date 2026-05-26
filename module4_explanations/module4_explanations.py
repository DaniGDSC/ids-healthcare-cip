#!/usr/bin/env python3
"""Module 4 — Generate Explanations (RQ1/RO1) — CLI entry point.

This file is now a thin CLI + backward-compatibility shim. The actual
implementation lives in:

  - ``module4_explanations.config``               — constants + templates
  - ``module4_explanations.feature_groups``       — _FEATURE_GROUPS taxonomy
  - ``module4_explanations.io``                   — load/save/path resolution
  - ``module4_explanations.compute``              — SHAP + DAE + top-k helpers
  - ``module4_explanations.stakeholder``          — analyst/clinician/admin builders
  - ``module4_explanations.nlg``                  — 6-step NLG + stakeholder router
  - ``module4_explanations.validation``           — 3 faithfulness validators
  - ``module4_explanations.plotting``             — all charts (output_dir param)
  - ``module4_explanations.example_explanations`` — worked thesis examples (Y3 fix)
  - ``module4_explanations.online_explainer``     — AlertExplainer (Y10 fix)
  - ``module4_explanations.batch_sim``            — batch sim + latency stats

Existing imports like ``from module4_explanations.module4_explanations
import compute_tree_shap`` continue to work via the re-exports below.

Usage:
    python -m module4_explanations.module4_explanations
    python -m module4_explanations.module4_explanations --split test
    python -m module4_explanations.module4_explanations --split demo --explanations-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Backward-compat re-exports ────────────────────────────────────────
from module4_explanations.compute import (  # noqa: E402,F401
    compute_dae_feature_errors,
    compute_global_importance,
    compute_tree_shap,
    _top_features_dae,
    _top_features_shap,
)
from module4_explanations.config import (  # noqa: E402,F401
    BIOMETRIC_FEATURES,
    CLINICIAN_TEMPLATES,
    FEATURE_CONCEPTS,
    NLG_TEMPLATES,
    SHAP_MODELS,
    TOP_K_FEATURES,
    TOP_N_WATERFALL,
    TRACK_A_MODELS,
)
from module4_explanations.example_explanations import (  # noqa: E402,F401
    generate_example_explanations,
)
from module4_explanations.feature_groups import (  # noqa: E402,F401
    _FEATURE_GROUPS,
    _feature_to_narrative,
)
from module4_explanations.io import (  # noqa: E402,F401
    CHARTS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    _split_paths,
    export_feature_concepts,
    export_nlg_templates,
    load_predictions,
    load_test_data,
    save_dae_errors,
    save_global_importance,
    save_shap_values,
    write_json_batch,
    write_json_sync,
)
from module4_explanations.nlg import (  # noqa: E402,F401
    generate_clinician_alert,
    route_explanation,
)
from module4_explanations.plotting import (  # noqa: E402,F401
    plot_beeswarm,
    plot_dae_breakdowns,
    plot_dae_global_weights,
    plot_force,
    plot_global_importance_bar,
    plot_per_category_importance,
    plot_waterfalls,
)
from module4_explanations.stakeholder import (  # noqa: E402,F401
    _severity,
    build_admin_dashboard,
    build_analyst_report,
    build_clinician_summaries,
)
from module4_explanations.validation import (  # noqa: E402,F401
    validate_consistency,
    validate_cross_model,
    validate_perturbation,
)

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="python -m module4_explanations.module4_explanations",
        description=(
            "Generate stakeholder explanations from Phase 2 predictions. "
            "Default mode produces all paper artefacts (SHAP, charts, "
            "admin dashboard, validation) for the test split. "
            "--explanations-only is a thin mode that produces ONLY the "
            "two JSONs Module 5 needs (analyst + clinician); intended "
            "for the demo split where the full paper artefact set is "
            "neither consumed nor needed."
        ),
    )
    parser.add_argument(
        "--split", choices=("test", "demo"), default="test",
        help="Frozen split (test=paper-clean, demo=operator-clean).",
    )
    parser.add_argument(
        "--explanations-only", action="store_true",
        help=(
            "Skip SHAP/DAE persistence, all charts, admin dashboard, "
            "feature_concepts/nlg_templates export, example explanations, "
            "and validation. Only writes analyst_report{suffix}.json and "
            "clinician_summaries{suffix}.json."
        ),
    )
    args = parser.parse_args()

    paths = _split_paths(args.split)
    suffix = paths["suffix"]
    explanations_only = args.explanations_only

    sep = "=" * 72
    t0 = time.perf_counter()

    logger.info(sep)
    logger.info(
        "MODULE 4 — GENERATE EXPLANATIONS (RQ1/RO1) — split=%s%s",
        args.split, " [explanations-only]" if explanations_only else "",
    )
    logger.info(sep)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not explanations_only:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test, attack_cats, feat_names = load_test_data(paths["parquet"])
    n_samples = len(y_test)
    logger.info(
        "Split %s: %d samples, %d features", args.split, n_samples, len(feat_names),
    )

    pred_paths = {
        "xgboost":       paths["xgboost_preds"],
        "random_forest": paths["random_forest_preds"],
        "decision_tree": paths["decision_tree_preds"],
    }

    # ── Track A: TreeSHAP ──
    all_shap: dict = {}
    all_preds: dict = {}
    global_importances: dict = {}

    for name, cfg in TRACK_A_MODELS.items():
        preds = load_predictions(pred_paths[name])
        all_preds[name] = preds

        if name not in SHAP_MODELS:
            logger.info("Skipping TreeSHAP for %s (not in SHAP_MODELS)", name)
            continue

        sv, expected = compute_tree_shap(
            name, PROJECT_ROOT / cfg["pipeline"], X_test, feat_names,
        )
        all_shap[name] = sv

        if not explanations_only:
            save_shap_values(name, sv, expected, feat_names)
            importance = compute_global_importance(sv, feat_names)
            save_global_importance(name, importance)
            global_importances[name] = importance

            plot_global_importance_bar(name, importance, output_dir=CHARTS_DIR)
            plot_waterfalls(
                name, sv, expected, X_test, feat_names,
                preds["y_pred"], preds["y_proba"], output_dir=CHARTS_DIR,
            )
            plot_beeswarm(name, sv, X_test, feat_names, output_dir=CHARTS_DIR)
            plot_force(
                name, sv, expected, X_test, feat_names,
                preds["y_pred"], preds["y_proba"], output_dir=CHARTS_DIR,
            )
            plot_per_category_importance(
                name, sv, y_test, attack_cats, feat_names,
                output_dir=CHARTS_DIR,
            )

    # ── Track B: DAE ──
    sq_err, weighted_err, feat_weights = compute_dae_feature_errors(
        X_test, feat_names,
    )
    dae_preds = load_predictions(paths["dae_preds"])

    if not explanations_only:
        save_dae_errors(sq_err, weighted_err, feat_weights, feat_names)
        plot_dae_global_weights(feat_weights, feat_names, output_dir=CHARTS_DIR)
        plot_dae_breakdowns(
            weighted_err, feat_names, dae_preds["y_pred"],
            dae_preds["reconstruction_error"], output_dir=CHARTS_DIR,
        )

    # ── Stakeholder outputs ──
    alerts = build_analyst_report(
        all_shap, all_preds, weighted_err, dae_preds, feat_names,
        suffix=suffix,
    )
    build_clinician_summaries(
        all_shap, all_preds, dae_preds, feat_names, suffix=suffix,
    )

    if not explanations_only:
        build_admin_dashboard(
            all_shap, all_preds, dae_preds, feat_names, feat_weights,
            global_importances, attack_cats,
        )
        export_feature_concepts()
        export_nlg_templates()
        generate_example_explanations(
            all_shap, all_preds, dae_preds, weighted_err, feat_names,
            y_test, attack_cats, split=args.split,
        )

        logger.info("")
        logger.info("── Explanation Validation ──")
        validate_consistency(all_shap, feat_names, project_root=PROJECT_ROOT)
        validate_perturbation(all_shap, X_test, y_test, feat_names)
        validate_cross_model(global_importances)

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info(
        "EXPLANATIONS COMPLETE — %.1fs (split=%s%s)",
        elapsed, args.split, ", thin" if explanations_only else "",
    )
    logger.info(sep)
    logger.info("  Output dir    : %s", OUTPUT_DIR)
    if not explanations_only:
        logger.info(
            "  SHAP files    : %d models (%s)",
            len(SHAP_MODELS), ", ".join(SHAP_MODELS),
        )
        logger.info("  DAE errors    : dae_feature_errors.npz")
        logger.info("  Charts        : %s", CHARTS_DIR)
    logger.info(
        "  Analyst alerts: %d  → analyst_report%s.json", len(alerts), suffix,
    )
    logger.info("  Clinician sums: → clinician_summaries%s.json", suffix)
    logger.info(sep)


if __name__ == "__main__":
    main()
