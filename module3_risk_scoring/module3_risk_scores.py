#!/usr/bin/env python3
"""Module 3 — Composite Risk Scores (RQ2/RO2) — CLI entry point.

This file is now a thin CLI + backward-compatibility shim. The actual
implementation lives in:

  - ``module3_risk_scoring.config``       — constants
  - ``module3_risk_scoring.components``   — D_crit, S_data, D_clinical_tier
  - ``module3_risk_scoring.composition``  — composite R + tier assignment
  - ``module3_risk_scoring.feedback``     — apply_feedback, apply_weight_feedback
  - ``module3_risk_scoring.analysis``     — fusion, contribution, sensitivity, examples
  - ``module3_risk_scoring.plotting``     — figures (output_dir param)
  - ``module3_risk_scoring.io``           — load / save / config exports

Existing imports like ``from module3_risk_scoring.module3_risk_scores
import WEIGHTS`` continue to work via the re-exports below.

Composite formula:
    R = 0.40·C_detect + 0.25·D_crit + 0.15·S_data + 0.20·D_clinical_tier

Where C_detect is the cascaded Track A → Track B fusion produced by
``detection_engine.DetectionEngine``.

Usage:
    python -m module3_risk_scoring.module3_risk_scores
    python -m module3_risk_scoring.module3_risk_scores --split test
    python -m module3_risk_scoring.module3_risk_scores --split both
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Project root on sys.path for absolute imports when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

# ── Backward-compat re-exports ────────────────────────────────────────
from module3_risk_scoring.analysis import (  # noqa: E402,F401
    _build_example,
    component_contribution_analysis,
    dual_track_fusion_analysis,
    generate_worked_examples,
    weight_sensitivity_analysis,
)
from module3_risk_scoring.components import (  # noqa: E402,F401
    _get_bio_idx,
    compute_d_clinical_tier,
    compute_d_crit,
    compute_s_data,
)
from module3_risk_scoring.composition import (  # noqa: E402,F401
    assign_risk_levels,
    compute_composite_risk,
)
from module3_risk_scoring.config import (  # noqa: E402,F401
    BIOMETRIC_FEATURES,
    CIA_SCORE,
    CIA_THREATS,
    DAE_BINARY_THRESHOLD,
    DATA_SENSITIVITY,
    DEFAULT_CIA_SCORE,
    DEFAULT_DEVICE_TIER,
    DEVICE_TIERS,
    FEATURE_ACTIVE_EPSILON,
    RESPONSE_MAPPING,
    RISK_THRESHOLDS,
    SIGMA_THRESHOLD,
    WEIGHTS,
)
from module3_risk_scoring.feedback import (  # noqa: E402,F401
    apply_feedback,
    apply_weight_feedback,
)
from module3_risk_scoring.io import (  # noqa: E402,F401
    CHARTS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    _split_paths,
    export_config_jsons,
    load_test_data,
    load_xgboost_proba,
    save_outputs,
)
from module3_risk_scoring.plotting import (  # noqa: E402,F401
    plot_component_breakdown,
    plot_component_scatter,
    plot_dual_track_heatmap,
    plot_risk_by_category,
    plot_risk_by_label,
    plot_risk_distribution,
    plot_weight_sensitivity_curve,
)

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m module3_risk_scoring.module3_risk_scores",
        description=(
            "Module 3 — composite risk scoring. Operates on the selected "
            "frozen split (test=paper-clean, demo=operator-clean)."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["test", "demo", "both"],
        default="test",
        help=(
            "Frozen split to process. 'test' writes the paper-clean "
            "`risk_scores.npz`; 'demo' writes `demo_scores.npz`."
        ),
    )
    args = parser.parse_args()

    splits_to_run = ["test", "demo"] if args.split == "both" else [args.split]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sep = "=" * 72
    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits_to_run:
        _run_one_split(split, sep)
    logger.info(
        "Module 3 complete (%.1fs, splits=%s)",
        time.perf_counter() - t0,
        splits_to_run,
    )


def _run_one_split(split: str, sep: str) -> None:
    paths = _split_paths(split)
    logger.info(sep)
    logger.info("MODULE 3 — COMPOSITE RISK SCORES (RQ2/RO2) — split=%s", split)
    logger.info(sep)

    # ── Load data ──
    X_test, y_test, attack_cats, feat_names = load_test_data(paths["parquet"])
    n_samples = len(y_test)
    n_attacks = (y_test == 1).sum()
    logger.info(
        "Data: %d samples (%d attacks) from %s",
        n_samples, n_attacks, paths["parquet"].name,
    )

    # ── Compute components ──
    logger.info("Computing risk components...")

    # XGBoost threshold log — only available for test (cached test predictions).
    # N4 fix: explicit variable, no locals() introspection.
    xgb_threshold: float | None = None
    if split == "test":
        try:
            _, xgb_threshold = load_xgboost_proba()
            logger.info("  Track A: XGBoost proba, threshold=%.3f", xgb_threshold)
        except FileNotFoundError:
            logger.warning(
                "  XGBoost cached test predictions absent; skipping threshold log"
            )

    from detection_engine import DetectionEngine
    det_result = DetectionEngine().predict(X_test)
    c_track_a = det_result.c_track_a
    c_track_b = det_result.c_track_b
    c_detect = det_result.c_detect
    logger.info(
        "  C_detect (cascaded fusion): range [%.4f, %.4f]",
        c_detect.min(), c_detect.max(),
    )

    d_crit = compute_d_crit(attack_cats)
    logger.info(
        "  D_crit: device tier=%s, %.0f elevated (attacks)",
        DEFAULT_DEVICE_TIER,
        (d_crit > DEVICE_TIERS[DEFAULT_DEVICE_TIER] * 0.5).sum(),
    )

    s_data = compute_s_data(X_test, feat_names)
    logger.info("  S_data: range [%.4f, %.4f]", s_data.min(), s_data.max())

    d_clinical_tier = compute_d_clinical_tier(X_test, feat_names)
    logger.info(
        "  D_clinical_tier: %.1f%% samples have abnormal biometrics",
        (d_clinical_tier > 0).mean() * 100,
    )

    # ── Composite risk ──
    R = compute_composite_risk(c_detect, d_crit, s_data, d_clinical_tier)
    levels = assign_risk_levels(R)
    logger.info("")
    logger.info(
        "Composite risk R: mean=%.4f, median=%.4f, std=%.4f",
        R.mean(), np.median(R), R.std(),
    )

    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        count = (levels == level).sum()
        pct = count / n_samples * 100
        logger.info("  %-10s %5d (%5.1f%%)", level, count, pct)

    # ── Dual-track fusion ──
    logger.info("")
    logger.info("── Dual-Track Fusion Analysis ──")
    fusion_threshold = xgb_threshold if xgb_threshold is not None else 0.5
    fusion = dual_track_fusion_analysis(
        c_track_a, c_track_b, y_test, attack_cats, fusion_threshold,
    )
    r = fusion["recall"]
    logger.info("  XGBoost recall: %.4f", r["xgboost_alone"])
    logger.info("  DAE recall:     %.4f", r["dae_alone"])
    logger.info(
        "  Union recall:   %.4f (fusion gain: +%.4f)",
        r["union_fusion"], r["fusion_gain"],
    )
    for qname, qdata in fusion["quadrants"].items():
        logger.info(
            "  %-15s %4d total, %3d attacks %s",
            qname, qdata["total"], qdata["true_attacks"],
            qdata.get("attack_categories", ""),
        )

    # ── Component contribution ──
    contributions = component_contribution_analysis(
        c_detect, d_crit, s_data, d_clinical_tier, levels,
    )

    # ── Sensitivity analysis ──
    logger.info("")
    sensitivity = weight_sensitivity_analysis(
        c_detect, d_crit, s_data, d_clinical_tier, y_test,
    )

    # ── Worked examples ──
    logger.info("")
    logger.info("Generating worked examples...")
    worked_examples = generate_worked_examples(
        R, c_detect, d_crit, s_data, d_clinical_tier,
        c_track_a, c_track_b, levels, y_test, attack_cats,
    )
    for ex in worked_examples:
        logger.info(
            "  %s (sample %d): R=%.4f → %s",
            ex["title"], ex["sample_index"], ex["R"], ex["risk_level"],
        )

    # ── Save outputs ──
    logger.info("")
    logger.info("Saving outputs...")
    save_outputs(
        R, c_detect, d_crit, s_data, d_clinical_tier, c_track_a, c_track_b,
        levels, y_test, attack_cats, fusion, contributions,
        sensitivity, worked_examples,
        out_npz=paths["out_npz"],
    )

    # ── Visualizations + config JSON exports (test split only) ──
    if split == "test":
        logger.info("Generating charts...")
        plot_risk_distribution(R, levels, output_dir=CHARTS_DIR)
        plot_component_breakdown(contributions, output_dir=CHARTS_DIR)
        plot_dual_track_heatmap(fusion, output_dir=CHARTS_DIR)
        plot_component_scatter(c_track_a, c_track_b, y_test, output_dir=CHARTS_DIR)
        plot_risk_by_category(R, attack_cats, y_test, output_dir=CHARTS_DIR)
        plot_risk_by_label(R, y_test, output_dir=CHARTS_DIR)
        plot_weight_sensitivity_curve(
            sensitivity["per_component_sensitivity"],
            sensitivity["best_auroc"],
            output_dir=CHARTS_DIR,
        )
        logger.info("Exporting config JSONs...")
        export_config_jsons()

    logger.info("")
    logger.info(sep)
    logger.info("SPLIT %s COMPLETE", split.upper())
    logger.info(sep)
    logger.info(
        "  Formula   : R = %.2f·C_detect + %.2f·D_crit + %.2f·S_data + %.2f·D_clinical_tier",
        WEIGHTS["w1"], WEIGHTS["w2"], WEIGHTS["w3"], WEIGHTS["w4"],
    )
    logger.info("  Fusion    : C_detect = cascaded(Track_A → Track_B)")
    logger.info("  Output    : %s", paths["out_npz"])
    logger.info(sep)


if __name__ == "__main__":
    main()
