#!/usr/bin/env python3
"""Feedback-Loop Demonstration (Tasks C.3–C.6).

Closed-loop iteration that:
  C.3  Single feedback iteration   — before/after comparison
  C.4  Multi-iteration convergence — 5 cycles, plots FPR/FNR & thresholds
  C.5  Weight adjustment via AUROC — reduce low-variance components
  C.6  Thesis-ready outputs        — tables, convergence plot, adjusted config

Usage:
    python -m pipeline.feedback_loop_demo
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.module3_risk_scoring.module3_risk_scores import (
    WEIGHTS,
    assign_risk_levels,
    apply_feedback,
    apply_weight_feedback,
    compute_composite_risk,
)
from pipeline.module5_responses.module5_pipeline import FeedbackLoop

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "reports"
CHARTS_DIR = PROJECT_ROOT / "results" / "charts"

DEFAULT_THRESHOLDS = {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}
N_ITERATIONS = 5


# ── Helpers ────────────────────────────────────────────────────────────

def _load_data() -> dict:
    """Load risk score components and ground truth."""
    npz = dict(np.load(OUTPUT_DIR / "risk_scores.npz", allow_pickle=True))
    return {
        "c_detect":  npz["c_detect"],
        "d_crit":    npz["d_crit"],
        "s_data":    npz["s_data"],
        "a_patient": npz["a_patient"],
        "y_true":    npz["y_true"],
        "R":         npz["R"],
    }


def _compute_rates(
    levels: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Compute FPR, FNR, precision, recall, F1 treating MEDIUM+ as positive."""
    pred_pos = np.isin(levels, ["MEDIUM", "HIGH", "CRITICAL"])
    actual_pos = y_true == 1
    total = len(y_true)

    tp = int((pred_pos & actual_pos).sum())
    fp = int((pred_pos & ~actual_pos).sum())
    fn = int((~pred_pos & actual_pos).sum())
    tn = int((~pred_pos & ~actual_pos).sum())

    fpr = fp / total if total else 0.0
    fnr = fn / total if total else 0.0

    prec = precision_score(actual_pos, pred_pos, zero_division=0)
    rec  = recall_score(actual_pos, pred_pos, zero_division=0)
    f1   = f1_score(actual_pos, pred_pos, zero_division=0)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1": round(float(f1), 6),
    }


def _run_feedback_loop(
    R: np.ndarray,
    levels: np.ndarray,
    y_true: np.ndarray,
    thresholds: dict,
) -> dict:
    """Run FeedbackLoop over all samples and return adjustment dict."""
    fb = FeedbackLoop()
    for idx in range(len(R)):
        tier = str(levels[idx])
        gt = "attack" if y_true[idx] == 1 else "benign"
        fb.record(f"ALERT-{idx:05d}", gt, tier, float(R[idx]), [])
    return fb.compute_adjustments(current_thresholds=thresholds)


# ── C.3  Single Feedback Iteration ────────────────────────────────────

def single_iteration(data: dict) -> dict:
    """One feedback cycle: compute → adjust → re-compute → compare."""
    logger.info("── C.3  Single Feedback Iteration ──")

    R = data["R"]
    y_true = data["y_true"]
    thresholds = dict(DEFAULT_THRESHOLDS)

    # Before
    levels_before = assign_risk_levels(R, thresholds)
    rates_before = _compute_rates(levels_before, y_true)
    logger.info("  BEFORE  FPR=%.4f  FNR=%.4f  F1=%.4f",
                rates_before["fpr"], rates_before["fnr"], rates_before["f1"])

    # Feedback
    adjustment = _run_feedback_loop(R, levels_before, y_true, thresholds)

    # Apply
    new_thresholds = apply_feedback(thresholds, adjustment)
    logger.info("  Thresholds: %s → %s", thresholds, new_thresholds)

    # After
    levels_after = assign_risk_levels(R, new_thresholds)
    rates_after = _compute_rates(levels_after, y_true)
    logger.info("  AFTER   FPR=%.4f  FNR=%.4f  F1=%.4f",
                rates_after["fpr"], rates_after["fnr"], rates_after["f1"])

    return {
        "thresholds_before": thresholds,
        "thresholds_after": new_thresholds,
        "rates_before": rates_before,
        "rates_after": rates_after,
        "feedback": adjustment,
    }


# ── C.4  Multi-Iteration Convergence ─────────────────────────────────

def multi_iteration_convergence(data: dict) -> list[dict]:
    """Run N_ITERATIONS feedback cycles and track convergence."""
    logger.info("")
    logger.info("── C.4  Multi-Iteration Convergence (%d iterations) ──",
                N_ITERATIONS)

    R = data["R"]
    y_true = data["y_true"]
    thresholds = dict(DEFAULT_THRESHOLDS)
    history = []

    for i in range(N_ITERATIONS):
        levels = assign_risk_levels(R, thresholds)
        rates = _compute_rates(levels, y_true)
        adjustment = _run_feedback_loop(R, levels, y_true, thresholds)
        new_thresholds = apply_feedback(thresholds, adjustment)

        record = {
            "iteration": i,
            "thresholds": dict(thresholds),
            "rates": rates,
        }
        history.append(record)
        logger.info("  Iter %d  FPR=%.4f  FNR=%.4f  F1=%.4f  thresh=%s",
                     i, rates["fpr"], rates["fnr"], rates["f1"], thresholds)
        thresholds = new_thresholds

    # Final state after last adjustment
    levels_final = assign_risk_levels(R, thresholds)
    rates_final = _compute_rates(levels_final, y_true)
    history.append({
        "iteration": N_ITERATIONS,
        "thresholds": dict(thresholds),
        "rates": rates_final,
    })
    logger.info("  Final   FPR=%.4f  FNR=%.4f  F1=%.4f  thresh=%s",
                rates_final["fpr"], rates_final["fnr"], rates_final["f1"],
                thresholds)

    return history


def plot_convergence(history: list[dict]) -> None:
    """Plot FPR/FNR and threshold values over iterations."""
    iters = [h["iteration"] for h in history]
    fprs  = [h["rates"]["fpr"] for h in history]
    fnrs  = [h["rates"]["fnr"] for h in history]
    t_med = [h["thresholds"]["MEDIUM"] for h in history]
    t_hi  = [h["thresholds"]["HIGH"] for h in history]
    t_cr  = [h["thresholds"]["CRITICAL"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: FPR / FNR
    ax1.plot(iters, fprs, "o-", color="#e74c3c", linewidth=2, label="FPR")
    ax1.plot(iters, fnrs, "s-", color="#3274A1", linewidth=2, label="FNR")
    ax1.axhline(0.10, color="#e74c3c", linestyle=":", alpha=0.5, label="FPR target (10%)")
    ax1.axhline(0.05, color="#3274A1", linestyle=":", alpha=0.5, label="FNR target (5%)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Rate")
    ax1.set_title("FPR / FNR Convergence")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: Thresholds
    ax2.plot(iters, t_med, "o-", color="#e67e22", linewidth=2, label="MEDIUM")
    ax2.plot(iters, t_hi,  "s-", color="#e74c3c", linewidth=2, label="HIGH")
    ax2.plot(iters, t_cr,  "^-", color="#8e44ad", linewidth=2, label="CRITICAL")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Threshold Value")
    ax2.set_title("Threshold Convergence")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "feedback_convergence.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: feedback_convergence.png")


# ── C.5  Weight Adjustment via Feedback ───────────────────────────────

def weight_adjustment(data: dict) -> dict:
    """Adjust Module 3 weights using AUROC optimization."""
    logger.info("")
    logger.info("── C.5  Weight Adjustment via AUROC ──")

    c_detect  = data["c_detect"]
    d_crit    = data["d_crit"]
    s_data    = data["s_data"]
    a_patient = data["a_patient"]
    y_true    = data["y_true"]

    # Component variances (weighted)
    w = WEIGHTS
    variances = {
        "w1": float(np.var(w["w1"] * c_detect)),
        "w2": float(np.var(w["w2"] * d_crit)),
        "w3": float(np.var(w["w3"] * s_data)),
        "w4": float(np.var(w["w4"] * a_patient)),
    }
    logger.info("  Weighted variances: %s",
                {k: round(v, 6) for k, v in variances.items()})

    # AUROC with default weights
    R_default = compute_composite_risk(c_detect, d_crit, s_data, a_patient)
    auroc_before = roc_auc_score(y_true, R_default)
    logger.info("  AUROC (default weights): %.6f", auroc_before)

    # Apply weight feedback
    new_weights = apply_weight_feedback(
        dict(WEIGHTS), variances,
        y_true, c_detect, d_crit, s_data, a_patient,
    )
    R_adjusted = compute_composite_risk(
        c_detect, d_crit, s_data, a_patient, new_weights,
    )
    auroc_after = roc_auc_score(y_true, R_adjusted)
    logger.info("  AUROC (adjusted weights): %.6f", auroc_after)
    logger.info("  Default weights: %s", WEIGHTS)
    logger.info("  Adjusted weights: %s", new_weights)

    return {
        "weights_before": dict(WEIGHTS),
        "weights_after": new_weights,
        "variances": {k: round(v, 6) for k, v in variances.items()},
        "auroc_before": round(auroc_before, 6),
        "auroc_after": round(auroc_after, 6),
    }


# ── C.6  Thesis-Ready Outputs ─────────────────────────────────────────

def generate_comparison_table(single: dict) -> str:
    """Markdown table: before vs after for a single feedback iteration."""
    rb = single["rates_before"]
    ra = single["rates_after"]
    tb = single["thresholds_before"]
    ta = single["thresholds_after"]

    rows = [
        "| Metric | Before | After | Delta |",
        "|--------|-------:|------:|------:|",
    ]
    for tier in ["CRITICAL", "HIGH", "MEDIUM"]:
        delta = ta[tier] - tb[tier]
        rows.append(f"| Threshold {tier} | {tb[tier]:.4f} | {ta[tier]:.4f} | {delta:+.4f} |")

    for metric in ["fpr", "fnr", "precision", "recall", "f1"]:
        bv, av = rb[metric], ra[metric]
        delta = av - bv
        rows.append(f"| {metric.upper()} | {bv:.4f} | {av:.4f} | {delta:+.4f} |")

    return "\n".join(rows)


def export_adjusted_config(
    thresholds: dict,
    weights: dict,
) -> Path:
    """Write risk_config_adjusted.json with updated parameters."""
    cfg = {
        "formula": "R = w1*C_detect + w2*D_crit + w3*S_data + w4*A_patient",
        "fusion": "C_detect = max(Track_A_proba, Track_B_normalized_RE)",
        "weights": weights,
        "thresholds": thresholds,
        "alert_tiers": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        "source": "feedback_loop_demo — auto-tuned via closed-loop iteration",
    }
    path = OUTPUT_DIR / "risk_config_adjusted.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    logger.info("  Saved: risk_config_adjusted.json")
    return path


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sep = "=" * 72
    t0 = time.perf_counter()

    logger.info(sep)
    logger.info("FEEDBACK-LOOP DEMONSTRATION (Tasks C.3–C.6)")
    logger.info(sep)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    data = _load_data()
    logger.info("Loaded %d samples (%d attacks)",
                len(data["y_true"]), int((data["y_true"] == 1).sum()))

    # C.3: Single iteration
    single = single_iteration(data)

    # C.4: Multi-iteration convergence
    history = multi_iteration_convergence(data)
    plot_convergence(history)

    # C.5: Weight adjustment
    weight_result = weight_adjustment(data)

    # C.6: Thesis-ready outputs
    logger.info("")
    logger.info("── C.6  Thesis-Ready Outputs ──")

    # (a) Before/after comparison table
    table = generate_comparison_table(single)
    logger.info("\n%s", table)

    # (b) Convergence plot already saved by plot_convergence()

    # (c) Updated config
    final_thresholds = history[-1]["thresholds"]
    export_adjusted_config(final_thresholds, weight_result["weights_after"])

    # Save comprehensive results
    results = {
        "single_iteration": single,
        "convergence_history": history,
        "weight_adjustment": weight_result,
        "comparison_table_md": table,
        "final_thresholds": final_thresholds,
        "final_weights": weight_result["weights_after"],
    }
    results_path = OUTPUT_DIR / "feedback_loop_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8",
    )
    logger.info("  Saved: feedback_loop_results.json")

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("FEEDBACK-LOOP DEMO COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    logger.info("  feedback_loop_results.json")
    logger.info("  risk_config_adjusted.json")
    logger.info("  feedback_convergence.png")
    logger.info("  Output: %s", OUTPUT_DIR)
    logger.info(sep)


if __name__ == "__main__":
    main()
