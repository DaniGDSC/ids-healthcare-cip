#!/usr/bin/env python3
"""Dynamic Threshold Simulation (Phases B1 + B3).

Simulates a temporal stream over the test set to compare static vs
adaptive thresholds for both DAE anomaly detection and Module 3 risk
tier classification.

Tasks
-----
B1.1  Sort test set by time proxy (row index — no timestamp columns).
B1.2  Sliding-window median/MAD statistics for benign RE.
B1.3  Adaptive threshold: threshold_t = median + k * MAD.
B1.4  Static vs adaptive comparison over the stream.
B1.5  Sensitivity grid search: W x k.
B1.6  Comparison figures.
B3.1  Adaptive risk tier thresholds via rolling percentiles.
B3.2  Static vs adaptive tier comparison.
B3.3  Integration with Option C feedback loop.
B3.4  Master comparison table.

Usage:
    python -m dynamic_threshold_sim
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from module3_risk_scoring.module3_risk_scores import (
    assign_risk_levels,
    apply_feedback,
)
from module5_responses.module5_pipeline import FeedbackLoop

logger = logging.getLogger(__name__)

# Project root is two directories up: tools/diagnostics/ → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "reports"
CHARTS_DIR = PROJECT_ROOT / "results" / "charts"

# Defaults
DEFAULT_WINDOW = 100
DEFAULT_K = 3
WINDOW_GRID = [50, 100, 200, 500]
K_GRID = [2, 3, 4]

DEFAULT_THRESHOLDS = {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40, "LOW": 0.30}


# ═══════════════════════════════════════════════════════════════════════
# Data loading (B1.1)
# ═══════════════════════════════════════════════════════════════════════

def _split_paths(split: str) -> dict:
    """Resolve per-split input/output paths.

    Thin wrapper over :mod:`common.split_paths` — call sites keep their
    dict-access shape while the canonical path mapping lives in common.
    """
    from common import split_paths as sp
    return {
        "dae_preds": sp.dae_predictions(split),
        "risk_npz":  sp.risk_scores(split),
        "suffix":    sp.suffix(split),
    }


def load_stream_data(split: str = "test") -> dict:
    """Load a split's stream data, sorted by row index as temporal proxy.

    The WUSTL-EHMS-2020 parquet files have no timestamp columns, so row
    order serves as the time proxy.  This is acknowledged as a
    limitation in the thesis.
    """
    paths = _split_paths(split)
    npz = dict(np.load(paths["dae_preds"]))
    from common.risk_scores_loader import load_risk_scores as _load_risk_scores
    risk_npz = _load_risk_scores(paths["risk_npz"])
    # DAE is persisted as dae_detector.json + dae_model.weights.h5 (no
    # pickle on the load path — see DAE.from_artefacts). The model_registry
    # singleton caches it.
    from common.model_registry import get_dae
    detector = get_dae()
    if detector._train_errors is None:
        raise RuntimeError(
            "DAE artifact is missing the `train_errors` array — adaptive "
            "threshold simulation needs it to seed the sliding-window "
            "baseline. Retrain the DAE (module2_detection.dae_training) so "
            "the JSON sidecar persists train_errors."
        )

    re_scores = npz["reconstruction_error"]  # raw RE per sample
    y_true = risk_npz["y_true"]

    # Static DAE threshold (95th percentile of benign training RE)
    static_threshold = float(detector._threshold)
    train_errors = np.array(detector._train_errors)

    return {
        "re_scores": re_scores,
        "y_true": y_true,
        "static_threshold": static_threshold,
        "train_benign_re": train_errors,
        # Risk score components for B3
        "R": risk_npz["R"],
        "c_detect": risk_npz["c_detect"],
        "d_crit": risk_npz["d_crit"],
        "s_data": risk_npz["s_data"],
        "d_clinical_tier": risk_npz["d_clinical_tier"],
    }


# ═══════════════════════════════════════════════════════════════════════
# B1.2  Sliding-window statistics
# ═══════════════════════════════════════════════════════════════════════

def _median_mad(window: np.ndarray) -> tuple[float, float]:
    """Return (median, MAD) of the values in window.

    DT-2: np.partition gives O(W) median without a full O(W log W) sort.
    DT-7: single pre-allocated buffer for abs-deviation avoids two temporaries.
    """
    mid = len(window) // 2
    med = float(np.partition(window, mid)[mid])
    dev = np.empty_like(window)
    np.abs(window - med, out=dev)
    mad = float(np.partition(dev, mid)[mid])
    return med, mad


# ═══════════════════════════════════════════════════════════════════════
# B1.3 + B1.4  Stream processing — static vs adaptive
# ═══════════════════════════════════════════════════════════════════════

def run_stream(
    re_scores: np.ndarray,
    y_true: np.ndarray,
    static_threshold: float,
    train_benign_re: np.ndarray,
    W: int = DEFAULT_WINDOW,
    k: float = DEFAULT_K,
) -> dict:
    """Process the test set as a stream, comparing static and adaptive
    classification at every step.

    Returns per-step arrays and summary metrics.
    """
    n = len(re_scores)

    # Seed the adaptive window with benign training RE tail
    seed = train_benign_re[-W:] if len(train_benign_re) >= W else train_benign_re
    benign_window: deque[float] = deque(seed.tolist(), maxlen=W)

    # Output arrays
    static_preds = np.zeros(n, dtype=int)
    adaptive_preds = np.zeros(n, dtype=int)
    adaptive_thresh = np.zeros(n, dtype=float)
    static_thresh_arr = np.full(n, static_threshold)

    cum_f1_static = np.zeros(n, dtype=float)
    cum_f1_adaptive = np.zeros(n, dtype=float)
    cum_fpr_static = np.zeros(n, dtype=float)
    cum_fpr_adaptive = np.zeros(n, dtype=float)
    cum_fnr_static = np.zeros(n, dtype=float)
    cum_fnr_adaptive = np.zeros(n, dtype=float)

    # DT-1: O(1) running confusion-matrix counters replace the O(t) slice +
    # f1_score scan that made the loop O(N²).  F1/FPR/FNR derived each tick
    # from four integers — no array allocation inside the hot path.
    tp_s = fp_s = fn_s = tn_s = 0
    tp_a = fp_a = fn_a = tn_a = 0

    for t in range(n):
        re = re_scores[t]
        yt = int(y_true[t])

        # Static prediction
        sp = int(re > static_threshold)
        static_preds[t] = sp

        # Adaptive threshold
        window_arr = np.array(benign_window)
        med, mad = _median_mad(window_arr)
        thresh_t = med + k * mad if mad > 0 else med * (1 + k * 0.1)
        adaptive_thresh[t] = thresh_t
        ap = int(re > thresh_t)
        adaptive_preds[t] = ap

        # Update benign window: add sample only if classified benign by
        # the *static* threshold (avoids feedback loop contamination)
        if re <= static_threshold:
            benign_window.append(re)

        # DT-1: update confusion matrix counters O(1)
        if sp == 1 and yt == 1:
            tp_s += 1
        elif sp == 1 and yt == 0:
            fp_s += 1
        elif sp == 0 and yt == 1:
            fn_s += 1
        else:
            tn_s += 1

        if ap == 1 and yt == 1:
            tp_a += 1
        elif ap == 1 and yt == 0:
            fp_a += 1
        elif ap == 0 and yt == 1:
            fn_a += 1
        else:
            tn_a += 1

        # Cumulative metrics: need at least one positive and one negative seen
        if tp_s + fn_s > 0 and tp_s + fp_s >= 0 and (tp_s > 0 or fp_s > 0 or fn_s > 0):
            n_seen = t + 1
            # Static F1
            prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
            rec_s  = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
            cum_f1_static[t] = (2 * prec_s * rec_s / (prec_s + rec_s)
                                 if (prec_s + rec_s) > 0 else 0.0)
            # Adaptive F1
            prec_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
            rec_a  = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
            cum_f1_adaptive[t] = (2 * prec_a * rec_a / (prec_a + rec_a)
                                   if (prec_a + rec_a) > 0 else 0.0)
            cum_fpr_static[t]   = fp_s / n_seen
            cum_fnr_static[t]   = fn_s / n_seen
            cum_fpr_adaptive[t] = fp_a / n_seen
            cum_fnr_adaptive[t] = fn_a / n_seen

    # Final metrics
    final = {}
    for label, preds in [("static", static_preds), ("adaptive", adaptive_preds)]:
        final[label] = {
            "f1": round(float(f1_score(y_true, preds, zero_division=0)), 6),
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, preds, zero_division=0)), 6),
            "fpr": round(float(((preds == 1) & (y_true == 0)).sum() / len(y_true)), 6),
            "fnr": round(float(((preds == 0) & (y_true == 1)).sum() / len(y_true)), 6),
        }

    return {
        "W": W, "k": k,
        "static_preds": static_preds,
        "adaptive_preds": adaptive_preds,
        "adaptive_thresh": adaptive_thresh,
        "static_thresh": static_thresh_arr,
        "cum_f1_static": cum_f1_static,
        "cum_f1_adaptive": cum_f1_adaptive,
        "cum_fpr_static": cum_fpr_static,
        "cum_fpr_adaptive": cum_fpr_adaptive,
        "cum_fnr_static": cum_fnr_static,
        "cum_fnr_adaptive": cum_fnr_adaptive,
        "final_metrics": final,
    }


# ═══════════════════════════════════════════════════════════════════════
# B1.5  Sensitivity grid search
# ═══════════════════════════════════════════════════════════════════════

def sensitivity_grid(
    re_scores: np.ndarray,
    y_true: np.ndarray,
    static_threshold: float,
    train_benign_re: np.ndarray,
) -> pd.DataFrame:
    """Grid search over W × k, returning final F1 and FPR for each.

    DT-6: 12 independent run_stream calls executed in parallel via
    ThreadPoolExecutor — numpy releases the GIL for most array ops so
    threads run concurrently.  max_workers capped at combo count so we
    never spin up idle threads.
    """
    combos = [(W, k_val) for W in WINDOW_GRID for k_val in K_GRID]
    n_workers = min(len(combos), os.cpu_count() or 4)

    def _run_one(args: tuple) -> dict:
        W, k_val = args
        return run_stream(re_scores, y_true, static_threshold,
                          train_benign_re, W=W, k=k_val)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_run_one, combos))

    rows = []
    for (W, k_val), result in zip(combos, results):
        fm = result["final_metrics"]
        rows.append({
            "W": W, "k": k_val,
            "F1_static": fm["static"]["f1"],
            "F1_adaptive": fm["adaptive"]["f1"],
            "FPR_static": fm["static"]["fpr"],
            "FPR_adaptive": fm["adaptive"]["fpr"],
            "FNR_adaptive": fm["adaptive"]["fnr"],
            "delta_F1": round(fm["adaptive"]["f1"] - fm["static"]["f1"], 6),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# B1.6  Comparison figures
# ═══════════════════════════════════════════════════════════════════════

def plot_threshold_over_time(result: dict, suffix: str = "") -> None:
    """(a) Static horizontal line vs adaptive curve."""
    fig, ax = plt.subplots(figsize=(14, 5))
    n = len(result["adaptive_thresh"])
    x = np.arange(n)

    ax.plot(x, result["static_thresh"], color="#e74c3c", linewidth=1.5,
            label=f"Static (={result['static_thresh'][0]:.2e})")
    ax.plot(x, result["adaptive_thresh"], color="#3274A1", linewidth=0.8,
            alpha=0.9, label=f"Adaptive (W={result['W']}, k={result['k']})")
    ax.set_xlabel("Sample Index (time proxy)")
    ax.set_ylabel("Threshold (RE scale)")
    ax.set_title("DAE Anomaly Threshold Over Time: Static vs Adaptive")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = CHARTS_DIR / f"threshold_over_time{suffix}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)


def plot_cumulative_f1(result: dict, suffix: str = "") -> None:
    """(b) Cumulative F1 over time."""
    fig, ax = plt.subplots(figsize=(14, 5))
    n = len(result["cum_f1_static"])
    x = np.arange(n)

    ax.plot(x, result["cum_f1_static"], color="#e74c3c", linewidth=1.2,
            label="Static F1")
    ax.plot(x, result["cum_f1_adaptive"], color="#3274A1", linewidth=1.2,
            label="Adaptive F1")
    ax.set_xlabel("Sample Index (time proxy)")
    ax.set_ylabel("Cumulative F1")
    ax.set_title("Cumulative F1 Score: Static vs Adaptive Threshold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = CHARTS_DIR / f"cumulative_f1{suffix}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)


def plot_sensitivity_heatmap(grid_df: pd.DataFrame, suffix: str = "") -> None:
    """(c) Heatmap of F1 by W × k."""
    pivot = grid_df.pivot(index="k", columns="W", values="F1_adaptive")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto",
                   vmin=pivot.values.min() - 0.01,
                   vmax=pivot.values.max() + 0.01)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Window Size W")
    ax.set_ylabel("Multiplier k")
    ax.set_title("Adaptive Threshold F1 — Sensitivity to W and k")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.4f}",
                    ha="center", va="center", fontsize=10, fontweight="bold")

    plt.colorbar(im, label="F1 Score")
    plt.tight_layout()
    out = CHARTS_DIR / f"sensitivity_heatmap{suffix}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)


# ═══════════════════════════════════════════════════════════════════════
# B3.1  Adaptive risk tier thresholds
# ═══════════════════════════════════════════════════════════════════════

def run_adaptive_tiers(
    R: np.ndarray,
    y_true: np.ndarray,
    W: int = DEFAULT_WINDOW,
) -> dict:
    """Rolling percentile-based tier thresholds over the test stream.

    Adaptive tiers: MEDIUM = p75, HIGH = p90, CRITICAL = p95 of the
    benign risk score distribution within the window.
    """
    n = len(R)

    # Seed with first W benign samples
    benign_mask_init = y_true[:W] == 0
    init_scores = R[:W][benign_mask_init] if benign_mask_init.any() else R[:W]
    benign_window: deque[float] = deque(init_scores.tolist(), maxlen=W)

    static_levels = assign_risk_levels(R, DEFAULT_THRESHOLDS)

    adaptive_levels = np.empty(n, dtype="<U10")
    tier_history = {"MEDIUM": [], "HIGH": [], "CRITICAL": []}

    for t in range(n):
        score = R[t]

        # Compute adaptive thresholds from benign window
        # DT-3: single np.percentile call — one sort instead of three
        warr = np.array(benign_window) if len(benign_window) >= 10 else R[:W]
        t_med, t_hi, t_cr = np.percentile(warr, [75, 90, 95])
        thresholds_t = {"CRITICAL": t_cr, "HIGH": t_hi, "MEDIUM": t_med}

        tier_history["MEDIUM"].append(t_med)
        tier_history["HIGH"].append(t_hi)
        tier_history["CRITICAL"].append(t_cr)

        # Classify
        if score >= t_cr:
            adaptive_levels[t] = "CRITICAL"
        elif score >= t_hi:
            adaptive_levels[t] = "HIGH"
        elif score >= t_med:
            adaptive_levels[t] = "MEDIUM"
        else:
            adaptive_levels[t] = "LOW"

        # Update window with benign samples (static label to avoid contamination)
        if y_true[t] == 0:
            benign_window.append(score)

    # Compute comparison metrics
    def _tier_metrics(levels, y_true):
        pred_pos = np.isin(levels, ["MEDIUM", "HIGH", "CRITICAL"])
        actual_pos = y_true == 1
        total = len(y_true)
        tp = int((pred_pos & actual_pos).sum())
        fp = int((pred_pos & ~actual_pos).sum())
        fn = int((~pred_pos & actual_pos).sum())
        return {
            "f1": round(float(f1_score(actual_pos, pred_pos, zero_division=0)), 6),
            "fpr": round(fp / total, 6),
            "fnr": round(fn / total, 6),
            "precision": round(float(precision_score(actual_pos, pred_pos, zero_division=0)), 6),
            "recall": round(float(recall_score(actual_pos, pred_pos, zero_division=0)), 6),
        }

    # Alert distribution
    def _tier_dist(levels: np.ndarray) -> dict:
        # DT-4: np.unique one-pass replaces 4 × O(N) (levels==tier).sum() calls
        vals, cnts = np.unique(levels, return_counts=True)
        base = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        base.update({str(v): int(c) for v, c in zip(vals, cnts)})
        return base

    return {
        "static_metrics": _tier_metrics(static_levels, y_true),
        "adaptive_metrics": _tier_metrics(adaptive_levels, y_true),
        "static_distribution": _tier_dist(static_levels),
        "adaptive_distribution": _tier_dist(adaptive_levels),
        "tier_history": {k: np.array(v) for k, v in tier_history.items()},
        "static_levels": static_levels,
        "adaptive_levels": adaptive_levels,
    }


# ═══════════════════════════════════════════════════════════════════════
# B3.3  Integrate with Option C feedback loop
# ═══════════════════════════════════════════════════════════════════════

def run_combined(
    R: np.ndarray,
    y_true: np.ndarray,
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    d_clinical_tier: np.ndarray,
    W: int = DEFAULT_WINDOW,
    n_feedback_iters: int = 3,
) -> dict:
    """Combined approach: sliding-window (short-term) + feedback (long-term).

    1. Run adaptive tiers on raw R for short-term adaptation.
    2. Run feedback loop for long-term threshold calibration.
    3. Combine: use feedback-adjusted thresholds as the *base* for the
       adaptive percentile window.
    """
    # --- Feedback-only (Option C) ---
    # DT-5: pre-compute label strings once with vectorised np.where —
    # avoids N×n_feedback_iters Python-level ternary evaluations.
    gt_labels = np.where(y_true == 1, "attack", "benign")

    thresholds = dict(DEFAULT_THRESHOLDS)
    for _ in range(n_feedback_iters):
        levels = assign_risk_levels(R, thresholds)
        fb = FeedbackLoop()
        for idx in range(len(R)):
            fb.record(f"A-{idx}", gt_labels[idx], str(levels[idx]), float(R[idx]), [])
        adj = fb.compute_adjustments(current_thresholds=thresholds)
        thresholds = apply_feedback(thresholds, adj)
    feedback_thresholds = dict(thresholds)
    feedback_levels = assign_risk_levels(R, feedback_thresholds)

    # --- Sliding-window only ---
    adaptive_result = run_adaptive_tiers(R, y_true, W=W)

    # --- Combined: sliding window with feedback-calibrated baseline ---
    n = len(R)
    benign_window: deque[float] = deque(maxlen=W)
    for idx in range(min(W, n)):
        if y_true[idx] == 0:
            benign_window.append(R[idx])

    combined_levels = np.empty(n, dtype="<U10")
    for t in range(n):
        score = R[t]
        if len(benign_window) >= 10:
            warr = np.array(benign_window)
            p75, p90, p95 = np.percentile(warr, [75, 90, 95])
            # Blend: average of adaptive percentile and feedback threshold
            t_med = (p75 + feedback_thresholds["MEDIUM"]) / 2
            t_hi = (p90 + feedback_thresholds["HIGH"]) / 2
            t_cr = (p95 + feedback_thresholds["CRITICAL"]) / 2
        else:
            t_med = feedback_thresholds["MEDIUM"]
            t_hi = feedback_thresholds["HIGH"]
            t_cr = feedback_thresholds["CRITICAL"]

        if score >= t_cr:
            combined_levels[t] = "CRITICAL"
        elif score >= t_hi:
            combined_levels[t] = "HIGH"
        elif score >= t_med:
            combined_levels[t] = "MEDIUM"
        else:
            combined_levels[t] = "LOW"

        if y_true[t] == 0:
            benign_window.append(score)

    def _metrics(levels):
        pred_pos = np.isin(levels, ["MEDIUM", "HIGH", "CRITICAL"])
        actual_pos = y_true == 1
        total = len(y_true)
        fp = int((pred_pos & ~actual_pos).sum())
        fn = int((~pred_pos & actual_pos).sum())
        return {
            "f1": round(float(f1_score(actual_pos, pred_pos, zero_division=0)), 6),
            "fpr": round(fp / total, 6),
            "fnr": round(fn / total, 6),
            "precision": round(float(precision_score(actual_pos, pred_pos, zero_division=0)), 6),
            "recall": round(float(recall_score(actual_pos, pred_pos, zero_division=0)), 6),
        }

    def _dist(levels: np.ndarray) -> dict:
        # DT-4: np.unique one-pass replaces 4 × O(N) equality scans
        vals, cnts = np.unique(levels, return_counts=True)
        base = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        base.update({str(v): int(c) for v, c in zip(vals, cnts)})
        return base

    static_levels = assign_risk_levels(R, DEFAULT_THRESHOLDS)

    return {
        "static":   {"metrics": _metrics(static_levels),   "distribution": _dist(static_levels)},
        "adaptive":  {"metrics": _metrics(adaptive_result["adaptive_levels"]),
                      "distribution": _dist(adaptive_result["adaptive_levels"])},
        "feedback":  {"metrics": _metrics(feedback_levels), "distribution": _dist(feedback_levels),
                      "thresholds": feedback_thresholds},
        "combined":  {"metrics": _metrics(combined_levels), "distribution": _dist(combined_levels)},
    }


# ═══════════════════════════════════════════════════════════════════════
# B3.4  Master comparison table
# ═══════════════════════════════════════════════════════════════════════

def build_master_table(comparison: dict, tier_result: dict) -> str:
    """Markdown table comparing all four approaches."""
    approaches = [
        ("Static-only", comparison["static"]),
        ("Sliding-window", comparison["adaptive"]),
        ("Feedback-loop", comparison["feedback"]),
        ("Combined", comparison["combined"]),
    ]

    rows = [
        "| Approach | F1 | FPR | FNR | Precision | Recall | LOW | MED | HIGH | CRIT |",
        "|----------|---:|----:|----:|----------:|-------:|----:|----:|-----:|-----:|",
    ]
    for name, data in approaches:
        m = data["metrics"]
        d = data["distribution"]
        rows.append(
            f"| {name} | {m['f1']:.4f} | {m['fpr']:.4f} | {m['fnr']:.4f} "
            f"| {m['precision']:.4f} | {m['recall']:.4f} "
            f"| {d['LOW']} | {d['MEDIUM']} | {d['HIGH']} | {d['CRITICAL']} |"
        )
    return "\n".join(rows)


# ═══════════════════════════════════════════════════════════════════════
# Additional B3 figures
# ═══════════════════════════════════════════════════════════════════════

def plot_adaptive_tier_thresholds(tier_result: dict, suffix: str = "") -> None:
    """Plot adaptive tier threshold values over time."""
    fig, ax = plt.subplots(figsize=(14, 5))
    n = len(tier_result["tier_history"]["MEDIUM"])
    x = np.arange(n)

    colors = {"MEDIUM": "#e67e22", "HIGH": "#e74c3c", "CRITICAL": "#8e44ad"}
    for tier, color in colors.items():
        ax.plot(x, tier_result["tier_history"][tier], color=color,
                linewidth=0.8, alpha=0.9, label=f"Adaptive {tier}")
        static_val = DEFAULT_THRESHOLDS[tier]
        ax.axhline(static_val, color=color, linestyle="--", alpha=0.4,
                    label=f"Static {tier} ({static_val})")

    ax.set_xlabel("Sample Index (time proxy)")
    ax.set_ylabel("Threshold Value")
    ax.set_title("Risk Tier Thresholds Over Time: Static vs Adaptive Percentiles")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = CHARTS_DIR / f"adaptive_tier_thresholds{suffix}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)


def plot_master_comparison(comparison: dict, suffix: str = "") -> None:
    """Bar chart comparing F1 across the four approaches."""
    approaches = ["Static-only", "Sliding-window", "Feedback-loop", "Combined"]
    keys = ["static", "adaptive", "feedback", "combined"]
    colors = ["#95a5a6", "#3274A1", "#e67e22", "#2ecc71"]

    f1s = [comparison[k]["metrics"]["f1"] for k in keys]
    fprs = [comparison[k]["metrics"]["fpr"] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(approaches, f1s, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("F1 Score")
    ax1.set_title("F1 Score by Approach")
    ax1.set_ylim(min(f1s) - 0.02, max(f1s) + 0.02)
    for i, v in enumerate(f1s):
        ax1.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    ax2.bar(approaches, fprs, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("False Positive Rate")
    ax2.set_title("FPR by Approach")
    for i, v in enumerate(fprs):
        ax2.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = CHARTS_DIR / f"master_comparison{suffix}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Dynamic threshold simulation (Phases B1 + B3). Test = paper-clean "
            "(unsuffixed outputs preserved for thesis); demo = operator-clean "
            "(outputs suffixed _demo so the Online Simulation panel reflects "
            "the operator stream without overwriting the test baseline)."
        )
    )
    parser.add_argument(
        "--split",
        choices=("test", "demo"),
        default="test",
        help="Frozen split to stream-process. Default: test.",
    )
    args = parser.parse_args()

    paths = _split_paths(args.split)
    suffix = paths["suffix"]

    sep = "=" * 72
    t0 = time.perf_counter()

    logger.info(sep)
    logger.info(
        "DYNAMIC THRESHOLD SIMULATION (Phases B1 + B3) — split=%s",
        args.split,
    )
    logger.info(sep)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_stream_data(args.split)
    n = len(data["y_true"])
    n_attacks = int((data["y_true"] == 1).sum())
    logger.info("Loaded %d samples (%d attacks)", n, n_attacks)
    logger.info("Static DAE threshold: %.6e", data["static_threshold"])
    logger.info("Time proxy: row index (no timestamp columns in dataset)")

    # ── B1.4  Static vs adaptive (default W, k) ──
    logger.info("")
    logger.info("── B1.4  Static vs Adaptive (W=%d, k=%d) ──",
                DEFAULT_WINDOW, DEFAULT_K)
    result = run_stream(
        data["re_scores"], data["y_true"],
        data["static_threshold"], data["train_benign_re"],
        W=DEFAULT_WINDOW, k=DEFAULT_K,
    )
    for label in ["static", "adaptive"]:
        m = result["final_metrics"][label]
        logger.info("  %-10s F1=%.4f  Prec=%.4f  Recall=%.4f  FPR=%.4f  FNR=%.4f",
                     label, m["f1"], m["precision"], m["recall"], m["fpr"], m["fnr"])

    # ── B1.5  Sensitivity grid ──
    logger.info("")
    logger.info("── B1.5  Sensitivity Grid (W × k) ──")
    grid_df = sensitivity_grid(
        data["re_scores"], data["y_true"],
        data["static_threshold"], data["train_benign_re"],
    )
    logger.info("\n%s", grid_df.to_string(index=False))

    # ── B1.6  Figures ──
    logger.info("")
    logger.info("── B1.6  Generating Figures ──")
    plot_threshold_over_time(result, suffix=suffix)
    plot_cumulative_f1(result, suffix=suffix)
    plot_sensitivity_heatmap(grid_df, suffix=suffix)

    # ── B3.1 + B3.2  Adaptive risk tiers ──
    logger.info("")
    logger.info("── B3.1/B3.2  Adaptive Risk Tier Thresholds ──")
    tier_result = run_adaptive_tiers(data["R"], data["y_true"], W=DEFAULT_WINDOW)
    logger.info("  Static  tiers: %s  metrics: %s",
                tier_result["static_distribution"], tier_result["static_metrics"])
    logger.info("  Adaptive tiers: %s  metrics: %s",
                tier_result["adaptive_distribution"], tier_result["adaptive_metrics"])
    plot_adaptive_tier_thresholds(tier_result, suffix=suffix)

    # ── B3.3  Combined ──
    logger.info("")
    logger.info("── B3.3  Combined (Sliding-Window + Feedback) ──")
    comparison = run_combined(
        data["R"], data["y_true"],
        data["c_detect"], data["d_crit"],
        data["s_data"], data["d_clinical_tier"],
        W=DEFAULT_WINDOW,
    )
    for approach in ["static", "adaptive", "feedback", "combined"]:
        m = comparison[approach]["metrics"]
        logger.info("  %-15s F1=%.4f  FPR=%.4f  FNR=%.4f",
                     approach, m["f1"], m["fpr"], m["fnr"])

    # ── B3.4  Master comparison ──
    logger.info("")
    logger.info("── B3.4  Master Comparison Table ──")
    table = build_master_table(comparison, tier_result)
    logger.info("\n%s", table)
    plot_master_comparison(comparison, suffix=suffix)

    # ── Save all results ──
    results = {
        "split": args.split,
        "b1_static_vs_adaptive": {
            "W": DEFAULT_WINDOW, "k": DEFAULT_K,
            "final_metrics": result["final_metrics"],
        },
        "b1_sensitivity_grid": grid_df.to_dict(orient="records"),
        "b3_adaptive_tiers": {
            "static_metrics": tier_result["static_metrics"],
            "adaptive_metrics": tier_result["adaptive_metrics"],
            "static_distribution": tier_result["static_distribution"],
            "adaptive_distribution": tier_result["adaptive_distribution"],
        },
        "b3_combined_comparison": comparison,
        "master_comparison_table_md": table,
        "time_proxy_note": (
            f"Row index used as temporal proxy — WUSTL-EHMS-2020 {args.split} "
            "parquet contains no timestamp columns."
        ),
    }
    out_name = f"dynamic_threshold_results{suffix}.json"
    out_path = OUTPUT_DIR / out_name
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("")
    logger.info("Saved: %s", out_name)

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("DYNAMIC THRESHOLD SIM COMPLETE — %.1fs (split=%s)", elapsed, args.split)
    logger.info(sep)
    logger.info("  %s", out_name)
    logger.info("  threshold_over_time%s.png", suffix)
    logger.info("  cumulative_f1%s.png", suffix)
    logger.info("  sensitivity_heatmap%s.png", suffix)
    logger.info("  adaptive_tier_thresholds%s.png", suffix)
    logger.info("  master_comparison%s.png", suffix)
    logger.info(sep)


if __name__ == "__main__":
    main()
