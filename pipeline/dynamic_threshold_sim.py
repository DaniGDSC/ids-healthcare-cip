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
    python -m pipeline.dynamic_threshold_sim
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

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

# Defaults
DEFAULT_WINDOW = 100
DEFAULT_K = 3
WINDOW_GRID = [50, 100, 200, 500]
K_GRID = [2, 3, 4]

DEFAULT_THRESHOLDS = {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}


# ═══════════════════════════════════════════════════════════════════════
# Data loading (B1.1)
# ═══════════════════════════════════════════════════════════════════════

def load_stream_data() -> dict:
    """Load test data sorted by row index as temporal proxy.

    The WUSTL-EHMS-2020 test parquet has no timestamp columns, so row
    order serves as the time proxy.  This is acknowledged as a
    limitation in the thesis.
    """
    npz = dict(np.load(
        PROJECT_ROOT / "results" / "models" / "dae_test_predictions.npz",
        allow_pickle=True,
    ))
    risk_npz = dict(np.load(
        OUTPUT_DIR / "risk_scores.npz", allow_pickle=True,
    ))
    detector = joblib.load(
        PROJECT_ROOT / "results" / "models" / "dae_detector.pkl",
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
        "a_patient": risk_npz["a_patient"],
    }


# ═══════════════════════════════════════════════════════════════════════
# B1.2  Sliding-window statistics
# ═══════════════════════════════════════════════════════════════════════

def _median_mad(window: np.ndarray) -> tuple[float, float]:
    """Return (median, MAD) of the values in window."""
    med = float(np.median(window))
    mad = float(np.median(np.abs(window - med)))
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

    for t in range(n):
        re = re_scores[t]

        # Static prediction
        static_preds[t] = int(re > static_threshold)

        # Adaptive threshold
        window_arr = np.array(benign_window)
        med, mad = _median_mad(window_arr)
        thresh_t = med + k * mad if mad > 0 else med * (1 + k * 0.1)
        adaptive_thresh[t] = thresh_t
        adaptive_preds[t] = int(re > thresh_t)

        # Update benign window: add sample only if classified benign by
        # the *static* threshold (avoids feedback loop contamination)
        if re <= static_threshold:
            benign_window.append(re)

        # Cumulative metrics (need at least 1 sample of each class)
        sl = slice(0, t + 1)
        yt = y_true[sl]
        if len(np.unique(yt)) > 1:
            cum_f1_static[t] = f1_score(yt, static_preds[sl], zero_division=0)
            cum_f1_adaptive[t] = f1_score(yt, adaptive_preds[sl], zero_division=0)

            n_seen = t + 1
            fp_s = int(((static_preds[sl] == 1) & (yt == 0)).sum())
            fn_s = int(((static_preds[sl] == 0) & (yt == 1)).sum())
            fp_a = int(((adaptive_preds[sl] == 1) & (yt == 0)).sum())
            fn_a = int(((adaptive_preds[sl] == 0) & (yt == 1)).sum())
            cum_fpr_static[t] = fp_s / n_seen
            cum_fnr_static[t] = fn_s / n_seen
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
    """Grid search over W × k, returning final F1 and FPR for each."""
    rows = []
    for W in WINDOW_GRID:
        for k_val in K_GRID:
            result = run_stream(re_scores, y_true, static_threshold,
                                train_benign_re, W=W, k=k_val)
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

def plot_threshold_over_time(result: dict) -> None:
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
    plt.savefig(CHARTS_DIR / "threshold_over_time.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: threshold_over_time.png")


def plot_cumulative_f1(result: dict) -> None:
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
    plt.savefig(CHARTS_DIR / "cumulative_f1.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: cumulative_f1.png")


def plot_sensitivity_heatmap(grid_df: pd.DataFrame) -> None:
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
    plt.savefig(CHARTS_DIR / "sensitivity_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: sensitivity_heatmap.png")


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
        warr = np.array(benign_window) if len(benign_window) >= 10 else R[:W]
        t_med = float(np.percentile(warr, 75))
        t_hi = float(np.percentile(warr, 90))
        t_cr = float(np.percentile(warr, 95))
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
    def _tier_dist(levels):
        dist = {}
        for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            dist[tier] = int((levels == tier).sum())
        return dist

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
    a_patient: np.ndarray,
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
    thresholds = dict(DEFAULT_THRESHOLDS)
    for _ in range(n_feedback_iters):
        levels = assign_risk_levels(R, thresholds)
        fb = FeedbackLoop()
        for idx in range(len(R)):
            gt = "attack" if y_true[idx] == 1 else "benign"
            fb.record(f"A-{idx}", gt, str(levels[idx]), float(R[idx]), [])
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

    def _dist(levels):
        return {t: int((levels == t).sum()) for t in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]}

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

def plot_adaptive_tier_thresholds(tier_result: dict) -> None:
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
    plt.savefig(CHARTS_DIR / "adaptive_tier_thresholds.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: adaptive_tier_thresholds.png")


def plot_master_comparison(comparison: dict) -> None:
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
    plt.savefig(CHARTS_DIR / "master_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: master_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sep = "=" * 72
    t0 = time.perf_counter()

    logger.info(sep)
    logger.info("DYNAMIC THRESHOLD SIMULATION (Phases B1 + B3)")
    logger.info(sep)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_stream_data()
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
    plot_threshold_over_time(result)
    plot_cumulative_f1(result)
    plot_sensitivity_heatmap(grid_df)

    # ── B3.1 + B3.2  Adaptive risk tiers ──
    logger.info("")
    logger.info("── B3.1/B3.2  Adaptive Risk Tier Thresholds ──")
    tier_result = run_adaptive_tiers(data["R"], data["y_true"], W=DEFAULT_WINDOW)
    logger.info("  Static  tiers: %s  metrics: %s",
                tier_result["static_distribution"], tier_result["static_metrics"])
    logger.info("  Adaptive tiers: %s  metrics: %s",
                tier_result["adaptive_distribution"], tier_result["adaptive_metrics"])
    plot_adaptive_tier_thresholds(tier_result)

    # ── B3.3  Combined ──
    logger.info("")
    logger.info("── B3.3  Combined (Sliding-Window + Feedback) ──")
    comparison = run_combined(
        data["R"], data["y_true"],
        data["c_detect"], data["d_crit"],
        data["s_data"], data["a_patient"],
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
    plot_master_comparison(comparison)

    # ── Save all results ──
    results = {
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
            "Row index used as temporal proxy — WUSTL-EHMS-2020 test "
            "parquet contains no timestamp columns."
        ),
    }
    out_path = OUTPUT_DIR / "dynamic_threshold_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("")
    logger.info("Saved: dynamic_threshold_results.json")

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("DYNAMIC THRESHOLD SIM COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    logger.info("  dynamic_threshold_results.json")
    logger.info("  threshold_over_time.png")
    logger.info("  cumulative_f1.png")
    logger.info("  sensitivity_heatmap.png")
    logger.info("  adaptive_tier_thresholds.png")
    logger.info("  master_comparison.png")
    logger.info(sep)


if __name__ == "__main__":
    main()
