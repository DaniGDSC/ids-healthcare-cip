#!/usr/bin/env python3
"""Behavioral Drift Detection (Phase B2).

Monitors the DAE reconstruction-error distribution over a simulated
temporal stream and detects distributional drift using PSI and the
Kolmogorov–Smirnov test.

Tasks
-----
B2.1  PSI calculator.
B2.2  KS test detector.
B2.3  Run drift detection over the test stream.
B2.4  Simulate recalibration trigger.
B2.5  Generate drift detection figures.

Usage:
    python -m pipeline.drift_detection
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

import joblib
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "reports"
CHARTS_DIR = PROJECT_ROOT / "results" / "charts"

DEFAULT_WINDOW = 200
N_BINS = 10
PSI_THRESHOLD = 0.10
KS_ALPHA = 0.05


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_drift_data() -> dict:
    """Load DAE reconstruction errors and training benign baseline."""
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

    return {
        "re_scores": npz["reconstruction_error"],
        "y_true": risk_npz["y_true"],
        "train_benign_re": np.array(detector._train_errors),
        "static_threshold": float(detector._threshold),
    }


# ═══════════════════════════════════════════════════════════════════════
# B2.1  PSI calculator
# ═══════════════════════════════════════════════════════════════════════

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = N_BINS,
) -> float:
    """Population Stability Index between two distributions.

    PSI = sum_i (p_i - q_i) * ln(p_i / q_i)

    where p is the reference distribution and q is the current window.
    Bins are computed from the reference quantiles.
    """
    # Use reference quantile-based bins for stability
    bin_edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    # Normalize to proportions with small epsilon to avoid log(0)
    eps = 1e-8
    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))
    return psi


# ═══════════════════════════════════════════════════════════════════════
# B2.2  KS test detector
# ═══════════════════════════════════════════════════════════════════════

def compute_ks(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    """Two-sample KS test.  Returns (statistic, p_value)."""
    stat, p_val = ks_2samp(reference, current)
    return float(stat), float(p_val)


# ═══════════════════════════════════════════════════════════════════════
# B2.3  Run drift detection over the stream
# ═══════════════════════════════════════════════════════════════════════

def run_drift_detection(
    re_scores: np.ndarray,
    y_true: np.ndarray,
    train_benign_re: np.ndarray,
    static_threshold: float,
    W: int = DEFAULT_WINDOW,
) -> dict:
    """Slide a window of size W over the test stream and compute PSI/KS
    at each window position vs the training benign baseline.
    """
    n = len(re_scores)
    n_windows = n - W + 1
    if n_windows <= 0:
        raise ValueError(f"Window size {W} exceeds stream length {n}")

    psi_values = np.zeros(n_windows)
    ks_stats = np.zeros(n_windows)
    ks_pvalues = np.zeros(n_windows)
    window_centers = np.arange(W // 2, W // 2 + n_windows)

    for i in range(n_windows):
        window = re_scores[i : i + W]
        psi_values[i] = compute_psi(train_benign_re, window)
        ks_stats[i], ks_pvalues[i] = compute_ks(train_benign_re, window)

    # Identify drift events
    psi_drift = psi_values > PSI_THRESHOLD
    ks_drift = ks_pvalues < KS_ALPHA
    either_drift = psi_drift | ks_drift

    # Cluster consecutive drift points into events
    drift_events = []
    in_event = False
    for i in range(n_windows):
        if either_drift[i] and not in_event:
            in_event = True
            event_start = window_centers[i]
        elif not either_drift[i] and in_event:
            in_event = False
            drift_events.append({
                "start": int(event_start),
                "end": int(window_centers[i - 1]),
                "psi_triggered": bool(psi_drift[i - 1]),
                "ks_triggered": bool(ks_drift[i - 1]),
            })
    if in_event:
        drift_events.append({
            "start": int(event_start),
            "end": int(window_centers[n_windows - 1]),
            "psi_triggered": bool(psi_drift[n_windows - 1]),
            "ks_triggered": bool(ks_drift[n_windows - 1]),
        })

    logger.info("  Drift events detected: %d", len(drift_events))
    logger.info("  PSI > %.2f at %d/%d positions (%.1f%%)",
                PSI_THRESHOLD, psi_drift.sum(), n_windows,
                psi_drift.mean() * 100)
    logger.info("  KS p < %.2f at %d/%d positions (%.1f%%)",
                KS_ALPHA, ks_drift.sum(), n_windows,
                ks_drift.mean() * 100)

    return {
        "psi_values": psi_values,
        "ks_stats": ks_stats,
        "ks_pvalues": ks_pvalues,
        "window_centers": window_centers,
        "psi_drift": psi_drift,
        "ks_drift": ks_drift,
        "drift_events": drift_events,
        "n_windows": n_windows,
    }


# ═══════════════════════════════════════════════════════════════════════
# B2.4  Recalibration trigger
# ═══════════════════════════════════════════════════════════════════════

def simulate_recalibration(
    re_scores: np.ndarray,
    y_true: np.ndarray,
    train_benign_re: np.ndarray,
    static_threshold: float,
    drift_result: dict,
    W: int = DEFAULT_WINDOW,
) -> dict:
    """When drift is detected, recalibrate the DAE threshold using the
    current window as the new benign baseline.

    Compare detection performance before and after recalibration.
    """
    n = len(re_scores)

    # Static predictions (no recalibration)
    static_preds = (re_scores > static_threshold).astype(int)
    static_f1 = f1_score(y_true, static_preds, zero_division=0)

    # Recalibrated predictions
    recal_preds = np.zeros(n, dtype=int)
    current_threshold = static_threshold
    recal_threshold_history = np.full(n, static_threshold)
    recal_events = []

    for t in range(n):
        recal_preds[t] = int(re_scores[t] > current_threshold)
        recal_threshold_history[t] = current_threshold

        # Check if this position is inside a drift event
        for event in drift_result["drift_events"]:
            if t == event["start"]:
                # Recalibrate: use recent benign samples as new baseline
                lookback = max(0, t - W)
                recent_re = re_scores[lookback:t]
                recent_labels = y_true[lookback:t]
                recent_benign = recent_re[recent_labels == 0]

                if len(recent_benign) >= 20:
                    new_threshold = float(np.percentile(recent_benign, 95))
                    recal_events.append({
                        "sample_index": t,
                        "old_threshold": round(current_threshold, 8),
                        "new_threshold": round(new_threshold, 8),
                        "window_benign_count": len(recent_benign),
                    })
                    current_threshold = new_threshold
                    logger.info("    Recalibration at t=%d: %.6e → %.6e "
                                "(%d benign in window)",
                                t, recal_events[-1]["old_threshold"],
                                new_threshold, len(recent_benign))
                break

    recal_f1 = f1_score(y_true, recal_preds, zero_division=0)

    # Segment-level comparison around drift events
    segment_comparisons = []
    for event in drift_result["drift_events"]:
        s, e = event["start"], min(event["end"] + W, n)
        if s >= n or e <= s:
            continue
        sl = slice(s, e)
        yt = y_true[sl]
        if len(np.unique(yt)) < 2:
            continue
        segment_comparisons.append({
            "segment": f"{s}-{e}",
            "static_f1": round(float(f1_score(yt, static_preds[sl], zero_division=0)), 6),
            "recal_f1": round(float(f1_score(yt, recal_preds[sl], zero_division=0)), 6),
        })

    return {
        "static_f1": round(float(static_f1), 6),
        "recalibrated_f1": round(float(recal_f1), 6),
        "recalibration_events": recal_events,
        "segment_comparisons": segment_comparisons,
        "recal_threshold_history": recal_threshold_history,
    }


# ═══════════════════════════════════════════════════════════════════════
# B2.5  Drift detection figures
# ═══════════════════════════════════════════════════════════════════════

def plot_psi_over_time(drift: dict) -> None:
    """(a) PSI over time with threshold line."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = drift["window_centers"]
    ax.plot(x, drift["psi_values"], color="#3274A1", linewidth=0.8, alpha=0.9)
    ax.axhline(PSI_THRESHOLD, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"PSI threshold ({PSI_THRESHOLD})")
    # Shade drift regions
    for event in drift["drift_events"]:
        ax.axvspan(event["start"], event["end"], alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Sample Index (time proxy)")
    ax.set_ylabel("PSI")
    ax.set_title("Population Stability Index Over Time")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "drift_psi.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: drift_psi.png")


def plot_ks_over_time(drift: dict) -> None:
    """(b) KS p-value over time with threshold line."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = drift["window_centers"]
    ax.plot(x, drift["ks_pvalues"], color="#3274A1", linewidth=0.8, alpha=0.9)
    ax.axhline(KS_ALPHA, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"KS alpha ({KS_ALPHA})")
    for event in drift["drift_events"]:
        ax.axvspan(event["start"], event["end"], alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Sample Index (time proxy)")
    ax.set_ylabel("KS p-value")
    ax.set_title("Kolmogorov–Smirnov p-value Over Time")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "drift_ks.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: drift_ks.png")


def plot_annotated_timeline(
    drift: dict,
    recal: dict,
    y_true: np.ndarray,
) -> None:
    """(c) Annotated timeline showing drift events and recalibration points."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    x = drift["window_centers"]
    n = len(y_true)

    # Panel 1: PSI + KS combined
    ax1 = axes[0]
    ax1.plot(x, drift["psi_values"], color="#3274A1", linewidth=0.8,
             label="PSI", alpha=0.9)
    ax1.axhline(PSI_THRESHOLD, color="#e74c3c", linestyle="--",
                linewidth=1, alpha=0.6)
    ax1.set_ylabel("PSI")
    ax1.set_title("Drift Detection Timeline")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    for event in drift["drift_events"]:
        ax1.axvspan(event["start"], event["end"], alpha=0.15, color="#e74c3c")

    # Panel 2: Threshold with recalibration
    ax2 = axes[1]
    ax2.plot(np.arange(n), recal["recal_threshold_history"],
             color="#2ecc71", linewidth=1.2, label="Recalibrated threshold")
    ax2.axhline(recal["recal_threshold_history"][0], color="#95a5a6",
                linestyle="--", linewidth=1, label="Original threshold")
    for event in recal["recalibration_events"]:
        ax2.axvline(event["sample_index"], color="#e67e22", linestyle=":",
                     linewidth=1.5, alpha=0.8)
    ax2.set_ylabel("DAE Threshold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: Ground truth distribution
    ax3 = axes[2]
    attack_mask = y_true == 1
    # Rolling attack rate with window
    W = 200
    rolling_attack = np.convolve(attack_mask.astype(float),
                                  np.ones(W) / W, mode="same")
    ax3.plot(np.arange(n), rolling_attack, color="#e74c3c", linewidth=0.8,
             label=f"Rolling attack rate (W={W})")
    ax3.set_ylabel("Attack Rate")
    ax3.set_xlabel("Sample Index (time proxy)")
    ax3.legend(loc="upper right")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "drift_timeline.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: drift_timeline.png")


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
    logger.info("BEHAVIORAL DRIFT DETECTION (Phase B2)")
    logger.info(sep)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_drift_data()
    n = len(data["y_true"])
    logger.info("Loaded %d test samples, %d training benign RE values",
                n, len(data["train_benign_re"]))
    logger.info("Static DAE threshold: %.6e", data["static_threshold"])

    # ── B2.3  Drift detection over stream ──
    logger.info("")
    logger.info("── B2.3  Drift Detection (W=%d) ──", DEFAULT_WINDOW)
    drift = run_drift_detection(
        data["re_scores"], data["y_true"],
        data["train_benign_re"], data["static_threshold"],
        W=DEFAULT_WINDOW,
    )

    # ── B2.4  Recalibration ──
    logger.info("")
    logger.info("── B2.4  Recalibration Trigger ──")
    recal = simulate_recalibration(
        data["re_scores"], data["y_true"],
        data["train_benign_re"], data["static_threshold"],
        drift, W=DEFAULT_WINDOW,
    )
    logger.info("  Static F1:       %.6f", recal["static_f1"])
    logger.info("  Recalibrated F1: %.6f", recal["recalibrated_f1"])
    logger.info("  Recalibration events: %d", len(recal["recalibration_events"]))
    for seg in recal["segment_comparisons"]:
        logger.info("    Segment %s: static F1=%.4f → recal F1=%.4f",
                     seg["segment"], seg["static_f1"], seg["recal_f1"])

    # ── B2.5  Figures ──
    logger.info("")
    logger.info("── B2.5  Generating Figures ──")
    plot_psi_over_time(drift)
    plot_ks_over_time(drift)
    plot_annotated_timeline(drift, recal, data["y_true"])

    # ── Save results ──
    results = {
        "window_size": DEFAULT_WINDOW,
        "psi_threshold": PSI_THRESHOLD,
        "ks_alpha": KS_ALPHA,
        "drift_events": drift["drift_events"],
        "psi_summary": {
            "mean": round(float(drift["psi_values"].mean()), 6),
            "max": round(float(drift["psi_values"].max()), 6),
            "pct_above_threshold": round(float(drift["psi_drift"].mean() * 100), 2),
        },
        "ks_summary": {
            "mean_pvalue": round(float(drift["ks_pvalues"].mean()), 6),
            "min_pvalue": round(float(drift["ks_pvalues"].min()), 8),
            "pct_below_alpha": round(float(drift["ks_drift"].mean() * 100), 2),
        },
        "recalibration": {
            "static_f1": recal["static_f1"],
            "recalibrated_f1": recal["recalibrated_f1"],
            "events": recal["recalibration_events"],
            "segment_comparisons": recal["segment_comparisons"],
        },
        "time_proxy_note": (
            "Row index used as temporal proxy — WUSTL-EHMS-2020 test "
            "parquet contains no timestamp columns."
        ),
    }
    out_path = OUTPUT_DIR / "drift_detection_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("")
    logger.info("Saved: drift_detection_results.json")

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("DRIFT DETECTION COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    logger.info("  drift_detection_results.json")
    logger.info("  drift_psi.png")
    logger.info("  drift_ks.png")
    logger.info("  drift_timeline.png")
    logger.info(sep)


if __name__ == "__main__":
    main()
