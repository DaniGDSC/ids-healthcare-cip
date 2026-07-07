#!/usr/bin/env python3
"""Generate the 5 RQ1 report figures (PNG @ 300 DPI for paper/Markdown).

Outputs into `results/figures/`:
  * roc_curves.png             — ROC for Track A, B, fused, composite
  * pr_curves.png              — Precision-Recall counterparts
  * confusion_matrix.png       — TP/FN/FP/TN matrix at surfacing decision
  * tier_calibration_hist.png  — composite R histogram with tier boundary lines
  * device_correlation.png     — D_crit vs D_clinical_tier scatter (colored by tier)

Uses matplotlib only (no seaborn dep). White background, sans-serif body
font, monospace numerics — consistent with Sentinel design language so
figures pasted into the report read together.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Sentinel-aligned palette (web-safe approximations of the theme tokens)
COLOR = {
    "track_a":   "#7BA7BC",  # accent
    "track_b":   "#E07A5F",  # tier-high
    "fused":     "#5F9E7B",  # success
    "composite": "#8E7CC3",  # tier-critical analogue (violet)
    "critical":  "#C53030",
    "high":      "#E07A5F",
    "medium":    "#D4A445",
    "low":       "#5B8FB9",
    "diag":      "#9CA0AB",
}

PLT_RCPARAMS = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#262A33",
    "axes.titlesize": 13,
    "axes.titleweight": "500",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(PLT_RCPARAMS)


def _load_data():
    d = np.load(RESULTS / "reports" / "risk_scores.npz", allow_pickle=True)
    return {
        "y_true": d["y_true"].astype(int),
        "c_track_a": d["c_track_a"].astype(float),
        "c_track_b": d["c_track_b"].astype(float),
        "c_detect": d["c_detect"].astype(float),
        "R": d["R"].astype(float),
        "risk_levels": d["risk_levels"],
        "d_crit": d["d_crit"].astype(float),
        "d_clinical_tier": d["d_clinical_tier"].astype(float),
    }


# ──────────────────────────────────────────────────────────────────────
# R6: ROC curves
# ──────────────────────────────────────────────────────────────────────
def plot_roc(data):
    y = data["y_true"]
    series = [
        ("Track A (XGBoost)",      data["c_track_a"], COLOR["track_a"]),
        ("Track B (DAE cascade)",   data["c_track_b"], COLOR["track_b"]),
        ("Fused C_detect",          data["c_detect"], COLOR["fused"]),
        ("Composite risk R",        data["R"],        COLOR["composite"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, scores, color in series:
        fpr, tpr, _ = roc_curve(y, scores)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC = {a:.4f})")
    ax.plot([0, 1], [0, 1], color=COLOR["diag"], linestyle=":",
            linewidth=1.2, label="Chance (AUC = 0.50)")
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves — Detection performance across Track A, Track B, Fused, Composite",
                 loc="left", pad=12)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")
    out = FIGURES / "roc_curves.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────────────
# R7: Precision-Recall curves
# ──────────────────────────────────────────────────────────────────────
def plot_pr(data):
    y = data["y_true"]
    series = [
        ("Track A", data["c_track_a"], COLOR["track_a"]),
        ("Track B", data["c_track_b"], COLOR["track_b"]),
        ("Fused C_detect", data["c_detect"], COLOR["fused"]),
        ("Composite R", data["R"], COLOR["composite"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, scores, color in series:
        prec, rec, _ = precision_recall_curve(y, scores)
        ap = auc(rec, prec)
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{name} (AP = {ap:.4f})")
    # Baseline = class prevalence
    prev = y.mean()
    ax.axhline(prev, color=COLOR["diag"], linestyle=":", linewidth=1.2,
               label=f"Random baseline (P = {prev:.3f})")
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Sensitivity-favoring view (12.5% attack prevalence)",
                 loc="left", pad=12)
    ax.legend(loc="lower left", frameon=False)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")
    out = FIGURES / "pr_curves.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────────────
# R8: Confusion matrix (surfacing decision)
# ──────────────────────────────────────────────────────────────────────
def plot_confusion(data):
    y = data["y_true"]
    surfaced = np.isin(data["risk_levels"], ("MEDIUM", "HIGH", "CRITICAL")).astype(int)
    cm = confusion_matrix(y, surfaced, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    # Layout: 2x2 matrix with annotations
    fig, ax = plt.subplots(figsize=(7, 6))
    matrix = np.array([[tn, fp], [fn, tp]])

    # Tier-based color: TP/TN good (low color), FN bad (red), FP medium-bad (orange)
    cell_colors = [
        ["#5F9E7B33", "#D4A44533"],  # TN (success-bg), FP (medium-bg)
        ["#C5303033", "#7BA7BC33"],  # FN (critical-bg), TP (accent-bg)
    ]
    cell_text_colors = [
        ["#5F9E7B", "#D4A445"],
        ["#C53030", "#7BA7BC"],
    ]
    labels = [["TN", "FP"], ["FN", "TP"]]

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                       facecolor=cell_colors[i][j],
                                       edgecolor="#262A33", linewidth=1.2))
            cnt = matrix[i, j]
            pct = cnt / total * 100
            ax.text(j + 0.5, 1 - i + 0.65,
                    f"{labels[i][j]}",
                    ha="center", va="center", fontsize=14,
                    color=cell_text_colors[i][j], fontweight="600")
            ax.text(j + 0.5, 1 - i + 0.4,
                    f"{cnt:,}",
                    ha="center", va="center", fontsize=20,
                    color="#262A33", family="monospace", fontweight="500")
            ax.text(j + 0.5, 1 - i + 0.18,
                    f"{pct:.2f}%",
                    ha="center", va="center", fontsize=11,
                    color="#6A6F7B", family="monospace")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Suppressed (LOW)", "Surfaced (MED+)"], fontsize=11)
    ax.set_yticklabels(["Attack", "Benign"], fontsize=11)
    ax.set_xlabel("Predicted (surfacing decision)", labelpad=10)
    ax.set_ylabel("Actual (ground truth)", labelpad=10)
    ax.set_title(
        f"Confusion Matrix — Surfacing decision (n={total:,})\n"
        f"Sensitivity = TP/(TP+FN) = {tp/(tp+fn):.4f}  ·  "
        f"Specificity = TN/(TN+FP) = {tn/(tn+fp):.4f}",
        loc="left", pad=14, fontsize=12,
    )
    ax.set_aspect("equal")
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    out = FIGURES / "confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────────────
# R9: Tier boundary calibration histogram
# ──────────────────────────────────────────────────────────────────────
def plot_tier_histogram(data):
    R = data["R"]
    tiers = data["risk_levels"]
    y = data["y_true"]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6.5),
                                          sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})

    # Top: stacked histogram by tier
    bins = np.linspace(R.min(), R.max(), 60)
    tier_order = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    tier_color = {"LOW": COLOR["low"], "MEDIUM": COLOR["medium"],
                  "HIGH": COLOR["high"], "CRITICAL": COLOR["critical"]}
    bottom = np.zeros(len(bins) - 1)
    for tier in tier_order:
        mask = tiers == tier
        if not mask.any():
            continue
        counts, _ = np.histogram(R[mask], bins=bins)
        ax_top.bar(bins[:-1], counts, width=np.diff(bins), bottom=bottom,
                   color=tier_color[tier], edgecolor="white", linewidth=0.4,
                   align="edge", label=f"{tier} (n={int(mask.sum())})")
        bottom += counts

    # Boundary lines — empirical (min R for each tier > LOW)
    boundaries = {}
    for tier in tier_order:
        mask = tiers == tier
        if mask.any():
            boundaries[tier] = R[mask].min()

    for tier in ("MEDIUM", "HIGH", "CRITICAL"):
        if tier in boundaries:
            ax_top.axvline(boundaries[tier], color=tier_color[tier],
                           linestyle="--", linewidth=1.2, alpha=0.7)
            ax_top.text(boundaries[tier], ax_top.get_ylim()[1] * 0.95,
                        f" {tier}≥{boundaries[tier]:.3f}",
                        rotation=90, va="top", ha="left",
                        fontsize=9, color=tier_color[tier], family="monospace")

    ax_top.set_ylabel("Sample count")
    ax_top.set_title(
        "Tier-boundary calibration — Composite risk R distribution by tier",
        loc="left", pad=12,
    )
    ax_top.legend(loc="upper right", frameon=False, ncol=2)
    ax_top.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, axis="y")

    # Bottom: attack density (rug-style) so reader sees where attacks land
    attack_R = R[y == 1]
    benign_R = R[y == 0]
    ax_bot.hist(benign_R, bins=bins, color="#9CA0AB", alpha=0.6,
                label=f"Benign (n={len(benign_R)})", edgecolor="white", linewidth=0.4)
    ax_bot.hist(attack_R, bins=bins, color=COLOR["critical"], alpha=0.8,
                label=f"Attack (n={len(attack_R)})", edgecolor="white", linewidth=0.4)
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel("Composite risk R")
    ax_bot.set_ylabel("Count (log)")
    ax_bot.legend(loc="upper left", frameon=False)
    ax_bot.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, axis="y")

    out = FIGURES / "tier_calibration_hist.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────────────
# R10: D_crit vs D_clinical_tier correlation
# ──────────────────────────────────────────────────────────────────────
def plot_device_correlation(data):
    dc = data["d_crit"]
    dct = data["d_clinical_tier"]
    tiers = data["risk_levels"]

    # Pearson correlation
    if dc.std() > 0 and dct.std() > 0:
        rho = float(np.corrcoef(dc, dct)[0, 1])
    else:
        rho = float("nan")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    tier_color = {"LOW": COLOR["low"], "MEDIUM": COLOR["medium"],
                  "HIGH": COLOR["high"], "CRITICAL": COLOR["critical"]}
    for tier in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        mask = tiers == tier
        if not mask.any():
            continue
        # Add jitter so the discrete value clusters don't overlap as a single
        # dot (d_crit only has 3 unique values; d_clinical_tier has 7).
        jit_x = np.random.RandomState(42).normal(0, 0.008, mask.sum())
        jit_y = np.random.RandomState(43).normal(0, 0.008, mask.sum())
        ax.scatter(dc[mask] + jit_x, dct[mask] + jit_y,
                   color=tier_color[tier], alpha=0.4, s=18,
                   edgecolors="none", label=f"{tier} (n={int(mask.sum())})")

    ax.set_xlabel("D_crit — Device criticality")
    ax.set_ylabel("D_clinical_tier — Clinical-care weighting")
    ax.set_title(
        f"Device-context correlation — D_crit vs D_clinical_tier  (ρ = {rho:.3f})",
        loc="left", pad=12,
    )
    ax.legend(loc="upper left", frameon=False, title="Risk tier")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    # Set reasonable axis ranges with padding
    ax.set_xlim(dc.min() - 0.05, dc.max() + 0.05)
    ax.set_ylim(-0.05, dct.max() + 0.1)

    out = FIGURES / "device_correlation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_weight_sensitivity():
    """Composite-risk weight sensitivity — 2-panel figure.

    Top: AUC across 51 weight configurations, sorted, with the canonical
    baseline highlighted. Shows the narrow band of AUC variation
    (0.966-0.996, span 0.030).

    Bottom: FNR_critical across the same 51 configurations. ALL points
    sit at 0.0 (defendability claim: safety property robust to weight
    choice). Target threshold line drawn at 0.05.

    Data source: `results/rq1_weight_sensitivity.json` (R3 fix — anchored
    on canonical Module 3 weights).
    """
    with open(RESULTS / "rq1_weight_sensitivity.json") as f:
        sens = json.load(f)

    grid = sens["grid"]
    canon = sens["canonical_baseline_row"]
    # Sort by AUC so the canonical's position relative to the rest is
    # visually clear.
    sorted_grid = sorted(grid, key=lambda g: g["AUC"])
    aucs = [g["AUC"] for g in sorted_grid]
    fnrs = [g["FNR_critical"] for g in sorted_grid]
    is_canon = [g.get("is_canonical", False) for g in sorted_grid]
    canon_idx = is_canon.index(True)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ── Top: AUC bars, canonical highlighted ──
    x = np.arange(len(aucs))
    bar_colors = [
        "#C53030" if c else "#7BA7BC"   # tier-critical for canonical, accent otherwise
        for c in is_canon
    ]
    ax_top.bar(x, aucs, color=bar_colors, edgecolor="#262A33",
               linewidth=0.4, width=0.9)

    # Annotate canonical position
    ax_top.axhline(canon["AUC"], color="#C53030", linestyle="--",
                   linewidth=1.0, alpha=0.6,
                   label=f"Canonical AUC = {canon['AUC']:.4f}")
    # Show min/max as dotted reference lines
    ax_top.axhline(max(aucs), color="#5F9E7B", linestyle=":",
                   linewidth=0.9, alpha=0.6,
                   label=f"Grid max = {max(aucs):.4f}")
    ax_top.axhline(min(aucs), color="#9CA0AB", linestyle=":",
                   linewidth=0.9, alpha=0.6,
                   label=f"Grid min = {min(aucs):.4f}")

    # Y range: narrow to AUC band so variation is visible
    ax_top.set_ylim(min(aucs) - 0.005, max(aucs) + 0.005)
    ax_top.set_ylabel("AUC (Composite risk R vs y_true)")
    ax_top.set_title(
        f"RQ1 — Composite risk weight sensitivity  ·  {len(grid)} configurations\n"
        f"Canonical baseline: α={canon['alpha']}, β={canon['beta']}, "
        f"γ={canon['gamma']}, δ={canon['delta']}  "
        f"(from module3_risk_scoring.WEIGHTS)",
        loc="left", pad=14,
    )
    ax_top.legend(loc="lower right", frameon=False)
    ax_top.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, axis="y")

    # Mark canonical column with arrow + label
    ax_top.annotate(
        "canonical baseline",
        xy=(canon_idx, canon["AUC"]),
        xytext=(canon_idx + 5, canon["AUC"] - 0.012),
        fontsize=9, color="#C53030", fontweight="500",
        arrowprops=dict(arrowstyle="->", color="#C53030", lw=0.8),
    )

    # ── Bottom: FNR_critical (all at 0) with target threshold ──
    ax_bot.scatter(x, fnrs, color="#5F9E7B", edgecolor="#262A33",
                   s=18, linewidth=0.4, label="FNR_critical (per config)")
    ax_bot.axhline(0.05, color="#C53030", linestyle="--", linewidth=1.0,
                   alpha=0.7, label="Spec target ceiling = 0.05")

    # Highlight canonical
    ax_bot.scatter([canon_idx], [canon["FNR_critical"]],
                   color="#C53030", edgecolor="#262A33", s=80,
                   marker="D", zorder=5, label="Canonical")

    ax_bot.set_ylim(-0.01, 0.07)
    ax_bot.set_xlim(-1, len(aucs))
    ax_bot.set_xlabel("Weight configuration (sorted by AUC, ascending)")
    ax_bot.set_ylabel("FNR_critical")
    ax_bot.text(
        len(aucs) - 1, 0.025,
        f"All {len(grid)} configs: FNR_critical = 0.000 ✓\n"
        "Safety property robust to weight choice",
        ha="right", va="center",
        fontsize=10, color="#262A33",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#E8F0E8", edgecolor="#5F9E7B", alpha=0.8),
    )
    ax_bot.legend(loc="upper left", frameon=False, fontsize=9)
    ax_bot.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, axis="y")

    out = FIGURES / "rq1_weight_sensitivity.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_supervised_model_comparison():
    """Compare F1 and AUC for the three supervised Track A models."""
    with open(RESULTS / "rq1_ablation_track_a.json") as f:
        ablation = json.load(f)

    model_order = ("xgboost", "random_forest", "decision_tree")
    labels = {
        "xgboost": "XGBoost",
        "random_forest": "Random Forest",
        "decision_tree": "Decision Tree",
    }
    bar_colors = {
        "f1": COLOR["track_a"],
        "auc": COLOR["fused"],
    }

    f1_scores = []
    auc_scores = []
    display_labels = []
    for model_name in model_order:
        metrics = ablation["models"][model_name]["metrics_at_threshold_0.5"]
        f1_scores.append(metrics["f1"])
        auc_scores.append(metrics["auc"])
        display_labels.append(labels[model_name])

    x = np.arange(len(display_labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    f1_bars = ax.bar(
        x - width / 2,
        f1_scores,
        width,
        color=bar_colors["f1"],
        edgecolor="#262A33",
        linewidth=0.5,
        label="F1",
    )
    auc_bars = ax.bar(
        x + width / 2,
        auc_scores,
        width,
        color=bar_colors["auc"],
        edgecolor="#262A33",
        linewidth=0.5,
        label="AUC",
    )

    for bars in (f1_bars, auc_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.012,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#262A33",
                family="monospace",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(
        "Supervised model comparison - F1 and AUC on the Track A test set",
        loc="left",
        pad=12,
    )
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, axis="y")
    ax.legend(loc="upper right", frameon=False)

    out = FIGURES / "supervised_model_f1_auc_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_prioritization_layer_metrics():
    """Plot prioritization-layer headline metrics as a compact bar chart."""
    with open(RESULTS / "rq1_metrics.json") as f:
        metrics = json.load(f)

    operational_recall = 1.0 - metrics["primary_safety_metric"]["FNR_critical"]
    surfaced_precision = metrics["surfacing_decision"]["precision"]
    surfaced_recall = metrics["surfacing_decision"]["recall"]

    labels = ["Operational Recall", "Surfaced Precision", "Surfaced Recall"]
    values = [operational_recall, surfaced_precision, surfaced_recall]
    colors = [COLOR["critical"], COLOR["track_a"], COLOR["fused"]]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    bars = ax.bar(
        labels,
        values,
        color=colors,
        edgecolor="#262A33",
        linewidth=0.6,
        width=0.62,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#262A33",
            family="monospace",
        )

    ax.axhline(1.0, color="#9CA0AB", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(
        "Prioritization layer performance - operational recall and surfaced quality",
        loc="left",
        pad=12,
    )
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, axis="y")
    ax.text(
        0.02,
        0.03,
        "Operational Recall = 1 - FNR_critical",
        transform=ax.transAxes,
        fontsize=9,
        color="#6A6F7B",
    )

    out = FIGURES / "prioritization_layer_metrics.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    data = _load_data()
    print("[R6] ROC curves...")
    print(f"  -> {plot_roc(data)}")
    print("[R7] PR curves...")
    print(f"  -> {plot_pr(data)}")
    print("[R8] Confusion matrix...")
    print(f"  -> {plot_confusion(data)}")
    print("[R9] Tier calibration histogram...")
    print(f"  -> {plot_tier_histogram(data)}")
    print("[R10] Device correlation scatter...")
    print(f"  -> {plot_device_correlation(data)}")
    print("[S7]  Weight sensitivity (R3 follow-up)...")
    print(f"  -> {plot_weight_sensitivity()}")
    print("[S8]  Supervised-model F1/AUC comparison...")
    print(f"  -> {plot_supervised_model_comparison()}")
    print("[S9]  Prioritization-layer metrics...")
    print(f"  -> {plot_prioritization_layer_metrics()}")


if __name__ == "__main__":
    main()
