#!/usr/bin/env python3
"""Generate RQ3 report figures (PNG @ 300 DPI for paper/Markdown).

Outputs:
  * results/figures/rq3_mve_comparison.png   — Group A (baseline) vs B (with MVE)
  * results/figures/rq3_per_role_accuracy.png — per-role with/without XAI bars
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sentinel palette (matches RQ1 figures for cross-paper consistency)
COLOR = {
    "baseline":   "#9CA0AB",  # neutral gray
    "with_mve":   "#7BA7BC",  # accent
    "with_xai":   "#5F9E7B",  # success
    "without_xai":"#E07A5F",  # tier-high (warning tone)
    "diag":       "#9CA0AB",
}

plt.rcParams.update({
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
})


def plot_mve_comparison():
    """Group A (baseline, no MVE) vs Group B (with MVE) — 4 panel metric set."""
    with open(PROJECT_ROOT / "analysis/outputs/rq3_primary.json") as f:
        data = json.load(f)

    p = data["primary_metric_composite_accuracy"]
    summ_A = data["summary_group_A"]
    summ_B = data["summary_group_B"]

    metrics = [
        ("Composite\naccuracy", summ_A["mean_composite_accuracy"], summ_B["mean_composite_accuracy"], 0.0, 1.0),
        ("Severity\naccuracy",  summ_A["mean_severity_accuracy"],  summ_B["mean_severity_accuracy"],  0.0, 1.0),
        ("Action\naccuracy",    summ_A["mean_action_accuracy"],    summ_B["mean_action_accuracy"],    0.0, 1.0),
        ("Catastrophic\nmiss rate", summ_A["catastrophic_miss_rate"], summ_B["catastrophic_miss_rate"], 0.0, 0.10),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))

    for ax, (label, a_val, b_val, ymin, ymax) in zip(axes, metrics):
        x = np.array([0, 1])
        vals = [a_val, b_val]
        colors_bar = [COLOR["baseline"], COLOR["with_mve"]]
        bars = ax.bar(x, vals, color=colors_bar, edgecolor="#262A33", linewidth=0.8, width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (ymax - ymin) * 0.025,
                    f"{v:.3f}" if v < 1 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10, family="monospace",
                    color="#262A33", fontweight="500")
        ax.set_xticks(x)
        ax.set_xticklabels(["Group A\n(baseline)", "Group B\n(with MVE)"])
        ax.set_title(label, fontsize=11)
        ax.set_ylim(ymin, ymax * 1.15)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, axis="y")

    fig.suptitle(
        f"RQ3 — User study group comparison (n={summ_A['n_participants']} per group)\n"
        f"Composite accuracy: Mann-Whitney p = {p['mann_whitney_p_value']:.5f}, "
        f"Cohen's d = {p['cohens_d']:.2f} ({p['effect_size']} effect) · verdict = {p['verdict']}",
        fontsize=13, y=1.05, x=0.04, ha="left", fontweight="500",
    )

    out = OUT_DIR / "rq3_mve_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_per_role_accuracy():
    """Per-role with-XAI vs without-XAI bars, with p-values annotated."""
    with open(PROJECT_ROOT / "analysis/outputs/rq3_per_role.json") as f:
        data = json.load(f)

    comp = data["role_comparison_with_vs_without_xai"]
    roles = sorted(comp.keys())
    with_vals = [comp[r]["accuracy_with_xai"] for r in roles]
    without_vals = [comp[r]["accuracy_without_xai"] for r in roles]
    deltas = [comp[r]["delta_pp"] for r in roles]
    pvals = [comp[r]["mann_whitney_p_value"] for r in roles]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(roles))
    bar_w = 0.36
    bars_wo = ax.bar(x - bar_w/2, without_vals, bar_w,
                     color=COLOR["without_xai"], edgecolor="#262A33",
                     linewidth=0.8, label="Without XAI")
    bars_w = ax.bar(x + bar_w/2, with_vals, bar_w,
                    color=COLOR["with_xai"], edgecolor="#262A33",
                    linewidth=0.8, label="With XAI")

    # Annotate bar values
    for bars, vals in [(bars_wo, without_vals), (bars_w, with_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=10, family="monospace", color="#262A33")

    # Annotate Δpp + p-value above each role
    for i, (d, p) in enumerate(zip(deltas, pvals)):
        sig_marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(i, max(with_vals[i], without_vals[i]) + 0.06,
                f"Δ {d:+.1f}pp  {sig_marker}\np = {p:.4f}",
                ha="center", va="bottom", fontsize=9,
                family="monospace", color="#262A33")

    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in roles])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Decision accuracy")
    ax.set_title(
        "RQ3 per-role breakdown — with vs without XAI (M6 study)",
        loc="left", pad=14,
    )
    ax.legend(loc="lower right", frameon=False, ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, axis="y")
    ax.text(
        0.0, -0.18,
        "Significance: *** p<0.001  ·  ** p<0.01  ·  * p<0.05  ·  ns = not significant",
        transform=ax.transAxes, fontsize=9, color="#6A6F7B",
    )

    out = OUT_DIR / "rq3_per_role_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    print("[12] Plotting per-role accuracy...")
    print(f"  → {plot_per_role_accuracy()}")
    print("[13] Plotting MVE comparison...")
    print(f"  → {plot_mve_comparison()}")


if __name__ == "__main__":
    main()
