#!/usr/bin/env python3
"""RQ2 report figures — PNG @ 300 DPI for paper/Markdown."""
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

COLOR = {
    "stable":     "#5F9E7B",   # success
    "unstable":   "#E07A5F",   # tier-high
    "target":     "#C53030",   # tier-critical
    "mode_a":     "#8E7CC3",   # violet
    "mode_b":     "#7BA7BC",   # accent
    "diag":       "#9CA0AB",
    "background": "#F7F8FA",
}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#262A33",
    "axes.titlesize": 13, "axes.titleweight": "500",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
})


def plot_stability_histogram():
    with open(PROJECT_ROOT / "results/rq2_shap_stability.json") as f:
        data = json.load(f)
    scores = [s["stability_score"] for s in data["per_sample"]]
    s = data["summary"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bins = np.linspace(0, 1, 26)
    n, _, patches = ax.hist(scores, bins=bins, color=COLOR["mode_b"],
                            edgecolor="#262A33", linewidth=0.5, alpha=0.85)

    # Mark threshold
    thr = data["_meta"]["stability_threshold_jaccard"]
    ax.axvline(thr, color=COLOR["target"], linestyle="--", linewidth=1.3,
               label=f"Stability threshold (Jaccard ≥ {thr:.2f})")
    # Mark mean
    ax.axvline(s["mean_stability_score"], color="#262A33", linestyle=":",
               linewidth=1.3,
               label=f"Mean = {s['mean_stability_score']:.3f}")

    # Target line
    ax.axvline(s["target_mean_stability"], color=COLOR["diag"], linestyle="-.",
               linewidth=1.0, alpha=0.5,
               label=f"Spec target (mean ≥ {s['target_mean_stability']:.2f})")

    ax.set_xlim(0, 1)
    ax.set_xlabel("SHAP stability score (Jaccard top-k overlap under perturbation)")
    ax.set_ylabel("Number of samples")
    ax.set_title(
        "RQ2.b — SHAP stability distribution\n"
        f"n={s['n_stable']+s['n_unstable']} attack samples · pct_stable = {s['pct_stable']:.1f}% "
        f"(spec target {s['target_pct_stable']:.0f}%)",
        loc="left", pad=12,
    )
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, axis="y")

    out = OUT_DIR / "rq2_shap_stability_hist.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_mve_alignment():
    with open(PROJECT_ROOT / "results/rq2_mve_shap_alignment.json") as f:
        data = json.load(f)

    ma = data["mode_a_llm_narrative"]
    mb = data["mode_b_rule_based"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # 3 metric groups, 2 modes each
    x = np.arange(3)
    bar_w = 0.36
    mode_a_vals = [ma["contains_top1_pct"], ma["contains_at_least_2_pct"], ma["contains_all_3_pct"]]
    mode_b_vals = [mb["contains_top1_pct"], mb["contains_at_least_2_pct"], mb["contains_all_3_pct"]]

    bars_a = ax.bar(x - bar_w/2, mode_a_vals, bar_w,
                    color=COLOR["mode_a"], edgecolor="#262A33", linewidth=0.8,
                    label="Mode A (LLM narrative)")
    bars_b = ax.bar(x + bar_w/2, mode_b_vals, bar_w,
                    color=COLOR["mode_b"], edgecolor="#262A33", linewidth=0.8,
                    label="Mode B (rule-based)")

    for bars, vals in [(bars_a, mode_a_vals), (bars_b, mode_b_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                    f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=10, family="monospace", color="#262A33")

    # Target lines
    ax.axhline(95, color=COLOR["target"], linestyle="--", linewidth=1.0,
               alpha=0.6, label="Spec target (≥2 ≥ 95%)")
    ax.axhline(80, color=COLOR["target"], linestyle=":", linewidth=1.0,
               alpha=0.5, label="Spec target (all 3 ≥ 80%)")

    ax.set_xticks(x)
    ax.set_xticklabels(["Contains top-1\nSHAP feature",
                        "Contains ≥2 of 3\ntop SHAP features",
                        "Contains all 3\ntop SHAP features"])
    ax.set_ylabel("% of evaluated samples")
    ax.set_ylim(0, 115)
    ax.set_title(
        f"RQ2.b — MVE Layer 1 vs top SHAP features alignment\n"
        f"n={ma['n_total']} explanations per mode · gap between current "
        f"implementation and spec targets shown in red",
        loc="left", pad=12,
    )
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, axis="y")

    out = OUT_DIR / "rq2_mve_alignment.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    print("[11] Plotting SHAP stability histogram...")
    print(f"  → {plot_stability_histogram()}")
    print("[12] Plotting MVE-SHAP alignment bar chart...")
    print(f"  → {plot_mve_alignment()}")


if __name__ == "__main__":
    main()
