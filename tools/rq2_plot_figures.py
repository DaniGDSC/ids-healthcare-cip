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
    mb_large = data.get("mode_b_rule_based_large_n", {}).get("metrics", {})
    has_large = bool(mb_large)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 3 metric groups; 2 or 3 series depending on large-N availability
    x = np.arange(3)
    n_series = 3 if has_large else 2
    bar_w = 0.85 / n_series

    mode_a_vals = [ma["contains_top1_pct"], ma["contains_at_least_2_pct"], ma["contains_all_3_pct"]]
    mode_b_vals = [mb["contains_top1_pct"], mb["contains_at_least_2_pct"], mb["contains_all_3_pct"]]

    series: list[tuple[list, str, str]] = [
        (mode_a_vals, COLOR["mode_a"], f"Mode A LLM narrative (n={ma['n_total']})"),
        (mode_b_vals, COLOR["mode_b"], f"Mode B rule-based (n={mb['n_total']})"),
    ]
    large_vals = None
    if has_large:
        large_vals = [
            mb_large["contains_top1_pct"],
            mb_large["contains_at_least_2_pct"],
            mb_large["contains_all_3_pct"],
        ]
        series.append((large_vals, "#5F9E7B",
                       f"Mode B rule-based large-N (n={mb_large['n_total']})"))

    # Offsets center the side-by-side bars on each x tick.
    offsets = [bar_w * (i - (len(series) - 1) / 2) for i in range(len(series))]

    for offset, (vals, color, label) in zip(offsets, series):
        bars = ax.bar(x + offset, vals, bar_w,
                      color=color, edgecolor="#262A33", linewidth=0.8,
                      label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                    f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=9, family="monospace", color="#262A33")

    # 95% CI error bars on the large-N series, when present
    if has_large:
        ci_t1 = mb_large["ci95_top1_pct"]
        ci_2 = mb_large["ci95_at_least_2_pct"]
        ci_3 = mb_large["ci95_all_3_pct"]
        lows = [v - low for v, (low, high) in zip(large_vals, [ci_t1, ci_2, ci_3])]
        highs = [high - v for v, (low, high) in zip(large_vals, [ci_t1, ci_2, ci_3])]
        ax.errorbar(x + offsets[-1], large_vals, yerr=[lows, highs],
                    fmt="none", ecolor="#262A33", capsize=4, capthick=1.0,
                    elinewidth=1.0, alpha=0.7)

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

    subtitle_extra = ""
    if has_large:
        ci_2_l, ci_2_h = mb_large["ci95_at_least_2_pct"]
        subtitle_extra = (
            f" · 95% CI on Mode B large-N ≥2: [{ci_2_l:.1f}%, {ci_2_h:.1f}%] "
            "rules out small-sample fluke"
        )
    n_parts = f"{ma['n_total']}/{mb['n_total']}"
    if has_large:
        n_parts += f"/{mb_large.get('n_total', '')}"
    ax.set_title(
        "RQ2.b — MVE Layer 1 vs top SHAP features alignment\n"
        f"Post G1+G2+G6 fix · n={n_parts} per series{subtitle_extra}",
        loc="left", pad=12,
    )
    ax.legend(loc="lower right", frameon=False, ncol=1, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, axis="y")

    out = OUT_DIR / "rq2_mve_alignment.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_explanation_quality_summary():
    """Plot a compact explanation-quality summary across technical and user metrics."""
    with open(PROJECT_ROOT / "results/rq2_mve_shap_alignment.json") as f:
        alignment = json.load(f)
    with open(PROJECT_ROOT / "results/rq2_shap_stability.json") as f:
        stability = json.load(f)
    with open(PROJECT_ROOT / "results/reports/evaluation_results.json") as f:
        evaluation = json.load(f)

    values = [
        alignment["mode_b_rule_based_large_n"]["metrics"]["contains_at_least_2_pct"] / 100.0,
        stability["summary"]["pct_stable"] / 100.0,
        evaluation["metrics"]["with_xai"]["likert_comprehensibility"] / 5.0,
        evaluation["metrics"]["with_xai"]["likert_actionability"] / 5.0,
    ]
    raw_labels = [
        "Faithfulness\n(>=2 of top-3 SHAP)",
        "Stability\n(% stable)",
        "Comprehensibility\n(Likert / 5)",
        "Actionability\n(Likert / 5)",
    ]
    raw_value_labels = ["100.0%", "86.2%", "4.03/5", "4.31/5"]
    colors = [COLOR["mode_b"], COLOR["stable"], COLOR["mode_a"], "#D4A445"]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    x = np.arange(len(values))
    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor="#262A33",
        linewidth=0.6,
        width=0.64,
    )

    for bar, label in zip(bars, raw_value_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            family="monospace",
            color="#262A33",
        )

    ax.axhline(0.95, color=COLOR["target"], linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(3.45, 0.955, "faithfulness target", ha="right", va="bottom", fontsize=8.5, color=COLOR["target"])
    ax.axhline(0.80, color=COLOR["diag"], linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(3.45, 0.805, "stability target", ha="right", va="bottom", fontsize=8.5, color=COLOR["diag"])

    ax.set_xticks(x)
    ax.set_xticklabels(raw_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Normalized score")
    ax.set_title(
        "Explanation quality summary - technical faithfulness and user-facing clarity",
        loc="left",
        pad=12,
    )
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, axis="y")
    ax.text(
        0.02,
        0.03,
        "Likert metrics are normalized to a 0-1 scale for comparison.",
        transform=ax.transAxes,
        fontsize=9,
        color="#6A6F7B",
    )

    out = OUT_DIR / "rq2_explanation_quality_summary.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_decision_support_metrics():
    """Plot core decision-support metrics with vs without XAI."""
    with open(PROJECT_ROOT / "results/reports/evaluation_results.json") as f:
        evaluation = json.load(f)["metrics"]

    labels = [
        "Accuracy",
        "Confidence",
        "Trust",
        "Usefulness",
        "Actionability",
    ]
    with_xai = [
        evaluation["with_xai"]["decision_accuracy"],
        evaluation["with_xai"]["mean_confidence"] / 5.0,
        evaluation["with_xai"]["likert_trust"] / 5.0,
        evaluation["with_xai"]["likert_usefulness"] / 5.0,
        evaluation["with_xai"]["likert_actionability"] / 5.0,
    ]
    without_xai = [
        evaluation["without_xai"]["decision_accuracy"],
        evaluation["without_xai"]["mean_confidence"] / 5.0,
        evaluation["without_xai"]["likert_trust"] / 5.0,
        evaluation["without_xai"]["likert_usefulness"] / 5.0,
        evaluation["without_xai"]["likert_actionability"] / 5.0,
    ]
    with_labels = ["0.92", "3.55/5", "4.35/5", "4.49/5", "4.31/5"]
    without_labels = ["0.75", "2.97/5", "3.07/5", "3.43/5", "3.25/5"]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.2, 5.9))

    bars_without = ax.bar(
        x - width / 2,
        without_xai,
        width,
        color=COLOR["diag"],
        edgecolor="#262A33",
        linewidth=0.6,
        label="Without XAI",
    )
    bars_with = ax.bar(
        x + width / 2,
        with_xai,
        width,
        color=COLOR["mode_b"],
        edgecolor="#262A33",
        linewidth=0.6,
        label="With XAI",
    )

    for bars, value_labels in ((bars_without, without_labels), (bars_with, with_labels)):
        for bar, text_label in zip(bars, value_labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                text_label,
                ha="center",
                va="bottom",
                fontsize=9,
                family="monospace",
                color="#262A33",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Normalized score")
    ax.set_title(
        "Decision support metrics - operator performance with vs without XAI",
        loc="left",
        pad=12,
    )
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, axis="y")
    ax.text(
        0.02,
        0.03,
        "Confidence and Likert metrics are normalized to a 0-1 scale for comparison.",
        transform=ax.transAxes,
        fontsize=9,
        color="#6A6F7B",
    )

    out = OUT_DIR / "rq2_decision_support_metrics.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    print("[11] Plotting SHAP stability histogram...")
    print(f"  → {plot_stability_histogram()}")
    print("[12] Plotting MVE-SHAP alignment bar chart...")
    print(f"  → {plot_mve_alignment()}")
    print("[13] Plotting explanation-quality summary chart...")
    print(f"  → {plot_explanation_quality_summary()}")
    print("[14] Plotting decision-support metrics chart...")
    print(f"  → {plot_decision_support_metrics()}")


if __name__ == "__main__":
    main()
