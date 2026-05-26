"""Module 6 thesis figures — Task 6.8."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = PROJECT_ROOT / "results/charts"


def generate_thesis_figures(metrics: dict, stats: dict, responses: list) -> None:
    """Generate thesis-ready figures."""
    df = pd.DataFrame(responses)

    # Figure 1: Likert scores comparison (with vs without XAI)
    dimensions = ["trust", "usefulness", "comprehensibility", "actionability"]
    with_scores = [metrics["with_xai"][f"likert_{d}"] for d in dimensions]
    without_scores = [metrics["without_xai"][f"likert_{d}"] for d in dimensions]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dimensions))
    w = 0.35
    bars1 = ax.bar(x - w / 2, without_scores, w, label="Without XAI", color="#95a5a6", alpha=0.8)
    bars2 = ax.bar(x + w / 2, with_scores, w, label="With XAI", color="#3274A1", alpha=0.8)

    for bar, val in zip(bars1, without_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, with_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", fontsize=9)

    for i, d in enumerate(dimensions):
        key = f"likert_{d}"
        if key in stats and stats[key].get("significant"):
            ax.annotate("*", xy=(i, max(with_scores[i], without_scores[i]) + 0.3),
                        ha="center", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dimensions])
    ax.set_ylabel("Mean Likert Score (1-5)")
    ax.set_title("Explanation Quality: With vs Without XAI")
    ax.set_ylim(0, 5.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "likert_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: likert_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    acc_with = metrics["with_xai"]["decision_accuracy"] * 100
    acc_without = metrics["without_xai"]["decision_accuracy"] * 100
    bars = ax.bar(["Without XAI", "With XAI"], [acc_without, acc_with],
                  color=["#95a5a6", "#3274A1"], alpha=0.8, width=0.5)
    for bar, val in zip(bars, [acc_without, acc_with]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Decision Accuracy (%)")
    ax.set_title("Decision Accuracy: With vs Without XAI Explanations")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: accuracy_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    data_with = df[df["condition"] == "with_xai"]["decision_time_sec"].values
    data_without = df[df["condition"] == "without_xai"]["decision_time_sec"].values
    bp = ax.boxplot([data_without, data_with],
                    tick_labels=["Without XAI", "With XAI"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#95a5a6")
    bp["boxes"][1].set_facecolor("#3274A1")
    for b in bp["boxes"]:
        b.set_alpha(0.7)
    ax.set_ylabel("Decision Time (seconds)")
    ax.set_title("Time-to-Decision: With vs Without XAI")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "decision_time_boxplot.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: decision_time_boxplot.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    roles = sorted(metrics["per_role"].keys())
    x = np.arange(len(roles))
    w = 0.35
    acc_with_role = [metrics["per_role"][r]["with_xai_accuracy"] * 100 for r in roles]
    acc_without_role = [metrics["per_role"][r]["without_xai_accuracy"] * 100 for r in roles]
    ax.bar(x - w / 2, acc_without_role, w, label="Without XAI", color="#95a5a6", alpha=0.8)
    ax.bar(x + w / 2, acc_with_role, w, label="With XAI", color="#3274A1", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in roles])
    ax.set_ylabel("Decision Accuracy (%)")
    ax.set_title("Decision Accuracy by Stakeholder Role")
    ax.set_ylim(0, 105)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "accuracy_by_role.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: accuracy_by_role.png")

    _plot_radar_chart(df, stats)
    _plot_decision_time_by_tier(df)
    _plot_accuracy_by_tier(df)
    _plot_effect_sizes(stats)


def _plot_radar_chart(df: pd.DataFrame, stats: dict) -> None:
    """Radar chart: 4 Likert dimensions × 3 roles (with-XAI condition)."""
    dimensions = ["trust", "usefulness", "comprehensibility", "actionability"]
    labels = [d.capitalize() for d in dimensions]
    roles = sorted(df["participant_role"].unique())
    role_colors = {"analyst": "#3274A1", "clinician": "#2ecc71", "administrator": "#e67e22"}

    with_df = df[df["condition"] == "with_xai"]
    role_means = {}
    for role in roles:
        rdf = with_df[with_df["participant_role"] == role]
        role_means[role] = [float(rdf[f"likert_{d}"].mean()) for d in dimensions]

    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for role in roles:
        values = role_means[role] + role_means[role][:1]
        ax.plot(angles, values, "o-", linewidth=2,
                color=role_colors.get(role, "#999"),
                label=role.capitalize())
        ax.fill(angles, values, alpha=0.15,
                color=role_colors.get(role, "#999"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
    ax.set_title("Likert Ratings by Role (With XAI)", y=1.08, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "radar_likert_by_role.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("  Chart: radar_likert_by_role.png")


def _plot_decision_time_by_tier(df: pd.DataFrame) -> None:
    """Boxplot of decision time grouped by alert tier and A/B condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = ["without_xai", "with_xai"]
    colors = ["#95a5a6", "#3274A1"]
    positions = []
    data_groups = []
    tick_labels = []

    for i, cond in enumerate(conditions):
        cond_df = df[df["condition"] == cond]
        for j, role in enumerate(sorted(df["participant_role"].unique())):
            role_df = cond_df[cond_df["participant_role"] == role]
            data_groups.append(role_df["decision_time_sec"].values)
            positions.append(j * 3 + i)
            if i == 0:
                tick_labels.append(role.capitalize())

    bp = ax.boxplot(data_groups, positions=positions, widths=0.8,
                    patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % 2])
        patch.set_alpha(0.7)

    ax.set_xticks([j * 3 + 0.5 for j in range(len(tick_labels))])
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Decision Time (seconds)")
    ax.set_title("Decision Time by Role and Condition")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#95a5a6", alpha=0.7, label="Without XAI"),
        Patch(facecolor="#3274A1", alpha=0.7, label="With XAI"),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "decision_time_by_role.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: decision_time_by_role.png")


def _plot_accuracy_by_tier(df: pd.DataFrame) -> None:
    """Per-condition accuracy broken down by correct_action (proxy for tier)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    action_order = ["dismiss", "monitor", "investigate", "isolate", "escalate"]
    action_labels = [a.capitalize() for a in action_order]

    x = np.arange(len(action_order))
    w = 0.35

    acc_table = (
        df.groupby(["condition", "correct_action"])["decision_correct"]
        .mean()
        .unstack(level="correct_action", fill_value=0.0)
    )

    for i, (cond, color, label) in enumerate([
        ("without_xai", "#95a5a6", "Without XAI"),
        ("with_xai", "#3274A1", "With XAI"),
    ]):
        cond_row = acc_table.loc[cond] if cond in acc_table.index else None
        accs = [
            float(cond_row[a]) * 100 if (cond_row is not None and a in cond_row) else 0.0
            for a in action_order
        ]
        offset = -w / 2 + i * w
        bars = ax.bar(x + offset, accs, w, label=label, color=color, alpha=0.8)
        for bar, val in zip(bars, accs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(action_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Decision Accuracy by Correct Action (Tier Proxy)")
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "accuracy_by_tier.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: accuracy_by_tier.png")


def _plot_effect_sizes(stats: dict) -> None:
    """Forest plot of Cohen's d effect sizes for all measures."""
    measures = []
    effects = []
    ci_lo = []
    ci_hi = []

    for measure, result in stats.items():
        if "cohens_d" not in result:
            continue
        d = result["cohens_d"]
        n = 15
        se = np.sqrt(2 / n + d ** 2 / (2 * n))
        measures.append(measure.replace("likert_", "").replace("_", " ").title())
        effects.append(d)
        ci_lo.append(d - 1.96 * se)
        ci_hi.append(d + 1.96 * se)

    if not measures:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(measures) * 0.6)))
    y = np.arange(len(measures))

    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in effects]
    ax.barh(y, effects, color=colors, alpha=0.7, height=0.5)
    ax.errorbar(effects, y, xerr=[
        [e - lo for e, lo in zip(effects, ci_lo)],
        [hi - e for e, hi in zip(effects, ci_hi)],
    ], fmt="none", ecolor="black", capsize=3)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axvline(-0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axvline(0.8, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axvline(-0.8, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(measures)
    ax.set_xlabel("Cohen's d (With XAI − Without XAI)")
    ax.set_title("Effect Sizes: XAI Impact on Evaluation Measures")
    ax.text(0.55, -0.6, "medium", fontsize=7, color="gray", ha="center")
    ax.text(0.85, -0.6, "large", fontsize=7, color="gray", ha="center")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "effect_size_forest.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: effect_size_forest.png")


__all__ = ["generate_thesis_figures", "CHARTS_DIR"]
