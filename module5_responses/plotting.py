"""Module 5 visualisations — distribution, precision, funnel, sankey proxy."""
from __future__ import annotations

import logging
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import ACTION_CATALOGUE  # noqa: E402
from .loaders import CHARTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)


def plot_response_distribution(records: list) -> None:
    """Bar chart of response actions by risk level."""
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    all_actions = sorted(
        ACTION_CATALOGUE.keys(), key=lambda a: ACTION_CATALOGUE[a]["cost"]
    )
    colors_list = plt.cm.Set2(np.linspace(0, 1, len(all_actions)))

    action_by_level = {lv: {a: 0 for a in all_actions} for lv in levels}
    for rec in records:
        level = rec["risk_level"]
        if level in action_by_level:
            for a in rec["response"]["actions"]:
                if a in action_by_level[level]:
                    action_by_level[level][a] += 1

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(levels))
    width = 0.8 / len(all_actions)

    for i, action in enumerate(all_actions):
        vals = [action_by_level[lv][action] for lv in levels]
        if max(vals) > 0:
            ax.bar(
                x + i * width, vals, width,
                label=action.replace("_", " "),
                color=colors_list[i], alpha=0.85,
            )

    ax.set_xticks(x + width * len(all_actions) / 2)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Number of Alerts")
    ax.set_title("Adaptive Response Actions by Risk Level")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "response_actions_by_level.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: response_actions_by_level.png")


def plot_precision_by_level(stats: dict) -> None:
    """Precision (true attack rate) per risk level."""
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    tp = [stats["true_positives_by_level"].get(lv, 0) for lv in levels]
    fp = [stats["false_positives_by_level"].get(lv, 0) for lv in levels]
    prec = [stats["precision_by_level"].get(lv, 0) for lv in levels]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(levels))
    w = 0.35
    ax.bar(x - w / 2, tp, w, label="True Attacks", color="#e74c3c", alpha=0.8)
    ax.bar(x + w / 2, fp, w, label="False Positives", color="#95a5a6", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, prec, "ko-", linewidth=2, markersize=8, label="Precision")
    ax2.set_ylabel("Precision")
    ax2.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Count")
    ax.set_title("Alert Precision by Risk Level")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "precision_by_level.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: precision_by_level.png")


def plot_escalation_funnel(stats: dict) -> None:
    """Horizontal funnel of alert volumes per tier."""
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts = [stats["alerts_by_level"].get(lv, 0) for lv in levels]
    colors_map = {
        "LOW": "#2ecc71", "MEDIUM": "#f1c40f", "HIGH": "#e74c3c", "CRITICAL": "#8e44ad",
    }
    # Derived from TIER_POLICIES["X"]["max_response_min"] so they stay in sync.
    from .config import TIER_POLICIES
    sla = [TIER_POLICIES[lv]["max_response_min"] for lv in levels]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(
        levels, counts,
        color=[colors_map[lv] for lv in levels],
        alpha=0.8, edgecolor="black", linewidth=0.5,
    )
    for bar, _level, count, s in zip(bars, levels, counts, sla):
        ax.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            f"n={count} | SLA ≤{s}min",
            va="center", fontsize=9,
        )
    ax.set_xlabel("Number of Alerts")
    ax.set_title("Response Escalation Funnel")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "response_escalation_funnel.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: response_escalation_funnel.png")


def plot_effectiveness_by_action(effectiveness: dict) -> None:
    """Precision per mitigation action."""
    prop = effectiveness["proportionality_analysis"]
    prop = [p for p in prop if p["total"] > 0]
    if not prop:
        return

    names = [p["action"].replace("_", "\n") for p in prop]
    precs = [p["precision"] for p in prop]
    costs = [p["cost"] for p in prop]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        names, precs,
        color=plt.cm.RdYlGn_r(np.asarray(costs)),
        alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    ax.set_ylabel("Precision (true attack rate)")
    ax.set_title(
        "Response Proportionality — Costly Actions Should Have Higher Precision"
    )
    ax.set_ylim(0, 1.05)
    for bar, p in zip(bars, prop):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={p['total']}", ha="center", fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "effectiveness_by_action.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: effectiveness_by_action.png")


def plot_response_sankey(audit_records: list) -> None:
    """Grouped-bar Sankey proxy: risk level × outcome flow."""
    flows: dict = defaultdict(int)
    for rec in audit_records:
        level = rec["risk_level"]
        actions = rec["recommended_actions"]
        costs = [(a, ACTION_CATALOGUE.get(a, {}).get("cost", 0)) for a in actions]
        primary = max(costs, key=lambda x: x[1])[0] if costs else "log_event"
        outcome = rec["simulated_outcome"]["outcome"]
        flows[(level, primary, outcome)] += 1

    # M5-7: pre-aggregate to (level, outcome) → count in one pass.
    level_outcome: dict = defaultdict(int)
    for (lv, _a, oc), v in flows.items():
        level_outcome[(lv, oc)] += v

    outcomes = sorted(set(k[2] for k in flows))
    outcome_colors = {
        "threat_contained": "#2ecc71",
        "benign_logged": "#3498db",
        "false_positive_isolated": "#e67e22",
        "threat_logged_not_mitigated": "#e74c3c",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    x = np.arange(len(levels))
    width = 0.8 / max(len(outcomes), 1)

    for i, outcome in enumerate(outcomes):
        vals = [level_outcome[(lv, outcome)] for lv in levels]
        ax.bar(
            x + i * width, vals, width,
            label=outcome.replace("_", " "),
            color=outcome_colors.get(outcome, "#999"),
            alpha=0.85,
        )

    ax.set_xticks(x + width * len(outcomes) / 2)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Count")
    ax.set_title("Risk Level → Simulated Outcome Flow")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "response_sankey.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: response_sankey.png")


__all__ = [
    "plot_response_distribution",
    "plot_precision_by_level",
    "plot_escalation_funnel",
    "plot_effectiveness_by_action",
    "plot_response_sankey",
]
