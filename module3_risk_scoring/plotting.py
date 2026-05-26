"""Plot generators for Module 3 risk-scoring artefacts.

Every plot function accepts an explicit ``output_dir`` parameter so
tests can redirect outputs to ``tmp_path``. Production callers (the
CLI in ``module3_risk_scores.py``) pass the canonical
``results/charts/`` directory.

Categorical color palettes are derived from the input data rather than
hardcoded to ``{Normal, Spoofing, Data Alteration}`` — new attack
categories cycle through a default palette instead of falling back to
ambiguous blue.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import RISK_THRESHOLDS  # noqa: E402

logger = logging.getLogger(__name__)

# Categorical-category color cycle (used when a new attack category
# appears that's not in the canonical {Normal, Spoofing, Data Alteration}).
_CATEGORY_PALETTE: tuple = (
    "#2ecc71",  # Normal — green
    "#e74c3c",  # Spoofing — red
    "#8e44ad",  # Data Alteration — purple
    "#3498db",
    "#f39c12",
    "#1abc9c",
    "#e67e22",
    "#34495e",
)


def _category_color_map(categories: List[str]) -> Dict[str, str]:
    """Stable color assignment derived from the actual category list."""
    canonical = {
        "Normal": "#2ecc71",
        "Spoofing": "#e74c3c",
        "Data Alteration": "#8e44ad",
    }
    palette_iter = itertools.cycle(_CATEGORY_PALETTE)
    mapping: Dict[str, str] = {}
    for c in categories:
        if c in canonical:
            mapping[c] = canonical[c]
        else:
            mapping[c] = next(palette_iter)
    return mapping


# ── plot_risk_distribution ───────────────────────────────────────────


def plot_risk_distribution(
    R: np.ndarray,
    levels: np.ndarray,
    *,
    output_dir: Path,
) -> Path:
    """Histogram of risk scores with level boundaries."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"LOW": "#f1c40f", "MEDIUM": "#e67e22",
              "HIGH": "#e74c3c", "CRITICAL": "#8e44ad"}
    boundaries = [(0, 0.4, "LOW"), (0.4, 0.6, "MEDIUM"),
                  (0.6, 0.8, "HIGH"), (0.8, 1.0, "CRITICAL")]
    for lo, hi, label in boundaries:
        ax.axvspan(lo, hi, alpha=0.15, color=colors[label])

    ax.hist(R, bins=100, edgecolor="black", linewidth=0.5, alpha=0.8, color="#3274A1")
    for thresh, label in RISK_THRESHOLDS:
        count = (levels == label).sum()
        ax.axvline(thresh, color=colors[label], linestyle="--", linewidth=1.5)
        ax.text(thresh + 0.01, ax.get_ylim()[1] * 0.9, f"{label}\n(n={count})",
                fontsize=8, color=colors[label], fontweight="bold")
    low_count = (levels == "LOW").sum()
    ax.text(0.02, ax.get_ylim()[1] * 0.9, f"LOW\n(n={low_count})",
            fontsize=8, color=colors["LOW"], fontweight="bold")

    ax.set_xlabel("Composite Risk Score R")
    ax.set_ylabel("Count")
    ax.set_title("Risk Score Distribution with Alert Priority Levels")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "risk_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


# ── plot_component_breakdown ─────────────────────────────────────────


def plot_component_breakdown(
    contributions: dict,
    *,
    output_dir: Path,
) -> Path | None:
    """Stacked bar of mean weighted contributions per risk level."""
    comp_names = ["C_detect", "D_crit", "S_data", "D_clinical_tier"]
    colors = ["#C44E52", "#3274A1", "#55A868", "#CCB974"]
    level_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    active_levels = [
        l for l in level_order
        if contributions["per_level"][l].get("count", 0) > 0
    ]
    if not active_levels:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(active_levels))
    bottom = np.zeros(len(active_levels))

    for cn, color in zip(comp_names, colors):
        vals = [
            contributions["per_level"][l]["mean_contributions"].get(cn, 0)
            for l in active_levels
        ]
        ax.bar(x, vals, bottom=bottom, color=color, label=cn, width=0.6)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{l}\n(n={contributions['per_level'][l]['count']})"
        for l in active_levels
    ])
    ax.set_ylabel("Mean Weighted Contribution")
    ax.set_title("Component Breakdown by Risk Level")
    ax.legend(loc="upper left")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "component_breakdown.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


# ── plot_dual_track_heatmap ──────────────────────────────────────────


def plot_dual_track_heatmap(fusion: dict, *, output_dir: Path) -> Path:
    """2×2 heatmap showing dual-track detection quadrants."""
    q = fusion["quadrants"]
    matrix_total = np.array([
        [q["both_flag"]["true_attacks"], q["only_dae"]["true_attacks"]],
        [q["only_xgboost"]["true_attacks"], q["neither"]["true_attacks"]],
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix_total, cmap="YlOrRd", aspect="auto")
    labels = [
        [
            f"Both flag\n{q['both_flag']['true_attacks']} attacks\n({q['both_flag']['total']} total)",
            f"Only DAE\n{q['only_dae']['true_attacks']} attacks\n({q['only_dae']['total']} total)",
        ],
        [
            f"Only XGBoost\n{q['only_xgboost']['true_attacks']} attacks\n({q['only_xgboost']['total']} total)",
            f"Neither\n{q['neither']['true_attacks']} attacks\n({q['neither']['total']} total)",
        ],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=10, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["XGBoost Flags", "XGBoost Clear"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["DAE Flags", "DAE Clear"])
    ax.set_title("Dual-Track Detection Quadrants (True Attacks)")
    plt.colorbar(im, label="True Attacks")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "dual_track_venn.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


# ── plot_component_scatter ───────────────────────────────────────────


def plot_component_scatter(
    c_sup: np.ndarray,
    c_anom: np.ndarray,
    y_true: np.ndarray,
    *,
    output_dir: Path,
) -> Path:
    """Scatter of C_supervised vs C_anomaly colored by ground truth."""
    fig, ax = plt.subplots(figsize=(10, 8))
    benign = y_true == 0
    attack = y_true == 1
    ax.scatter(c_sup[benign], c_anom[benign], c="#2ecc71",
               alpha=0.3, s=10, label="Benign")
    ax.scatter(c_sup[attack], c_anom[attack], c="#e74c3c",
               alpha=0.6, s=20, label="Attack")
    ax.set_xlabel("C_supervised (XGBoost probability)")
    ax.set_ylabel("C_anomaly (DAE normalized score)")
    ax.set_title("Track A vs Track B — Complementary Detection Zones")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "component_scatter.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


# ── plot_risk_by_category (Y6 — colors derived from input) ──────────


def plot_risk_by_category(
    R: np.ndarray,
    attack_cats: np.ndarray,
    y_true: np.ndarray,
    *,
    output_dir: Path,
) -> Path:
    """Box plot of risk scores by attack category."""
    categories = []
    scores = []
    normal_mask = y_true == 0
    categories.extend(["Normal"] * int(normal_mask.sum()))
    scores.extend(R[normal_mask].tolist())

    if attack_cats is not None:
        cats_str = attack_cats.astype(str)
        attack_mask = y_true == 1
        for cat in sorted(np.unique(cats_str[attack_mask])):
            mask = (cats_str == cat) & attack_mask
            categories.extend([cat] * int(mask.sum()))
            scores.extend(R[mask].tolist())

    df = pd.DataFrame({"Category": categories, "Risk Score": scores})
    fig, ax = plt.subplots(figsize=(10, 6))
    cats = df["Category"].unique().tolist()
    color_map = _category_color_map(cats)
    bp_data = [df[df["Category"] == c]["Risk Score"].values for c in cats]
    bp = ax.boxplot(bp_data, tick_labels=cats, patch_artist=True, widths=0.5)
    for patch, cat in zip(bp["boxes"], cats):
        patch.set_facecolor(color_map.get(cat, "#3274A1"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Composite Risk Score R")
    ax.set_title("Risk Score Distribution by Attack Category")
    ax.axhline(0.4, color="orange", linestyle="--", alpha=0.5, label="MEDIUM threshold")
    ax.axhline(0.6, color="red", linestyle="--", alpha=0.5, label="HIGH threshold")
    ax.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "risk_by_category.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


# ── plot_risk_by_label ───────────────────────────────────────────────


def plot_risk_by_label(
    R: np.ndarray,
    y_true: np.ndarray,
    *,
    output_dir: Path,
) -> Path:
    """Overlaid histograms of R for benign vs attack — verify separation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(R[y_true == 0], bins=80, alpha=0.6, color="#2ecc71",
            label="Benign", density=True)
    ax.hist(R[y_true == 1], bins=80, alpha=0.6, color="#e74c3c",
            label="Attack", density=True)
    ax.axvline(0.40, color="orange", linestyle="--", label="MEDIUM threshold")
    ax.axvline(0.60, color="red", linestyle="--", label="HIGH threshold")
    ax.set_xlabel("Composite Risk Score R")
    ax.set_ylabel("Density")
    ax.set_title("Risk Score Distribution by True Label — Separation Quality")
    ax.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "risk_by_label.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


# ── plot_weight_sensitivity_curve (extracted from analysis) ─────────


def plot_weight_sensitivity_curve(
    per_component: Dict[str, List[Dict[str, float]]],
    best_auroc: float,
    *,
    output_dir: Path,
) -> Path:
    """Plot per-component AUROC sweep curve from weight sensitivity analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#C44E52", "#3274A1", "#55A868", "#CCB974"]
    for (label, curve), color in zip(per_component.items(), colors):
        ws = [c["weight"] for c in curve]
        aucs = [c["auroc"] for c in curve]
        ax.plot(ws, aucs, "o-", color=color, label=label,
                linewidth=2, markersize=5)
    ax.axhline(best_auroc, color="black", linestyle=":", alpha=0.5,
               label=f"Best={best_auroc:.4f}")
    ax.set_xlabel("Component Weight")
    ax.set_ylabel("AUROC (R as binary classifier)")
    ax.set_title("Weight Sensitivity Analysis — AUROC vs Component Weight")
    ax.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "weight_sensitivity.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("  Chart: %s", out.name)
    return out


__all__ = [
    "plot_risk_distribution",
    "plot_component_breakdown",
    "plot_dual_track_heatmap",
    "plot_component_scatter",
    "plot_risk_by_category",
    "plot_risk_by_label",
    "plot_weight_sensitivity_curve",
]
