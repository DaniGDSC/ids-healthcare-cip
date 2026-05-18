"""RQ1 figures (RQ1_pipeline.md §7.1).

Reads ``results/reports/risk_scores.npz`` + ``results/rq1_metrics.json``,
writes four PDFs into ``results/figures/``:

  - roc_curves.pdf
  - pr_curves.pdf
  - confusion_matrix.pdf
  - tier_boundary_histogram.pdf

Pure plotting — no model loading, no metrics re-computation beyond what
sklearn needs for curve construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NPZ = REPO_ROOT / "results/reports/risk_scores.npz"
METRICS = REPO_ROOT / "results/rq1_metrics.json"
FIG_DIR = REPO_ROOT / "results/figures"


def make_roc(data, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, key in [
        ("Track A (XGBoost)", "c_track_a"),
        ("Track B (DAE)", "c_track_b"),
        ("Fused (max)", "c_detect"),
    ]:
        fpr, tpr, _ = roc_curve(data["y_true"], data[key])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Track A, Track B, Fused")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close(fig)


def make_pr(data, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, key in [
        ("Track A (XGBoost)", "c_track_a"),
        ("Track B (DAE)", "c_track_b"),
        ("Fused (max)", "c_detect"),
    ]:
        p, r, _ = precision_recall_curve(data["y_true"], data[key])
        ax.plot(r, p, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — Track A, Track B, Fused")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close(fig)


def make_confusion(metrics: dict, out: Path) -> None:
    cm_dict = metrics["headline"]["confusion_matrix"]
    cm = np.array(
        [[cm_dict["tn"], cm_dict["fp"]],
         [cm_dict["fn"], cm_dict["tp"]]]
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Benign", "Attack"]
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Test Split")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close(fig)


def make_tier_histogram(data, out: Path) -> None:
    """Histogram of R values with tier boundary lines.

    Visual argument for the calibration of the 0.40 / 0.60 / 0.80
    boundaries (L4 in the limitations block).
    """
    R = data["R"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(R, bins=50, alpha=0.7, edgecolor="black")
    for boundary, label, color in [
        (0.40, "MEDIUM", "gold"),
        (0.60, "HIGH", "orange"),
        (0.80, "CRITICAL", "red"),
    ]:
        ax.axvline(
            boundary, linestyle="--", color=color, linewidth=1.5,
            label=f"{label} threshold ({boundary})",
        )
    ax.set_xlabel("Composite risk R")
    ax.set_ylabel("Count")
    ax.set_title("Risk score distribution with tier boundaries (test split)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(NPZ, allow_pickle=False)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))

    make_roc(data, FIG_DIR / "roc_curves.pdf")
    make_pr(data, FIG_DIR / "pr_curves.pdf")
    make_confusion(metrics, FIG_DIR / "confusion_matrix.pdf")
    make_tier_histogram(data, FIG_DIR / "tier_boundary_histogram.pdf")
    print(f"Wrote 4 PDFs to {FIG_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
