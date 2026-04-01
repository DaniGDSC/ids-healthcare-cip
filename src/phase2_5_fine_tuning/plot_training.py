"""Plot learning curves from finetuned_results.json.

Usage::
    python -m src.phase2_5_fine_tuning.plot_training
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def plot_training_curves(
    results_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    """Plot training loss curves from finetuned_results.json."""
    if results_path is None:
        results_path = PROJECT_ROOT / "data" / "phase2_5" / "finetuned_results.json"
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "phase2_5" / "training_curves.png"

    with open(results_path) as f:
        results = json.load(f)

    history = results.get("training_history", [])
    if not history:
        print("No training history found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss per stage
    stages = []
    for h in history:
        stages.append({
            "name": h.get("phase", "Unknown"),
            "train_loss": h.get("final_train_loss", 0),
            "val_loss": h.get("final_val_loss", 0),
            "epochs": h.get("epochs_run", 0),
        })

    names = [s["name"][:20] for s in stages]
    train_losses = [s["train_loss"] for s in stages]
    val_losses = [s["val_loss"] for s in stages]

    x = range(len(stages))
    axes[0].bar([i - 0.15 for i in x], train_losses, 0.3, label="Train Loss", color="#3498db")
    axes[0].bar([i + 0.15 for i in x], val_losses, 0.3, label="Val Loss", color="#e74c3c")
    axes[0].set_xlabel("Training Stage")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss per Training Stage")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    axes[0].legend()

    # Plot metrics comparison (baseline vs finetuned)
    comparison = results.get("baseline_comparison", {})
    if comparison:
        metrics = list(comparison.keys())[:6]
        baseline_vals = [comparison[m].get("baseline", 0) for m in metrics]
        finetuned_vals = [comparison[m].get("finetuned", 0) for m in metrics]

        x2 = range(len(metrics))
        axes[1].bar([i - 0.15 for i in x2], baseline_vals, 0.3, label="Baseline", color="#95a5a6")
        axes[1].bar([i + 0.15 for i in x2], finetuned_vals, 0.3, label="Finetuned", color="#2ecc71")
        axes[1].set_xlabel("Metric")
        axes[1].set_ylabel("Value")
        axes[1].set_title("Baseline vs Finetuned")
        axes[1].set_xticks(list(x2))
        axes[1].set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=7)
        axes[1].legend()
        axes[1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_training_curves()
