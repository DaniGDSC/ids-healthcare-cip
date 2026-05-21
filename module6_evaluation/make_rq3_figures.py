"""Generate paper-ready PDFs for RQ3 from canonical sub-files.

Path C adaptations (vs RQ3_MERGE_AND_FIGURES_SPEC.md):
  - Lowercase role enums (biomed_engineer, IT_generalist, nurse_manager)
  - drop decision_time (absent in LLM-persona data)
  - escalation field names: rate_A, rate_B, p_value, test (not the spec's
    escalation_rate_A, chi2_p_value, recommended_test)
  - User-study figure title labels "Method 1: LLM-persona simulation"

Usage:
  python -m module6_evaluation.make_rq3_figures              # all
  python -m module6_evaluation.make_rq3_figures --only invariants
  python -m module6_evaluation.make_rq3_figures --list
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "results" / "figures"

FIGURES = {
    "invariants":  "rq3_invariant_matrix.pdf",
    "user_study":  "rq3_per_role_with_escalation.pdf",
}


def _load_json(rel: str) -> Optional[dict]:
    p = REPO_ROOT / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _skip(name: str, reason: str) -> None:
    print(f"  [SKIP] {name}: {reason}")


def _saved(name: str, path: Path) -> None:
    print(f"  [OK]   {name} -> {path.relative_to(REPO_ROOT)}")


# ─── Figure 1: Invariant pass/fail matrix ──────────────────────


def make_invariants(out_path: Path) -> None:
    data = _load_json("results/rq3_invariant_evidence.json")
    if not data:
        return _skip("invariants", "rq3_invariant_evidence.json missing")
    invs = data.get("invariants") or []
    if not invs:
        return _skip("invariants", "no invariants in evidence file")

    invs_sorted = sorted(invs, key=lambda i: i["id"])
    n = len(invs_sorted)
    color_map = {
        "pass": "#3aaa35",
        "fail": "#c0392b",
        "pending": "#bbbbbb",
        "documented": "#888888",
        "no_tests_found": "#f39c12",
        "unknown": "#dddddd",
    }
    marker_map = {
        "pass": "PASS", "fail": "FAIL", "pending": "pend",
        "documented": "doc", "no_tests_found": "?", "unknown": "?",
    }

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.5 * n + 2)))

    rqs = [1, 2, 3]
    y = np.arange(n)[::-1]
    cell_w, cell_h = 0.8, 0.7

    for i, inv in enumerate(invs_sorted):
        for j, rq in enumerate(rqs):
            applies = rq in (inv.get("serves_rqs") or [])
            if not applies:
                ax.add_patch(plt.Rectangle(
                    (j - cell_w / 2, y[i] - cell_h / 2),
                    cell_w, cell_h, fill=False,
                    edgecolor="#dddddd", linewidth=0.5,
                ))
            else:
                status = inv.get("_overall_status", "unknown")
                color = color_map.get(status, "#dddddd")
                ax.add_patch(plt.Rectangle(
                    (j - cell_w / 2, y[i] - cell_h / 2),
                    cell_w, cell_h, color=color, alpha=0.85,
                    edgecolor="black", linewidth=0.6,
                ))
                label = marker_map.get(status, "?")
                ax.text(j, y[i], label, ha="center", va="center",
                        fontsize=8,
                        color="white" if status in {"pass", "fail"}
                        else "black", fontweight="bold")

    for i, inv in enumerate(invs_sorted):
        title = (inv["title"] if len(inv["title"]) <= 55
                 else inv["title"][:52] + "...")
        severity = inv.get("severity", "")
        sev_marker = "[!]" if severity == "safety_critical" else ""
        ax.text(-0.7, y[i], f"{sev_marker} Inv {inv['id']}: {title}",
                ha="right", va="center", fontsize=8)

    for j, rq in enumerate(rqs):
        ax.text(j, max(y) + 0.7, f"RQ{rq}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_xlim(-5.0, len(rqs) - 0.5 + 0.2)
    ax.set_ylim(-0.7, max(y) + 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "box")
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map["pass"], label="Pass"),
        plt.Rectangle((0, 0), 1, 1, color=color_map["fail"], label="Fail"),
        plt.Rectangle((0, 0), 1, 1, color=color_map["pending"], label="Pending"),
        plt.Rectangle((0, 0), 1, 1, color=color_map["documented"],
                      label="Documented"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8, frameon=False)

    ax.set_title(
        "Architectural Invariants - Cross-RQ Coverage Matrix\n"
        "[!] = safety-critical invariant",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    _saved("invariants", out_path)


# ─── Figure 2: User study per role with escalation (Path C) ────


_PATH_C_ROLES = ("biomed_engineer", "IT_generalist", "nurse_manager")


def make_user_study(out_path: Path) -> None:
    data = _load_json("analysis/outputs/rq3_user_study.json")
    if not data:
        return _skip("user_study", "rq3_user_study.json missing")

    accuracy_block = data.get("per_role_accuracy_confidence") or {}
    escalation_block = data.get("per_role_escalation") or {}
    if not accuracy_block and not escalation_block:
        return _skip("user_study", "no per-role data available")

    roles = [r for r in _PATH_C_ROLES
             if r in accuracy_block or r in escalation_block]
    if not roles:
        return _skip("user_study", "no Path C roles present in JSON")

    # 3 panels: accuracy, confidence, escalation (drop decision_time per Path C)
    panels = [
        ("accuracy", "Accuracy", accuracy_block, ["median_A", "median_B"], "n_warning"),
        ("confidence", "Confidence", accuracy_block, ["median_A", "median_B"], "n_warning"),
        ("escalation", "Appropriate Escalation Rate", escalation_block,
         ["rate_A", "rate_B"], "n_warning"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 5.0))
    if len(panels) == 1:
        axes = [axes]

    for ax, (metric, label, block, val_keys, warn_key) in zip(axes, panels):
        x = np.arange(len(roles))
        w = 0.35

        a_vals, b_vals = [], []
        warnings = []
        fishers = []
        p_values = []
        for r in roles:
            cell = block.get(r) or {}
            if metric == "escalation":
                a_vals.append(cell.get("rate_A") or 0.0)
                b_vals.append(cell.get("rate_B") or 0.0)
                p_values.append(cell.get("p_value"))
                fishers.append(bool(cell.get("fisher_fallback")))
            else:
                # accuracy / confidence: nested under metric inside the role
                metric_cell = cell.get(metric) or {}
                a_vals.append(metric_cell.get(val_keys[0]) or 0.0)
                b_vals.append(metric_cell.get(val_keys[1]) or 0.0)
                p_values.append(metric_cell.get("p_value"))
                warnings.append(bool(metric_cell.get(warn_key)))
                fishers.append(False)
            if metric == "escalation":
                warnings.append(bool(cell.get(warn_key)))

        ax.bar(x, a_vals, w, label="Condition A (no MVE)", color="#cc6677")
        ax.bar(x + w, b_vals, w, label="Condition B (with MVE)", color="#4477aa")
        if metric == "escalation":
            ax.set_ylim(0, 1.05)
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels([r.replace("_", "\n") for r in roles], fontsize=9)
        ax.set_title(label, fontsize=10)

        # Annotate p-values / warnings / fisher
        for i, r in enumerate(roles):
            top = max(a_vals[i], b_vals[i])
            p = p_values[i]
            badges = []
            if warnings and warnings[i]:
                badges.append("low-n")
            if fishers and fishers[i]:
                badges.append("Fisher")
            if p is not None:
                p_text = f"p={p:.2g}"
                color = "darkred" if p < 0.05 else "gray"
                ax.annotate(p_text,
                            xy=(i + w / 2, top + 0.04),
                            ha="center", fontsize=7, color=color)
            if badges:
                ax.annotate(" ".join(badges),
                            xy=(i + w / 2, top - 0.04),
                            ha="center", fontsize=6, color="red")

        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "RQ3.5 - User Study: per-role accuracy/confidence + appropriate "
        "escalation rate\n"
        "Method 1: LLM-persona simulation (gpt-4o-mini), not human study",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    _saved("user_study", out_path)


# ─── Dispatch ──────────────────────────────────────────────────


GENERATORS = {
    "invariants":  make_invariants,
    "user_study":  make_user_study,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(FIGURES.keys()),
                    help="Generate only one figure")
    ap.add_argument("--list", action="store_true",
                    help="List figure IDs and exit")
    args = ap.parse_args()

    if args.list:
        for fid, fname in FIGURES.items():
            print(f"  {fid:12s} -> results/figures/{fname}")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    to_run = [args.only] if args.only else list(FIGURES.keys())
    for fid in to_run:
        GENERATORS[fid](FIG_DIR / FIGURES[fid])


if __name__ == "__main__":
    main()
