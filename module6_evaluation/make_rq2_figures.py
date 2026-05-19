"""Generate paper-ready PDFs for RQ2 from canonical sub-files.

Usage:
  python -m module6_evaluation.make_rq2_figures              # all figures
  python -m module6_evaluation.make_rq2_figures --only stability
  python -m module6_evaluation.make_rq2_figures --list

Path C: the user-study figure uses LLM-persona role enums and metrics
(accuracy + confidence; decision_time is absent in LLM data) and labels
the source explicitly as a persona simulation.
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
    "stability":   "rq2_shap_stability_histogram.pdf",
    "alignment":   "rq2_mve_alignment_modes.pdf",
    "mitre":       "rq2_mitre_grounding_per_category.pdf",
    "user_study":  "rq2_user_study_per_role.pdf",
    "failures":    "rq2_failure_categories.pdf",
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


# ─── Figure 1: SHAP stability histogram ────────────────────────

def make_stability(out_path: Path) -> None:
    data = _load_json("results/rq2_shap_stability.json")
    if not data:
        return _skip("stability", "rq2_shap_stability.json missing")
    agg = (data.get("results") or {}).get("aggregate") or {}
    counts = agg.get("histogram_counts")
    edges = agg.get("histogram_edges")
    if not counts or not edges or len(edges) != len(counts) + 1:
        return _skip("stability", "histogram_counts/edges missing or malformed")

    threshold = (data.get("results", {}).get("computation_params", {})
                 .get("stability_threshold", 0.90))
    n = agg.get("n_alerts", sum(counts))
    mean = agg.get("mean_stability")
    pct_stable = agg.get("pct_stable")

    fig, ax = plt.subplots(figsize=(6, 4))
    bin_centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    width = (edges[1] - edges[0]) * 0.9
    ax.bar(bin_centers, counts, width=width, edgecolor="black", alpha=0.7)
    ax.axvline(threshold, linestyle="--", color="red",
               label=f"Stability threshold = {threshold}")
    ax.set_xlabel("Per-alert stability score (mean top-3 Jaccard over perturbations)")
    ax.set_ylabel("Number of alerts")
    subtitle = []
    if mean is not None:
        subtitle.append(f"mean={mean:.3f}")
    if pct_stable is not None:
        subtitle.append(f"pct_stable={pct_stable:.0%}")
    ax.set_title(f"SHAP Stability — {n} surfaced alerts"
                 + (f" ({', '.join(subtitle)})" if subtitle else ""))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    _saved("stability", out_path)


# ─── Figure 2: MVE-SHAP alignment by fusion class ──────────────

def make_alignment(out_path: Path) -> None:
    """Real file stratifies by fusion_class, not by_mode — figure pivots."""
    data = _load_json("results/rq2_mve_shap_alignment.json")
    if not data:
        return _skip("alignment", "rq2_mve_shap_alignment.json missing")
    by_class = (data.get("results") or {}).get("by_fusion_class") or {}
    if not by_class:
        return _skip("alignment", "results.by_fusion_class missing")

    classes = sorted(by_class.keys())
    all3 = [by_class[c].get("all_3_present", 0) for c in classes]
    two_plus = [by_class[c].get("two_plus_present", 0) for c in classes]
    any_ = [by_class[c].get("any_present", 0) for c in classes]
    ns = [by_class[c].get("n_alerts", 0) for c in classes]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(classes))
    w = 0.27
    ax.bar(x - w, all3, w, label="All 3 SHAP features in Layer 1", color="#4477aa")
    ax.bar(x, two_plus, w, label=">=2 features", color="#ddcc77")
    ax.bar(x + w, any_, w, label=">=1 feature", color="#cc6677")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(classes, ns)],
                       fontsize=9)
    ax.set_ylabel("Proportion of alerts")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.80, linestyle=":", color="gray",
               label="Target (all 3) >= 0.80")
    ax.axhline(0.95, linestyle=":", color="lightgray",
               label="Target (>=2) >= 0.95")
    ax.set_title("MVE-SHAP Alignment — per fusion class")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    _saved("alignment", out_path)


# ─── Figure 3: MITRE grounding per attack category ──────────────

def make_mitre(out_path: Path) -> None:
    data = _load_json("results/rq2_mitre_grounding.json")
    if not data:
        return _skip("mitre", "rq2_mitre_grounding.json missing")
    by_cat = data.get("by_attack_category") or {}
    if not by_cat:
        return _skip("mitre", "by_attack_category missing")

    cats = sorted(by_cat.keys())
    grounded = [by_cat[c].get("grounded_pct", 0) for c in cats]
    strict = [by_cat[c].get("strict_grounded_pct", 0) for c in cats]
    ns = [by_cat[c].get("n_evaluated", 0) for c in cats]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x, grounded, w, label="T-ID OR human name (lenient)",
           color="#4477aa")
    ax.bar(x + w, strict, w, label="T-ID AND human name (strict)",
           color="#117733")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cats, ns)],
                       rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Proportion of alerts grounding MITRE")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.90, linestyle=":", color="gray", label="Target >= 0.90")
    ax.set_title("MITRE Layer 1 Grounding — per attack category")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    _saved("mitre", out_path)


# ─── Figure 4: User study per role (Path C — LLM-persona) ──────

# Lowercase enums — the real persona role keys.
_PATH_C_ROLES = ("biomed_engineer", "IT_generalist", "nurse_manager")
# decision_time is absent in LLM-persona data.
_PATH_C_METRICS = ("accuracy", "confidence")


def make_user_study(out_path: Path) -> None:
    data = _load_json("analysis/outputs/rq2c_per_role.json")
    if not data:
        return _skip("user_study", "rq2c_per_role.json missing")
    per_role = data.get("per_role") or {}
    roles = [r for r in _PATH_C_ROLES if r in per_role
             and isinstance(per_role[r], dict)
             and "accuracy" in per_role[r]]
    if not roles:
        return _skip("user_study", "no Path C roles populated")

    metrics = [m for m in _PATH_C_METRICS
               if all(per_role[r].get(m, {}).get("median_A") is not None
                      for r in roles)]
    if not metrics:
        return _skip("user_study", "no metrics have median_A across all roles")

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(4.5 * len(metrics), 4.5),
                             sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        x = np.arange(len(roles))
        w = 0.35
        a_vals = [per_role[r][metric].get("median_A", 0) for r in roles]
        b_vals = [per_role[r][metric].get("median_B", 0) for r in roles]
        ax.bar(x, a_vals, w, label="Condition A (no MVE)",
               color="#cc6677")
        ax.bar(x + w, b_vals, w, label="Condition B (with MVE)",
               color="#4477aa")
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels([r.replace("_", "\n") for r in roles], fontsize=9)
        ax.set_title(metric.title())
        for i, r in enumerate(roles):
            if per_role[r][metric].get("n_warning"):
                ax.annotate("low-n", xy=(i + w / 2,
                                          max(a_vals[i], b_vals[i]) * 1.02),
                            ha="center", fontsize=7, color="red")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "User Study Outcomes — per role × condition\n"
        "Method 1: LLM-persona simulation (gpt-4o-mini), not human study",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    _saved("user_study", out_path)


# ─── Figure 5: Failure category counts ─────────────────────────

def make_failures(out_path: Path) -> None:
    data = _load_json("results/rq2_failure_mode_catalog.json")
    if not data:
        return _skip("failures", "rq2_failure_mode_catalog.json missing")
    by_cat = (data.get("summary") or {}).get("by_category") or {}
    if not by_cat:
        return _skip("failures", "summary.by_category missing")

    cats = list(by_cat.keys())
    counts = [by_cat[c] for c in cats]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4477aa", "#cc6677", "#117733", "#ddcc77", "#aaaaaa"]
    ax.bar(cats, counts, color=colors[:len(cats)], edgecolor="black")
    for i, n in enumerate(counts):
        ax.text(i, n + 0.05 * max(counts + [1]), str(n),
                ha="center", fontsize=9)
    ax.set_ylabel("Observations")
    ax.set_ylim(0, max(counts + [1]) * 1.25)
    ax.set_title("Failure Mode Catalog — observations per category\n"
                 "(observation, not improvement; see §7.2.3 for future work)",
                 fontsize=10)
    plt.xticks(rotation=15, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    _saved("failures", out_path)


# ─── Dispatch ──────────────────────────────────────────────────

GENERATORS = {
    "stability":   make_stability,
    "alignment":   make_alignment,
    "mitre":       make_mitre,
    "user_study":  make_user_study,
    "failures":    make_failures,
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
        out_path = FIG_DIR / FIGURES[fid]
        GENERATORS[fid](out_path)


if __name__ == "__main__":
    main()
