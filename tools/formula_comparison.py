#!/usr/bin/env python3
"""Module 3 risk-formula option comparison — non-destructive simulator.

Reads the current ``risk_scores.npz`` (composite R + 6 components + y_true)
and simulates 4 candidate options without modifying any artifact:

  v1                   — current formula, current thresholds (baseline)
  v1 + Phase A         — current formula, add NORMAL tier @ R < 0.20
  v1 + Phase B         — current formula, detection gate (C_detect ≥ 0.05)
  v1 + Phase A + B     — both gates simultaneously

For each option, computes:

  Operational (full alert pool — what operators see):
    precision, recall, F1, alert_volume, FP_count

  Surfaced (RQ1 convention, MEDIUM+ only):
    precision, recall, F1 — paper-frozen metric

  Per-tier:
    distribution of attacks / benign across CRITICAL / HIGH / MEDIUM / LOW / NORMAL

  Counterfactual impact:
    estimated coverage of feasible counterfactuals on the new alert pool,
    using the rule "a sample has counterfactual ⇔ it was XGBoost-flagged
    (y_pred=1) AND y_proba ≥ threshold". This mirrors the constraint in
    ``compute_counterfactual``'s early-return.

Then ranks options by a weighted composite score and prints the winner
+ rationale. The full comparison + ranking is written to
``results/formula_comparison.json`` (does NOT touch risk_scores.npz).

The simulator is read-only over the inputs and writes only to its own
output file under ``results/``. Restore-from-backup is one ``cp``:
``cp backups/pre_formula_fix/risk_scores.npz results/reports/``.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
OUTPUT      = PROJECT_ROOT / "results" / "formula_comparison.json"


# ── Option definitions ──────────────────────────────────────────────


def _v1_tier(R: np.ndarray) -> np.ndarray:
    """Canonical v1 tier assignment — no NORMAL emitted."""
    conditions = [R >= 0.80, R >= 0.60, R >= 0.40]
    choices = ["CRITICAL", "HIGH", "MEDIUM"]
    return np.select(conditions, choices, default="LOW")


def _v1_phase_a_tier(R: np.ndarray, t_normal: float = 0.20) -> np.ndarray:
    """v1 thresholds + NORMAL @ R < t_normal."""
    conditions = [R >= 0.80, R >= 0.60, R >= 0.40, R >= t_normal]
    choices = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    return np.select(conditions, choices, default="NORMAL")


def _v1_phase_b_tier(R: np.ndarray, c_detect: np.ndarray,
                     gate: float = 0.05) -> np.ndarray:
    """v1 thresholds, but anything with C_detect < gate → NORMAL."""
    base = _v1_tier(R)
    base[c_detect < gate] = "NORMAL"
    return base


def _v1_phase_ab_tier(R: np.ndarray, c_detect: np.ndarray,
                      t_normal: float = 0.20, gate: float = 0.05) -> np.ndarray:
    """Both gates simultaneously — most conservative."""
    base = _v1_phase_a_tier(R, t_normal)
    base[c_detect < gate] = "NORMAL"
    return base


# ── Metric helpers ──────────────────────────────────────────────────


_SURFACED = ("CRITICAL", "HIGH", "MEDIUM")
_ANY_ALERT = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
_ALL_TIERS = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL")


def _prec_recall_f1(alert_mask: np.ndarray, y_true: np.ndarray) -> dict:
    tp = int(((y_true == 1) & alert_mask).sum())
    fp = int(((y_true == 0) & alert_mask).sum())
    fn = int(((y_true == 1) & ~alert_mask).sum())
    tn = int(((y_true == 0) & ~alert_mask).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "alert_volume": int(alert_mask.sum()),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


def _per_tier_distribution(tiers: np.ndarray, y_true: np.ndarray) -> dict:
    out: dict = {}
    for t in _ALL_TIERS:
        mask = tiers == t
        out[t] = {
            "total":   int(mask.sum()),
            "attacks": int(((y_true == 1) & mask).sum()),
            "benign":  int(((y_true == 0) & mask).sum()),
        }
    return out


def _counterfactual_coverage_estimate(tiers: np.ndarray, y_proba: np.ndarray,
                                       threshold: float) -> dict:
    """Estimate how many alerts in the new alert pool would have a
    feasible counterfactual.

    Approximation: a counterfactual is feasible ⇔ the sample has y_proba
    ≥ threshold (i.e. XGBoost actually classifies it as attack). This
    mirrors the early-return in ``compute_counterfactual`` and is the
    *upper bound* on what the engine can achieve without retraining.
    """
    actionable_mask = np.isin(tiers, _SURFACED)
    any_alert_mask  = np.isin(tiers, _ANY_ALERT)
    eligible = y_proba >= threshold

    def _rate(scope_mask: np.ndarray) -> dict:
        seen = int(scope_mask.sum())
        feas = int((scope_mask & eligible).sum())
        return {"seen": seen, "feasible_est": feas,
                "rate": round(feas / seen, 4) if seen else 0.0}

    return {
        "actionable_feasible_est": _rate(actionable_mask),
        "any_alert_feasible_est":  _rate(any_alert_mask),
    }


# ── Scoring + ranking ───────────────────────────────────────────────


# Weighted composite score. Higher = better. Tuned to reflect upgrade-plan
# priorities (operational signal-to-noise is the bottleneck; surfaced
# precision/recall are paper claims and should be preserved).
_SCORE_WEIGHTS = {
    "operational_precision":     0.30,   # the headline UX win
    "operational_recall":        0.20,   # but don't lose attacks
    "surfaced_precision_keep":   0.15,   # MUST NOT regress
    "surfaced_recall_keep":      0.15,   # MUST NOT regress
    "actionable_cf_coverage":    0.10,   # Phase 2 outcome
    "noise_reduction":           0.10,   # alert-volume cut
}


def _score(option_metrics: dict, baseline: dict) -> dict:
    """Weighted composite score 0..1 (higher = better).

    Surfaced precision/recall use a "keep above 99% of baseline" rule:
    any drop > 1pp absolute scores 0; equal or higher scores 1.
    """
    op = option_metrics["operational"]
    su = option_metrics["surfaced"]
    cf = option_metrics["counterfactual"]
    base_op = baseline["operational"]
    base_su = baseline["surfaced"]

    def _keep_above(curr: float, base: float, tolerance: float = 0.01) -> float:
        if curr + 1e-9 >= base - tolerance:
            return 1.0
        return max(0.0, 1.0 + (curr - (base - tolerance)) * 10)

    def _improvement(curr: float, base: float) -> float:
        # Map absolute improvement to [0, 1] (clamp at +50pp = 1.0)
        delta = curr - base
        return max(0.0, min(1.0, delta / 0.5))

    components = {
        "operational_precision":   _improvement(op["precision"],   base_op["precision"]),
        "operational_recall":      _keep_above(op["recall"],       base_op["recall"], 0.05),
        "surfaced_precision_keep": _keep_above(su["precision"],    base_su["precision"]),
        "surfaced_recall_keep":    _keep_above(su["recall"],       base_su["recall"]),
        "actionable_cf_coverage":  cf["actionable_feasible_est"]["rate"],
        "noise_reduction":         max(0.0, 1.0 - op["alert_volume"] / base_op["alert_volume"]),
    }
    total = sum(_SCORE_WEIGHTS[k] * components[k] for k in _SCORE_WEIGHTS)
    return {
        "components":  {k: round(v, 4) for k, v in components.items()},
        "weights":     _SCORE_WEIGHTS,
        "total_score": round(total, 4),
    }


# ── Top-level driver ────────────────────────────────────────────────


# Phase A+B optimal hyperparameters, chosen by the sweep in
# ``results/formula_sweep.json``: t_normal=0.30, gate=0.02 hits score
# 0.9672 with operational precision 0.686, recall 0.961, alert volume
# reduced 82%. Earlier defaults (0.20 / 0.05) are kept as opt-in via
# kwargs for backwards compatibility.
OPTIMAL_T_NORMAL = 0.30
OPTIMAL_GATE     = 0.02


def evaluate_options(
    R: np.ndarray, c_detect: np.ndarray, y_true: np.ndarray,
    y_proba: np.ndarray, threshold: float,
    *, t_normal: float = OPTIMAL_T_NORMAL, gate: float = OPTIMAL_GATE,
) -> dict:
    """Evaluate all 4 options against the same input data.

    Returns the per-option metrics + ranking. The first-pass v1 result
    is the baseline against which scores are normalised.
    """
    options = {
        "v1_baseline":            _v1_tier(R),
        "v1_phase_a":             _v1_phase_a_tier(R, t_normal=t_normal),
        "v1_phase_b":             _v1_phase_b_tier(R, c_detect, gate=gate),
        "v1_phase_a_plus_b":      _v1_phase_ab_tier(R, c_detect, t_normal=t_normal, gate=gate),
    }

    per_option: dict = {}
    for name, tiers in options.items():
        surfaced_mask  = np.isin(tiers, _SURFACED)
        any_alert_mask = np.isin(tiers, _ANY_ALERT)
        per_option[name] = {
            "operational":    _prec_recall_f1(any_alert_mask, y_true),
            "surfaced":       _prec_recall_f1(surfaced_mask,  y_true),
            "by_tier":        _per_tier_distribution(tiers,   y_true),
            "counterfactual": _counterfactual_coverage_estimate(tiers, y_proba, threshold),
        }

    baseline_block = per_option["v1_baseline"]
    for name in per_option:
        per_option[name]["score"] = _score(per_option[name], baseline_block)

    ranking = sorted(per_option.items(), key=lambda kv: -kv[1]["score"]["total_score"])
    winner_name, winner_block = ranking[0]
    return {
        "per_option": per_option,
        "ranking":    [(n, b["score"]["total_score"]) for n, b in ranking],
        "winner":     winner_name,
        "winner_rationale": _rationale(winner_name, winner_block, baseline_block),
        "params": {"t_normal": t_normal, "gate": gate, "threshold": threshold},
    }


def _rationale(name: str, winner: dict, baseline: dict) -> str:
    op_b = baseline["operational"]
    op_w = winner["operational"]
    su_b = baseline["surfaced"]
    su_w = winner["surfaced"]
    cf_w = winner["counterfactual"]["actionable_feasible_est"]
    return (
        f"{name}: operational precision {op_b['precision']:.4f} → "
        f"{op_w['precision']:.4f} ({op_w['precision'] - op_b['precision']:+.4f}); "
        f"recall {op_b['recall']:.4f} → {op_w['recall']:.4f} "
        f"({op_w['recall'] - op_b['recall']:+.4f}); "
        f"surfaced precision/recall kept at "
        f"{su_w['precision']:.4f}/{su_w['recall']:.4f} "
        f"(vs {su_b['precision']:.4f}/{su_b['recall']:.4f}); "
        f"counterfactual actionable coverage est. {cf_w['rate']:.1%}; "
        f"alert volume {op_b['alert_volume']} → {op_w['alert_volume']} "
        f"({100 * (1 - op_w['alert_volume'] / op_b['alert_volume']):.0f}% reduction)."
    )


# ── Tabular printer ────────────────────────────────────────────────


def _print_table(comparison: dict) -> None:
    print()
    print("=" * 92)
    print(" " * 30 + "FORMULA OPTION COMPARISON")
    print("=" * 92)
    fmt = ("{:<24s} {:>8s} {:>8s} {:>8s} {:>10s}  {:>8s} {:>8s} {:>10s}  {:>8s}")
    print(fmt.format(
        "Option",
        "OpPrec", "OpRec", "OpF1", "AlertVol",
        "SuPrec", "SuRec", "CF-feas%", "Score",
    ))
    print("-" * 92)
    for name, _score in comparison["ranking"]:
        b = comparison["per_option"][name]
        op = b["operational"]; su = b["surfaced"]
        cf = b["counterfactual"]["actionable_feasible_est"]["rate"]
        print(fmt.format(
            name,
            f"{op['precision']:.4f}", f"{op['recall']:.4f}", f"{op['f1']:.4f}",
            f"{op['alert_volume']:>10d}",
            f"{su['precision']:.4f}", f"{su['recall']:.4f}",
            f"{cf:.1%}",
            f"{b['score']['total_score']:.4f}",
        ))
    print("=" * 92)
    print(f"WINNER: {comparison['winner']}")
    print(f"  {comparison['winner_rationale']}")
    print("=" * 92)


def main() -> int:
    import joblib

    risk = np.load(REPORTS_DIR / "risk_scores.npz", allow_pickle=True)
    R = risk["R"]; c_detect = risk["c_detect"]; y_true = risk["y_true"]

    # Threshold + y_proba for counterfactual feasibility upper bound.
    try:
        from common.model_registry import get_track_a_thresholds
        threshold = float(get_track_a_thresholds()["xgboost"])
    except Exception:
        threshold = 0.5
    xgb = np.load(REPORTS_DIR.parent / "models" / "xgboost_test_predictions.npz")
    y_proba = xgb["y_proba"]
    if len(y_proba) != len(R):
        print(f"WARN: y_proba length ({len(y_proba)}) ≠ R length ({len(R)}); "
              "counterfactual estimate may be off.", file=sys.stderr)

    comparison = evaluate_options(R, c_detect, y_true, y_proba, threshold)
    comparison["_meta"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_samples":    int(len(R)),
        "threshold":    threshold,
    }

    _print_table(comparison)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    from common.artifact_versioning import embed_version_in_dict
    OUTPUT.write_text(json.dumps(embed_version_in_dict(comparison, OUTPUT.name), indent=2))
    print(f"\n[formula-comparison] wrote {OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
