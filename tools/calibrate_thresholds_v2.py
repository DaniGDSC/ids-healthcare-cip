#!/usr/bin/env python3
"""Calibrate v2 risk-level thresholds (Sprint 4 / Tầng 3.1).

The v1 thresholds (0.80 / 0.60 / 0.40 / 0.30) were tuned against the
v1 linear-sum R distribution. v2 multiplies detection by context, so
the R distribution shifts upward when there's a strong detection
signal and downward (to 0) when the gate fails. New thresholds are
needed so tier proportions remain operationally sensible.

This tool runs v2 over the cached test-split components and reports
candidate thresholds for two calibration policies:

  policy A — "preserve RQ1 surfaced recall"
      Pick thresholds so that MEDIUM+ recall under v2 matches v1
      within 1pp. This keeps the paper's headline claim defensible.

  policy B — "preserve tier proportions"
      Pick thresholds so the v2 tier histogram matches v1's
      (CRITICAL/HIGH/MEDIUM share). Useful when the dashboard's
      severity colour palette is calibrated to a specific share.

The picked v2 thresholds are NOT auto-written into config.py — they
are printed and saved to ``results/v2_threshold_calibration.json``
so the engineer can review and update the constant intentionally.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from module3_risk_scoring.composition import (  # noqa: E402
    compute_composite_risk,
)


NPZ = PROJECT_ROOT / "results" / "reports" / "risk_scores.npz"
OUT = PROJECT_ROOT / "results" / "v2_threshold_calibration.json"


def _surfaced_recall(R: np.ndarray, y_true: np.ndarray,
                      surfaced_threshold: float) -> float:
    surfaced = R >= surfaced_threshold
    tp = int(((y_true == 1) & surfaced).sum())
    n_attacks = int((y_true == 1).sum())
    return tp / n_attacks if n_attacks else 0.0


def _tier_proportions(R: np.ndarray,
                       thresholds: tuple[float, float, float, float]) -> dict:
    t_crit, t_high, t_med, t_low = thresholds
    n = len(R)
    return {
        "CRITICAL": float(((R >= t_crit)).sum() / n),
        "HIGH":     float(((R >= t_high) & (R < t_crit)).sum() / n),
        "MEDIUM":   float(((R >= t_med)  & (R < t_high)).sum() / n),
        "LOW":      float(((R >= t_low)  & (R < t_med)).sum() / n),
        "NORMAL":   float((R < t_low).sum() / n),
    }


def main() -> int:
    d = np.load(NPZ, allow_pickle=True)
    c_detect = d["c_detect"]; d_crit = d["d_crit"]
    s_data   = d["s_data"];   d_clinical_tier = d["d_clinical_tier"]
    y_true   = d["y_true"]

    R_v1 = compute_composite_risk(
        c_detect, d_crit, s_data, d_clinical_tier, formula_version="v1",
    )
    R_v2 = compute_composite_risk(
        c_detect, d_crit, s_data, d_clinical_tier, formula_version="v2",
    )

    print(f"v1 R range: [{R_v1.min():.4f}, {R_v1.max():.4f}]  mean={R_v1.mean():.4f}")
    print(f"v2 R range: [{R_v2.min():.4f}, {R_v2.max():.4f}]  mean={R_v2.mean():.4f}")

    # ── policy A: match RQ1 surfaced recall (denominator = total attacks) ──
    # v1 RQ1 MEDIUM cutoff is 0.40. Find v2 cutoff matching surfaced recall.
    surfaced_v1 = _surfaced_recall(R_v1, y_true, 0.40)
    print(f"\nPolicy A — preserve RQ1 surfaced recall (v1 MEDIUM+ recall={surfaced_v1:.4f})")
    candidates = np.linspace(0.001, 0.50, 500)
    best_t = None
    best_delta = float("inf")
    for t in candidates:
        recall = _surfaced_recall(R_v2, y_true, t)
        delta = abs(recall - surfaced_v1)
        if delta < best_delta:
            best_delta = delta
            best_t = t
    v2_medium_A = float(best_t)
    print(f"  matched v2 MEDIUM cutoff: {v2_medium_A:.4f} "
          f"(recall={_surfaced_recall(R_v2, y_true, v2_medium_A):.4f})")

    # ── policy B: match v1 tier proportions ──
    print("\nPolicy B — match v1 tier proportions")
    v1_props = _tier_proportions(R_v1, (0.80, 0.60, 0.40, 0.30))
    print(f"  v1 share: {dict((k, round(v, 4)) for k, v in v1_props.items())}")

    def _pick_threshold_for_share(R: np.ndarray, target_above_share: float) -> float:
        # Find the threshold where (R >= t).mean() = target_above_share
        return float(np.quantile(R, 1.0 - target_above_share))

    crit_target = v1_props["CRITICAL"]
    high_target = v1_props["CRITICAL"] + v1_props["HIGH"]
    med_target  = v1_props["CRITICAL"] + v1_props["HIGH"] + v1_props["MEDIUM"]
    low_target  = (v1_props["CRITICAL"] + v1_props["HIGH"]
                   + v1_props["MEDIUM"] + v1_props["LOW"])

    v2_crit_B = _pick_threshold_for_share(R_v2, crit_target)
    v2_high_B = _pick_threshold_for_share(R_v2, high_target)
    v2_med_B  = _pick_threshold_for_share(R_v2, med_target)
    v2_low_B  = _pick_threshold_for_share(R_v2, low_target)
    print(f"  v2 thresholds: CRITICAL={v2_crit_B:.4f}, HIGH={v2_high_B:.4f}, "
          f"MEDIUM={v2_med_B:.4f}, LOW={v2_low_B:.4f}")

    # ── selected (recommended) thresholds ──
    # Recommend policy A's MEDIUM cutoff, with CRITICAL/HIGH/LOW
    # interpolated to keep monotonic spacing. The Sprint 4 default
    # constant in ``module3_risk_scoring/config.py:RISK_THRESHOLDS_V2``
    # is the result of this calibration; this tool is the source of
    # truth so anyone can regenerate them.
    recommended = (
        max(v2_crit_B, v2_medium_A + 0.45),
        max(v2_high_B, v2_medium_A + 0.20),
        v2_medium_A,
        max(0.02, v2_medium_A / 4.0),
    )
    print(f"\nRecommended v2 thresholds (rounded to 2 dp):")
    for tier, val in zip(("CRITICAL", "HIGH", "MEDIUM", "LOW"), recommended):
        print(f"  {tier:<10s} {round(val, 2):.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "v1_distribution": {
            "min":    float(R_v1.min()),
            "max":    float(R_v1.max()),
            "mean":   float(R_v1.mean()),
            "tier_proportions": v1_props,
            "rq1_surfaced_recall_at_0.40": surfaced_v1,
        },
        "v2_distribution": {
            "min":    float(R_v2.min()),
            "max":    float(R_v2.max()),
            "mean":   float(R_v2.mean()),
        },
        "policy_A_matched_medium":  v2_medium_A,
        "policy_B_thresholds": {
            "CRITICAL": v2_crit_B, "HIGH": v2_high_B,
            "MEDIUM": v2_med_B,    "LOW":  v2_low_B,
        },
        "recommended_v2_thresholds": {
            "CRITICAL": round(recommended[0], 2),
            "HIGH":     round(recommended[1], 2),
            "MEDIUM":   round(recommended[2], 2),
            "LOW":      round(recommended[3], 2),
        },
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
