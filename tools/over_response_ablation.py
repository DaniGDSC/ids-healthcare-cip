#!/usr/bin/env python3
"""Counterfactual ablation for the M5 over-response rate.

Loads the existing M3 risk components from ``results/reports/risk_scores.npz``
and re-simulates the tier-assignment + adaptive-response pipeline under
several hypothetical configurations, **without modifying any code files**.
For each strategy, reports:

* surfaced count (MEDIUM+ tier)
* over_response (benign with isolate/restrict in recommended actions)
* under_response (attack without isolate/restrict in recommended actions)
* threat_contained (attack with isolate/restrict in recommended actions)
* FNR_critical (fraction of critical-device attacks not surfaced)
* composite score = (1 - over) * recall_attack - 10 * FNR_critical_excess

Strategies:
  E0  baseline (current config)
  A   weights = (0.50, 0.25, 0.05, 0.20)  — drop S_data weight 0.15 → 0.05,
       give 0.10 back to C_detect
  B   MEDIUM default_actions drops restrict_traffic
  C   RISK_THRESHOLDS: MEDIUM 0.40 → 0.50 (raise surfacing bar)
  D   c_detect grey-zone shrinkage (p in [0.3, 0.7] pulled 50% toward 0.5)
  AB  A + B combined
  AC  A + C combined
  BC  B + C combined
  ABC all three

D is the most invasive in production (it edits inference-time probas);
A/B/C are config-only edits.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module3_risk_scoring.composition import (
    assign_risk_levels,
    compute_composite_risk,
)
from module3_risk_scoring.config import (
    MIN_DETECTION_GATE,
    RISK_THRESHOLDS,
    WEIGHTS,
)
from module5_responses import config as m5_config
from module5_responses.adaptive import select_adaptive_response

CRITICAL_DEVICE_THRESHOLD = 0.8


def load_inputs(split: str = "test"):
    """Load per-sample inputs needed for end-to-end simulation.

    Args:
        split: "test" → reads ``risk_scores.npz`` + ``test_phase1.parquet``.
               "demo" → reads ``demo_scores.npz`` + ``demo_phase1.parquet``.
    """
    if split not in ("test", "demo"):
        raise ValueError(f"unknown split: {split!r}")
    risk_filename = "risk_scores.npz" if split == "test" else "demo_scores.npz"
    parquet_filename = f"{split}_phase1.parquet"

    risk_npz = np.load(
        PROJECT_ROOT / "results/reports" / risk_filename, allow_pickle=True,
    )
    y_true = risk_npz["y_true"].astype(int)
    c_detect = risk_npz["c_detect"].astype(float)
    d_crit = risk_npz["d_crit"].astype(float)
    s_data = risk_npz["s_data"].astype(float)
    d_clinical_tier = risk_npz["d_clinical_tier"].astype(float)

    df = pd.read_parquet(PROJECT_ROOT / "data/processed" / parquet_filename)
    if "Attack Category" in df.columns:
        attack_cats = df["Attack Category"].astype(str).fillna("normal").values
    else:
        attack_cats = np.array(["normal"] * len(y_true), dtype=object)
    # Normalise empties / missing → "normal"
    attack_cats = np.where(
        (attack_cats == "") | (attack_cats == "nan") | (attack_cats == "None"),
        "normal",
        attack_cats,
    )

    return {
        "y_true": y_true,
        "c_detect": c_detect,
        "d_crit": d_crit,
        "s_data": s_data,
        "d_clinical_tier": d_clinical_tier,
        "attack_cats": attack_cats,
    }


def thresholds_dict(threshold_list):
    """Convert RISK_THRESHOLDS-style list to the dict shape assign_risk_levels expects."""
    return {name: th for th, name in threshold_list}


# Action sets that count as "mitigation" for the over/under_response metric.
# Matches executor.simulate_outcome's `has_mitigation` definition.
MITIGATION_ACTIONS = {"isolate_device", "restrict_traffic", "re_authenticate"}


def simulate(
    inputs: dict,
    *,
    weights: dict | None = None,
    risk_thresholds: list | None = None,
    tier_policies_patch: dict | None = None,
    calibrate_c_detect=None,
    label: str = "",
) -> dict:
    """Run one ablation scenario end-to-end and return aggregate metrics.

    Returns dict with: n, n_surfaced, over_resp, under_resp, threat_contained,
    fnr_critical, attack_recall_surfaced, attack_recall_mitigated, score.
    """
    c_detect = inputs["c_detect"].copy()
    if calibrate_c_detect is not None:
        c_detect = calibrate_c_detect(c_detect)

    R = compute_composite_risk(
        c_detect=c_detect,
        d_crit=inputs["d_crit"],
        s_data=inputs["s_data"],
        d_clinical_tier=inputs["d_clinical_tier"],
        weights=weights,
    )
    thresholds = thresholds_dict(risk_thresholds) if risk_thresholds else None
    levels = assign_risk_levels(
        R, thresholds=thresholds, c_detect=c_detect,
        detection_gate=MIN_DETECTION_GATE,
    )

    # Apply tier_policies patch on the M5 module so select_adaptive_response
    # picks up the modified default_actions.
    patched_policies = deepcopy(m5_config.TIER_POLICIES)
    if tier_policies_patch:
        for tier, override in tier_policies_patch.items():
            patched_policies[tier] = {**patched_policies[tier], **override}

    n = len(R)
    y = inputs["y_true"]
    cats = inputs["attack_cats"]

    # crit_mask: attacks on life-critical devices (per d_crit ≥ 0.8).
    is_attack = (y == 1)
    is_crit_attack = is_attack & (inputs["d_crit"] >= CRITICAL_DEVICE_THRESHOLD)

    n_surfaced = 0
    n_over_resp = 0     # benign + mitigation
    n_under_resp = 0    # attack + no mitigation (but surfaced)
    n_threat_contained = 0    # attack + mitigation
    n_attacks_surfaced = 0
    n_crit_attacks_unsurfaced = 0
    n_benign_surfaced = 0
    n_benign_log_only = 0     # benign + surfaced + no mitigation

    # M5 surfacing criterion: level != NORMAL.
    with patch.object(m5_config, "TIER_POLICIES", patched_policies):
        # adaptive imports TIER_POLICIES directly, refresh import-time binding
        from module5_responses import adaptive as _adaptive_mod
        _adaptive_mod.TIER_POLICIES = patched_policies

        for idx in range(n):
            lvl = str(levels[idx])
            if lvl == "NORMAL":
                if is_crit_attack[idx]:
                    n_crit_attacks_unsurfaced += 1
                continue
            n_surfaced += 1
            if is_attack[idx]:
                n_attacks_surfaced += 1
            else:
                n_benign_surfaced += 1

            cat = str(cats[idx]) if not is_attack[idx] else str(cats[idx])
            response = select_adaptive_response(
                risk_level=lvl,
                risk_score=float(R[idx]),
                attack_category=cat,
                biometric_in_top_features=False,
            )
            actions = set(response.get("actions", []) or [])
            has_mitigation = bool(actions & MITIGATION_ACTIONS)

            if is_attack[idx]:
                if has_mitigation:
                    n_threat_contained += 1
                else:
                    n_under_resp += 1
            else:
                if has_mitigation:
                    n_over_resp += 1
                else:
                    n_benign_log_only += 1

        # Restore module binding
        _adaptive_mod.TIER_POLICIES = m5_config.TIER_POLICIES

    n_attacks_total = int(is_attack.sum())
    n_crit_attacks_total = int(is_crit_attack.sum())
    over_response_rate = n_over_resp / max(n_surfaced, 1)
    under_response_rate = n_under_resp / max(n_attacks_surfaced, 1)
    attack_recall_surfaced = n_attacks_surfaced / max(n_attacks_total, 1)
    attack_recall_mitigated = n_threat_contained / max(n_attacks_total, 1)
    fnr_critical = n_crit_attacks_unsurfaced / max(n_crit_attacks_total, 1)

    # Composite score (higher = better):
    #   penalise over_response (1 - over_response_rate),
    #   reward containment (attack_recall_mitigated),
    #   harshly penalise any FNR_critical above the 0.05 target.
    fnr_excess = max(0.0, fnr_critical - 0.05)
    score = (1 - over_response_rate) * attack_recall_mitigated - 10 * fnr_excess

    return {
        "label": label,
        "n_total": n,
        "n_surfaced": n_surfaced,
        "n_attacks": n_attacks_total,
        "n_attacks_surfaced": n_attacks_surfaced,
        "n_benign_surfaced": n_benign_surfaced,
        "n_threat_contained": n_threat_contained,
        "n_over_response": n_over_resp,
        "n_under_response": n_under_resp,
        "n_benign_log_only": n_benign_log_only,
        "n_crit_attacks": n_crit_attacks_total,
        "n_crit_attacks_unsurfaced": n_crit_attacks_unsurfaced,
        "over_response_rate": round(over_response_rate, 4),
        "under_response_rate": round(under_response_rate, 4),
        "attack_recall_surfaced": round(attack_recall_surfaced, 4),
        "attack_recall_mitigated": round(attack_recall_mitigated, 4),
        "fnr_critical": round(fnr_critical, 4),
        "score": round(score, 4),
    }


def calibrate_grey_zone_shrink(c, lo=0.30, hi=0.70, factor=0.5):
    """Pull grey-zone XGB probas toward 0.5 by ``1 - factor`` (default 50%).

    Emulates a more conservative XGBoost without retraining. Outside the
    grey zone the proba is left intact, so confident attacks and confident
    benigns are unaffected.
    """
    out = c.copy()
    mask = (c > lo) & (c < hi)
    out[mask] = 0.5 + (c[mask] - 0.5) * factor
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Over-response ablation harness for M5.",
    )
    parser.add_argument(
        "--split", choices=("test", "demo"), default="test",
        help="Which frozen split to load (default: test).",
    )
    args = parser.parse_args()

    inputs = load_inputs(args.split)
    print(f"Split: {args.split}  |  Loaded {len(inputs['y_true'])} samples "
          f"({int(inputs['y_true'].sum())} attacks).")
    print()

    # Pre-B baseline: MEDIUM default included restrict_traffic. Since the
    # production config now reflects Strategy B (the live winner), the
    # "E0 baseline" row below explicitly patches restrict_traffic back in
    # so the comparison is counterfactual against the pre-B state.
    BASELINE_MEDIUM = {
        "MEDIUM": {
            "default_actions": [
                "log_event", "restrict_traffic", "enhanced_monitoring",
            ],
        },
    }

    strategies: list[dict] = []

    # E0 — Counterfactual baseline (pre-B): restores restrict_traffic to
    # MEDIUM default so we can measure what the over-response was before
    # the Strategy B fix was applied to production config.
    strategies.append(simulate(
        inputs,
        tier_policies_patch=BASELINE_MEDIUM,
        label="E0 baseline (pre-B)",
    ))

    # A — Drop w3, give to w1 (still against pre-B baseline)
    strategies.append(simulate(
        inputs,
        weights={"w1": 0.50, "w2": 0.25, "w3": 0.05, "w4": 0.20},
        tier_policies_patch=BASELINE_MEDIUM,
        label="A   w3 0.15→0.05 (→w1)",
    ))

    # B — MEDIUM defaults drop restrict_traffic (= current production)
    strategies.append(simulate(
        inputs,
        label="B   MEDIUM no restrict_traffic ◀ current",
    ))

    # C — Raise MEDIUM threshold 0.40 → 0.50 (counterfactual to B; restores
    # pre-B MEDIUM default so the comparison is "what if I picked C instead
    # of B", not "what if I picked C on top of B").
    C_THRESHOLDS = [(0.80, "CRITICAL"), (0.60, "HIGH"), (0.50, "MEDIUM"), (0.30, "LOW")]
    strategies.append(simulate(
        inputs,
        risk_thresholds=C_THRESHOLDS,
        tier_policies_patch=BASELINE_MEDIUM,
        label="C   MEDIUM threshold 0.40→0.50",
    ))

    # D — Grey-zone proba shrinkage (counterfactual to B)
    strategies.append(simulate(
        inputs,
        calibrate_c_detect=calibrate_grey_zone_shrink,
        tier_policies_patch=BASELINE_MEDIUM,
        label="D   c_detect grey-zone shrink 50%",
    ))

    # Combinations
    strategies.append(simulate(
        inputs,
        weights={"w1": 0.50, "w2": 0.25, "w3": 0.05, "w4": 0.20},
        tier_policies_patch={
            "MEDIUM": {"default_actions": ["log_event", "enhanced_monitoring"]},
        },
        label="AB  A + B",
    ))

    strategies.append(simulate(
        inputs,
        weights={"w1": 0.50, "w2": 0.25, "w3": 0.05, "w4": 0.20},
        risk_thresholds=C_THRESHOLDS,
        tier_policies_patch=BASELINE_MEDIUM,  # counterfactual to B
        label="AC  A + C",
    ))

    strategies.append(simulate(
        inputs,
        tier_policies_patch={
            "MEDIUM": {"default_actions": ["log_event", "enhanced_monitoring"]},
        },
        risk_thresholds=C_THRESHOLDS,
        label="BC  B + C",
    ))

    strategies.append(simulate(
        inputs,
        weights={"w1": 0.50, "w2": 0.25, "w3": 0.05, "w4": 0.20},
        tier_policies_patch={
            "MEDIUM": {"default_actions": ["log_event", "enhanced_monitoring"]},
        },
        risk_thresholds=C_THRESHOLDS,
        label="ABC A + B + C",
    ))

    # E — Drop isolate_device from HIGH default (counterfactual to B).
    # HIGH ATTACK_ROUTING still adds isolate via Data Alteration; Spoofing
    # at HIGH still gets restrict via routing.
    strategies.append(simulate(
        inputs,
        tier_policies_patch={
            **BASELINE_MEDIUM,
            "HIGH": {
                "default_actions": [
                    "log_event", "forensic_snapshot", "enhanced_monitoring",
                ],
            },
        },
        label="E   HIGH no isolate_device",
    ))

    # B+E — drop default mitigation from both MEDIUM and HIGH.
    strategies.append(simulate(
        inputs,
        tier_policies_patch={
            "MEDIUM": {"default_actions": ["log_event", "enhanced_monitoring"]},
            "HIGH": {
                "default_actions": [
                    "log_event", "forensic_snapshot", "enhanced_monitoring",
                ],
            },
        },
        label="B+E MEDIUM+HIGH defaults trimmed",
    ))

    # ── Render comparison table ──────────────────────────────────────
    hdr = (
        f"{'strategy':30s}  {'srf':>4s}  {'cont':>4s}  {'over':>4s}  "
        f"{'under':>5s}  {'OR%':>5s}  {'UR%':>5s}  {'recall':>6s}  "
        f"{'fnrC':>5s}  {'score':>6s}"
    )
    sep = "─" * len(hdr)
    print(hdr)
    print(sep)
    for s in strategies:
        print(
            f"{s['label']:30s}  {s['n_surfaced']:4d}  "
            f"{s['n_threat_contained']:4d}  {s['n_over_response']:4d}  "
            f"{s['n_under_response']:5d}  "
            f"{s['over_response_rate']*100:5.1f}  "
            f"{s['under_response_rate']*100:5.1f}  "
            f"{s['attack_recall_mitigated']*100:5.1f}%  "
            f"{s['fnr_critical']*100:4.1f}%  "
            f"{s['score']:6.3f}"
        )
    print(sep)
    print(
        "Legend: srf=surfaced  cont=threat_contained  over=over-response  "
        "under=under-response\n"
        "        OR%=over_response_rate  UR%=under_response_rate  "
        "recall=attack_recall_mitigated  fnrC=FNR_critical  "
        "score=(1-OR)·recall − 10·max(0, fnrC−0.05)"
    )

    # Best strategy
    best = max(strategies, key=lambda s: s["score"])
    print(f"\nBest by composite score: {best['label']}  (score={best['score']:.3f})")

    # Write JSON for downstream inspection
    out_path = PROJECT_ROOT / "results/over_response_ablation.json"
    out_path.write_text(json.dumps(strategies, indent=2))
    print(f"\nWrote per-strategy metrics → {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
