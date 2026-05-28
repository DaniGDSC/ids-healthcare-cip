#!/usr/bin/env python3
"""Side-by-side v1 vs v2 comparison for Sprint 4 / Tầng 3.1.

Loads the v1 (paper-frozen) and v2 (deployed-snapshot) artifact sets
from their respective backup directories and prints a cross-table of
the metrics that matter for the migration decision:

  - RQ1 surfaced precision/recall (paper claim)
  - operational precision/recall (production UX)
  - alert volume
  - tier distribution
  - counterfactual coverage
  - LOW-tier attack density (formula-bug sentinel)

Writes ``results/v1_v2_comparison.json`` for the dashboard /
documentation generator.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


V1_DIR = PROJECT_ROOT / "backups" / "v1_paper_frozen"
V2_DIR = PROJECT_ROOT / "backups" / "v2_deployed_snapshot"
OUT    = PROJECT_ROOT / "results" / "v1_v2_comparison.json"


def _load(dir_: Path) -> dict:
    responses_path = dir_ / "alert_responses.json"
    baseline_path  = dir_ / "phase0_baseline.json"
    if not responses_path.exists() or not baseline_path.exists():
        raise FileNotFoundError(f"{dir_} missing required snapshot files")
    envelope = json.loads(responses_path.read_text())
    records = envelope.get("records", envelope) if isinstance(envelope, dict) else envelope
    baseline = json.loads(baseline_path.read_text())
    return {"records": records, "baseline": baseline}


def _summarise(records: list[dict], baseline: dict) -> dict:
    n = len(records)
    tier_dist = Counter(r["risk_level"] for r in records)
    n_attacks_in_pool = sum(1 for r in records if r.get("ground_truth") == "attack")
    n_low = sum(1 for r in records if r["risk_level"] == "LOW")
    n_low_attacks = sum(
        1 for r in records
        if r["risk_level"] == "LOW" and r.get("ground_truth") == "attack"
    )
    surfaced = [r for r in records if r["risk_level"] in ("CRITICAL", "HIGH", "MEDIUM")]
    n_surf_attacks = sum(1 for r in surfaced if r.get("ground_truth") == "attack")

    oh = baseline.get("operational_health", {})
    cc = baseline.get("counterfactual_coverage", {})

    return {
        "n_alert_records":          n,
        "tier_distribution":        dict(tier_dist),
        "operational_precision":    oh.get("operational_precision"),
        "operational_recall":       oh.get("operational_recall"),
        "surfaced_precision":       oh.get("surfaced_precision"),
        "surfaced_recall":          oh.get("surfaced_recall"),
        "low_tier_attack_density":  oh.get("low_tier_attack_density"),
        "counterfactual_overall_rate":            cc.get("rate"),
        "counterfactual_actionable_feasible_rate": cc.get("actionable_feasible_rate"),
        "n_attacks_in_pool":        n_attacks_in_pool,
        "n_low":                    n_low,
        "n_low_attacks":            n_low_attacks,
        "n_surfaced_attacks":       n_surf_attacks,
    }


def _print_table(v1: dict, v2: dict) -> None:
    print()
    print("=" * 88)
    print(" v1 (paper) ↔ v2 (deployed) — side by side")
    print("=" * 88)
    metric_rows = [
        ("Alert records emitted",        "n_alert_records",          ":>6d"),
        ("Attacks in pool",              "n_attacks_in_pool",        ":>6d"),
        ("Surfaced (MEDIUM+) attacks",   "n_surfaced_attacks",       ":>6d"),
        ("LOW records",                  "n_low",                    ":>6d"),
        ("LOW attack density",           "low_tier_attack_density",  ":>6.1%"),
        ("",                             None,                       None),
        ("Operational precision",        "operational_precision",    ":>6.1%"),
        ("Operational recall",           "operational_recall",       ":>6.1%"),
        ("Surfaced precision",           "surfaced_precision",       ":>6.1%"),
        ("Surfaced recall",              "surfaced_recall",          ":>6.1%"),
        ("",                             None,                       None),
        ("CF actionable feasible",       "counterfactual_actionable_feasible_rate", ":>6.1%"),
    ]
    fmt = "{:<32s} {:>14s} {:>14s} {:>14s}"
    print(fmt.format("", "v1", "v2", "Δ (v2 − v1)"))
    print("-" * 88)
    for label, key, spec in metric_rows:
        if key is None:
            print("")
            continue
        v1_val = v1.get(key)
        v2_val = v2.get(key)
        if v1_val is None or v2_val is None:
            v1s = "—" if v1_val is None else f"{v1_val}"
            v2s = "—" if v2_val is None else f"{v2_val}"
            ds  = "—"
        else:
            v1s = format(v1_val, spec.lstrip(":"))
            v2s = format(v2_val, spec.lstrip(":"))
            delta = v2_val - v1_val
            ds  = format(delta, spec.lstrip(":") if isinstance(delta, int) else ":>+6.1%")
        print(fmt.format(label, v1s, v2s, ds))

    print()
    print("Tier distribution:")
    all_tiers = sorted(set(v1["tier_distribution"]) | set(v2["tier_distribution"]),
                       key=lambda t: ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"].index(t)
                       if t in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL") else 9)
    print(fmt.format("  Tier", "v1", "v2", "Δ"))
    for tier in all_tiers:
        v1_c = v1["tier_distribution"].get(tier, 0)
        v2_c = v2["tier_distribution"].get(tier, 0)
        delta = v2_c - v1_c
        print(fmt.format(f"  {tier}", str(v1_c), str(v2_c),
                         f"{delta:+d}"))
    print("=" * 88)


def main() -> int:
    v1 = _load(V1_DIR)
    v2 = _load(V2_DIR)
    v1_summary = _summarise(v1["records"], v1["baseline"])
    v2_summary = _summarise(v2["records"], v2["baseline"])

    _print_table(v1_summary, v2_summary)

    report = {
        "v1_paper_frozen": v1_summary,
        "v2_deployed":     v2_summary,
        "delta": {
            k: (v2_summary[k] - v1_summary[k])
            if isinstance(v1_summary.get(k), (int, float))
            and isinstance(v2_summary.get(k), (int, float))
            else None
            for k in v1_summary
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    from common.artifact_versioning import embed_version_in_dict
    OUT.write_text(json.dumps(embed_version_in_dict(report, OUT.name), indent=2))
    print(f"\nWrote {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
