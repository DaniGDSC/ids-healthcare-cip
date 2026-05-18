"""RQ1 tier × surfacing truth table (RQ1_pipeline.md §8.1).

Enumerates every (risk_tier × patchable × maintenance_active) cell and
reports the expected ``should_surface`` decision plus its reason, by
invoking the real ``src.risk_scorer.score_alert`` function with
synthetic ``AlertContext`` dicts.

Outputs:
  - results/rq1_tier_surfacing_truth_table.csv
  - results/rq1_tier_surfacing_truth_table.md

Used as RQ1 Appendix B evidence and as RQ3 invariant evidence (the
safety floor at CRITICAL+unpatchable).
"""

from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

from src.risk_scorer import score_alert

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = REPO_ROOT / "results/rq1_tier_surfacing_truth_table.csv"
OUT_MD = REPO_ROOT / "results/rq1_tier_surfacing_truth_table.md"

TIERS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
PATCHABLE_OPTIONS = [True, False]
MAINTENANCE_OPTIONS = [True, False]

# Tier → representative anomaly score within that tier's band.  These
# are the values fed to score_alert as ``anomaly_score`` so the function
# can compute the adjusted score / threshold correctly.
SCORE_BY_TIER = {
    "CRITICAL": 0.85,
    "HIGH": 0.65,
    "MEDIUM": 0.45,
    "LOW": 0.20,
}

# Patchable → representative device class.  ``ehr_workstation`` is the
# canonical patchable device class; ``infusion_pump`` the canonical
# unpatchable life-critical class.
DEVICE_BY_PATCHABLE = {
    True: "ehr_workstation",
    False: "infusion_pump",
}


def evaluate_cell(
    tier: str, patchable: bool, maintenance_active: bool
) -> dict:
    """Invoke score_alert and classify the surfacing reason.

    Returns: dict with ``should_surface`` bool and a ``reason`` tag that
    distinguishes the safety floor, the maintenance-window path, and the
    threshold-comparison path.
    """
    device_class = DEVICE_BY_PATCHABLE[patchable]
    device_context = {
        "criticality": tier,
        "patchable": patchable,
        "device_class": device_class,
        "clinical_function": "truth_table_synthetic",
    }
    # The known-vendor-IP flag determines whether the maintenance-window
    # path triggers; we pair it with maintenance_active to exercise the
    # full ``is_maintenance_window AND is_known_vendor_ip`` branch in
    # src.risk_scorer.score_alert.
    event_context = {
        "is_maintenance_window": maintenance_active,
        "is_known_vendor_ip": maintenance_active,
        "similar_events_past_30d": 0,
    }
    result = score_alert(
        anomaly_score=SCORE_BY_TIER[tier],
        device_context=device_context,
        event_context=event_context,
    )

    should_surface = bool(result.should_surface)

    # Classify the reason by re-tracing the canonical decision branches
    # in src/risk_scorer.py.  The actual decision is owned by
    # score_alert; this is just human-readable annotation.
    if tier == "CRITICAL" and not patchable:
        reason = "safety_floor"
    elif maintenance_active:
        # Maintenance + vendor IP path: surfaces only if reduced score
        # still clears the base threshold *or* safety-floor applies.
        reason = (
            "maintenance_window_reduced" if should_surface
            else "suppressed_maintenance"
        )
    else:
        reason = (
            "above_threshold" if should_surface else "below_threshold"
        )
    return {
        "should_surface": should_surface,
        "reason": reason,
        "adjusted_score": float(result.adjusted_score),
        "threshold": float(result.threshold),
        "risk_multiplier": float(result.risk_multiplier),
        "device_class": device_class,
    }


def main() -> None:
    rows: list[dict] = []
    for tier, p, m in product(TIERS, PATCHABLE_OPTIONS, MAINTENANCE_OPTIONS):
        result = evaluate_cell(tier, p, m)
        rows.append({
            "risk_tier": tier,
            "patchable": p,
            "maintenance_active": m,
            "device_class": result["device_class"],
            "anomaly_score": SCORE_BY_TIER[tier],
            "adjusted_score": round(result["adjusted_score"], 4),
            "threshold": round(result["threshold"], 4),
            "risk_multiplier": round(result["risk_multiplier"], 4),
            "should_surface": result["should_surface"],
            "reason": result["reason"],
        })

    # CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    lines = [
        "# RQ1 / RQ3 — Tier × Patchable × Maintenance Truth Table",
        "",
        ("Derived by invoking ``src.risk_scorer.score_alert`` with "
         "synthetic ``AlertContext`` dicts (one per cell).  The "
         "``should_surface`` column reflects the real decision "
         "function — no mocking."),
        "",
        ("Safety floor (RQ3 invariant): CRITICAL + unpatchable always "
         "surfaces, even during a maintenance window."),
        "",
        ("| risk_tier | patchable | maintenance | device_class | "
         "anomaly | adjusted | threshold | mult | should_surface | "
         "reason |"),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['risk_tier']} | {r['patchable']} | "
            f"{r['maintenance_active']} | {r['device_class']} | "
            f"{r['anomaly_score']} | {r['adjusted_score']} | "
            f"{r['threshold']} | {r['risk_multiplier']} | "
            f"{r['should_surface']} | {r['reason']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")
    print(f"Total cells: {len(rows)}")
    n_surfaced = sum(1 for r in rows if r["should_surface"])
    print(f"Surfaced: {n_surfaced} / {len(rows)}")


if __name__ == "__main__":
    main()
