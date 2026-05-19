"""Verify the picked fusion threshold generalises from val → test.

Loads the picked thresholds from results/models/_fusion_thresholds.json
(produced by analysis/calibrate_fusion_threshold.py), then evaluates them
on:
  - val_phase1.parquet (tuning split — reproduces the calibration metrics)
  - test_phase1.parquet (production headline split — the real report)
  - stratified_holdout.parquet (legacy held-out split with known
    distribution drift vs current Phase-1 splits; reported as a "tougher"
    sanity check rather than a gate)

The script does NOT mutate any artefacts; it only writes a report at
results/rq1_fusion_threshold_holdout.json.

Failure mode: a >5 percentage-point degradation in either sensitivity or
specificity from val → test on production Phase-1 splits indicates a
generalisation gap. The script exits non-zero in that case so CI can fail.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import fbeta_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

THRESHOLDS_PATH = PROJECT_ROOT / "results/models/_fusion_thresholds.json"
OUT_PATH = PROJECT_ROOT / "results/rq1_fusion_threshold_holdout.json"

# Splits to evaluate. The first two are the canonical val/test pair on the
# current Phase-1 pipeline; the third is the legacy stratified holdout.
SPLITS = [
    ("val",                "data/processed/val_phase1.parquet",
                           "tuning split — reproduces calibration metrics"),
    ("test",               "data/processed/test_phase1.parquet",
                           "production headline split — RQ1 report"),
    ("stratified_holdout", "results/reports/stratified_holdout.parquet",
                           "legacy held-out split; informational only "
                           "(known preprocessing drift vs current Phase-1)"),
]

DEGRADATION_LIMIT = 0.05  # >5pp sens or spec drop val → test → fail


def evaluate_split(scores, a_high, a_low, b) -> dict:
    """Return sens/spec/F2/cm given a SplitScores and thresholds."""
    ta, tb, y = scores.c_track_a, scores.c_track_b, scores.y_true
    high_conf = ta >= a_high
    in_confirm_band = (ta >= a_low) & (ta < a_high)
    below_low = ta < a_low
    dae_flags = tb >= b
    y_pred = (high_conf | (in_confirm_band & dae_flags) |
              (below_low & dae_flags)).astype(int)

    import numpy as np
    tp = int(((y == 1) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    tn = int(((y == 0) & (y_pred == 0)).sum())

    sens = float(recall_score(y, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f2 = float(fbeta_score(y, y_pred, beta=2.0, zero_division=0))
    return {
        "n_rows": int(len(y)),
        "n_attacks": int((y == 1).sum()),
        "sensitivity": sens,
        "specificity": spec,
        "f2": f2,
        "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def main() -> int:
    from module3_risk_scoring.score_split import score_parquet

    if not THRESHOLDS_PATH.exists():
        raise SystemExit(
            f"{THRESHOLDS_PATH} not found — run "
            "analysis/calibrate_fusion_threshold.py first."
        )
    th = json.loads(THRESHOLDS_PATH.read_text())
    picked = th["picked"]
    a_high, a_low, b = picked["a_high"], picked["a_low"], picked["b"]
    print(f"Verifying thresholds: a_high={a_high}, a_low={a_low}, b={b}")

    results = {}
    for name, rel, note in SPLITS:
        path = PROJECT_ROOT / rel
        if not path.exists():
            results[name] = {"_status": f"split not found: {rel}"}
            print(f"  {name}: SKIPPED (not found)")
            continue
        scores = score_parquet(path)
        if scores.y_true is None:
            results[name] = {"_status": f"no Label column in {rel}"}
            continue
        metrics = evaluate_split(scores, a_high, a_low, b)
        metrics["_note"] = note
        metrics["split_sha256"] = scores.parquet_sha256
        results[name] = metrics
        print(f"  {name:22s}: sens={metrics['sensitivity']:.4f}  "
              f"spec={metrics['specificity']:.4f}  "
              f"f2={metrics['f2']:.4f}  (n={metrics['n_rows']})")

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "analysis/verify_fusion_threshold_holdout.py",
        "thresholds_applied": {"a_high": a_high, "a_low": a_low, "b": b},
        "thresholds_source": str(THRESHOLDS_PATH.relative_to(PROJECT_ROOT)),
        "results": results,
    }

    # Generalisation gate: val → test degradation.
    if all(k in results and "sensitivity" in results[k]
           for k in ("val", "test")):
        d_sens = results["val"]["sensitivity"] - results["test"]["sensitivity"]
        d_spec = results["val"]["specificity"] - results["test"]["specificity"]
        payload["val_to_test_delta"] = {
            "delta_sensitivity": d_sens,
            "delta_specificity": d_spec,
            "limit": DEGRADATION_LIMIT,
        }
        if d_sens > DEGRADATION_LIMIT or d_spec > DEGRADATION_LIMIT:
            payload["val_to_test_delta"]["pass"] = False
            OUT_PATH.write_text(json.dumps(payload, indent=2))
            raise SystemExit(
                f"GENERALISATION GAP: val → test "
                f"Δsens={d_sens:+.4f}, Δspec={d_spec:+.4f} "
                f"(limit ±{DEGRADATION_LIMIT}). "
                "Threshold tuned on val does not transfer to test."
            )
        payload["val_to_test_delta"]["pass"] = True

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
