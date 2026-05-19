"""Calibrate `a_high` (KNOWN_ATTACK boundary) on the val_phase1 split.

Closes the `a_high` half of Stage 5B (RQ1_pipeline.md §6.1).
Joint sensitivity over a_low and b is still future work — this script holds
a_low=0.40 and b=0.70 fixed and only sweeps a_high.

Selection rule (deterministic, locked):
  smallest a_high s.t. (a) a_high > a_low (so the CONFIRM band is non-empty
  and the four-class fusion semantics are preserved), and (b) sens > 0.90
  AND spec > 0.95 on the tuning split.
  Tiebreak: prefer higher F2. If no a_high satisfies all of (a)+(b),
  fail loudly.

Inputs:
  data/processed/val_phase1.parquet (2,448 rows). Per split_metadata.yaml,
  val is disjoint from train/test/demo and is the canonical Phase-1 held-out
  validation split. It was previously used by Module 2 for fitting the
  isotonic XGBoost calibrator (a different decision than fusion-threshold
  selection); reusing it here for threshold selection does not contaminate
  the test split.

  results/reports/stratified_calibration.parquet was considered but rejected:
  built from a deprecated script (docs/_archive/build_stratified_eval_set.py)
  whose row counts no longer match the current Phase-1 splits, with Track A
  AUC = 0.949 vs 0.995 on current test — a distribution-shift signal.

Output:
  results/models/_fusion_thresholds.json — used at runtime by classify_fusion
  via module3_risk_scoring.module3_risk_scores.load_fusion_thresholds().
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import fbeta_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TUNING_PARQUET = PROJECT_ROOT / "data/processed/val_phase1.parquet"
OUT_PATH = PROJECT_ROOT / "results/models/_fusion_thresholds.json"

# Sweep grid (a_high). 0.01 step gives 56 points; cheap.
A_HIGH_MIN, A_HIGH_MAX, A_HIGH_STEP = 0.30, 0.85, 0.01

# Fixed thresholds (defaults from src.data_models / classify_fusion docstring).
A_LOW_FIXED = 0.40
B_FIXED = 0.70

# Selection-rule targets (mirror tests/acceptance_tests.py::test_rq1_targets_met).
SENS_TARGET = 0.90
SPEC_TARGET = 0.95


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def evaluate(c_track_a, c_track_b, y_true, a_high, a_low, b) -> dict:
    """Apply fusion rule, return sens/spec/F2/cm. Mirrors classify_fusion."""
    high_conf = c_track_a >= a_high
    in_confirm_band = (c_track_a >= a_low) & (c_track_a < a_high)
    below_low = c_track_a < a_low
    dae_flags = c_track_b >= b

    fired = high_conf | (in_confirm_band & dae_flags) | (below_low & dae_flags)
    y_pred = fired.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    sens = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    return {
        "a_high": float(a_high),
        "sensitivity": sens,
        "specificity": spec,
        "f2": f2,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def main() -> int:
    from module3_risk_scoring.score_split import score_parquet

    if not TUNING_PARQUET.exists():
        raise SystemExit(f"Tuning split not found: {TUNING_PARQUET}")

    print(f"Scoring tuning split: {TUNING_PARQUET.name}")
    scores = score_parquet(TUNING_PARQUET)
    if scores.y_true is None:
        raise SystemExit("Tuning parquet has no 'Label' column.")
    print(f"  n_rows={len(scores.y_true)}, "
          f"n_attacks={int(scores.y_true.sum())}, "
          f"n_benign={int((scores.y_true == 0).sum())}, "
          f"sha256={scores.parquet_sha256[:16]}...")

    # Sweep — round to grid step to avoid float-precision artefacts.
    n_steps = int(round((A_HIGH_MAX - A_HIGH_MIN) / A_HIGH_STEP)) + 1
    a_highs = np.round(np.linspace(A_HIGH_MIN, A_HIGH_MAX, n_steps), 4)
    sweep = [
        evaluate(scores.c_track_a, scores.c_track_b, scores.y_true,
                 a_high=float(a), a_low=A_LOW_FIXED, b=B_FIXED)
        for a in a_highs
    ]

    # Selection: smallest a_high > a_low + one grid step (preserves 4-class
    # semantics — a_high == a_low collapses the CONFIRM band to empty),
    # meeting both targets; tiebreak by higher F2.
    min_a_high = round(A_LOW_FIXED + A_HIGH_STEP, 4)
    qualifying = [row for row in sweep
                  if row["a_high"] >= min_a_high
                  and row["sensitivity"] > SENS_TARGET
                  and row["specificity"] > SPEC_TARGET]
    if not qualifying:
        best = max(sweep, key=lambda r: r["f2"])
        raise SystemExit(
            f"NO a_high in [{A_HIGH_MIN}, {A_HIGH_MAX}] satisfies "
            f"sens>{SENS_TARGET} AND spec>{SPEC_TARGET} on tuning split.\n"
            f"  Best F2 row: a_high={best['a_high']:.2f}, "
            f"sens={best['sensitivity']:.4f}, spec={best['specificity']:.4f}, "
            f"f2={best['f2']:.4f}\n"
            f"  → Single-knob (a_high) tuning is insufficient. "
            f"Escalate to joint sweep over (a_low, b) per Stage 5B."
        )

    # Smallest a_high; tiebreak: max F2.
    qualifying.sort(key=lambda r: (r["a_high"], -r["f2"]))
    picked = qualifying[0]

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "analysis/calibrate_fusion_threshold.py",
        "git_commit": _git_commit(),
        "tuning_split": str(TUNING_PARQUET.relative_to(PROJECT_ROOT)),
        "tuning_split_sha256": scores.parquet_sha256,
        "tuning_n_rows": int(len(scores.y_true)),
        "selection_rule": (
            f"smallest a_high s.t. sens > {SENS_TARGET} AND "
            f"spec > {SPEC_TARGET}; tiebreak max F2"
        ),
        "sweep_grid": {
            "a_high_min": A_HIGH_MIN,
            "a_high_max": A_HIGH_MAX,
            "step": A_HIGH_STEP,
        },
        "fixed_thresholds": {"a_low": A_LOW_FIXED, "b": B_FIXED},
        "targets": {"sensitivity": SENS_TARGET, "specificity": SPEC_TARGET},
        "tuning_metrics_at_picked": {
            "sensitivity": picked["sensitivity"],
            "specificity": picked["specificity"],
            "f2": picked["f2"],
            "confusion_matrix": {
                "tp": picked["tp"], "fn": picked["fn"],
                "fp": picked["fp"], "tn": picked["tn"],
            },
        },
        "picked": {
            "a_high": picked["a_high"],
            "a_low": A_LOW_FIXED,
            "b": B_FIXED,
        },
        "sweep_table": sweep,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Picked a_high = {picked['a_high']:.2f}")
    print(f"  Tuning sensitivity = {picked['sensitivity']:.4f} "
          f"(target > {SENS_TARGET})")
    print(f"  Tuning specificity = {picked['specificity']:.4f} "
          f"(target > {SPEC_TARGET})")
    print(f"  Tuning F2          = {picked['f2']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
