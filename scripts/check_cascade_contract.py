"""Verify the cascade contract on the test set:

  - Trees (Track A) detect KNOWN attacks: P_xgb >= a_high
  - DAE (Track B) detects UNKNOWN attacks + verifies normal:
        P_xgb < a_low  → DAE flag → NOVEL_ANOMALY
        a_low <= P_xgb < a_high → DAE flag → CONFIRMED_ANOMALY
        otherwise → BENIGN

Outputs a per-regime breakdown of how attacks and benigns route through
each branch, plus DAE-only metrics on the Track-A-silent residual where
the "DAE catches unknown attacks" claim must be measured.

Run:
    python scripts/check_cascade_contract.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Spec defaults from src/data_models.py
A_HIGH = 0.85
A_LOW = 0.40
B = 0.70

# F2-tuned XGB surfacing threshold used in the deployment pipeline
XGB_SURFACING = 0.05

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    models_dir = PROJECT_ROOT / "results/models"
    xgb_pred = np.load(models_dir / "xgboost_test_predictions.npz")
    dae_pred = np.load(models_dir / "dae_test_predictions.npz")

    y_true = xgb_pred["y_true"].astype(int)
    p_xgb = xgb_pred["y_proba"]
    dae_err = dae_pred["reconstruction_error"]
    # Map DAE error to a [0, 1] anomaly score by passing through the
    # detector threshold: score = (error / threshold) clipped to [0, 1]
    dae_report = json.load(open(models_dir / "dae_final_report.json"))
    dae_threshold = dae_report["test_metrics"]["threshold"]
    dae_score = np.minimum(dae_err / dae_threshold, 2.0) / 2.0  # rough [0,1]
    # Actual flagging uses DAE's predict() (binary) which is err >= threshold.
    dae_flag = (dae_err >= dae_threshold).astype(int)

    n = len(y_true)
    print(f"n_test={n}, n_attacks={y_true.sum()}, n_benign={(y_true == 0).sum()}")
    print(f"thresholds: a_high={A_HIGH}, a_low={A_LOW}, dae_thr={dae_threshold:.6f}\n")

    # ── Regime masks based ONLY on Track A's confidence ──
    high_conf = p_xgb >= A_HIGH
    confirm_band = (p_xgb >= A_LOW) & (p_xgb < A_HIGH)
    silent = p_xgb < A_LOW

    print("=" * 78)
    print(" REGIME 1 — Track A high-confidence (P_xgb >= 0.85): trees own this")
    print("=" * 78)
    n_hc = high_conf.sum()
    atk_hc = (high_conf & (y_true == 1)).sum()
    ben_hc = (high_conf & (y_true == 0)).sum()
    tree_caught = atk_hc       # all are flagged KNOWN_ATTACK
    tree_fp = ben_hc           # all benigns here become FPs
    print(f"  count={n_hc}  attacks={atk_hc}  benigns={ben_hc}")
    print(f"  → routed as KNOWN_ATTACK: TP={tree_caught} (recall_known={atk_hc/y_true.sum():.4f})")
    print(f"  → benigns mis-flagged as KNOWN_ATTACK (FP): {tree_fp}")
    print()

    print("=" * 78)
    print(" REGIME 2 — Track A confirm band (0.40 <= P_xgb < 0.85): DAE corroborates")
    print("=" * 78)
    n_cb = confirm_band.sum()
    atk_cb = (confirm_band & (y_true == 1)).sum()
    ben_cb = (confirm_band & (y_true == 0)).sum()
    flagged_cb = (confirm_band & (dae_flag == 1)).sum()
    cb_tp = (confirm_band & (dae_flag == 1) & (y_true == 1)).sum()
    cb_fp = (confirm_band & (dae_flag == 1) & (y_true == 0)).sum()
    cb_fn = (confirm_band & (dae_flag == 0) & (y_true == 1)).sum()
    print(f"  count={n_cb}  attacks={atk_cb}  benigns={ben_cb}")
    print(f"  DAE flagged (→ CONFIRMED_ANOMALY): {flagged_cb}")
    print(f"    TP (real attack DAE caught)            : {cb_tp}")
    print(f"    FP (benign DAE incorrectly flagged)    : {cb_fp}")
    print(f"    FN (attack DAE missed in this band)    : {cb_fn}")
    if atk_cb > 0:
        print(f"  DAE recall WITHIN this band: {cb_tp/atk_cb:.4f}")
    print()

    print("=" * 78)
    print(" REGIME 3 — Track A SILENT (P_xgb < 0.40): DAE is the only signal")
    print("           ★ This is where DAE earns its keep on UNKNOWN attacks ★")
    print("=" * 78)
    n_si = silent.sum()
    atk_si = (silent & (y_true == 1)).sum()
    ben_si = (silent & (y_true == 0)).sum()
    flagged_si = (silent & (dae_flag == 1)).sum()
    si_tp = (silent & (dae_flag == 1) & (y_true == 1)).sum()
    si_fp = (silent & (dae_flag == 1) & (y_true == 0)).sum()
    si_fn = (silent & (dae_flag == 0) & (y_true == 1)).sum()
    si_tn = (silent & (dae_flag == 0) & (y_true == 0)).sum()
    print(f"  count={n_si}  attacks={atk_si}  benigns={ben_si}")
    print(f"  DAE flagged (→ NOVEL_ANOMALY): {flagged_si}")
    print(f"    TP (UNKNOWN attack DAE caught)            : {si_tp}")
    print(f"    FP (benign DAE incorrectly flagged)       : {si_fp}")
    print(f"    FN (UNKNOWN attack DAE missed)            : {si_fn}")
    print(f"    TN (benign correctly suppressed)          : {si_tn}")
    if atk_si > 0:
        rec = si_tp / atk_si
        prec = si_tp / max(flagged_si, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"  DAE residual-regime metrics: recall={rec:.4f}  precision={prec:.4f}  F1={f1:.4f}")
    if ben_si > 0:
        print(f"  DAE FPR on Track-A-silent benigns: {si_fp/ben_si:.4f}")
    print()

    print("=" * 78)
    print(" CASCADE END-TO-END (KNOWN ∪ NOVEL ∪ CONFIRMED vs everything else)")
    print("=" * 78)
    surfaced = high_conf | (confirm_band & (dae_flag == 1)) | (silent & (dae_flag == 1))
    suppressed = ~surfaced

    tp = (surfaced & (y_true == 1)).sum()
    fp = (surfaced & (y_true == 0)).sum()
    fn = (suppressed & (y_true == 1)).sum()
    tn = (suppressed & (y_true == 0)).sum()
    rec = tp / y_true.sum()
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    fpr = fp / (y_true == 0).sum()
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Cascade recall={rec:.4f}  precision={prec:.4f}  F1={f1:.4f}  FPR={fpr:.4f}")
    print()

    # ── Per-attack-category breakdown in the silent regime ──
    test_parquet = PROJECT_ROOT / "data/processed/test_phase1.parquet"
    if test_parquet.exists() and "Attack Category" in pd.read_parquet(
        test_parquet, columns=None
    ).columns:
        df = pd.read_parquet(test_parquet, columns=["Attack Category"])
        cats = df["Attack Category"].values
        print("=" * 78)
        print(" Per-attack-category catch by DAE in REGIME 3 (Track A silent)")
        print("=" * 78)
        for cat in pd.unique(cats):
            if cat == "normal":
                continue
            mask = silent & (cats == cat)
            n_cat = mask.sum()
            caught = (mask & (dae_flag == 1)).sum()
            if n_cat > 0:
                print(f"  {cat:30s}  {caught}/{n_cat} caught  ({caught/n_cat:.2%})")
        # Also report per-category total presence vs Track-A-silent presence
        print()
        print("  (How often Track A is silent on each attack category overall)")
        for cat in pd.unique(cats):
            if cat == "normal":
                continue
            mask_cat = (cats == cat)
            n_total = mask_cat.sum()
            n_silent = (silent & mask_cat).sum()
            if n_total > 0:
                print(f"  {cat:30s}  Track A silent on {n_silent}/{n_total} ({n_silent/n_total:.2%})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
