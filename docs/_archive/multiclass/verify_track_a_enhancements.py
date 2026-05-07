"""End-to-end verification of Track A enhancements on EHMS test set.

Exercises:
  - Enhancement 1 (calibration):   raw vs calibrated probas → Brier, AUC, AUPRC
  - Enhancement 2 (per-device):    surfacing rate per device class
  - Enhancement 4 (diversity):     DISAGREEMENT_ANOMALY count and routing

Reads existing artefacts under results/models/ — does not retrain anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from module3_risk_scoring.multiclass_fusion import (  # noqa: E402
    classify_fusion_with_diversity,
    diversity_score,
    ensemble_softmax,
)
from src.data_models import (  # noqa: E402
    FusionClass,
    MULTICLASS_LABEL_ORDER_EHMS,
    P_XGB_HIGH_CONF,
    normal_index,
)
from src.risk_scorer import get_track_a_surfacing_threshold  # noqa: E402


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def main() -> int:
    models_dir = PROJECT_ROOT / "results/models"
    test_df = pd.read_parquet(
        PROJECT_ROOT / "data/processed/test_phase1.parquet",
        columns=["Label", "Attack Category", "device_class"],
    )
    y_true = test_df["Label"].values.astype(int)
    cat_true = test_df["Attack Category"].astype(str).values
    device = test_df["device_class"].astype(str).values

    # ── Enhancement 1: raw vs calibrated probas ──
    section("Enhancement 1: calibration impact (binary trees on EHMS test)")
    print(f"  {'model':<15s}  {'raw_brier':>10s}  {'cal_brier':>10s}  {'Δ':>8s}  "
          f"{'raw_auc':>9s}  {'cal_auc':>9s}")
    for name in ("xgboost", "random_forest", "decision_tree"):
        rep_path = models_dir / f"{name}_calibration_report.json"
        if not rep_path.exists():
            print(f"  {name}: no calibration report — skip")
            continue
        rep = json.load(open(rep_path))
        print(
            f"  {name:<15s}  {rep['raw']['test_brier']:>10.4f}  "
            f"{rep['calibrated']['test_brier']:>10.4f}  "
            f"{rep['raw']['test_brier']-rep['calibrated']['test_brier']:>+8.4f}  "
            f"{rep['raw']['test_auc_roc']:>9.4f}  "
            f"{rep['calibrated']['test_auc_roc']:>9.4f}"
        )

    # ── Enhancement 2: per-device surfacing rates ──
    section("Enhancement 2: per-device Track A surfacing thresholds")
    p_xgb_test = np.load(models_dir / "xgboost_test_predictions.npz")["y_proba"]
    print(f"  {'device_class':<20s}  {'thr':>5s}  {'n':>5s}  "
          f"{'attacks':>8s}  {'benigns':>8s}  {'surfaced':>8s}  "
          f"{'recall':>7s}  {'fpr':>7s}")
    for dc in sorted(set(device)):
        thr = get_track_a_surfacing_threshold(dc)
        mask = device == dc
        n = mask.sum()
        atk = ((y_true == 1) & mask).sum()
        ben = ((y_true == 0) & mask).sum()
        surfaced = (p_xgb_test >= thr) & mask
        tp = ((y_true == 1) & surfaced).sum()
        fp = ((y_true == 0) & surfaced).sum()
        rec = tp / max(atk, 1)
        fpr = fp / max(ben, 1)
        print(f"  {dc:<20s}  {thr:>5.2f}  {n:>5d}  "
              f"{atk:>8d}  {ben:>8d}  {surfaced.sum():>8d}  "
              f"{rec:>7.4f}  {fpr:>7.4f}")

    # ── Enhancement 4: diversity + DISAGREEMENT_ANOMALY ──
    section("Enhancement 4: ensemble diversity + DISAGREEMENT_ANOMALY routing")
    label_order = MULTICLASS_LABEL_ORDER_EHMS
    norm_idx = normal_index(label_order)
    sm_xgb = np.load(models_dir / "xgboost_multiclass_test_proba.npy")
    sm_rf = np.load(models_dir / "random_forest_multiclass_test_proba.npy")
    sm_dt = np.load(models_dir / "decision_tree_multiclass_test_proba.npy")
    sm_ens = ensemble_softmax(sm_xgb, sm_rf, sm_dt, method="mean")

    pa_xgb = 1 - sm_xgb[:, norm_idx]
    pa_rf = 1 - sm_rf[:, norm_idx]
    pa_dt = 1 - sm_dt[:, norm_idx]
    div = diversity_score(pa_xgb, pa_rf, pa_dt, metric="std")
    print(f"  Diversity statistics on test ({len(div)} rows):")
    print(f"    mean={div.mean():.4f}  median={np.median(div):.4f}  "
          f"max={div.max():.4f}  p95={np.percentile(div, 95):.4f}")
    print(f"    rows with diversity ≥ 0.20: {int((div >= 0.20).sum())}")

    dae = np.load(models_dir / "dae_multiclass_test_predictions.npz")
    dae_flag = dae["y_pred"].astype(int)

    fusion, pred_attack, _ = classify_fusion_with_diversity(
        softmax_a=sm_ens,
        dae_score=dae_flag,
        p_attack_per_model=(pa_xgb, pa_rf, pa_dt),
        label_order=label_order,
        a_high=P_XGB_HIGH_CONF,
        b_diversity=0.20,
        normal_idx=norm_idx,
        gate_normal_through_dae=True,
    )

    print()
    print(f"  Fusion class distribution (multiclass + diversity):")
    counts = pd.Series(fusion).value_counts().to_dict()
    for fc in [c.value for c in FusionClass]:
        print(f"    {fc:25s}  {counts.get(fc, 0):5d}")

    # Compare disagreement rate by ground truth
    disagreement_mask = fusion == FusionClass.DISAGREEMENT_ANOMALY.value
    if disagreement_mask.sum() > 0:
        print()
        print(f"  Among {disagreement_mask.sum()} DISAGREEMENT_ANOMALY rows:")
        print(f"    attacks: {int(((y_true == 1) & disagreement_mask).sum())}")
        print(f"    benigns: {int(((y_true == 0) & disagreement_mask).sum())}")
        cats_in_dis = pd.Series(cat_true[disagreement_mask]).value_counts().to_dict()
        print(f"    Attack Category distribution: {cats_in_dis}")
        # Predicted (would-have-been) attack class
        pred_str = np.array([label_order[i] if i >= 0 else "?"
                              for i in pred_attack])
        pred_dist = pd.Series(pred_str[disagreement_mask]).value_counts().to_dict()
        print(f"    Predicted attack class distribution: {pred_dist}")

    # ── Cascade end-to-end with all enhancements ──
    section("Cascade end-to-end: multi-class + diversity + (uncalibrated) DAE")
    surfaced = fusion != FusionClass.BENIGN.value
    tp = int((surfaced & (y_true == 1)).sum())
    fp = int((surfaced & (y_true == 0)).sum())
    fn = int((~surfaced & (y_true == 1)).sum())
    tn = int((~surfaced & (y_true == 0)).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  recall={rec:.4f}  precision={prec:.4f}  F1={f1:.4f}  FPR={fpr:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
