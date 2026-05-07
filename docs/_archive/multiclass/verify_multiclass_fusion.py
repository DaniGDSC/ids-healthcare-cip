"""Verify the multi-class cascade fusion contract on the EHMS test set.

Reads:
  - results/models/{xgboost,random_forest,decision_tree}_multiclass_test_proba.npy
  - results/models/dae_multiclass_test_predictions.npz
  - data/processed/test_phase1.parquet (ground truth + Attack Category)

Computes:
  1. Per-fusion-class breakdown (KNOWN_ATTACK / NOVEL_ANOMALY /
     CONFIRMED_ANOMALY / BENIGN counts and per-class composition).
  2. Confusion: how predicted_attack_class matches true Attack Category.
  3. Comparison against the binary fusion's cascade output (binary
     P_xgb is loaded from the test_predictions.npz of the binary
     pipeline).

Side-by-side comparison answers: did the multi-class refactor actually
realise the cascade contract better than the binary one?
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
    classify_fusion_multiclass,
    ensemble_softmax,
)
from src.data_models import (  # noqa: E402
    FusionClass,
    MULTICLASS_LABEL_ORDER_EHMS,
    P_XGB_HIGH_CONF,
    normal_index,
)


def main() -> int:
    models_dir = PROJECT_ROOT / "results/models"
    label_order = MULTICLASS_LABEL_ORDER_EHMS
    norm_idx = normal_index(label_order)

    # ── Load multi-class softmax + ensemble ──
    softmaxes = [
        np.load(models_dir / f"{name}_multiclass_test_proba.npy")
        for name in ("xgboost", "random_forest", "decision_tree")
    ]
    softmax_ens = ensemble_softmax(*softmaxes, method="mean")
    n = len(softmax_ens)

    # ── Load DAE flag ──
    dae = np.load(models_dir / "dae_multiclass_test_predictions.npz")
    dae_flag = dae["y_pred"].astype(int)

    # ── Ground truth ──
    test_df = pd.read_parquet(PROJECT_ROOT / "data/processed/test_phase1.parquet",
                              columns=["Label", "Attack Category"])
    y_true = test_df["Label"].values.astype(int)
    cat_true = test_df["Attack Category"].astype(str).values

    print(f"n_test={n}, attacks={y_true.sum()}, benigns={(y_true==0).sum()}")
    print(f"label_order={label_order}, normal_idx={norm_idx}")
    print(f"a_high={P_XGB_HIGH_CONF}, b=binary DAE flag")
    print()

    # ── Run multi-class fusion (literal cascade contract) ──
    fusion, pred_attack = classify_fusion_multiclass(
        softmax_a=softmax_ens,
        dae_score=dae_flag,
        label_order=label_order,
        a_high=P_XGB_HIGH_CONF,
        normal_idx=norm_idx,
        gate_normal_through_dae=True,
    )

    # ── Also run the precision-biased variant for comparison ──
    fusion_pb, pred_attack_pb = classify_fusion_multiclass(
        softmax_a=softmax_ens,
        dae_score=dae_flag,
        label_order=label_order,
        a_high=P_XGB_HIGH_CONF,
        normal_idx=norm_idx,
        gate_normal_through_dae=False,
    )

    # ── Section 1: fusion class distribution ──
    print("=" * 78)
    print(" Section 1: Multi-class fusion output distribution")
    print("=" * 78)
    fc_counts = pd.Series(fusion).value_counts().to_dict()
    for fc in [c.value for c in FusionClass]:
        print(f"  {fc:25s}  {fc_counts.get(fc, 0):5d}")
    print()

    # ── Section 2: per-fusion-class composition vs ground truth ──
    print("=" * 78)
    print(" Section 2: Per-fusion-class true-label composition")
    print("=" * 78)
    for fc in [c.value for c in FusionClass]:
        mask = fusion == fc
        if mask.sum() == 0:
            continue
        n_atk = int(((y_true == 1) & mask).sum())
        n_ben = int(((y_true == 0) & mask).sum())
        cats_in_class = pd.Series(cat_true[mask]).value_counts().to_dict()
        print(f"  {fc}:")
        print(f"    total={mask.sum()}, attacks={n_atk}, benigns={n_ben}")
        print(f"    by Attack Category: {cats_in_class}")
    print()

    # ── Section 3: KNOWN_ATTACK class-prediction accuracy ──
    print("=" * 78)
    print(" Section 3: KNOWN_ATTACK predicted-class vs true Attack Category")
    print("=" * 78)
    known_mask = fusion == FusionClass.KNOWN_ATTACK.value
    if known_mask.sum() > 0:
        pred_str = np.array([label_order[i] if i >= 0 else "?"
                             for i in pred_attack])
        confusion = pd.crosstab(
            pd.Series(cat_true[known_mask], name="true"),
            pd.Series(pred_str[known_mask], name="predicted"),
        )
        print(confusion)
        # Of correctly-fired KNOWN_ATTACK rows, how many had the right class?
        known_attacks = known_mask & (y_true == 1)
        if known_attacks.sum() > 0:
            agree = (pred_str[known_attacks] == cat_true[known_attacks]).sum()
            print(f"\n  Correct class among KNOWN_ATTACK true-attacks: "
                  f"{agree}/{known_attacks.sum()} "
                  f"({agree/known_attacks.sum()*100:.2f}%)")
    print()

    # ── Section 4: cascade end-to-end ──
    print("=" * 78)
    print(" Section 4: Cascade end-to-end (any non-BENIGN = surfaced)")
    print("=" * 78)
    surfaced = fusion != FusionClass.BENIGN.value
    tp = int(((y_true == 1) & surfaced).sum())
    fp = int(((y_true == 0) & surfaced).sum())
    fn = int(((y_true == 1) & ~surfaced).sum())
    tn = int(((y_true == 0) & ~surfaced).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  recall={rec:.4f}  precision={prec:.4f}  F1={f1:.4f}  FPR={fpr:.4f}")
    print()

    # ── Section 5: side-by-side vs binary fusion ──
    print("=" * 78)
    print(" Section 5: Multi-class vs binary fusion comparison")
    print("=" * 78)
    bin_pred = np.load(models_dir / "xgboost_test_predictions.npz")
    p_xgb = bin_pred["y_proba"]
    bin_dae = np.load(models_dir / "dae_test_predictions.npz")
    bin_dae_flag = bin_dae["y_pred"].astype(int)

    A_HIGH = 0.85
    A_LOW = 0.40
    high_conf_b = p_xgb >= A_HIGH
    silent_b = p_xgb < A_LOW
    confirm_b = (~high_conf_b) & (~silent_b)
    bin_surfaced = high_conf_b | (confirm_b & (bin_dae_flag == 1)) | (silent_b & (bin_dae_flag == 1))
    btp = int(((y_true == 1) & bin_surfaced).sum())
    bfp = int(((y_true == 0) & bin_surfaced).sum())
    bfn = int(((y_true == 1) & ~bin_surfaced).sum())
    btn = int(((y_true == 0) & ~bin_surfaced).sum())
    brec = btp / max(btp + bfn, 1)
    bprec = btp / max(btp + bfp, 1)
    bf1 = 2 * btp / max(2 * btp + bfp + bfn, 1)
    bfpr = bfp / max(bfp + btn, 1)

    print(f"  binary cascade:       TP={btp}  FP={bfp}  FN={bfn}  TN={btn}")
    print(f"                        recall={brec:.4f}  precision={bprec:.4f}  F1={bf1:.4f}  FPR={bfpr:.4f}")
    print(f"  multi-class cascade:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"                        recall={rec:.4f}  precision={prec:.4f}  F1={f1:.4f}  FPR={fpr:.4f}")
    print()
    print(f"  Δ recall:    {rec-brec:+.4f}")
    print(f"  Δ precision: {prec-bprec:+.4f}")
    print(f"  Δ F1:        {f1-bf1:+.4f}")
    print(f"  Δ FPR:       {fpr-bfpr:+.4f}")
    print()

    # ── Section 6: how often does cascade-contract uncertainty trigger? ──
    print("=" * 78)
    print(" Section 6: How often is Track A 'uncertain' (top_p < a_high)?")
    print("=" * 78)
    top_p = softmax_ens.max(axis=1)
    uncertain = top_p < P_XGB_HIGH_CONF
    print(f"  uncertain rows: {int(uncertain.sum())}/{n} ({uncertain.mean()*100:.2f}%)")
    print(f"    of those, attacks: {int((uncertain & (y_true==1)).sum())}")
    print(f"    of those, benigns: {int((uncertain & (y_true==0)).sum())}")
    print(f"    of those, DAE flagged: {int((uncertain & (dae_flag==1)).sum())}")
    print(f"  → DAE arbitration window size: {int(uncertain.sum())} rows")
    print(f"    (vs binary: confirm-band has {int((confirm_b).sum())} rows + silent has {int(silent_b.sum())})")

    # ── Section 7: precision-biased variant (gate_normal_through_dae=False) ──
    print()
    print("=" * 78)
    print(" Section 7: Precision-biased variant (legacy: confident-normal → BENIGN)")
    print("=" * 78)
    surfaced_pb = fusion_pb != FusionClass.BENIGN.value
    tp_pb = int(((y_true == 1) & surfaced_pb).sum())
    fp_pb = int(((y_true == 0) & surfaced_pb).sum())
    fn_pb = int(((y_true == 1) & ~surfaced_pb).sum())
    tn_pb = int(((y_true == 0) & ~surfaced_pb).sum())
    rec_pb = tp_pb / max(tp_pb + fn_pb, 1)
    prec_pb = tp_pb / max(tp_pb + fp_pb, 1)
    f1_pb = 2 * tp_pb / max(2 * tp_pb + fp_pb + fn_pb, 1)
    fpr_pb = fp_pb / max(fp_pb + tn_pb, 1)
    print(f"  TP={tp_pb}  FP={fp_pb}  FN={fn_pb}  TN={tn_pb}")
    print(f"  recall={rec_pb:.4f}  precision={prec_pb:.4f}  F1={f1_pb:.4f}  FPR={fpr_pb:.4f}")
    print()
    print("=" * 78)
    print(" SIDE-BY-SIDE: 3 fusion designs")
    print("=" * 78)
    rows = [
        ("binary cascade",                 brec, bprec, bf1, bfpr, btp, bfp, bfn, btn),
        ("multi-class (gate normal=True)", rec,  prec,  f1,  fpr,  tp,  fp,  fn,  tn),
        ("multi-class (gate normal=False)", rec_pb, prec_pb, f1_pb, fpr_pb,
                                            tp_pb, fp_pb, fn_pb, tn_pb),
    ]
    print(f"  {'design':<35s}  {'rec':>6s}  {'prec':>6s}  {'F1':>6s}  {'FPR':>6s}  {'TP':>5s}  {'FP':>4s}  {'FN':>4s}")
    for name, r, p, f, f_, t, fp_, fn_, _ in rows:
        print(f"  {name:<35s}  {r:.4f}  {p:.4f}  {f:.4f}  {f_:.4f}  {t:>5d}  {fp_:>4d}  {fn_:>4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
