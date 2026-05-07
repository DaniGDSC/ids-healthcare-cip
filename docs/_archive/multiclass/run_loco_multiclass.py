"""LOCO cascade-validation on MedSec-25 with MULTI-CLASS Track A.

This is the cascade-contract test the binary LOCO could not run.

For each attack category H ∈ {Recon, Initial access, Exfil, Lateral}:
  1. Train multi-class Track A on {Benign, all attacks except H}.
     - K = 4 classes total per fold (1 benign + 3 known attacks).
  2. At inference, the held-out test rows of category H produce SPREAD
     softmax (no class confident, since trees never learned an "H pattern").
  3. Cascade fusion: spread softmax → uncertain → DAE arbitrates →
     NOVEL_ANOMALY if DAE flags, BENIGN otherwise.

The headline metric:
  - **`top_p < a_high` rate on H** — how often is multi-class Track A
    "uncertain" on the unseen category? Higher is better — it means
    the trees correctly *don't recognise* the novel category.
  - **DAE recall on uncertain H** — of the rows the trees correctly
    forwarded to DAE, how many does DAE actually flag as anomalous?

If multi-class gets >> 0.5–1.5% uncertain rate (the binary baseline),
the cascade design is realised. If DAE then catches >50% of those, the
contract empirically holds.

Run:
    python experiments/medsec25_loco/run_loco_multiclass.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module3_risk_scoring.multiclass_fusion import (  # noqa: E402
    classify_fusion_multiclass,
    ensemble_softmax,
)
from src.data_models import FusionClass, P_XGB_HIGH_CONF  # noqa: E402

PROCESSED = PROJECT_ROOT / "data/processed/medsec25"
OUT_DIR = PROJECT_ROOT / "results/medsec25_loco_multiclass"

A_HIGH = P_XGB_HIGH_CONF
DAE_PCT = 99.0

logger = logging.getLogger(__name__)


def _load_split(name: str) -> tuple:
    df = pd.read_parquet(PROCESSED / f"{name}.parquet")
    y = df["Label"].values.astype(int)
    m = df["Attack Category"].values.astype(str)
    X = df.drop(columns=["Label", "Attack Category"]).values.astype(np.float32)
    return X, y, m


def _to_native(x):
    if isinstance(x, dict):
        return {str(k): _to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_native(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return _to_native(x.tolist())
    if isinstance(x, np.str_):
        return str(x)
    return x


def _train_multiclass_track_a(
    X_train: np.ndarray, y_multi_train: np.ndarray,
    label_order: list[str],
    *, random_state: int, sample_cap: int,
) -> tuple:
    """Fit multi-class XGB-surrogate + RF + DT on the K-class label."""
    if sample_cap is not None and len(X_train) > sample_cap:
        rng = np.random.default_rng(random_state)
        # Stratified by multi-class label
        indices_per_class = [np.where(y_multi_train == c)[0]
                              for c in label_order]
        per_class_cap = max(sample_cap // len(label_order), 1)
        sel = []
        for ix in indices_per_class:
            n = min(per_class_cap, ix.size)
            sel.append(rng.choice(ix, size=n, replace=False))
        idx = np.concatenate(sel)
        rng.shuffle(idx)
        X_train = X_train[idx]
        y_multi_train = y_multi_train[idx]

    label_to_id = {s: i for i, s in enumerate(label_order)}
    y_int = np.array([label_to_id[s] for s in y_multi_train], dtype=np.int64)
    sample_weight = compute_sample_weight("balanced", y_int)

    xgb = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.9, random_state=random_state,
    )
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=None,
        min_samples_split=5, min_samples_leaf=1,
        max_features=0.5, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    dt = DecisionTreeClassifier(
        max_depth=None, min_samples_split=2, min_samples_leaf=2,
        class_weight="balanced", random_state=random_state,
    )
    logger.info("  Fit on %d rows (per-class %s)", len(X_train),
                pd.Series(y_multi_train).value_counts().to_dict())
    xgb.fit(X_train, y_int, sample_weight=sample_weight)
    rf.fit(X_train, y_int)
    dt.fit(X_train, y_int)
    return xgb, rf, dt, label_to_id


def _train_dae(X_benign_aug: np.ndarray, *, random_state: int):
    from module2_detection.models.DAE import DAEDetector
    n_feat = X_benign_aug.shape[1]
    enc = max(int(n_feat * 0.85), n_feat - 4)
    bot = max(min(int(n_feat * 0.4), n_feat - 2), 4)
    arch = [enc, bot, enc]
    det = DAEDetector(
        encoding_dims=arch,
        noise_rate=0.10, learning_rate=1e-4,
        threshold_percentile=DAE_PCT,
        epochs=60, batch_size=256,
        random_state=random_state,
    )
    det.fit(X_benign_aug, validation_split=0.0)
    return det


def _run_fold(
    held_out: str,
    X_train: np.ndarray, y_train: np.ndarray, m_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, m_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, m_test: np.ndarray,
    args,
) -> dict:
    fold_dir = OUT_DIR / "per_fold" / held_out.replace(" ", "_")
    fold_dir.mkdir(parents=True, exist_ok=True)
    sep = "─" * 78
    logger.info(sep)
    logger.info("LOCO multi-class fold: HOLDING OUT '%s'", held_out)
    logger.info(sep)

    # Build per-fold label_order (Benign + all attack cats except held_out).
    # Use a canonical order: "Benign" first, then attack cats sorted.
    keep_attack = sorted({c for c in pd.unique(m_train)
                          if c not in ("Benign", held_out)})
    label_order = ["Benign"] + keep_attack
    logger.info("  Per-fold label_order (%d-class): %s",
                len(label_order), label_order)

    # Train rows: keep only rows whose category is in label_order
    mask_train = np.isin(m_train, label_order)
    mask_val = np.isin(m_val, label_order)
    Xt = X_train[mask_train]
    mt = m_train[mask_train]
    Xv = X_val[mask_val]
    mv = m_val[mask_val]
    yv = y_val[mask_val]
    logger.info("  Train rows: %d", len(Xt))

    # ── Train multi-class Track A ──
    xgb, rf, dt, label_to_id = _train_multiclass_track_a(
        Xt, mt, label_order,
        random_state=args.random_state,
        sample_cap=args.train_sample_cap,
    )
    norm_idx = label_to_id["Benign"]

    # Predict softmax on val (to derive P(attack) for DAE input + train DAE)
    sm_val_xgb = xgb.predict_proba(Xv).astype(np.float32)
    sm_val_rf = rf.predict_proba(Xv).astype(np.float32)
    sm_val_dt = dt.predict_proba(Xv).astype(np.float32)
    p_val_xgb = 1 - sm_val_xgb[:, norm_idx]
    p_val_rf = 1 - sm_val_rf[:, norm_idx]
    p_val_dt = 1 - sm_val_dt[:, norm_idx]
    P_val = np.column_stack([p_val_xgb, p_val_rf, p_val_dt])

    benign_val_mask = yv == 0
    Xv_benign_aug = np.column_stack([Xv[benign_val_mask],
                                      P_val[benign_val_mask]]).astype(np.float32)
    logger.info("  DAE training set: %d benign val rows × %d cascaded features",
                len(Xv_benign_aug), Xv_benign_aug.shape[1])
    det = _train_dae(Xv_benign_aug, random_state=args.random_state)

    # ── Predict softmax on test ──
    sm_test_xgb = xgb.predict_proba(X_test).astype(np.float32)
    sm_test_rf = rf.predict_proba(X_test).astype(np.float32)
    sm_test_dt = dt.predict_proba(X_test).astype(np.float32)
    sm_test_ens = ensemble_softmax(sm_test_xgb, sm_test_rf, sm_test_dt,
                                    method="mean")

    p_test_xgb = 1 - sm_test_xgb[:, norm_idx]
    p_test_rf = 1 - sm_test_rf[:, norm_idx]
    p_test_dt = 1 - sm_test_dt[:, norm_idx]
    P_test = np.column_stack([p_test_xgb, p_test_rf, p_test_dt])
    X_test_aug = np.column_stack([X_test, P_test]).astype(np.float32)

    dae_err = det.reconstruction_error(X_test_aug)
    dae_thr = det.threshold
    dae_flag = (dae_err >= dae_thr).astype(int)

    # Persist
    np.savez(
        fold_dir / "test_predictions.npz",
        y_true=y_test, m_test=m_test,
        sm_test_xgb=sm_test_xgb, sm_test_rf=sm_test_rf, sm_test_dt=sm_test_dt,
        sm_test_ens=sm_test_ens,
        dae_err=dae_err, dae_thr=dae_thr,
        label_order=np.array(label_order, dtype=object),
    )

    # ── Apply multi-class fusion ──
    fusion, pred_attack = classify_fusion_multiclass(
        softmax_a=sm_test_ens,
        dae_score=dae_flag,
        label_order=label_order,
        a_high=A_HIGH,
        normal_idx=norm_idx,
        gate_normal_through_dae=True,
    )

    # ── Cascade-contract metrics ──
    is_unknown = m_test == held_out
    is_benign = y_test == 0
    is_known_attack = (y_test == 1) & ~is_unknown
    n_unknown = int(is_unknown.sum())

    top_p = sm_test_ens.max(axis=1)
    top_class = sm_test_ens.argmax(axis=1)

    # On unknown attacks, how often is Track A "uncertain" (top_p < a_high)?
    high_conf_on_unknown = int((is_unknown & (top_p >= A_HIGH)).sum())
    uncertain_on_unknown = int((is_unknown & (top_p < A_HIGH)).sum())

    # Within uncertain unknown attacks, how often does DAE flag?
    uncertain_unknown_mask = is_unknown & (top_p < A_HIGH)
    dae_on_uncertain_unknown = int((uncertain_unknown_mask & (dae_flag == 1)).sum())

    # Of the high-conf-on-unknown rows, what does Track A predict?
    high_conf_top_class_dist = pd.Series(
        [label_order[c] for c in top_class[is_unknown & (top_p >= A_HIGH)]]
    ).value_counts().to_dict()

    # DAE FPR on Track-A-uncertain benigns
    uncertain_benign_mask = is_benign & (top_p < A_HIGH)
    dae_fp_on_uncertain_benign = int(
        (uncertain_benign_mask & (dae_flag == 1)).sum()
    )
    n_uncertain_benign = int(uncertain_benign_mask.sum())

    # Cascade end-to-end (any non-BENIGN = surfaced)
    surfaced = fusion != FusionClass.BENIGN.value
    tp = int((surfaced & (y_test == 1)).sum())
    fp = int((surfaced & is_benign).sum())
    fn = int((~surfaced & (y_test == 1)).sum())
    tn = int((~surfaced & is_benign).sum())

    # Fusion class distribution
    fc_counts = pd.Series(fusion).value_counts().to_dict()

    fold_result = {
        "held_out_category": str(held_out),
        "label_order_in_fold": [str(s) for s in label_order],
        "n_train_rows_used": int(len(Xt)),
        "n_unknown_in_test": n_unknown,
        "n_benign_in_test": int(is_benign.sum()),
        "n_known_attack_in_test": int(is_known_attack.sum()),

        "track_a_on_unknown": {
            "high_confidence_count": high_conf_on_unknown,
            "high_confidence_rate": round(
                high_conf_on_unknown / max(n_unknown, 1), 4),
            "uncertain_count": uncertain_on_unknown,
            "uncertain_rate": round(
                uncertain_on_unknown / max(n_unknown, 1), 4),
            "high_conf_top_class_distribution": high_conf_top_class_dist,
            "comment": (
                "Multi-class Track A is supposed to be UNCERTAIN on the "
                "held-out category (since it's a class never seen). "
                "Higher uncertain_rate = better cascade-contract behaviour. "
                "Compare to the BINARY LOCO baseline: <2% uncertain_rate."
            ),
        },

        "dae_on_uncertain_unknown": {
            "n_uncertain_unknown": uncertain_on_unknown,
            "n_caught_by_dae": dae_on_uncertain_unknown,
            "recall_on_uncertain_unknown": round(
                dae_on_uncertain_unknown / max(uncertain_on_unknown, 1), 4),
            "comment": (
                "Of the unknown attacks Track A correctly flagged as uncertain, "
                "how many does DAE recover? This is the cascade-contract "
                "completion metric — Track A passed the buck, DAE either "
                "catches them or doesn't."
            ),
        },

        "dae_on_uncertain_benign": {
            "n_uncertain_benign": n_uncertain_benign,
            "n_dae_fp": dae_fp_on_uncertain_benign,
            "fpr_on_uncertain_benign": round(
                dae_fp_on_uncertain_benign / max(n_uncertain_benign, 1), 4),
        },

        "cascade_end_to_end": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": round(tp / max(tp + fn, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
            "fpr": round(fp / max(fp + tn, 1), 4),
        },

        "fusion_class_distribution": {
            str(k): int(v) for k, v in fc_counts.items()
        },
        "dae_threshold": float(dae_thr),
    }

    logger.info("  Track A high-conf on unknown '%s': %d/%d (%.2f%%)",
                held_out, high_conf_on_unknown, n_unknown,
                100 * high_conf_on_unknown / max(n_unknown, 1))
    logger.info("  Track A UNCERTAIN on unknown '%s': %d/%d (%.2f%%) ★",
                held_out, uncertain_on_unknown, n_unknown,
                100 * uncertain_on_unknown / max(n_unknown, 1))
    logger.info("  DAE catch on uncertain unknown: %d/%d (%.2f%%) ★",
                dae_on_uncertain_unknown, uncertain_on_unknown,
                100 * dae_on_uncertain_unknown / max(uncertain_on_unknown, 1))
    logger.info("  DAE FPR on uncertain benign: %d/%d (%.2f%%)",
                dae_fp_on_uncertain_benign, n_uncertain_benign,
                100 * dae_fp_on_uncertain_benign / max(n_uncertain_benign, 1))
    return fold_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LOCO multi-class cascade-validation on MedSec-25",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--train-sample-cap", type=int, default=80000,
        help="Stratified subsample cap (default 80,000 for ~3-5min/fold)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not (PROCESSED / "train.parquet").exists():
        logger.error("Run experiments/medsec25_loco/preprocess.py first.")
        return 1

    logger.info("Loading splits from %s", PROCESSED)
    X_train, y_train, m_train = _load_split("train")
    X_val, y_val, m_val = _load_split("val")
    X_test, y_test, m_test = _load_split("test")

    attack_cats = sorted({c for c in pd.unique(m_train) if c != "Benign"})
    logger.info("Attack categories: %s", attack_cats)

    folds: list[dict] = []
    t0 = time.perf_counter()
    for cat in attack_cats:
        try:
            r = _run_fold(
                cat,
                X_train, y_train, m_train,
                X_val, y_val, m_val,
                X_test, y_test, m_test,
                args,
            )
            folds.append(r)
        except Exception as exc:
            logger.exception("Fold %s failed: %s", cat, exc)
            folds.append({"held_out_category": cat, "error": str(exc)})

    summary = {
        "experiment": "MedSec-25 LOCO cascade validation (MULTI-CLASS Track A)",
        "purpose": (
            "Verify the cascade contract — trees are specific-pattern "
            "matchers; novel-category attacks produce spread softmax → DAE "
            "checks. Compares directly to the binary LOCO experiment "
            "(results/medsec25_loco/loco_results.yaml)."
        ),
        "thresholds": {
            "a_high": A_HIGH,
            "dae_threshold_percentile": DAE_PCT,
        },
        "categories_tested": attack_cats,
        "folds": folds,
        "verdict_per_fold": {
            f["held_out_category"]: (
                "PASS" if (
                    f.get("track_a_on_unknown", {}).get("uncertain_rate", 0) >= 0.50
                    and f.get("dae_on_uncertain_unknown", {}).get(
                        "recall_on_uncertain_unknown", 0) >= 0.50
                ) else "PARTIAL" if (
                    f.get("track_a_on_unknown", {}).get("uncertain_rate", 0) >= 0.50
                ) else "FAIL"
            )
            for f in folds if "error" not in f
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }

    summary_native = _to_native(summary)
    yaml_path = OUT_DIR / "loco_multiclass_results.yaml"
    json_path = OUT_DIR / "loco_multiclass_results.json"
    yaml_path.write_text(yaml.safe_dump(summary_native, sort_keys=False),
                         encoding="utf-8")
    json_path.write_text(json.dumps(summary_native, indent=2), encoding="utf-8")
    logger.info("Wrote %s", yaml_path)
    logger.info("Wrote %s", json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
