"""Leave-One-Category-Out (LOCO) cascade-validation experiment on MedSec-25.

Goal
----
Falsify or confirm the cascade contract:

    Track A (XGBoost) detects KNOWN attacks.
    DAE (Track B) detects UNKNOWN attacks AND verifies normal.

EHMS-2020 has only 2 attack categories, so it cannot test "unknown" in any
meaningful sense. MedSec-25 has 5 (Reconnaissance, Initial access,
Exfiltration, Lateral movement, plus Benign), so we can hold one attack
category out of Track A's training set entirely and ask whether the DAE
catches it during inference.

Protocol (per fold, 4 folds — one per attack category)
-------------------------------------------------------
1. Pick one attack category H to be the "unknown".
2. Build train_known = train rows whose Attack Category ∈ {Benign} ∪
   ({all attack categories} \\ {H}).
3. Train XGBoost on train_known (binary attack/benign).
4. Train DAE on benign-only rows from train_known (cascaded input
   = [features || P_xgb_val_known, P_rf_val_known, P_dt_val_known]).
5. Evaluate on the test set, restricted to:
     - test_known_attacks (categories ≠ H)
     - test_unknown_attacks (category = H)
     - test_benigns (Label = 0)
6. Report:
     - Track A high-confidence-rate on H (should be LOW for "unknown")
     - DAE flag-rate on H (should be HIGH if cascade holds)
     - DAE FPR on benigns (should be LOW)
     - Cascade-as-a-whole recall + FPR on test

Outputs
-------
results/medsec25_loco/loco_results.yaml — per-fold metrics + summary
results/medsec25_loco/per_fold/<H>/{xgb_proba.npy, dae_error.npy, ...}

Run
---
    python experiments/medsec25_loco/run_loco.py
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED = PROJECT_ROOT / "data/processed/medsec25"
OUT_DIR = PROJECT_ROOT / "results/medsec25_loco"

# Cascade thresholds (spec defaults from src/data_models.py)
A_HIGH = 0.85
A_LOW = 0.40
DAE_PCT = 99.0  # threshold percentile, matched to the EHMS post-retune value

logger = logging.getLogger(__name__)


def _load_split(name: str) -> tuple:
    df = pd.read_parquet(PROCESSED / f"{name}.parquet")
    y = df["Label"].values.astype(int)
    m = df["Attack Category"].values.astype(str)
    X = df.drop(columns=["Label", "Attack Category"]).values.astype(np.float32)
    return X, y, m


def _train_track_a(
    X_train: np.ndarray, y_train: np.ndarray,
    *, random_state: int, sample_cap: int,
) -> tuple[GradientBoostingClassifier, RandomForestClassifier, DecisionTreeClassifier]:
    """Train XGB-surrogate (GBM) + RF + DT on the binary task; return fitted models.

    The project uses ``sklearn.ensemble.GradientBoostingClassifier`` as an
    XGBoost surrogate (see module2_detection/models/XGBoost.py header) —
    we mirror that choice here so the LOCO experiment runs in the same
    environment as the EHMS pipeline. For speed, optionally stratified-
    subsample training rows to ``sample_cap``.
    """
    if sample_cap is not None and len(X_train) > sample_cap:
        rng = np.random.default_rng(random_state)
        idx_a = np.where(y_train == 1)[0]
        idx_b = np.where(y_train == 0)[0]
        cap_a = int(sample_cap * (idx_a.size / len(X_train)))
        cap_b = sample_cap - cap_a
        sel_a = rng.choice(idx_a, size=min(cap_a, idx_a.size), replace=False)
        sel_b = rng.choice(idx_b, size=min(cap_b, idx_b.size), replace=False)
        sel = np.concatenate([sel_a, sel_b])
        rng.shuffle(sel)
        X_train, y_train = X_train[sel], y_train[sel]

    n_atk = int((y_train == 1).sum())
    n_ben = int((y_train == 0).sum())
    # MedSec-25 is 97% attacks / 3% benigns — invert imbalance via
    # `class_weight='balanced'` on RF/DT and an explicit sample_weight
    # vector for GBM (which doesn't accept a class_weight arg).
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    xgb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        random_state=random_state,
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_split=5, min_samples_leaf=1,
        max_features=0.5, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    dt = DecisionTreeClassifier(
        max_depth=None, min_samples_split=2, min_samples_leaf=2,
        class_weight="balanced", random_state=random_state,
    )

    logger.info(
        "  Track A fit on %d rows (atk=%d, ben=%d, balanced sample_weight)",
        len(X_train), n_atk, n_ben,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weight)
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    return xgb, rf, dt


def _train_dae_cascaded(
    X_benign: np.ndarray,
    p_benign: np.ndarray,   # shape (n_benign, 3) — XGB / RF / DT val-set probas
    *, random_state: int,
):
    """Train cascaded DAE. Returns (det, threshold)."""
    from module2_detection.models.DAE import DAEDetector
    n_feat = X_benign.shape[1] + p_benign.shape[1]
    enc = max(int(n_feat * 0.85), n_feat - 4)
    bot = max(min(int(n_feat * 0.4), n_feat - 2), 4)
    arch = [enc, bot, enc]
    det = DAEDetector(
        encoding_dims=arch,
        noise_rate=0.10,
        learning_rate=1e-4,
        threshold_percentile=DAE_PCT,
        epochs=60,
        batch_size=256,
        random_state=random_state,
    )
    X_aug = np.column_stack([X_benign, p_benign]).astype(np.float32)
    logger.info("  DAE fit on %d benigns, %d features (arch=%s)",
                len(X_aug), n_feat, arch)
    det.fit(X_aug, validation_split=0.0)
    return det


def _scores(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray,
             y_score: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "f2":        float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        "auprc":     float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
    }


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
    logger.info("LOCO fold: HOLDING OUT '%s' from Track A's training set", held_out)
    logger.info(sep)

    # Mask: keep all train rows whose category != held_out
    keep_train = m_train != held_out
    keep_val = m_val != held_out
    Xt = X_train[keep_train]
    yt = y_train[keep_train]
    Xv = X_val[keep_val]
    yv = y_val[keep_val]
    logger.info("  Train rows after holding out '%s': %d (atk=%.2f%%)",
                held_out, len(Xt), yt.mean() * 100)

    # ── Train Track A ──
    xgb, rf, dt = _train_track_a(Xt, yt, random_state=args.random_state,
                                  sample_cap=args.train_sample_cap)

    # Predict on val (the half of the data the held-out attack was also
    # carved out of, so DAE proba columns match Track A's training distribution).
    p_xgb_val = _scores(xgb, Xv)
    p_rf_val = _scores(rf, Xv)
    p_dt_val = _scores(dt, Xv)
    val_probas = np.column_stack([p_xgb_val, p_rf_val, p_dt_val])

    # ── Train cascaded DAE on benign val rows ──
    val_benign_mask = yv == 0
    Xv_benign = Xv[val_benign_mask]
    val_probas_benign = val_probas[val_benign_mask]
    det = _train_dae_cascaded(
        Xv_benign, val_probas_benign,
        random_state=args.random_state,
    )

    # ── Evaluate on test set ──
    p_xgb_test = _scores(xgb, X_test)
    p_rf_test = _scores(rf, X_test)
    p_dt_test = _scores(dt, X_test)
    test_probas = np.column_stack([p_xgb_test, p_rf_test, p_dt_test])
    X_test_aug = np.column_stack([X_test, test_probas]).astype(np.float32)

    dae_err = det.reconstruction_error(X_test_aug)
    dae_thr = det.threshold
    dae_flag = (dae_err >= dae_thr).astype(int)

    # Persist arrays for later analysis
    np.savez(
        fold_dir / "test_predictions.npz",
        y_true=y_test, m_test=m_test,
        p_xgb=p_xgb_test, p_rf=p_rf_test, p_dt=p_dt_test,
        dae_err=dae_err, dae_thr=dae_thr,
    )

    # ── Compute regime-specific metrics ──
    # 1. Track A coverage: how often is XGB high-confident (>= a_high)?
    # 2. Track A silent: how often is XGB silent (< a_low)?
    high_conf = p_xgb_test >= A_HIGH
    silent = p_xgb_test < A_LOW
    confirm = (~high_conf) & (~silent)

    is_unknown = m_test == held_out
    is_benign = y_test == 0
    is_known_attack = (y_test == 1) & (~is_unknown)

    def _rate(mask: np.ndarray, pop: np.ndarray) -> float:
        n = int(pop.sum())
        if n == 0:
            return float("nan")
        return float((mask & pop).sum() / n)

    # The headline LOCO question:
    #   On UNKNOWN-category attacks, how often is Track A silent?
    #   And of those silent ones, how often does DAE flag?
    silent_unknown_mask = silent & is_unknown
    n_silent_unknown = int(silent_unknown_mask.sum())
    n_unknown = int(is_unknown.sum())

    # Track A's catch rate on unknown attacks (high-confidence — should be LOW)
    track_a_caught_unknown = int((high_conf & is_unknown).sum())
    track_a_silent_on_unknown = int((silent & is_unknown).sum())

    # DAE catch rate within Track-A-silent UNKNOWN attacks
    dae_caught_silent_unknown = int((dae_flag & silent_unknown_mask).sum())

    # DAE FPR on Track-A-silent benigns
    silent_benign_mask = silent & is_benign
    n_silent_benign = int(silent_benign_mask.sum())
    dae_fp_silent_benign = int((dae_flag & silent_benign_mask).sum())

    # Cascade end-to-end (KNOWN ∨ NOVEL ∨ CONFIRMED vs BENIGN)
    surfaced = high_conf | (confirm & (dae_flag == 1)) | (silent & (dae_flag == 1))
    tp = int((surfaced & (y_test == 1)).sum())
    fp = int((surfaced & is_benign).sum())
    fn = int((~surfaced & (y_test == 1)).sum())
    tn = int((~surfaced & is_benign).sum())

    # Track A standalone metrics on test (binary, threshold = A_HIGH for KNOWN)
    xgb_pred_high = (p_xgb_test >= A_HIGH).astype(int)
    track_a_metrics = _metrics(y_test, xgb_pred_high, p_xgb_test)

    fold_result = {
        "held_out_category": held_out,
        "n_train_rows_used": int(len(Xt)),
        "n_test_rows": int(len(X_test)),
        "n_unknown_in_test": n_unknown,
        "n_benign_in_test": int(is_benign.sum()),
        "n_known_attack_in_test": int(is_known_attack.sum()),

        "track_a_on_unknown": {
            "n_unknown_attacks": n_unknown,
            "high_confidence_count": track_a_caught_unknown,
            "high_confidence_rate": track_a_caught_unknown / max(n_unknown, 1),
            "silent_count": track_a_silent_on_unknown,
            "silent_rate": track_a_silent_on_unknown / max(n_unknown, 1),
            "comment": (
                "Higher silent_rate means Track A correctly fails to recognise "
                "the held-out category — necessary precondition for the DAE to "
                "do its job."
            ),
        },

        "dae_on_silent_unknown": {
            "n_silent_unknown": n_silent_unknown,
            "n_caught_by_dae": dae_caught_silent_unknown,
            "recall_on_silent_unknown": (
                dae_caught_silent_unknown / max(n_silent_unknown, 1)
            ),
            "comment": (
                "This is the cascade contract under test. The DAE is supposed "
                "to flag attacks that Track A misses (P_xgb < a_low)."
            ),
        },

        "dae_on_silent_benign": {
            "n_silent_benign": n_silent_benign,
            "n_dae_fp": dae_fp_silent_benign,
            "fpr_on_silent_benign": (
                dae_fp_silent_benign / max(n_silent_benign, 1)
            ),
        },

        "cascade_end_to_end": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "f1": (2 * tp / max(2 * tp + fp + fn, 1)),
            "fpr": fp / max(fp + tn, 1),
        },

        "track_a_standalone_at_a_high": track_a_metrics,
        "dae_threshold": float(dae_thr),
    }

    logger.info("  Track A high-conf on unknown '%s': %d/%d (%.2f%%)",
                held_out, track_a_caught_unknown, n_unknown,
                100 * track_a_caught_unknown / max(n_unknown, 1))
    logger.info("  Track A silent on unknown '%s': %d/%d (%.2f%%)",
                held_out, track_a_silent_on_unknown, n_unknown,
                100 * track_a_silent_on_unknown / max(n_unknown, 1))
    logger.info("  DAE catch on Track-A-silent unknown: %d/%d (%.2f%%) ★",
                dae_caught_silent_unknown, n_silent_unknown,
                100 * dae_caught_silent_unknown / max(n_silent_unknown, 1))
    logger.info("  DAE FPR on Track-A-silent benign: %d/%d (%.2f%%)",
                dae_fp_silent_benign, n_silent_benign,
                100 * dae_fp_silent_benign / max(n_silent_benign, 1))

    return fold_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LOCO cascade-validation experiment on MedSec-25",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--train-sample-cap", type=int, default=80000,
        help="Cap training rows per fold (stratified subsample, "
             "default 80,000 for ~3-5min/fold). None for full data.",
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

    # Aggregate summary
    summary = {
        "experiment": "MedSec-25 LOCO cascade validation",
        "purpose": (
            "Falsify or confirm the cascade contract — Track A detects "
            "KNOWN attacks; DAE detects UNKNOWN attacks + verifies normal — "
            "on a dataset with 5 attack categories so 'unknown' has meaning."
        ),
        "thresholds": {
            "a_high": A_HIGH,
            "a_low": A_LOW,
            "dae_threshold_percentile": DAE_PCT,
        },
        "categories_tested": attack_cats,
        "folds": folds,
        "verdict_per_fold": {
            f["held_out_category"]: (
                "PASS" if f.get("dae_on_silent_unknown", {}).get(
                    "recall_on_silent_unknown", 0) >= 0.50
                else "FAIL"
            )
            for f in folds if "error" not in f
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }

    # Cast numpy scalars/strings to native Python so yaml.safe_dump works
    # (np.str_ has no representer in PyYAML's SafeDumper). Mirrors
    # summarize_loco._to_native; kept inline to avoid a cross-module
    # dependency for a one-off conversion.
    def _native(x):
        if isinstance(x, dict):
            return {str(k): _native(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_native(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return _native(x.tolist())
        if isinstance(x, np.str_):
            return str(x)
        return x

    summary_native = _native(summary)
    yaml_path = OUT_DIR / "loco_results.yaml"
    yaml_path.write_text(yaml.safe_dump(summary_native, sort_keys=False),
                         encoding="utf-8")
    json_path = OUT_DIR / "loco_results.json"
    json_path.write_text(json.dumps(summary_native, indent=2), encoding="utf-8")
    logger.info("Wrote %s and %s", yaml_path, json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
