"""MedSec-25 preprocessing for the LOCO cascade-validation experiment.

This pipeline is intentionally separate from `module1_preprocessing/phase1`
— it operates on a different dataset (network-flow IDS, no biometrics)
and produces a different artifact set under `data/processed/medsec25/`.

Pipeline:
  1. Load `data/raw/MedSec-25/MedSec-25.csv` (554k rows × 84 cols).
  2. Drop high-cardinality identifiers (Flow ID, Src/Dst IP, Timestamp).
  3. Drop constant columns (zero variance — empirically 14 of them).
  4. Coerce to numeric, replace any residual ±inf with NaN, then drop rows
     with any NaN (CICFlowMeter rarely produces NaNs after step 1).
  5. Stratified 70/20/10 train/val/test split on the multi-class label.
  6. Fit a RobustScaler on train, transform val + test.
  7. Persist parquets + the multi-class label and a benign-only train slice
     for the DAE.

The DAE (Track B) and Track A trees in run_loco.py consume these
artifacts directly; no integrity baseline / Phase 0 audit is run for
this experiment because the LOCO study sits outside the thesis's main
methodological contract — it's a falsification check on the cascade
contract under a richer attack space.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data/raw/MedSec-25/MedSec-25.csv"
OUT_DIR = PROJECT_ROOT / "data/processed/medsec25"

DROP_IDS = ("Flow ID", "Src IP", "Dst IP", "Timestamp")
LABEL_COL = "Label"

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MedSec-25 preprocessing for LOCO cascade experiment",
    )
    parser.add_argument("--test-frac", type=float, default=0.10,
                        help="fraction held out as test (default 0.10)")
    parser.add_argument("--val-frac", type=float, default=0.20,
                        help="fraction (within trainval) carved as val (default 0.20)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--subsample", type=int, default=None,
                        help="random subsample to N rows (default: full dataset)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not RAW_CSV.exists():
        logger.error("Missing %s — re-run the kaggle download.", RAW_CSV)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", RAW_CSV.name)
    df = pd.read_csv(RAW_CSV)
    logger.info("Loaded: %d rows × %d cols", len(df), df.shape[1])

    # Optional subsample (for fast iteration)
    if args.subsample is not None and args.subsample < len(df):
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(len(df), size=args.subsample, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        logger.info("Subsampled to %d rows", len(df))

    # ── Step 1: drop ID-style columns ──
    df = df.drop(columns=[c for c in DROP_IDS if c in df.columns])
    logger.info("After ID drop: %d cols", df.shape[1])

    # ── Step 2: separate label, derive binary + multiclass ──
    multi = df[LABEL_COL].astype(str).copy()
    binary = (multi != "Benign").astype(int)
    df = df.drop(columns=[LABEL_COL])

    # ── Step 3: drop constants ──
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        logger.info("Dropped %d constant cols: %s", len(constant_cols), constant_cols)

    # ── Step 4: numeric coerce + clean inf/nan ──
    df = df.apply(pd.to_numeric, errors="coerce")
    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    keep = ~df.isna().any(axis=1)
    df = df[keep].reset_index(drop=True)
    multi = multi[keep.values].reset_index(drop=True)
    binary = binary[keep.values].reset_index(drop=True)
    if n_before != len(df):
        logger.info("Dropped %d rows with NaN/Inf", n_before - len(df))

    feat_names = df.columns.tolist()
    logger.info("Feature matrix: %d rows × %d numeric features",
                len(df), len(feat_names))

    # ── Step 5: stratified split (multiclass-stratified) ──
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_frac, random_state=args.random_state,
    )
    trainval_idx, test_idx = next(sss_test.split(df.values, multi.values))

    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_frac, random_state=args.random_state,
    )
    inner_train_idx, inner_val_idx = next(
        sss_val.split(df.iloc[trainval_idx].values, multi.iloc[trainval_idx].values)
    )
    train_idx = trainval_idx[inner_train_idx]
    val_idx = trainval_idx[inner_val_idx]

    X_train = df.iloc[train_idx].copy()
    X_val = df.iloc[val_idx].copy()
    X_test = df.iloc[test_idx].copy()
    y_train = binary.iloc[train_idx].values
    y_val = binary.iloc[val_idx].values
    y_test = binary.iloc[test_idx].values
    m_train = multi.iloc[train_idx].values
    m_val = multi.iloc[val_idx].values
    m_test = multi.iloc[test_idx].values

    logger.info(
        "Splits — train=%d (atk=%.2f%%) | val=%d (atk=%.2f%%) | test=%d (atk=%.2f%%)",
        len(X_train), y_train.mean() * 100,
        len(X_val), y_val.mean() * 100,
        len(X_test), y_test.mean() * 100,
    )

    # ── Step 6: scaling ──
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)
    X_test_scaled = scaler.transform(X_test.values)

    # ── Step 7: persist ──
    def _to_parquet(arr_scaled: np.ndarray, y_bin: np.ndarray,
                    m_lab: np.ndarray, name: str) -> None:
        out = pd.DataFrame(arr_scaled, columns=feat_names)
        out["Label"] = y_bin
        out["Attack Category"] = m_lab
        out.to_parquet(OUT_DIR / name, index=False)

    _to_parquet(X_train_scaled, y_train, m_train, "train.parquet")
    _to_parquet(X_val_scaled, y_val, m_val, "val.parquet")
    _to_parquet(X_test_scaled, y_test, m_test, "test.parquet")

    # Benign-only train slice for the DAE (matches EHMS pipeline convention)
    benign_mask = y_train == 0
    train_benign_df = pd.DataFrame(X_train_scaled[benign_mask], columns=feat_names)
    train_benign_df["Label"] = 0
    train_benign_df["Attack Category"] = m_train[benign_mask]
    train_benign_df.to_parquet(OUT_DIR / "train_benign.parquet", index=False)

    val_benign_mask = y_val == 0
    val_benign_df = pd.DataFrame(X_val_scaled[val_benign_mask], columns=feat_names)
    val_benign_df["Label"] = 0
    val_benign_df["Attack Category"] = m_val[val_benign_mask]
    val_benign_df.to_parquet(OUT_DIR / "val_benign.parquet", index=False)

    # Persist scaler + manifest
    manifest = {
        "rows_total": int(len(df)),
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "splits": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "attack_rate": {
            "train": round(float(y_train.mean()), 4),
            "val": round(float(y_val.mean()), 4),
            "test": round(float(y_test.mean()), 4),
        },
        "category_counts_in_train": {
            cat: int((m_train == cat).sum())
            for cat in pd.unique(m_train)
        },
        "category_counts_in_val": {
            cat: int((m_val == cat).sum())
            for cat in pd.unique(m_val)
        },
        "category_counts_in_test": {
            cat: int((m_test == cat).sum())
            for cat in pd.unique(m_test)
        },
        "constants_dropped": constant_cols,
        "ids_dropped": list(DROP_IDS),
        "random_state": args.random_state,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    import joblib
    joblib.dump(scaler, OUT_DIR / "robust_scaler.pkl")

    logger.info("Wrote %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
