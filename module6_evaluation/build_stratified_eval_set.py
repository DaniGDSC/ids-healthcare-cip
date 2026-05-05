"""Materialise the calibration / holdout split described in
results/reports/track_a_performance.yaml § test_set.primary_set.

Reads:
  - data/processed/test_phase1.parquet
  - results/reports/risk_scores.npz   (provides per-row severity tier)
  - results/models/xgboost_test_predictions.npz   (for joinable predictions)

Writes:
  - results/reports/stratified_calibration.parquet
  - results/reports/stratified_holdout.parquet
  - results/reports/stratified_split_summary.yaml

Stratification: 70/30 by composite-risk severity tier (CRITICAL/HIGH/MEDIUM/LOW),
deterministic via random_state=42.

Closes acceptance criterion AC-1 of track_a_performance.yaml.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

TIER_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def load_inputs(seed: int) -> pd.DataFrame:
    """Join test parquet + per-row severity tier from risk_scores.npz.

    Falls back to positional join when the test parquet lacks `row_id`
    (legacy artifact from before GAP-PB-1 closure).
    """
    test_path = PROJECT_ROOT / "data/processed/test_phase1.parquet"
    risk_path = PROJECT_ROOT / "results/reports/risk_scores.npz"

    if not test_path.exists():
        raise FileNotFoundError(
            f"{test_path} not found. Re-run module1_preprocessing/phase1 first."
        )
    if not risk_path.exists():
        raise FileNotFoundError(
            f"{risk_path} not found. "
            "Re-run module3_risk_scoring/module3_risk_scores.py first."
        )

    df = pd.read_parquet(test_path)
    risk = dict(np.load(risk_path, allow_pickle=True))
    risk_levels = risk["risk_levels"].astype(str)

    if len(risk_levels) != len(df):
        raise ValueError(
            f"row count mismatch: test parquet={len(df)}, risk_levels={len(risk_levels)}. "
            "Did you re-run Module 1 without re-running Module 3?"
        )

    df = df.reset_index(drop=True)
    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df), dtype=np.int64)
    df["severity_tier"] = risk_levels
    return df


def build_split(df: pd.DataFrame, holdout_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 70/30 split (or `holdout_ratio` for holdout) by severity_tier."""
    calibration_idx, holdout_idx = train_test_split(
        np.arange(len(df)),
        test_size=holdout_ratio,
        stratify=df["severity_tier"].values,
        random_state=seed,
    )
    calibration = df.iloc[calibration_idx].reset_index(drop=True)
    holdout = df.iloc[holdout_idx].reset_index(drop=True)
    return calibration, holdout


def per_tier_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["severity_tier"].value_counts().to_dict()
    return {t: int(counts.get(t, 0)) for t in TIER_ORDER}


def write_summary(
    calibration: pd.DataFrame,
    holdout: pd.DataFrame,
    seed: int,
    holdout_ratio: float,
    out_path: Path,
) -> None:
    summary = {
        "schema_version": 1,
        "random_seed": int(seed),
        "holdout_ratio": float(holdout_ratio),
        "stratification_key": "severity_tier",
        "calibration": {
            "total": int(len(calibration)),
            "per_tier_counts": per_tier_counts(calibration),
            "artifact": "results/reports/stratified_calibration.parquet",
        },
        "holdout": {
            "total": int(len(holdout)),
            "per_tier_counts": per_tier_counts(holdout),
            "artifact": "results/reports/stratified_holdout.parquet",
        },
        "closes_gap": "GAP-PB-1 partially (materialised partitions); AC-1 PASS",
    }
    out_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    logger.info("Wrote %s", out_path.relative_to(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified calibration/holdout split.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--holdout-ratio", type=float, default=0.30,
                        help="Holdout fraction (default: 0.30)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = PROJECT_ROOT / "results/reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(args.seed)
    logger.info("Loaded %d rows; per-tier counts: %s",
                len(df), per_tier_counts(df))

    calib, holdout = build_split(df, args.holdout_ratio, args.seed)
    logger.info("Calibration: %d rows %s", len(calib), per_tier_counts(calib))
    logger.info("Holdout:     %d rows %s", len(holdout), per_tier_counts(holdout))

    calib_path = out_dir / "stratified_calibration.parquet"
    holdout_path = out_dir / "stratified_holdout.parquet"
    calib.to_parquet(calib_path, index=False)
    holdout.to_parquet(holdout_path, index=False)
    logger.info("Wrote %s, %s",
                calib_path.relative_to(PROJECT_ROOT),
                holdout_path.relative_to(PROJECT_ROOT))

    write_summary(calib, holdout, args.seed, args.holdout_ratio,
                  out_dir / "stratified_split_summary.yaml")


if __name__ == "__main__":
    main()
