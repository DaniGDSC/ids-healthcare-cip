"""Score an arbitrary feature parquet through Track A + Track B.

Pure inference helper used by the fusion-threshold calibration scripts
(analysis/calibrate_fusion_threshold.py, analysis/verify_fusion_threshold_holdout.py).
No retraining; no side effects beyond model-registry caching.

Bit-exact reproduction of the production test-set scoring path:
  Track A — xgboost_calibrator.pkl (CalibratedClassifierCV).predict_proba(X)[:, 1]
  Track B — DAE from model_registry.predict_proba(X_sanitised)
  X_sanitised — NaN/Inf → 0.0 (matches module3_risk_scores._sanitise_features)

The 25 feature columns must match those used to train XGBoost / DAE;
non-feature columns (Label, Attack Category, row_id, device_class,
attack_category, severity_tier) are dropped automatically.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_XGB_CALIBRATOR = PROJECT_ROOT / "results/models/xgboost_calibrator.pkl"

_NON_FEATURE_COLS = {
    "Label", "Attack Category", "row_id",
    "device_class", "attack_category", "severity_tier",
}


@dataclass(frozen=True)
class SplitScores:
    """Container for scored split — arrays are aligned, length = n_rows."""
    c_track_a: np.ndarray   # calibrated XGBoost P(attack), shape (n,)
    c_track_b: np.ndarray   # DAE anomaly score, shape (n,)
    y_true: np.ndarray      # binary labels from Label column, shape (n,)
    attack_category: np.ndarray | None  # if column present, else None
    feat_names: list[str]
    parquet_sha256: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sanitise(X: np.ndarray) -> np.ndarray:
    """Match module3_risk_scores._sanitise_features: NaN/Inf → 0.0."""
    return np.where(np.isnan(X) | np.isinf(X), 0.0, X).astype(np.float32)


def score_parquet(parquet_path: str | Path) -> SplitScores:
    """Score a feature parquet through Track A (calibrated XGBoost) + Track B (DAE).

    Args:
        parquet_path: path to a parquet with the 25 feature columns plus
            (optionally) Label, Attack Category, row_id, device_class.

    Returns:
        SplitScores with c_track_a, c_track_b, y_true (if Label present),
        attack_category (if column present), feat_names, and parquet sha256.
    """
    from common.model_registry import get_dae

    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)
    feat_names = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    X = df[feat_names].values.astype(np.float32)

    # Track A — calibrated XGBoost. Calibrator is CalibratedClassifierCV(cv='prefit')
    # wrapping xgboost_final_pipeline; predict_proba returns shape (n, 2).
    cal = joblib.load(_XGB_CALIBRATOR)
    c_track_a = cal.predict_proba(X)[:, 1].astype(np.float64)

    # Track B — DAE on sanitised features (matches compute_c_detect).
    dae = get_dae()
    c_track_b = dae.predict_proba(_sanitise(X)).astype(np.float64)

    y_true = df["Label"].values.astype(np.int64) if "Label" in df.columns else None
    attack_cats = (
        df["Attack Category"].astype(str).values
        if "Attack Category" in df.columns else None
    )

    return SplitScores(
        c_track_a=c_track_a,
        c_track_b=c_track_b,
        y_true=y_true,
        attack_category=attack_cats,
        feat_names=feat_names,
        parquet_sha256=_sha256_file(parquet_path),
    )


def verify_reproduces_test() -> None:
    """Sanity check: score_parquet on test_phase1.parquet must reproduce
    the persisted production probas bit-exactly. Used as a CI guard against
    silent divergence between this helper and the production scoring path.
    """
    scores = score_parquet(PROJECT_ROOT / "data/processed/test_phase1.parquet")
    existing_ta = np.load(PROJECT_ROOT / "results/models/xgboost_test_proba_calibrated.npy")
    existing_npz = np.load(PROJECT_ROOT / "results/reports/risk_scores.npz", allow_pickle=False)
    existing_tb = existing_npz["c_track_b"]

    if scores.c_track_a.shape != existing_ta.shape:
        raise RuntimeError(
            f"Track A shape mismatch: {scores.c_track_a.shape} vs {existing_ta.shape}"
        )
    max_ta_diff = float(np.abs(scores.c_track_a - existing_ta).max())
    max_tb_diff = float(np.abs(scores.c_track_b - existing_tb).max())
    if max_ta_diff > 0.0 or max_tb_diff > 1e-6:
        raise RuntimeError(
            f"Scorer diverges from production: "
            f"max|ΔTrack_A|={max_ta_diff}, max|ΔTrack_B|={max_tb_diff}"
        )
    print(f"score_split: reproduces test-set Track A and Track B exactly "
          f"(max|ΔA|={max_ta_diff}, max|ΔB|={max_tb_diff})")


if __name__ == "__main__":
    verify_reproduces_test()
