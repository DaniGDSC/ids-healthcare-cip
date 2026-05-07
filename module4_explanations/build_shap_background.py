"""Persist the TreeSHAP background sample (ARCHITECTURE.md Step [11]).

Writes ``results/models/shap_background.pkl`` — a 200-sample stratified
subset of the training set used as the SHAP background distribution.
Persisting it once means dashboard / batch SHAP runs don't have to
re-sample on every invocation, and the audit trail captures exactly
which rows were used as the baseline.

Run:

    python -m module4_explanations.build_shap_background
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train_path = PROJECT_ROOT / "data/processed/train_phase1.parquet"
    out_path = PROJECT_ROOT / "results/models/shap_background.pkl"
    if not train_path.exists():
        logger.error("Missing %s — run Module 1 first.", train_path)
        return 1

    df = pd.read_parquet(train_path)
    drop_cols = [
        c for c in ["Label", "Attack Category", "row_id",
                    "device_class", "attack_category"]
        if c in df.columns
    ]
    feat_names = [c for c in df.columns if c not in drop_cols]
    X = df[feat_names].values.astype(np.float32)
    y = df["Label"].values

    n_total = len(X)
    n_target = min(200, n_total)

    if n_total <= n_target:
        bg = X
    else:
        # Stratified sampling — keep benign/attack ratio close to train.
        _, bg, _, _ = train_test_split(
            X, y,
            test_size=n_target,
            random_state=42,
            stratify=y,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "background": bg.astype(np.float32),
            "feature_names": list(feat_names),
            "n_samples": int(len(bg)),
            "source": str(train_path.relative_to(PROJECT_ROOT)),
        },
        out_path,
        compress=3,
    )
    logger.info(
        "Wrote %s — %d samples × %d features",
        out_path.relative_to(PROJECT_ROOT), len(bg), bg.shape[1],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
