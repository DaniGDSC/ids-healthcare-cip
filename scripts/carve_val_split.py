"""Post-hoc validation-slice carve (GAP-L1-2 / B3 helper).

Closes GAP-L1-2 on installations where the raw WUSTL-EHMS dataset is
unavailable, so the full Module 1 pipeline cannot be re-run from raw
inputs. The script operates on the *existing* preprocessed train
parquet, splitting it into a smaller train + a held-out validation
slice with the same ``StratifiedShuffleSplit`` logic the in-pipeline
splitter would have applied (val_ratio=0.20 of the train side, ~14%
of the global, stratified on Attack Category).

What it writes (atomic — old files only replaced after successful write):

- ``data/processed/train_phase1.parquet``      (overwritten — smaller)
- ``data/processed/val_phase1.parquet``        (new)
- ``data/processed/benign_only_train.parquet`` (rebuilt subset)
- ``data/processed/benign_only_val.parquet``   (new)

The test parquet is untouched; carve happens only on train.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data/processed"

VAL_RATIO_WITHIN_TRAIN = 0.20
RANDOM_STATE = 42

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    train_path = PROCESSED / "train_phase1.parquet"
    if not train_path.exists():
        logger.error("Missing %s — run Module 1 first.", train_path)
        return 1

    df = pd.read_parquet(train_path)
    logger.info("Loaded %s: %d rows", train_path.name, len(df))

    # Stratify on Attack Category if present (richer balance), else Label
    strat_col = "Attack Category" if "Attack Category" in df.columns else "Label"
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_RATIO_WITHIN_TRAIN,
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(sss.split(np.zeros(len(df)), df[strat_col].values))

    new_train = df.iloc[train_idx].reset_index(drop=True)
    val = df.iloc[val_idx].reset_index(drop=True)

    # Disjointness sanity check via row_id when available
    if "row_id" in df.columns:
        overlap = set(new_train["row_id"]) & set(val["row_id"])
        if overlap:
            logger.error("row_id overlap: %d", len(overlap))
            return 2

    logger.info(
        "Carve: train=%d (atk=%.2f%%) | val=%d (atk=%.2f%%)",
        len(new_train), new_train["Label"].mean() * 100,
        len(val), val["Label"].mean() * 100,
    )

    # Atomic-ish write: stage to *.tmp then rename
    val_path = PROCESSED / "val_phase1.parquet"
    train_benign_path = PROCESSED / "benign_only_train.parquet"
    val_benign_path = PROCESSED / "benign_only_val.parquet"

    new_train_benign = new_train[new_train["Label"] == 0].reset_index(drop=True)
    val_benign = val[val["Label"] == 0].reset_index(drop=True)

    new_train.to_parquet(train_path.with_suffix(".parquet.tmp"), index=False)
    val.to_parquet(val_path.with_suffix(".parquet.tmp"), index=False)
    new_train_benign.to_parquet(train_benign_path.with_suffix(".parquet.tmp"), index=False)
    val_benign.to_parquet(val_benign_path.with_suffix(".parquet.tmp"), index=False)

    train_path.with_suffix(".parquet.tmp").replace(train_path)
    val_path.with_suffix(".parquet.tmp").replace(val_path)
    train_benign_path.with_suffix(".parquet.tmp").replace(train_benign_path)
    val_benign_path.with_suffix(".parquet.tmp").replace(val_benign_path)

    logger.info("Wrote: %s, %s, %s, %s",
                train_path.name, val_path.name,
                train_benign_path.name, val_benign_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
