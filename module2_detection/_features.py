"""Module-2-local helpers shared across training and calibration scripts.

Kept inside ``module2_detection/`` rather than ``common/`` because these
helpers are tied to the Module 2 input contract (the column set produced
by Phase 1 preprocessing) and have no consumers outside this module.
"""

from __future__ import annotations

import pandas as pd

NON_FEATURE_COLS: tuple[str, ...] = (
    "Label",
    "Attack Category",
    "row_id",
    "device_class",
    "attack_category",
)


def drop_non_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop label / metadata columns; keep only model-input features."""
    drop = [c for c in NON_FEATURE_COLS if c in df.columns]
    return df.drop(columns=drop)
