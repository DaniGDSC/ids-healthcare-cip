#!/usr/bin/env python3
"""Compute per-feature baseline statistics from the benign training split.

Output: ``artifacts/feature_baselines.json`` with one entry per feature::

    {
      "SYS": {
        "median": 121.0,
        "iqr_low": 112.0,
        "iqr_high": 130.0,
        "unit": "mmHg",
        "decimal_places": 0,
        "is_biometric": true
      },
      ...
    }

Phase 1.1 of the upgrade plan: clinician/analyst summaries embed
``observed_value (baseline X, ±Y%)`` instead of the bare narrative
phrase. This is the data source.

Median + IQR (not mean + std) because feature distributions are heavily
skewed — IQR is robust to the rare outliers that would warp a mean.

Units + decimal_places are hand-curated for the small biometric +
common-network set; everything else defaults to no unit and 2 decimals,
which is fine for SHAP-driver display purposes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH   = PROJECT_ROOT / "data" / "processed" / "train_phase1.parquet"
OUTPUT_PATH  = PROJECT_ROOT / "artifacts" / "feature_baselines.json"


_FEATURE_UNITS: dict[str, tuple[str, int]] = {
    # Biometric — units carry clinical meaning, precision matters
    "SYS":        ("mmHg", 0),
    "DIA":        ("mmHg", 0),
    "Pulse_Rate": ("bpm", 0),
    "Heart_rate": ("bpm", 0),
    "Resp_Rate":  ("/min", 0),
    "SpO2":       ("%", 0),
    "Temp":       ("°C", 1),
    "ST":         ("mV", 2),
    # Network — units help analysts but most are scale-free magnitudes
    "SrcBytes":   ("B", 0),
    "DstBytes":   ("B", 0),
    "TotBytes":   ("B", 0),
    "SrcLoad":    ("bps", 0),
    "DstLoad":    ("bps", 0),
    "Load":       ("bps", 0),
    "Dur":        ("s", 3),
    "Sport":      ("port", 0),
    "DIntPkt":    ("ms", 2),
    "SIntPkt":    ("ms", 2),
    "SIntPktAct": ("ms", 2),
}

_BIOMETRIC = {
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
    "Heart_rate", "Resp_Rate", "ST",
}


def build_baselines(train_df: pd.DataFrame) -> dict:
    """Return ``{feat_name: {median, iqr_low, iqr_high, unit, ...}}``.

    Only benign rows (``Label == 0``) contribute to the baseline — by
    construction the alerts in the test set should deviate from this
    median, which is what the clinician needs to see.
    """
    benign = train_df[train_df["Label"] == 0]
    drop_cols = {"Label", "Attack Category", "row_id", "device_class"}
    feat_cols = [c for c in benign.columns if c not in drop_cols]

    stats: dict = {}
    for feat in feat_cols:
        col = benign[feat].astype(np.float64)
        p5, q25, med, q75, p95 = np.percentile(col, [5, 25, 50, 75, 95])
        unit, decimals = _FEATURE_UNITS.get(feat, ("", 2))
        stats[feat] = {
            "median":      round(float(med), decimals + 2),
            "iqr_low":     round(float(q25), decimals + 2),
            "iqr_high":    round(float(q75), decimals + 2),
            # Phase 2 — plausibility bounds for counterfactual search.
            # Counterfactual candidate values are clipped to [p05, p95] so
            # the proposed "what would have made this not alert" stays
            # inside the benign distribution rather than producing nonsense.
            "p05":         round(float(p5),  decimals + 2),
            "p95":         round(float(p95), decimals + 2),
            "unit":        unit,
            "decimal_places": decimals,
            "is_biometric": feat in _BIOMETRIC,
            "n_benign":    int(len(col)),
        }
    return stats


def main() -> int:
    if not TRAIN_PATH.exists():
        print(f"ERROR: {TRAIN_PATH} not found", flush=True)
        return 2
    df = pd.read_parquet(TRAIN_PATH)
    stats = build_baselines(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} "
          f"({len(stats)} features, {stats[next(iter(stats))]['n_benign']} benign rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
