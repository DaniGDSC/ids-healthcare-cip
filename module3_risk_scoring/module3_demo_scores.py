"""Module 3 demo-pool risk scoring (Strategy 1 — Frozen Demo Pool).

Per ARCHITECTURE.md: the dashboard / user-study path is independent of
the paper-metrics path. M3's main script (``module3_risk_scores.py``)
reads ``test_phase1.parquet`` → ``risk_scores.npz``; this driver reads
``demo_phase1.parquet`` → ``demo_scores.npz``. M6 sources
``evaluation_alerts.json`` from the demo path NEVER from the test path.

The two paths share scoring helpers (``compute_c_detect``,
``compute_composite_risk``, etc.) but persist disjoint NPZ outputs so a
reviewer can verify, byte-for-byte, that no test row appears in any
dashboard artefact.

Run::

    python -m module3_risk_scoring.module3_demo_scores
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from module3_risk_scoring.module3_risk_scores import (  # noqa: E402
    assign_risk_levels,
    compute_c_detect,
    compute_composite_risk,
    compute_d_clinical_tier,
    compute_d_crit,
    compute_s_data,
    load_split_data,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "results/reports"


def _load_xgb_demo_proba() -> tuple[np.ndarray, float]:
    """Load XGB demo probabilities + the F2-tuned threshold from the
    paper-side report (threshold is data-independent)."""
    import json

    models_dir = PROJECT_ROOT / "results/models"
    npz = np.load(models_dir / "xgboost_demo_predictions.npz")
    y_proba = npz["y_proba"]
    with open(models_dir / "xgboost_final_report.json") as f:
        threshold = json.load(f)["optimal_threshold"]
    return y_proba, threshold


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sep = "=" * 72
    logger.info(sep)
    logger.info("MODULE 3 — DEMO-POOL SCORES (Strategy 1)")
    logger.info(sep)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_demo, y_demo, attack_cats, feat_names = load_split_data("demo")
    logger.info(
        "Demo data: %d samples, %d attacks, attack_rate=%.4f",
        len(y_demo), int((y_demo == 1).sum()), float(y_demo.mean()),
    )

    c_track_a, xgb_threshold = _load_xgb_demo_proba()

    c_detect, c_track_b, fusion_class, data_quality = compute_c_detect(
        c_track_a, X_demo, xgb_threshold=xgb_threshold,
    )
    d_crit = compute_d_crit(attack_cats)
    s_data = compute_s_data(X_demo, feat_names)
    d_clinical_tier = compute_d_clinical_tier(X_demo, feat_names)

    R = compute_composite_risk(c_detect, d_crit, s_data, d_clinical_tier)
    levels = assign_risk_levels(R)

    # Persist row_id when the parquet carries it (graceful degradation
    # to positional index — same fallback Module 2 uses).
    df = pd.read_parquet(PROJECT_ROOT / "data/processed/demo_phase1.parquet")
    if "row_id" in df.columns:
        row_id = df["row_id"].values.astype(np.int64)
    else:
        row_id = np.arange(len(y_demo), dtype=np.int64)

    out_path = OUTPUT_DIR / "demo_scores.npz"
    np.savez(
        out_path,
        row_id=row_id,
        y_true=y_demo,
        c_track_a=c_track_a,
        c_track_b=c_track_b,
        c_detect=c_detect,
        d_crit=d_crit,
        s_data=s_data,
        d_clinical_tier=d_clinical_tier,
        R=R,
        risk_levels=levels.astype("U16"),
        fusion_class=fusion_class.astype("U24"),
        data_quality=data_quality.astype("U16"),
        attack_category=(
            attack_cats.astype("U32")
            if attack_cats is not None
            else np.array([], dtype="U32")
        ),
    )
    logger.info("Saved: %s", out_path.relative_to(PROJECT_ROOT))

    for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        n = int((levels == level).sum())
        logger.info("  %-10s %5d (%5.1f%%)", level, n, n / len(y_demo) * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
