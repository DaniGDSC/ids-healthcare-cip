"""Retrain the cascaded DAE on a 26-dim ``[25 raw || P_xgb_val]`` input.

Phase B of the v5 Track-A simplification (see
``docs/post_defense_track_a_simplification.md``). Phase A made
``classify_alert_v4`` independent of RandomForest / DecisionTree;
Phase B (this script) collapses the DAE cascade input from 28 → 26
dims so RF/DT no longer need to be inferred at runtime.

Inputs read from ``results/models/``:
  * ``xgboost_val_proba.npy`` (or its ``_calibrated`` variant) — the
    calibrated XGBoost probas on held-out validation benign rows.
  * ``dae_best_params.json`` — hyperparameters from the prior tuning
    run, reused unchanged.

Inputs read from ``data/processed/``:
  * ``val_phase1.parquet`` — used to extract the 25 raw features and
    the benign mask.

Outputs (overwrites):
  * ``results/models/dae_detector.pkl``
  * ``results/models/dae_calibration.json``
  * ``results/models/dae_thresholds.json``

**Backup the old artefacts before running.** This script does not
auto-backup. Suggested:

.. code-block:: bash

    mkdir -p results/models/_v4_backup
    cp results/models/dae_detector.pkl       results/models/_v4_backup/
    cp results/models/dae_calibration.json   results/models/_v4_backup/
    cp results/models/dae_thresholds.json    results/models/_v4_backup/

Usage::

    python -m module2_detection.retrain_dae_26dim
    python -m module2_detection.retrain_dae_26dim --random-state 42

After running, re-validate::

    python run_tests.py
    # Expect: RECOMMENDATION: ✓ SHIP_TO_USER_STUDY
    # If M6 (test_false_positive_rate) drops below 0.20 minimum, run
    # ``module2_detection/calibrate.py`` to re-tune DAE thresholds.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import dumps_signed  # noqa: E402
from module2_detection._features import drop_non_feature_cols  # noqa: E402
from module2_detection.models.DAE import DAEDetector  # noqa: E402

logger = logging.getLogger(__name__)


def _load_xgb_val_proba(output_dir: Path) -> np.ndarray:
    """Prefer the calibrated XGB val probas; fall back to raw."""
    calibrated = output_dir / "xgboost_val_proba_calibrated.npy"
    raw = output_dir / "xgboost_val_proba.npy"
    if calibrated.exists():
        path = calibrated
    elif raw.exists():
        path = raw
    else:
        raise FileNotFoundError(
            f"Neither {calibrated.name} nor {raw.name} found in {output_dir}. "
            "Re-run module2_train_models.py before this script."
        )
    arr = np.load(path)
    if arr.ndim == 2 and arr.shape[1] == 2:
        # If saved as full softmax, take P(attack) column.
        arr = arr[:, 1]
    logger.info("Loaded XGB val probas from %s (n=%d)", path.name, len(arr))
    return arr.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain DAE on 26-dim cascade input")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="results/models",
        help="Where to write the new dae artefacts (default: results/models).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t0 = time.perf_counter()
    sep = "=" * 72
    logger.info(sep)
    logger.info("PHASE B — DAE RETRAIN ON 26-DIM CASCADE")
    logger.info(sep)

    output_dir = PROJECT_ROOT / args.output_dir

    # Load val parquet → 25 raw features for benign rows.
    val_path = PROJECT_ROOT / "data/processed/val_phase1.parquet"
    if not val_path.exists():
        logger.error("Missing %s — run Phase 1 preprocessing first.", val_path)
        return 1
    val_df = pd.read_parquet(val_path)
    y_val = val_df["Label"].values.astype(int)
    X_val = drop_non_feature_cols(val_df).values.astype(np.float32)
    benign_mask = y_val == 0
    X_benign = X_val[benign_mask]
    logger.info(
        "Loaded val parquet (n_rows=%d, n_benign=%d, n_features=%d)",
        len(y_val), int(benign_mask.sum()), X_val.shape[1],
    )

    # Load XGB val probas, mask to benigns.
    p_xgb_val = _load_xgb_val_proba(output_dir)
    if len(p_xgb_val) != len(y_val):
        logger.error(
            "XGB val proba length (%d) != val parquet rows (%d). "
            "Re-run xgboost calibration before this script.",
            len(p_xgb_val), len(y_val),
        )
        return 1
    p_xgb_benign = p_xgb_val[benign_mask].reshape(-1, 1)

    # Build 26-dim cascade: [25 raw || P_xgb].
    X_benign_aug = np.column_stack([X_benign, p_xgb_benign])
    logger.info(
        "26-dim cascade input: shape=%s (was 28 in v4)",
        X_benign_aug.shape,
    )
    assert X_benign_aug.shape[1] == 26, (
        f"Expected 26-dim cascade; got {X_benign_aug.shape[1]}"
    )

    # Load DAE hyperparams from the prior tuning run.
    params_path = output_dir / "dae_best_params.json"
    if not params_path.exists():
        logger.error("Missing %s. Run dae tuning first.", params_path)
        return 1
    best_hp = json.loads(params_path.read_text(encoding="utf-8"))
    logger.info("Best DAE hyperparams: %s", best_hp)

    # Instantiate DAEDetector with the v4-tuned hyperparams. The
    # detector auto-derives ``n_features_in_`` from the training data
    # shape, so the 26-dim input flows through without any explicit
    # dim arg. The ``encoding_dims`` from v4 (24/12/24) was sized for
    # the 28-dim cascade — see Phase B note: the bottleneck (12) is
    # still < n_features (26), so the architecture remains valid.
    # If a tighter compression ratio is desired, re-tune via
    # ``module2_detection/tuning/run_dae.py`` after this script lands.
    det = DAEDetector(
        encoding_dims=best_hp.get("encoding_dims"),
        noise_rate=best_hp.get("noise_rate", 0.1),
        learning_rate=best_hp.get("learning_rate", 1e-3),
        threshold_percentile=best_hp.get("threshold_percentile", 95.0),
        clip_percentile=best_hp.get("clip_percentile", 1.0),
        epochs=int(best_hp.get("epochs", 100)),
        batch_size=int(best_hp.get("batch_size", 256)),
        random_state=args.random_state,
    )

    logger.info("Fitting DAE on %d benign samples (26-dim cascade)…", X_benign_aug.shape[0])
    det.fit(X_benign_aug)

    # Save new artefacts.
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "dae_detector.pkl"
    dumps_signed(det, pkl_path)
    logger.info("Saved %s", pkl_path)

    # Calibration JSON: percentile-rank curve over benign train errors.
    train_errors = det._train_errors  # noqa: SLF001 — internal but stable
    assert train_errors is not None, "DAE.fit() did not populate _train_errors"
    sorted_errs = np.sort(train_errors)
    calib = {
        "format_version": 2,
        "cascade_input_dim": 26,
        "method": "percentile_rank",
        "n_benign": int(len(train_errors)),
        "errors_sorted": sorted_errs.tolist(),
    }
    calib_path = output_dir / "dae_calibration.json"
    calib_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    logger.info("Saved %s (%d benign error samples)", calib_path, len(sorted_errs))

    # Thresholds JSON: percentile cuts the rest of the system already
    # consumes.
    thresholds = {
        "format_version": 2,
        "cascade_input_dim": 26,
        "p50": float(np.percentile(train_errors, 50)),
        "p90": float(np.percentile(train_errors, 90)),
        "p95": float(np.percentile(train_errors, 95)),
        "p99": float(np.percentile(train_errors, 99)),
        "threshold_percentile_used": det._threshold_pct,  # noqa: SLF001
        "threshold": float(det._threshold),  # noqa: SLF001
    }
    thr_path = output_dir / "dae_thresholds.json"
    thr_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    logger.info("Saved %s (threshold=%.6f)", thr_path, thresholds["threshold"])

    elapsed = time.perf_counter() - t0
    logger.info(sep)
    logger.info("DAE RETRAIN SUMMARY")
    logger.info(sep)
    logger.info("  Cascade input dim : 26  (was 28)")
    logger.info("  Benign samples    : %d", X_benign_aug.shape[0])
    logger.info("  p95 threshold     : %.6f", thresholds["p95"])
    logger.info("  Elapsed           : %.1f s", elapsed)
    logger.info(sep)
    logger.info("Next steps:")
    logger.info("  1. Update layer2_detector.py cascade construction to 26-dim.")
    logger.info("  2. Run `python run_tests.py` and verify SHIP_TO_USER_STUDY holds.")
    logger.info("  3. If M6 (FPR) regresses, run module2_detection/calibrate.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
