#!/usr/bin/env python3
"""Cascaded-DAE hyperparameter re-tuning under the val-proba regime (post-GAP-L1-1).

The legacy ``run_dae.py`` tuner sweeps a non-cascaded DAE on raw
25-dim features. Once GAP-L1-1 closed and ``train_track_b_dae`` started
training on a 28-dim ``[25 raw || P_xgb_val, P_rf_val, P_dt_val]`` input,
the saved ``dae_best_params.json`` was no longer guaranteed to be
optimal — its noise/learning-rate/threshold values were chosen against
a different input distribution.

This script re-runs the grid search on the cascaded val-proba input:

  1. Load val benigns + val probas → 28-dim cascade input X_benign_aug.
  2. Split X_benign_aug 80/20 into ``X_benign_fit`` (~1,599)
     / ``X_benign_val`` (~400).
  3. Build the positive-class signal from val attacks (~286 rows) +
     their val probas.
  4. Grid-search architecture, noise rate, learning rate, and threshold
     percentile, scoring each candidate by attack-F2 on the validation
     mix. Test set is NOT touched (closed-leak protocol from ``run_dae.py``
     finding #1).
  5. Persist the winner to ``results/models/dae_best_params.json`` so
     the next ``module2_train_models.py`` invocation picks it up.

Run:
    python module2_detection/tuning/run_dae_cascade.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Architectures sized for the 28-dim cascade input. Bottleneck < 28
# (compression), encoder/decoder ≥ bottleneck. Four shapes spanning
# a tight bottleneck (12) to a near-identity bottleneck (24).
HP_GRID = {
    "encoding_dims": [
        [24, 12, 24],
        [28, 14, 28],
        [28, 18, 28],   # current params (control)
        [32, 20, 32],
    ],
    "noise_rate": [0.05, 0.10, 0.20],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "threshold_percentile": [90.0, 95.0, 99.0],
}


def load_cascade_data(
    val_parquet: Path,
    models_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build [25 raw || 3 val probas] inputs for val benigns and val attacks.

    Returns:
        (X_benign_aug, X_attack_aug, feat_names) — both arrays in 28-dim
        cascade space, ready for DAE consumption.
    """
    val_df = pd.read_parquet(val_parquet)
    drop_cols = [
        c for c in ("Label", "Attack Category", "row_id",
                    "device_class", "attack_category")
        if c in val_df.columns
    ]
    y_val = val_df["Label"].values.astype(int)
    X_val = val_df.drop(columns=drop_cols).values.astype(np.float32)
    feat_names = [c for c in val_df.columns if c not in drop_cols]

    probas = np.column_stack([
        np.load(models_dir / f"{name}_val_proba.npy")
        for name in ("xgboost", "random_forest", "decision_tree")
    ])
    if len(probas) != len(X_val):
        raise ValueError(
            f"val proba length ({len(probas)}) != val parquet length ({len(X_val)})"
        )

    X_aug = np.column_stack([X_val, probas]).astype(np.float32)
    benign_mask = (y_val == 0)
    X_benign_aug = X_aug[benign_mask]
    X_attack_aug = X_aug[~benign_mask]
    logger.info(
        "Cascade input: 28-dim (25 raw + 3 val probas); "
        "val benign=%d, val attack=%d",
        len(X_benign_aug), len(X_attack_aug),
    )
    return X_benign_aug, X_attack_aug, feat_names + [
        "track_a_xgb", "track_a_rf", "track_a_dt"
    ]


def make_cascade_validation_split(
    X_benign_aug: np.ndarray,
    X_attack_aug: np.ndarray,
    *,
    benign_val_frac: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80/20 split of val benigns into fit/val; concat val attacks.

    Returns:
        (X_benign_fit, X_val, y_val) — same shape contract as run_dae.py's
        ``make_validation_split``.
    """
    rng = np.random.default_rng(random_state)
    n_benign = len(X_benign_aug)
    n_val = int(round(n_benign * benign_val_frac))
    perm = rng.permutation(n_benign)
    val_idx, fit_idx = perm[:n_val], perm[n_val:]

    X_benign_fit = X_benign_aug[fit_idx]
    X_benign_val = X_benign_aug[val_idx]

    X_val = np.vstack([X_benign_val, X_attack_aug]).astype(np.float32)
    y_val = np.concatenate([
        np.zeros(len(X_benign_val), dtype=np.int32),
        np.ones(len(X_attack_aug), dtype=np.int32),
    ])
    logger.info(
        "Cascade HP-selection split: benign_fit=%d, "
        "benign_val=%d, attack_val=%d",
        len(X_benign_fit), len(X_benign_val), len(X_attack_aug),
    )
    return X_benign_fit, X_val, y_val


def _fit_one_candidate(args: tuple) -> tuple[dict, dict]:
    """Worker: fit one DAE candidate, return (hp, val_metrics).

    Lazy TF import inside the worker keeps the module-level import
    clean and isolates Keras graph state across processes.
    """
    hp, X_benign_fit, X_val, y_val, epochs, batch_size, random_state = args
    from module2_detection.models.DAE import DAEDetector  # noqa: PLC0415
    det = DAEDetector(
        encoding_dims=hp["encoding_dims"],
        noise_rate=hp["noise_rate"],
        learning_rate=hp["learning_rate"],
        threshold_percentile=hp["threshold_percentile"],
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    )
    det.fit(X_benign_fit, validation_split=0.0)
    val_metrics = det.evaluate(X_val, y_val)
    return hp, val_metrics


def grid_search(
    X_benign_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    random_state: int,
) -> tuple[dict, list[dict]]:
    keys = list(HP_GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*HP_GRID.values())]
    n = len(combos)
    max_workers = min(n, os.cpu_count() or 4)
    logger.info(
        "Cascade DAE grid search: %d configs, %d workers",
        n, max_workers,
    )

    best_f2 = -1.0
    best_hp: dict = {}
    results_by_idx: dict[int, dict] = {}

    worker_args = [
        (hp, X_benign_fit, X_val, y_val, epochs, batch_size, random_state)
        for hp in combos
    ]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_fit_one_candidate, arg): i
            for i, arg in enumerate(worker_args)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            hp, val_metrics = future.result()
            results_by_idx[i] = {
                **hp,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            f2 = val_metrics["attack_f2"]
            logger.info(
                "  [%d/%d] dims=%s noise=%.2f lr=%.4f pct=%.0f → "
                "val_F2=%.4f val_AUC=%.4f",
                i + 1, n,
                hp["encoding_dims"], hp["noise_rate"],
                hp["learning_rate"], hp["threshold_percentile"],
                f2, val_metrics["auc_roc"],
            )
            if f2 > best_f2:
                best_f2 = f2
                best_hp = hp

    all_results = [results_by_idx[i] for i in range(n)]
    logger.info("Best HP by val attack_F2 (=%.4f): %s", best_f2, best_hp)
    return best_hp, all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cascaded DAE re-tuning for val-proba regime (GAP-L1-1)",
    )
    parser.add_argument(
        "--val-parquet",
        default="data/processed/val_phase1.parquet",
    )
    parser.add_argument(
        "--models-dir",
        default="results/models",
    )
    parser.add_argument(
        "--output-params",
        default="results/models/dae_best_params.json",
    )
    parser.add_argument(
        "--results-out",
        default="results/models/dae_grid_results_cascade.json",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sep = "=" * 72
    logger.info(sep)
    logger.info("CASCADED DAE RE-TUNING (post-GAP-L1-1)")
    logger.info(sep)

    val_path = PROJECT_ROOT / args.val_parquet
    models_dir = PROJECT_ROOT / args.models_dir
    if not val_path.exists():
        raise FileNotFoundError(
            f"Missing {val_path}. Run Module 1 with val_ratio>0 first."
        )
    if not all((models_dir / f"{n}_val_proba.npy").exists()
               for n in ("xgboost", "random_forest", "decision_tree")):
        raise FileNotFoundError(
            f"Missing *_val_proba.npy in {models_dir}. "
            "Run module2_detection/module2_train_models.py first."
        )

    X_benign_aug, X_attack_aug, feat_names = load_cascade_data(val_path, models_dir)
    X_benign_fit, X_val, y_val = make_cascade_validation_split(
        X_benign_aug, X_attack_aug,
        random_state=args.random_state,
    )

    t0 = time.perf_counter()
    best_hp, all_results = grid_search(
        X_benign_fit, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )
    elapsed = time.perf_counter() - t0

    # Preserve compatibility with module2_train_models.py
    # (clip_percentile present in legacy json but unused by DAEDetector ctor).
    out = dict(best_hp)
    out.setdefault("clip_percentile", 1.0)

    out_path = PROJECT_ROOT / args.output_params
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)

    results_path = PROJECT_ROOT / args.results_out
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d candidates, %.1fs)",
                results_path, len(all_results), elapsed)

    logger.info(sep)
    logger.info("Re-tune complete. Re-run:")
    logger.info("  python module2_detection/module2_train_models.py")
    logger.info(sep)


if __name__ == "__main__":
    main()
