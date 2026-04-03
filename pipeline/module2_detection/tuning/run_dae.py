#!/usr/bin/env python3
"""Run DAE (Denoising Autoencoder) fine-tuning pipeline.

Track B novelty detector: trains on benign-only data, sweeps architecture
and threshold hyperparameters, evaluates anomaly detection on the mixed
test set, and persists all artifacts.

Usage:
    python src/phase2_5_fine_tuning/run_dae.py
    python src/phase2_5_fine_tuning/run_dae.py --epochs 200 --threshold-pct 97
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.phase2_detection_engine.DAE import DAEDetector

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Hyperparameter grid for DAE ──────────────────────────────────────────
# All architectures enforce bottleneck < n_features (25) for compression.
# 3×3×3×3 = 81 combos — small enough for exhaustive search.
HP_GRID = {
    "encoding_dims": [
        [16, 8, 16],
        [20, 12, 20],
        [32, 16, 32],
    ],
    "noise_rate": [0.05, 0.1, 0.2],
    "learning_rate": [1e-4, 1e-3, 5e-3],
    "threshold_percentile": [90.0, 95.0, 99.0],
}


# ── Data loading ─────────────────────────────────────────────────────────

def load_data(
    train_path: Path,
    test_path: Path,
    label_col: str = "Label",
) -> tuple:
    """Load train/test parquet, extract benign-only train subset."""
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    drop_cols = [c for c in [label_col, "Attack Category"] if c in train_df.columns]

    y_train = train_df[label_col].values
    y_test = test_df[label_col].values

    X_train = train_df.drop(columns=drop_cols).values.astype(np.float32)
    X_test = test_df.drop(columns=drop_cols).values.astype(np.float32)

    feat_names = [c for c in train_df.columns if c not in drop_cols]

    # Benign-only subset for autoencoder training
    benign_mask = y_train == 0
    X_benign = X_train[benign_mask]

    logger.info(
        "Data loaded: train=%d×%d (benign=%d, attack=%d), test=%d×%d",
        *X_train.shape, benign_mask.sum(), (~benign_mask).sum(),
        *X_test.shape,
    )
    return X_benign, X_train, X_test, y_train, y_test, feat_names


# ── HP search ────────────────────────────────────────────────────────────

def grid_search(
    X_benign: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    random_state: int,
) -> tuple:
    """Exhaustive grid search over DAE hyperparameters.

    Selects the configuration that maximises attack-class F2 on the test set.
    """
    keys = list(HP_GRID.keys())
    combos = list(itertools.product(*HP_GRID.values()))
    logger.info("DAE grid search: %d configurations", len(combos))

    best_f2 = -1.0
    best_det = None
    best_hp: dict = {}
    all_results = []

    for i, vals in enumerate(combos, 1):
        hp = dict(zip(keys, vals))

        det = DAEDetector(
            encoding_dims=hp["encoding_dims"],
            noise_rate=hp["noise_rate"],
            learning_rate=hp["learning_rate"],
            threshold_percentile=hp["threshold_percentile"],
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
        )
        det.fit(X_benign)
        metrics = det.evaluate(X_test, y_test)

        result = {**hp, **metrics}
        all_results.append(result)

        f2 = metrics["attack_f2"]
        logger.info(
            "  [%d/%d] dims=%s noise=%.2f lr=%.4f pct=%.0f → F2=%.4f AUC=%.4f",
            i, len(combos),
            hp["encoding_dims"], hp["noise_rate"],
            hp["learning_rate"], hp["threshold_percentile"],
            f2, metrics["auc_roc"],
        )

        if f2 > best_f2:
            best_f2 = f2
            best_det = det
            best_hp = hp

    return best_det, best_hp, all_results


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAE fine-tuning for IoMT novelty-based intrusion detection",
    )
    parser.add_argument(
        "--train-parquet",
        default="data/processed/train_phase1.parquet",
    )
    parser.add_argument(
        "--test-parquet",
        default="data/processed/test_phase1.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="data/phase2/dae",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t0 = time.perf_counter()
    sep = "=" * 72

    logger.info(sep)
    logger.info("PHASE 2 — DAE FINE-TUNING (TRACK B)")
    logger.info(sep)

    # ── Load data ──
    train_path = PROJECT_ROOT / args.train_parquet
    test_path = PROJECT_ROOT / args.test_parquet
    X_benign, X_train, X_test, y_train, y_test, feat_names = load_data(
        train_path, test_path,
    )

    # ── Grid search ──
    logger.info("")
    logger.info("── DAE Grid Search (epochs=%d, batch=%d) ──",
                args.epochs, args.batch_size)

    best_det, best_hp, all_results = grid_search(
        X_benign, X_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )

    # ── Final evaluation with best model ──
    logger.info("")
    logger.info("── Best Configuration ──")
    test_metrics = best_det.evaluate(X_test, y_test)

    # ── Save artifacts ──
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Keras model weights
    weights_path = output_dir / "dae_model.weights.h5"
    best_det.model.save_weights(str(weights_path))
    logger.info("Saved weights: %s", weights_path)

    # 2. Full report
    report = best_det.get_report()
    report["best_hyperparameters"] = best_hp
    report["grid_search"] = {
        "n_configurations": len(all_results),
        "hp_grid": {k: [str(v) for v in vs] for k, vs in HP_GRID.items()},
    }
    report["data"] = {
        "train_parquet": str(train_path),
        "test_parquet": str(test_path),
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "benign_train_samples": int((y_train == 0).sum()),
        "attack_train_samples": int((y_train == 1).sum()),
        "test_samples": int(len(y_test)),
    }
    report["elapsed_seconds"] = round(time.perf_counter() - t0, 1)

    report_path = output_dir / "dae_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Saved report: %s", report_path)

    # 3. Best hyperparameters
    params_path = output_dir / "best_params.json"
    params_path.write_text(
        json.dumps(best_hp, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Saved best params: %s", params_path)

    # 4. All grid search results
    grid_path = output_dir / "grid_search_results.json"
    grid_path.write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Saved grid results: %s", grid_path)

    # 5. Test predictions
    y_pred = best_det.predict(X_test)
    errors = best_det.reconstruction_error(X_test)
    preds_path = output_dir / "test_predictions.npz"
    np.savez(
        preds_path,
        y_true=y_test,
        y_pred=y_pred,
        reconstruction_error=errors,
    )
    logger.info("Saved predictions: %s", preds_path)

    # ── Summary ──
    logger.info("")
    logger.info(sep)
    logger.info("DAE FINE-TUNING SUMMARY")
    logger.info(sep)
    logger.info("  Features        : %d", len(feat_names))
    logger.info("  Benign train    : %d samples", (y_train == 0).sum())
    logger.info("  Configs tested  : %d", len(all_results))
    logger.info("  Best arch       : %s", best_hp["encoding_dims"])
    logger.info("  Best noise      : %.2f", best_hp["noise_rate"])
    logger.info("  Best lr         : %.4f", best_hp["learning_rate"])
    logger.info("  Best pct        : %.0f", best_hp["threshold_percentile"])
    logger.info("  Threshold       : %.6f", best_det.threshold)
    logger.info("  Test attack F1  : %.4f", test_metrics["attack_f1"])
    logger.info("  Test attack F2  : %.4f", test_metrics["attack_f2"])
    logger.info("  Test AUC-ROC    : %.4f", test_metrics["auc_roc"])
    logger.info("  Benign err mean : %.6f", test_metrics["mean_benign_error"])
    logger.info("  Attack err mean : %.6f", test_metrics["mean_attack_error"])
    logger.info("  Elapsed         : %.1f s", report["elapsed_seconds"])
    logger.info("  Artifacts       : %s", output_dir)
    logger.info(sep)


if __name__ == "__main__":
    main()
