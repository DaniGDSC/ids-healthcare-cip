#!/usr/bin/env python3
"""Run DAE (Denoising Autoencoder) fine-tuning pipeline.

Track B novelty detector: trains on benign-only data, sweeps architecture
and threshold hyperparameters, and persists all artifacts.

HP selection protocol (research-integrity hardened, finding #1)
---------------------------------------------------------------
The previous version of this script picked the winning hyperparameter
configuration by ``det.evaluate(X_test, y_test)`` *inside the grid loop*
— meaning the held-out test set was used as the model-selection signal
for 81 candidate configurations, then re-used to "evaluate" the chosen
model. Every reported test metric was inflated by the optimisation
gain on the very labels it was meant to be benchmarked against.

The new protocol:

  1. The benign training set is split into ``X_benign_fit`` (80%) and
     ``X_benign_val`` (20%) with a deterministic seed.
  2. A held-out **attack** validation slice is sliced from
     ``X_train`` (NOT the test set) using ``y_train==1``. This gives
     the grid loop a real positive-class signal so it can pick a
     configuration that actually separates attacks from benigns.
  3. For each candidate config, the DAE is fit on ``X_benign_fit``
     and scored on ``X_val = vstack(X_benign_val, X_attack_val)`` with
     the corresponding ``y_val``. The selection metric is attack-F2 on
     this *validation* slice.
  4. The winner is then re-fit on the **full** benign training set
     (no inner split) and evaluated **once** on the untouched test set.

The test set is touched exactly once in step 4. The leak is closed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from pipeline.module2_detection.models.DAE import DAEDetector

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

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
    """Load train/test parquets and extract the benign-only train subset.

    Returns:
        (X_benign, X_train, X_test, y_train, y_test, feat_names)
    """
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    drop_cols = [c for c in [label_col, "Attack Category"] if c in train_df.columns]

    y_train = train_df[label_col].values
    y_test = test_df[label_col].values

    X_train = train_df.drop(columns=drop_cols).values.astype(np.float32)
    X_test = test_df.drop(columns=drop_cols).values.astype(np.float32)

    feat_names = [c for c in train_df.columns if c not in drop_cols]

    benign_mask = y_train == 0
    X_benign = X_train[benign_mask]

    logger.info(
        "Data loaded: train=%d×%d (benign=%d, attack=%d), test=%d×%d",
        *X_train.shape, benign_mask.sum(), (~benign_mask).sum(),
        *X_test.shape,
    )
    return X_benign, X_train, X_test, y_train, y_test, feat_names


def make_validation_split(
    X_benign: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    benign_val_frac: float = 0.20,
    random_state: int = 42,
) -> tuple:
    """Construct a train-only validation slice for HP selection.

    Splits ``X_benign`` into a fit portion and a benign-validation
    portion, then pulls all attacks out of ``X_train`` to form the
    attack-validation portion. The combined ``(X_val, y_val)`` is the
    signal the grid search optimises against — it is computed entirely
    from the training partition, so the test set is never seen during
    HP selection.

    Returns:
        (X_benign_fit, X_val, y_val) where:
          - ``X_benign_fit`` is what the DAE trains on per candidate
          - ``X_val`` is the mixed validation matrix
          - ``y_val`` is the corresponding {0,1} labels
    """
    rng = np.random.default_rng(random_state)
    n_benign = len(X_benign)
    n_val = int(round(n_benign * benign_val_frac))
    if n_val < 1:
        raise ValueError(
            f"Need at least 1 benign validation sample; "
            f"got benign_val_frac={benign_val_frac} on {n_benign} samples."
        )
    perm = rng.permutation(n_benign)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]

    X_benign_fit = X_benign[fit_idx]
    X_benign_val = X_benign[val_idx]
    X_attack_val = X_train[y_train == 1]
    if len(X_attack_val) < 1:
        raise ValueError(
            "No attack samples in the training partition — DAE HP "
            "selection requires a positive class signal in validation."
        )

    X_val = np.vstack([X_benign_val, X_attack_val]).astype(np.float32)
    y_val = np.concatenate([
        np.zeros(len(X_benign_val), dtype=np.int32),
        np.ones(len(X_attack_val),  dtype=np.int32),
    ])

    logger.info(
        "Validation split (TRAIN-ONLY): benign_fit=%d, "
        "benign_val=%d, attack_val=%d",
        len(X_benign_fit), len(X_benign_val), len(X_attack_val),
    )
    return X_benign_fit, X_val, y_val


# ── HP search ────────────────────────────────────────────────────────────

def grid_search(
    X_benign_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    random_state: int,
) -> tuple:
    """Exhaustive grid search over DAE hyperparameters.

    Selects the configuration that maximises attack-class F2 on the
    **train-only validation slice** (not the held-out test set). Each
    candidate is fitted on ``X_benign_fit`` and scored on ``X_val``,
    which is constructed from a slice of the training partition. The
    test set is NOT touched here — see finding #1 in the Phase 2
    security review.

    The function returns the best HP dict, the per-candidate val
    metrics, and the *unfitted* best HP for the caller to use as the
    seed of a final fit on the full benign training set.
    """
    keys = list(HP_GRID.keys())
    combos = list(itertools.product(*HP_GRID.values()))
    logger.info(
        "DAE grid search: %d configurations (selection on train-only val slice)",
        len(combos),
    )

    best_f2 = -1.0
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
        det.fit(X_benign_fit, validation_split=0.0)

        # Score on the validation slice — NEVER on the test set.
        val_metrics = det.evaluate(X_val, y_val)

        result = {**hp}
        for k, v in val_metrics.items():
            result[f"val_{k}"] = v
        all_results.append(result)

        f2 = val_metrics["attack_f2"]
        logger.info(
            "  [%d/%d] dims=%s noise=%.2f lr=%.4f pct=%.0f → val_F2=%.4f val_AUC=%.4f",
            i, len(combos),
            hp["encoding_dims"], hp["noise_rate"],
            hp["learning_rate"], hp["threshold_percentile"],
            f2, val_metrics["auc_roc"],
        )

        if f2 > best_f2:
            best_f2 = f2
            best_hp = hp

    logger.info(
        "Best HP by val attack_f2 (=%.4f): %s",
        best_f2, best_hp,
    )
    return best_hp, all_results


def fit_final_dae(
    best_hp: dict,
    X_benign: np.ndarray,
    epochs: int,
    batch_size: int,
    random_state: int,
) -> DAEDetector:
    """Re-fit the winning HP configuration on the FULL benign training set.

    The grid loop fitted on a 80% slice so a 20% benign-val slice was
    available for HP selection. The final model is trained on the full
    benign set so it sees every available training sample, exactly as
    the production-final model would. This is then evaluated **once**
    against the held-out test set in the caller.
    """
    det = DAEDetector(
        encoding_dims=best_hp["encoding_dims"],
        noise_rate=best_hp["noise_rate"],
        learning_rate=best_hp["learning_rate"],
        threshold_percentile=best_hp["threshold_percentile"],
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    )
    det.fit(X_benign, validation_split=0.0)
    return det


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

    # ── Train-only validation slice (closes finding #1 leakage) ──
    X_benign_fit, X_val, y_val = make_validation_split(
        X_benign, X_train, y_train,
        random_state=args.random_state,
    )

    # ── Grid search on the validation slice ──
    logger.info("")
    logger.info(
        "── DAE Grid Search (epochs=%d, batch=%d, selection=val_attack_f2) ──",
        args.epochs, args.batch_size,
    )
    best_hp, all_results = grid_search(
        X_benign_fit, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )

    # ── Re-fit the winner on the FULL benign training set ──
    logger.info("")
    logger.info("── Re-fitting best HP on full benign training set ──")
    best_det = fit_final_dae(
        best_hp, X_benign,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )

    # ── Single, final test-set evaluation (touched once) ──
    logger.info("")
    logger.info("── Held-out Test Set Evaluation (single touch) ──")
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
