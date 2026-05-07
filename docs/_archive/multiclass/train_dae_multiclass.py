"""Cascaded DAE trainer for the multi-class cascade-contract refactor.

The multi-class Track A in `module2_train_multiclass.py` writes
`{model}_multiclass_val_proba.npy` shape (n_val, K). For backward
compatibility with the existing 28-dim DAE input, we feed the DAE one
scalar per Track A model: ``P(attack) = 1 - softmax[:, normal_idx]``.

Cascade-contract claim being tested by this trainer
---------------------------------------------------
Under the multi-class trees, ``1 - P(normal)`` is no longer a binary
decision-boundary artefact — it's the total mass the trees assign to
*any* attack class. On rows where the trees are uncertain (spread
softmax across multiple classes), ``1 - P(normal)`` may still be high
even though no single attack class is confident. This is the new
"uncertain" regime where the DAE earns its keep.

Artefacts written
-----------------
    results/models/dae_multiclass_detector.json
    results/models/dae_multiclass_model.weights.h5
    results/models/dae_multiclass_final_report.json
    results/models/dae_multiclass_test_predictions.npz
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


from module2_detection._features import drop_non_feature_cols


def _p_attack(softmax: np.ndarray, normal_idx: int) -> np.ndarray:
    """Convert (n, K) softmax to (n,) P(attack) = 1 - P(normal)."""
    return (1.0 - softmax[:, normal_idx]).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train cascaded DAE on multi-class P(attack) columns",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/models")
    parser.add_argument(
        "--params-file",
        default="results/models/dae_best_params.json",
        help="Reuse the post-L1-1-retune DAE hyperparameters by default. "
             "These were tuned on val-proba cascade input that has the "
             "same shape (28-dim) as the multi-class cascade input.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sep = "=" * 72
    logger.info(sep)
    logger.info("CASCADED DAE TRAINING (multi-class Track A → P(attack) input)")
    logger.info(sep)

    from src.data_models import MULTICLASS_LABEL_ORDER_EHMS, normal_index
    from module2_detection.models.DAE import DAEDetector

    label_order = MULTICLASS_LABEL_ORDER_EHMS
    norm_idx = normal_index(label_order)
    logger.info("Label order: %s (benign idx=%d)", label_order, norm_idx)

    output_dir = PROJECT_ROOT / args.output_dir

    # ── Load val + test parquets ──
    val_df = pd.read_parquet(PROJECT_ROOT / "data/processed/val_phase1.parquet")
    test_df = pd.read_parquet(PROJECT_ROOT / "data/processed/test_phase1.parquet")
    y_val = val_df["Label"].values.astype(int)
    y_test = test_df["Label"].values.astype(int)
    X_val = drop_non_feature_cols(val_df).values.astype(np.float32)
    X_test = drop_non_feature_cols(test_df).values.astype(np.float32)
    logger.info("val=%d  test=%d  features=%d", len(X_val), len(X_test), X_val.shape[1])

    # ── Load multi-class softmax → P(attack) per model ──
    p_val_cols = []
    p_test_cols = []
    for name in ("xgboost", "random_forest", "decision_tree"):
        sm_val = np.load(output_dir / f"{name}_multiclass_val_proba.npy")
        sm_test = np.load(output_dir / f"{name}_multiclass_test_proba.npy")
        if sm_val.shape != (len(X_val), len(label_order)):
            raise RuntimeError(
                f"{name} val softmax shape {sm_val.shape} mismatches expected "
                f"({len(X_val)}, {len(label_order)})"
            )
        p_val_cols.append(_p_attack(sm_val, norm_idx))
        p_test_cols.append(_p_attack(sm_test, norm_idx))
        logger.info(
            "  %-15s val P(attack) mean=%.4f  test P(attack) mean=%.4f",
            name, p_val_cols[-1].mean(), p_test_cols[-1].mean(),
        )
    P_val = np.column_stack(p_val_cols)        # (n_val, 3)
    P_test = np.column_stack(p_test_cols)      # (n_test, 3)

    # ── Cascade input ──
    X_val_aug = np.column_stack([X_val, P_val]).astype(np.float32)
    X_test_aug = np.column_stack([X_test, P_test]).astype(np.float32)
    benign_mask = y_val == 0
    X_benign_aug = X_val_aug[benign_mask]
    logger.info("DAE training set: %d benign val rows × %d cascaded features",
                len(X_benign_aug), X_benign_aug.shape[1])

    # ── Hyperparameters (reuse the L1-1 retune winner; see dae_best_params.json) ──
    with open(PROJECT_ROOT / args.params_file) as f:
        best_hp = json.load(f)
    n_feat = X_benign_aug.shape[1]
    enc = max(best_hp.get("encoding_dims", [24, 12, 24])[0], n_feat - 4)
    bot = min(best_hp.get("encoding_dims", [24, 12, 24])[1], n_feat - 2)
    arch = [enc, bot, enc]
    logger.info("DAE hyperparameters: %s (architecture forced to %s for n_feat=%d)",
                best_hp, arch, n_feat)

    det = DAEDetector(
        encoding_dims=arch,
        noise_rate=best_hp.get("noise_rate", 0.10),
        learning_rate=best_hp.get("learning_rate", 1e-4),
        threshold_percentile=best_hp.get("threshold_percentile", 99.0),
        clip_percentile=best_hp.get("clip_percentile", 1.0),
        epochs=100,
        batch_size=256,
        random_state=args.seed,
    )

    t0 = time.perf_counter()
    det.fit(X_benign_aug, validation_split=0.0)
    elapsed = time.perf_counter() - t0
    logger.info("DAE fit in %.1fs", elapsed)

    # ── Test eval ──
    test_metrics = det.evaluate(X_test_aug, y_test)
    logger.info("Test metrics: %s",
                {k: round(float(v), 4) for k, v in test_metrics.items()
                 if isinstance(v, (int, float, np.floating))})

    y_pred = det.predict(X_test_aug)
    errors = det.reconstruction_error(X_test_aug)
    np.savez(
        output_dir / "dae_multiclass_test_predictions.npz",
        y_true=y_test, y_pred=y_pred, reconstruction_error=errors,
    )

    det.save_artefacts(
        json_path=output_dir / "dae_multiclass_detector.json",
        weights_path=output_dir / "dae_multiclass_model.weights.h5",
    )

    report = det.get_report()
    report["stage"] = "final_training_multiclass_cascade"
    report["architecture"] = "cascaded_multiclass"
    report["best_hyperparameters"] = best_hp
    report["adjusted_encoding_dims"] = arch
    report["data"] = {
        "n_raw_features": X_val.shape[1],
        "n_track_a_features": P_val.shape[1],
        "n_total_features": n_feat,
        "benign_train_samples": int(benign_mask.sum()),
        "test_samples": int(len(y_test)),
        "random_seed": int(args.seed),
        "track_a_proba_source": "multiclass_val",
        "label_order": list(label_order),
    }
    report["elapsed_seconds"] = round(elapsed, 1)

    report_path = output_dir / "dae_multiclass_final_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
