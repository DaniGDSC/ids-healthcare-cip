#!/usr/bin/env python3
"""Ablation: DAE-raw on benign-only training (no cascade).

Trains a fresh DAEDetector on the 25 raw features of benign-only train
rows — no Track A probability column appended — and evaluates on the
full test split (benign + attack). Mirrors the production cascaded
training in module2_detection.dae_training so the only varying axis is
the augmentation set.

Output: results/dae_raw_benign_only_ablation.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module2_detection.models.DAE import DAEDetector  # noqa: E402
from module2_detection.module2_train_models import load_data  # noqa: E402

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "results/models"
OUT_PATH = PROJECT_ROOT / "results/dae_raw_benign_only_ablation.json"
RANDOM_STATE = 42


def _operating_point(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Confusion-matrix metrics at a fixed threshold (errors > threshold == attack)."""
    y_pred = (scores > threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    pos = tp + fn
    neg = tn + fp
    sens = tp / pos if pos else float("nan")
    spec = tn / neg if neg else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) else float("nan")
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": round(sens, 6),
        "specificity": round(spec, 6),
        "precision": round(prec, 6) if prec == prec else float("nan"),
        "recall": round(sens, 6),
        "fnr": round(1 - sens, 6) if sens == sens else float("nan"),
        "fpr": round(1 - spec, 6) if spec == spec else float("nan"),
        "f1": round(f1, 6) if f1 == f1 else float("nan"),
    }


def run() -> dict:
    t0 = time.perf_counter()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Match the cascaded training's hyperparameter source so the only
    # axis varying is the augmentation set.
    best_hp = json.loads((MODELS_DIR / "dae_best_params.json").read_text())
    logger.info("Best params (shared with cascaded): %s", best_hp)

    X_train, X_test, y_train, y_test, feat_names = load_data()
    benign_mask = y_train == 0
    X_benign = X_train[benign_mask]
    n_feat = X_benign.shape[1]
    logger.info(
        "DAE-raw input: %d features, %d benign train samples, %d test samples (attack rate=%.1f%%)",
        n_feat, benign_mask.sum(), len(y_test), 100 * y_test.mean(),
    )
    assert n_feat == len(feat_names) == 25, f"expected 25 raw features, got {n_feat}"

    # Mirror the cascaded scaling rule:
    #   enc_dim = max(base_enc, n_feat - 4); bot_dim = min(base_bot, n_feat - 2)
    base_dims = best_hp.get("encoding_dims", [20, 12, 20])
    enc_dim = max(base_dims[0], n_feat - 4)
    bot_dim = min(base_dims[1], n_feat - 2)
    adjusted_dims = [enc_dim, bot_dim, enc_dim]
    logger.info("Adjusted architecture: %s (raw, no cascade)", adjusted_dims)

    det = DAEDetector(
        encoding_dims=adjusted_dims,
        noise_rate=best_hp.get("noise_rate", 0.05),
        learning_rate=best_hp.get("learning_rate", 0.001),
        threshold_percentile=best_hp.get("threshold_percentile", 90.0),
        clip_percentile=best_hp.get("clip_percentile", 1.0),
        epochs=100,
        batch_size=256,
        random_state=RANDOM_STATE,
    )
    det.fit(X_benign, validation_split=0.1)

    # ── Evaluate on test (benign + attack) ───────────────────────────
    # det.evaluate() uses the deterministic fixed threshold (the one
    # learned at the configured percentile of training reconstruction
    # error), matching how the paper reports metrics.
    eval_metrics = det.evaluate(X_test, y_test)

    # Additional operating points for cross-comparison.
    test_errors = det.reconstruction_error(X_test)
    fixed_thr = float(det.threshold)
    op_fixed = _operating_point(y_test, test_errors, fixed_thr)
    op_p95 = _operating_point(y_test, test_errors, float(np.percentile(det.train_errors, 95)))
    op_p99 = _operating_point(y_test, test_errors, float(np.percentile(det.train_errors, 99)))

    # AUC via DAE's own logic (errors as scores).
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = float(roc_auc_score(y_test, test_errors))
    ap = float(average_precision_score(y_test, test_errors))

    elapsed = round(time.perf_counter() - t0, 1)
    report = det.get_report()

    payload = {
        "_meta": {
            "description": "DAE-raw (no cascade) trained on benign-only — ablation against the production cascaded DAE.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "random_state": RANDOM_STATE,
        },
        "input": {
            "architecture": "raw",
            "n_features": int(n_feat),
            "feature_names": list(feat_names),
            "n_benign_train": int(benign_mask.sum()),
            "n_test": int(len(y_test)),
            "n_test_attacks": int(y_test.sum()),
            "n_test_benign": int((y_test == 0).sum()),
        },
        "hyperparameters": {
            "base_encoding_dims": base_dims,
            "adjusted_encoding_dims": adjusted_dims,
            "noise_rate": best_hp.get("noise_rate", 0.05),
            "learning_rate": best_hp.get("learning_rate", 0.001),
            "threshold_percentile": best_hp.get("threshold_percentile", 90.0),
            "clip_percentile": best_hp.get("clip_percentile", 1.0),
            "epochs": 100,
            "batch_size": 256,
        },
        "training": report["training"],
        "threshold_fixed": fixed_thr,
        "test_metrics_evaluate": eval_metrics,
        "test_auc_roc": round(auc, 6),
        "test_average_precision": round(ap, 6),
        "operating_points": {
            "fixed_p90_from_train": op_fixed,
            "p95_from_train": op_p95,
            "p99_from_train": op_p99,
        },
        "feature_weights": list(map(float, report.get("feature_weights") or det._feature_weights)),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Saved: %s (%.1fs)", OUT_PATH, elapsed)
    return payload


if __name__ == "__main__":
    run()
