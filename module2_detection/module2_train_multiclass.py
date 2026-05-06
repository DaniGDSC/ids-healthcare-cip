"""Multi-class Track A trainer (cascade-contract refactor).

Refactors Track A from binary `(attack, normal)` to multi-class
`(normal, known_attack_1, ..., known_attack_K)`. The trees become
specific-pattern matchers instead of boundary discriminators, so:

  - sharp softmax on a known attack class → KNOWN_ATTACK
  - sharp softmax on `normal`             → BENIGN
  - spread softmax (no class confident)   → uncertain → DAE checks

This file lives **side by side** with `module2_train_models.py`; nothing
in the binary pipeline is changed. Artifacts land at parallel filenames:

    results/models/{xgboost,random_forest,decision_tree}_multiclass_final.pkl
    results/models/{...}_multiclass_val_proba.npy   shape (n_val, K)
    results/models/{...}_multiclass_test_proba.npy  shape (n_test, K)
    results/models/{...}_multiclass_final_report.json

The deliberately lightweight wrapper here uses sklearn classes directly
(`GradientBoostingClassifier` as the XGBoost surrogate, matching the
project convention in module2_detection/models/XGBoost.py) instead of
the heavyweight wrappers in `module2_detection/models/`. The wrappers
were built around binary-specific machinery (F2-tuned thresholds, OOF
probas, attack-class metrics); multi-class doesn't need any of it.
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_models import MULTICLASS_LABEL_ORDER_EHMS, normal_index  # noqa: E402

logger = logging.getLogger(__name__)


def _load_split(name: str, label_col: str = "Attack Category") -> tuple:
    """Load a Phase 1 parquet and return (X, y_multi, feat_names).

    Mirrors `module2_train_models.load_data` but reads the multi-class
    label column instead of the binary `Label`. Drops the same non-feature
    columns (row_id, device_class, attack_category, Label).
    """
    path = PROJECT_ROOT / "data/processed" / f"{name}.parquet"
    df = pd.read_parquet(path)
    drop_cols = [c for c in (
        "Label", "Attack Category", "row_id", "device_class", "attack_category",
    ) if c in df.columns]
    y_multi = df[label_col].astype(str).values
    X = df.drop(columns=drop_cols).values.astype(np.float32)
    feat_names = [c for c in df.columns if c not in drop_cols]
    return X, y_multi, feat_names


def _encode_labels(y_multi_str: np.ndarray,
                   label_order: tuple[str, ...]) -> np.ndarray:
    """Map string labels → integer class ids per ``label_order``.

    Raises ``ValueError`` on any label not in the order — the caller
    must update ``MULTICLASS_LABEL_ORDER_EHMS`` in src/data_models.py
    before training on a dataset with new categories. Failing loud beats
    silently lumping a new attack category into "normal".
    """
    label_to_id = {s: i for i, s in enumerate(label_order)}
    unknown = set(y_multi_str) - set(label_to_id)
    if unknown:
        raise ValueError(
            f"Found labels not in MULTICLASS_LABEL_ORDER: {sorted(unknown)}. "
            f"Add them to src/data_models.MULTICLASS_LABEL_ORDER_EHMS."
        )
    return np.array([label_to_id[s] for s in y_multi_str], dtype=np.int64)


def _fit_one(name: str, model, X_train: np.ndarray, y_train: np.ndarray,
             *, sample_weight: np.ndarray | None) -> dict:
    t0 = time.perf_counter()
    if sample_weight is not None and isinstance(model, GradientBoostingClassifier):
        # GBM accepts sample_weight; RF/DT also do but their class_weight
        # already provides balanced weighting per fit.
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("  %s fit in %.1fs", name, elapsed)
    return {"name": name, "elapsed": round(elapsed, 1)}


def _scores(model, X: np.ndarray) -> np.ndarray:
    """Return (n, K) softmax / probability matrix."""
    p = model.predict_proba(X)
    if p.ndim != 2:
        raise RuntimeError(f"Expected (n, K) probas, got shape {p.shape}")
    return p.astype(np.float32)


def _evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray,
                         label_order: tuple[str, ...]) -> dict:
    """Compute multi-class metrics + per-class F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                        zero_division=0)),
        "per_class_f1": {
            cls: float(f1_score(y_true == idx, y_pred == idx, zero_division=0))
            for idx, cls in enumerate(label_order)
        },
        "classification_report": classification_report(
            y_true, y_pred,
            labels=list(range(len(label_order))),
            target_names=list(label_order),
            zero_division=0, output_dict=True,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train multi-class Track A (XGB-surrogate, RF, DT) on "
                    "Attack Category. Cascade-contract refactor: trees as "
                    "specific-pattern matchers instead of binary discriminator.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="results/models",
                        help="Where to drop *_multiclass_*.npy / .pkl artifacts")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    label_order = MULTICLASS_LABEL_ORDER_EHMS
    norm_idx = normal_index(label_order)
    sep = "=" * 72

    logger.info(sep)
    logger.info("MULTI-CLASS TRACK A TRAINING (cascade-contract refactor)")
    logger.info(sep)
    logger.info("Label order: %s (benign idx=%d)", label_order, norm_idx)

    # ── Load data ──
    X_train, y_train_str, feat_names = _load_split("train_phase1")
    X_val, y_val_str, _ = _load_split("val_phase1")
    X_test, y_test_str, _ = _load_split("test_phase1")
    y_train = _encode_labels(y_train_str, label_order)
    y_val = _encode_labels(y_val_str, label_order)
    y_test = _encode_labels(y_test_str, label_order)

    logger.info("train=%d  val=%d  test=%d  features=%d",
                len(X_train), len(X_val), len(X_test), len(feat_names))
    train_dist = pd.Series(y_train_str).value_counts().to_dict()
    logger.info("train Attack Category distribution: %s", train_dist)

    # ── Sample weights to counter EHMS imbalance (12.5% attack) ──
    # `compute_sample_weight('balanced')` upweights the rare classes
    # (Spoofing, Data Alteration) inversely to frequency.
    sample_weight = compute_sample_weight("balanced", y_train)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "xgboost": GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, random_state=args.seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            min_samples_split=5, min_samples_leaf=1,
            max_features=0.5, class_weight="balanced",
            random_state=args.seed, n_jobs=-1,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=None, min_samples_split=2, min_samples_leaf=2,
            class_weight="balanced", random_state=args.seed,
        ),
    }

    summaries = {}
    for name, clf in models.items():
        logger.info("─" * 60)
        logger.info("FIT: %s", name)
        logger.info("─" * 60)
        if isinstance(clf, GradientBoostingClassifier):
            _fit_one(name, clf, X_train, y_train, sample_weight=sample_weight)
        else:
            _fit_one(name, clf, X_train, y_train, sample_weight=None)

        # Softmax on val + test
        proba_val = _scores(clf, X_val)
        proba_test = _scores(clf, X_test)
        if proba_val.shape[1] != len(label_order):
            raise RuntimeError(
                f"{name}: predict_proba produced {proba_val.shape[1]} classes "
                f"but label_order has {len(label_order)}. Class collapse during fit."
            )

        # Persist softmax matrices
        np.save(output_dir / f"{name}_multiclass_val_proba.npy", proba_val)
        np.save(output_dir / f"{name}_multiclass_test_proba.npy", proba_test)

        # Persist model (joblib for downstream load — not signed_pickle since
        # this is a research artefact under the multi-class branch, not a
        # production deployable. Add signing later if this branch is shipped.)
        import joblib
        joblib.dump(clf, output_dir / f"{name}_multiclass_final_pipeline.pkl")

        # Multi-class evaluation on test
        y_pred_test = proba_test.argmax(axis=1)
        metrics = _evaluate_multiclass(y_test, y_pred_test, label_order)
        # Argmax distribution sanity
        argmax_dist = pd.Series([label_order[i] for i in y_pred_test]
                                ).value_counts().to_dict()

        # Cascade-relevant: softmax-confidence summary
        top_p = proba_test.max(axis=1)
        confidence_buckets = {
            "high_conf_>=0.85": int((top_p >= 0.85).sum()),
            "low_conf_<0.40":   int((top_p < 0.40).sum()),
            "uncertain_band":   int(((top_p >= 0.40) & (top_p < 0.85)).sum()),
        }

        report = {
            "model": name,
            "stage": "final_training_multiclass",
            "label_order": list(label_order),
            "best_hyperparameters": clf.get_params(),
            "data": {
                "n_features": len(feat_names),
                "feature_names": feat_names,
                "train_samples": int(len(X_train)),
                "val_samples": int(len(X_val)),
                "test_samples": int(len(X_test)),
                "random_seed": int(args.seed),
                "argmax_test_distribution": argmax_dist,
                "softmax_confidence_buckets_test": confidence_buckets,
            },
            "test_metrics": metrics,
        }
        report_path = output_dir / f"{name}_multiclass_final_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str),
                                encoding="utf-8")

        # ── Sanity assertions (will raise on regression) ──
        assert np.allclose(proba_val.sum(axis=1), 1.0, atol=1e-4), \
            f"{name} val softmax does not sum to 1 (max delta="\
            f"{abs(proba_val.sum(axis=1) - 1.0).max():.2e})"
        assert np.allclose(proba_test.sum(axis=1), 1.0, atol=1e-4), \
            f"{name} test softmax does not sum to 1"
        assert proba_val.shape == (len(X_val), len(label_order)), \
            f"{name} val proba shape mismatch: {proba_val.shape}"

        logger.info("  test acc=%.4f  macro_f1=%.4f  weighted_f1=%.4f",
                    metrics["accuracy"], metrics["macro_f1"],
                    metrics["weighted_f1"])
        logger.info("  per-class F1: %s",
                    {k: round(v, 4) for k, v in metrics["per_class_f1"].items()})
        logger.info("  test argmax distribution: %s", argmax_dist)
        logger.info("  test softmax confidence buckets: %s", confidence_buckets)
        summaries[name] = report["test_metrics"]

    logger.info(sep)
    logger.info("MULTI-CLASS TRACK A SUMMARY")
    logger.info(sep)
    logger.info("%-15s  %-10s  %-10s  %-10s",
                "model", "accuracy", "macro_f1", "weighted_f1")
    logger.info("-" * 50)
    for name, m in summaries.items():
        logger.info("%-15s  %.4f      %.4f      %.4f",
                    name, m["accuracy"], m["macro_f1"], m["weighted_f1"])
    logger.info(sep)

    # Manifest with cross-model summary
    manifest = {
        "label_order": list(label_order),
        "normal_index": norm_idx,
        "models": {
            name: {
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "per_class_f1": m["per_class_f1"],
            }
            for name, m in summaries.items()
        },
        "artifacts": {
            "val_proba": {
                name: f"{name}_multiclass_val_proba.npy"
                for name in models
            },
            "test_proba": {
                name: f"{name}_multiclass_test_proba.npy"
                for name in models
            },
            "models": {
                name: f"{name}_multiclass_final_pipeline.pkl"
                for name in models
            },
        },
        "seed": args.seed,
    }
    (output_dir / "multiclass_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s/multiclass_manifest.json", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
