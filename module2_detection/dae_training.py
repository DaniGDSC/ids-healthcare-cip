"""Track B (DAE) training — cascaded novelty detector.

Trains the DAE on **benign-only** rows whose input is
``[raw_features || Track_A_probas]``. The set of Track A models whose
probabilities are appended is owned by
:data:`common.dae_input.TRACK_A_FOR_DAE` — the single source of truth
shared with the inference engine (:mod:`detection_engine`).

This module is **pure training**:
  - Loads benign rows + Track A out-of-fold probabilities (produced by
    :func:`module2_detection.module2_train_models.train_track_a`).
  - Fits the DAE on the augmented benign matrix.
  - Persists ``dae_detector.json`` + ``dae_model.weights.h5`` +
    ``dae_final_report.json``.
  - Does **not** score the test set. Test-set scoring lives in
    :meth:`detection_engine.DetectionEngine.write_predictions`,
    because that needs the same cascaded inference path the rest of the
    pipeline uses.

Usage:
    from module2_detection.dae_training import train_dae
    metrics = train_dae()
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Make project root importable when invoked as a script.
_PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORT))

from common.dae_input import TRACK_A_FOR_DAE, augmented_feature_names  # noqa: E402
from module2_detection.models.DAE import DAEDetector  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "results/models"
RANDOM_STATE = 42


# ── Augmented training input ──────────────────────────────────────────

def _load_oof_probas(benign_mask: np.ndarray) -> np.ndarray:
    """Load Track A out-of-fold probabilities for benign rows.

    OOF probabilities (not in-sample) are critical: they're the
    classifier's *generalisation* signal on benign data, which is the
    distribution the DAE learns to reconstruct.

    Returns:
        Array of shape ``(n_benign, len(TRACK_A_FOR_DAE))``.
    """
    # Previously wrapped in ThreadPoolExecutor over 3 .npy files; the
    # thread-pool setup cost dominated the actual IO time (~80KB files
    # on local disk). Sequential is simpler and faster in practice.
    def _load_one(name: str) -> np.ndarray:
        path = MODELS_DIR / f"{name}_oof_proba.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path} — train Track A first via "
                f"module2_detection.module2_train_models.train_track_a"
            )
        return np.load(path)[benign_mask]

    cols = [_load_one(name) for name in TRACK_A_FOR_DAE]
    return np.column_stack(cols).astype(np.float32)


def build_training_input(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feat_names: list,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build the benign-only augmented training matrix.

    Returns:
        ``(X_benign_aug, aug_feat_names, benign_mask)`` where
        ``X_benign_aug`` is the cascaded matrix the DAE fits on.
    """
    benign_mask = y_train == 0
    X_benign = X_train[benign_mask]
    oof_probas = _load_oof_probas(benign_mask)

    logger.info(
        "Track A OOF probas (benign): shape=%s, means=%s",
        oof_probas.shape, np.round(oof_probas.mean(axis=0), 4),
    )

    X_benign_aug = np.column_stack([X_benign, oof_probas]).astype(np.float32)
    aug_feat_names = augmented_feature_names(feat_names)
    logger.info(
        "Cascaded DAE input: %d features (%d raw + %d Track A: %s)",
        X_benign_aug.shape[1], len(feat_names), oof_probas.shape[1],
        list(TRACK_A_FOR_DAE),
    )
    return X_benign_aug, aug_feat_names, benign_mask


# ── Training ──────────────────────────────────────────────────────────

def train_dae(seed: int = RANDOM_STATE, persist: bool = True) -> dict:
    """Train the cascaded DAE on benign data; persist artifacts.

    Args:
        seed: random state for the underlying Keras model.
        persist: if True, write ``dae_detector.json`` +
            ``dae_model.weights.h5`` + ``dae_final_report.json``.

    Returns:
        ``dict`` with the fit metrics + cascaded-input metadata.
    """
    t0 = time.perf_counter()
    sep = "-" * 60
    logger.info(sep)
    logger.info("DAE TRAINING (Track B — cascaded, benign-only)")
    logger.info(sep)

    # Lazy import to keep the heavy data-loader path out of the import
    # graph when callers only need the function reference.
    from module2_detection.module2_train_models import load_data

    X_train, _, y_train, _, feat_names = load_data()

    # Best hyperparameters from CV (tuning operated on raw features, but
    # the autoencoder architecture is scaled below to match the augmented
    # input width, so the hyperparameters carry across cleanly).
    params_path = MODELS_DIR / "dae_best_params.json"
    with open(params_path) as f:
        best_hp = json.load(f)
    logger.info("Best params: %s", best_hp)

    X_benign_aug, aug_feat_names, benign_mask = build_training_input(
        X_train, y_train, feat_names,
    )

    # Bottleneck must be < n_features; scale encoder/decoder to the
    # augmented input width.
    n_feat = X_benign_aug.shape[1]
    base_dims = best_hp.get("encoding_dims", [20, 12, 20])
    enc_dim = max(base_dims[0], n_feat - 4)
    bot_dim = min(base_dims[1], n_feat - 2)
    adjusted_dims = [enc_dim, bot_dim, enc_dim]
    logger.info(
        "Adjusted architecture: %s (for %d features)", adjusted_dims, n_feat,
    )

    det = DAEDetector(
        encoding_dims=adjusted_dims,
        noise_rate=best_hp.get("noise_rate", 0.2),
        learning_rate=best_hp.get("learning_rate", 0.0001),
        threshold_percentile=best_hp.get("threshold_percentile", 95.0),
        clip_percentile=best_hp.get("clip_percentile", 1.0),
        epochs=100,
        batch_size=256,
        random_state=seed,
    )
    det.fit(X_benign_aug, validation_split=0.1)

    elapsed = round(time.perf_counter() - t0, 1)

    if persist:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        det.save_artefacts(
            json_path=MODELS_DIR / "dae_detector.json",
            weights_path=MODELS_DIR / "dae_model.weights.h5",
        )

        report = det.get_report()
        report["stage"] = "final_training"
        report["architecture"] = "cascaded"
        report["best_hyperparameters"] = best_hp
        report["adjusted_encoding_dims"] = adjusted_dims
        report["data"] = {
            "n_raw_features": len(feat_names),
            "n_track_a_features": len(TRACK_A_FOR_DAE),
            "track_a_for_dae": list(TRACK_A_FOR_DAE),
            "n_total_features": n_feat,
            "feature_names": aug_feat_names,
            "benign_train_samples": int(benign_mask.sum()),
        }
        report["elapsed_seconds"] = elapsed

        report_path = MODELS_DIR / "dae_final_report.json"
        # Strict JSON serialisation — match the discipline applied across
        # Module 0/1 exporters. `default=str` would silently coerce numpy
        # arrays to repr strings and look like a valid JSON value.
        try:
            payload = json.dumps(report, indent=2)
        except TypeError as exc:
            raise TypeError(
                f"dae_final_report.json contains a non-JSON-serialisable "
                f"value (detail: {exc}). Fix the producer."
            ) from exc
        report_path.write_text(payload, encoding="utf-8")
        logger.info("Saved: %s (%.1fs)", MODELS_DIR, elapsed)

    return {
        "n_total_features": n_feat,
        "n_track_a_features": len(TRACK_A_FOR_DAE),
        "track_a_for_dae": list(TRACK_A_FOR_DAE),
        "adjusted_encoding_dims": adjusted_dims,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train_dae()
