"""Inference-time detection engine: Track A + Track B fusion.

The cascaded architecture (see :mod:`module2_detection.dae_training`):
  - Track A: supervised classifiers (XGBoost, RandomForest, DecisionTree).
  - Track B: DAE novelty detector trained on benign rows whose input is
    ``[raw_features || Track_A_probas]`` — the augmentation set is
    defined by :data:`common.dae_input.TRACK_A_FOR_DAE`.

This module is the *single* place that:
  1. Builds the augmented inference input.
  2. Asserts the DAE's expected input width matches that augmentation,
     so a stale artifact fails loudly instead of silently broadcasting.
  3. Returns ``c_detect = max(c_track_a, c_track_b)`` clipped to [0, 1].

Callers (Modules 3, 4, 6) should not reconstruct any of this themselves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from common.dae_input import TRACK_A_FOR_DAE

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DetectionResult:
    """One batch of detection outputs.

    Attributes:
        track_a_probas: dict ``{model_name: (n,) array}`` for every Track A
            model in the registry — used by analyst-report consensus, not
            just the DAE-input subset.
        track_a_preds: dict ``{model_name: (n,) array}`` of binary
            predictions after each model's optimal threshold.
        c_track_a: ``(n,)`` primary Track A confidence — the XGBoost
            probability, preserved from Module 3's original semantics.
        c_track_b: ``(n,)`` DAE anomaly score on [0, 1] (normalised
            reconstruction error).
        c_detect: ``(n,)`` fused detection confidence
            ``max(c_track_a, c_track_b)`` clipped to [0, 1].
        y_pred_dae: ``(n,)`` binary DAE prediction at its calibrated
            threshold.
        x_augmented: ``(n, n_raw + |TRACK_A_FOR_DAE|)`` cascaded input
            actually fed to the DAE. Exposed so explainers can decompose
            the per-feature reconstruction error against the same vector.
    """

    track_a_probas: dict
    track_a_preds: dict
    c_track_a: np.ndarray
    c_track_b: np.ndarray
    c_detect: np.ndarray
    y_pred_dae: np.ndarray
    x_augmented: np.ndarray


class DetectionEngine:
    """Process-scoped detection engine over Track A + DAE.

    Models are loaded lazily on first use via
    :mod:`common.model_registry` and cached for the lifetime of the
    process, so constructing a ``DetectionEngine()`` is free.

    The engine validates DAE input width on first call and raises
    ``ValueError`` if the trained DAE does not expect
    ``n_raw_features + len(TRACK_A_FOR_DAE)`` inputs. This is the
    intended failure mode when the DAE artifact is stale relative to
    :data:`common.dae_input.TRACK_A_FOR_DAE`.
    """

    PRIMARY_TRACK_A = "xgboost"

    def __init__(self):
        self._classifiers = None
        self._thresholds = None
        self._dae = None
        self._dae_expected_dim = None

    # ── Lazy registry access ───────────────────────────────────────────

    def _load(self):
        if self._classifiers is None:
            from common.model_registry import (
                get_dae,
                get_track_a_classifiers,
                get_track_a_thresholds,
            )
            self._classifiers = get_track_a_classifiers()
            self._thresholds = get_track_a_thresholds()
            self._dae = get_dae()
            self._dae_expected_dim = int(self._dae._clip_lo.shape[0])

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _sanitise(X: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf with zeros (OOD-05 guard — see Module 3).

        GradientBoostingClassifier raises on NaN; this keeps a single
        malformed row from crashing a batch.
        """
        finite_mask = np.isfinite(X)
        if not finite_mask.all():
            logger.warning(
                "detection_engine: NaN/Inf in features — replacing with zeros"
            )
            X = np.where(finite_mask, X, 0.0)
        return X.astype(np.float32, copy=False)

    def _track_a_probas_all(self, X: np.ndarray) -> dict:
        """Run every Track A classifier and return {name: proba}.

        Runs in a joblib threading pool — sklearn / xgboost release the
        GIL during predict_proba, so threading achieves real parallelism
        without pickling overhead.
        """
        def _one(name, clf):
            return name, clf.predict_proba(X)[:, 1]

        pairs = joblib.Parallel(n_jobs=len(self._classifiers), backend="threading")(
            joblib.delayed(_one)(name, clf)
            for name, clf in self._classifiers.items()
        )
        return dict(pairs)

    # ── Public API ─────────────────────────────────────────────────────

    def build_augmented(self, X_raw: np.ndarray) -> np.ndarray:
        """Build ``[X_raw || probas_for(TRACK_A_FOR_DAE)]``.

        This is the *only* function in the codebase that constructs the
        cascaded DAE input. Asserts width consistency against the loaded
        DAE so stale artifacts fail loudly here, not 100 lines downstream
        inside a numpy broadcast.
        """
        self._load()
        X_clean = self._sanitise(X_raw)
        n, n_raw = X_clean.shape

        probas = self._track_a_probas_all(X_clean)
        proba_cols = np.column_stack(
            [probas[name] for name in TRACK_A_FOR_DAE]
        ).astype(np.float32)

        expected = self._dae_expected_dim
        actual = n_raw + proba_cols.shape[1]
        if expected != actual:
            raise ValueError(
                f"DAE input width mismatch: trained DAE expects {expected} "
                f"dims, augmented input has {actual} (raw={n_raw} + "
                f"|TRACK_A_FOR_DAE|={proba_cols.shape[1]}). "
                f"Retrain the DAE via module2_detection.dae_training after "
                f"changing common.dae_input.TRACK_A_FOR_DAE."
            )

        X_aug = np.empty((n, expected), dtype=np.float32)
        X_aug[:, :n_raw] = X_clean
        X_aug[:, n_raw:] = proba_cols
        return X_aug

    def predict(self, X_raw: np.ndarray) -> DetectionResult:
        """Run Track A + Track B over a batch and return all per-sample arrays."""
        self._load()
        X_clean = self._sanitise(X_raw)

        # Track A — every classifier (not just DAE-input subset)
        probas = self._track_a_probas_all(X_clean)
        preds = {
            name: (probas[name] >= self._thresholds[name]).astype(np.int8)
            for name in probas
        }

        # c_track_a — primary supervised detector
        c_track_a = probas[self.PRIMARY_TRACK_A].astype(np.float32)

        # Augmented input for the DAE (probas already computed, reuse)
        n, n_raw = X_clean.shape
        proba_cols = np.column_stack(
            [probas[name] for name in TRACK_A_FOR_DAE]
        ).astype(np.float32)
        expected = self._dae_expected_dim
        actual = n_raw + proba_cols.shape[1]
        if expected != actual:
            raise ValueError(
                f"DAE input width mismatch: expects {expected}, "
                f"got {actual} (raw={n_raw} + "
                f"|TRACK_A_FOR_DAE|={proba_cols.shape[1]}). "
                f"Retrain DAE via module2_detection.dae_training."
            )
        x_aug = np.empty((n, expected), dtype=np.float32)
        x_aug[:, :n_raw] = X_clean
        x_aug[:, n_raw:] = proba_cols

        # Track B — DAE novelty
        c_track_b = self._dae.predict_proba(x_aug).astype(np.float32)
        y_pred_dae = self._dae.predict(x_aug).astype(np.int8)

        # Fusion: DAE elevates, never suppresses Track A
        c_detect = np.clip(np.maximum(c_track_a, c_track_b), 0.0, 1.0)

        return DetectionResult(
            track_a_probas=probas,
            track_a_preds=preds,
            c_track_a=c_track_a,
            c_track_b=c_track_b,
            c_detect=c_detect,
            y_pred_dae=y_pred_dae,
            x_augmented=x_aug,
        )

    def write_test_predictions(
        self,
        out_path: Path | None = None,
    ) -> Path:
        """Run the engine on ``test_phase1.parquet`` and write the DAE
        test-prediction npz that downstream modules expect.

        This replaces the side effect that previously lived in
        ``train_track_b_dae`` — keeping training pure (only artifact
        write) and centralising test-set scoring in the engine.
        """
        import pandas as pd

        if out_path is None:
            out_path = PROJECT_ROOT / "results/models/dae_test_predictions.npz"

        from module2_detection.module2_train_models import load_data
        _X_train, X_test, _y_train, y_test, _feat_names = load_data()

        result = self.predict(X_test)

        # Reconstruction error on the augmented input — the underlying
        # scalar that the DAE turns into a probability.
        recon_err = self._dae.reconstruction_error(result.x_augmented)

        np.savez(
            out_path,
            y_true=y_test,
            y_pred=result.y_pred_dae,
            reconstruction_error=recon_err,
        )
        logger.info(
            "detection_engine: wrote %s (%d samples, c_detect range "
            "[%.4f, %.4f])",
            out_path, len(y_test), result.c_detect.min(), result.c_detect.max(),
        )
        return out_path
