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

    # Cascade gate: when XGBoost proba ≥ TAU_SKIP_DAE, skip the DAE
    # forward pass for that row and set c_track_b = 0. Safe because
    # c_detect = max(c_track_a, c_track_b), so a high c_track_a already
    # dominates any DAE elevation. Kept distinct from
    # thresholds["xgboost"] (the calibrated decision threshold, often
    # well below 0.9) — this is a compute-side optimisation, not a
    # decision boundary. Lower at your peril: gated rows where DAE
    # might have scored higher will lose that elevation.
    TAU_SKIP_DAE: float = 0.90

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

    def predict(
        self,
        X_raw: np.ndarray,
        *,
        _force_full_dae: bool = False,
    ) -> DetectionResult:
        """Run Track A + Track B over a batch and return all per-sample arrays.

        The DAE is gated on ``c_track_a >= TAU_SKIP_DAE``: rows already
        confidently flagged by XGBoost skip the DAE forward pass and
        receive ``c_track_b = 0`` / ``y_pred_dae = 0``. Fusion is
        unchanged (``c_detect = max(c_track_a, c_track_b)`` clipped),
        so the gated rows' c_detect equals c_track_a — identical to
        the un-gated path for those rows.

        Args:
            X_raw: ``(n, n_raw_features)`` raw feature batch.
            _force_full_dae: engine-internal flag. When True, bypass the
                gate and score every row with the DAE. Used by
                :meth:`write_predictions` so that downstream
                evaluation artefacts (AUC, PSI, threshold sweeps) get
                a complete per-row ``reconstruction_error``. Not part
                of the public API.
        """
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

        # Augmented input for the DAE (probas already computed, reuse).
        # Built for ALL rows even when most will be gated — the
        # column_stack is cheap and Module 4 reads x_augmented for the
        # full batch.
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

        # Track B — DAE novelty, gated unless forced.
        c_track_b = np.zeros(n, dtype=np.float32)
        y_pred_dae = np.zeros(n, dtype=np.int8)
        if _force_full_dae:
            to_score_mask = np.ones(n, dtype=bool)
        else:
            to_score_mask = c_track_a < self.TAU_SKIP_DAE
        n_scored = int(to_score_mask.sum())
        if n_scored > 0:
            x_sub = x_aug[to_score_mask]
            c_track_b[to_score_mask] = self._dae.predict_proba(x_sub).astype(np.float32)
            y_pred_dae[to_score_mask] = self._dae.predict(x_sub).astype(np.int8)
        logger.info(
            "detection_engine.predict: %d/%d rows scored by DAE "
            "(%d gated by XGBoost>=%.2f, %.1f%% compute saved)%s",
            n_scored, n, n - n_scored, self.TAU_SKIP_DAE,
            100.0 * (n - n_scored) / max(n, 1),
            " [forced full-DAE]" if _force_full_dae else "",
        )

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

    def write_predictions(
        self,
        out_path: Path | None = None,
        split: str = "test",
    ) -> Path:
        """Run the engine on a labelled split parquet and write the DAE
        prediction npz that downstream modules expect.

        ``split`` selects which frozen parquet to score:
          - ``"test"`` → ``test_phase1.parquet`` → ``dae_test_predictions.npz``
          - ``"demo"`` → ``demo_phase1.parquet`` → ``dae_demo_predictions.npz``

        This replaces the side effect that previously lived in
        ``train_track_b_dae`` — keeping training pure (only artifact
        write) and centralising scoring in the engine.
        """
        from module2_detection.module2_train_models import load_split_data

        if split not in ("test", "demo"):
            raise ValueError(f"unknown split: {split!r} (expected 'test' or 'demo')")

        if out_path is None:
            out_path = PROJECT_ROOT / f"results/models/dae_{split}_predictions.npz"

        X, y, _feat_names = load_split_data(split)

        # Force full DAE coverage for the evaluation export: AUC,
        # PSI (drift_detection), and threshold sweeps
        # (dynamic_threshold_sim) need a per-sample reconstruction
        # error over the entire split — gated zeros would
        # corrupt those metrics.
        result = self.predict(X, _force_full_dae=True)

        # Reconstruction error on the augmented input — the underlying
        # scalar that the DAE turns into a probability.
        recon_err = self._dae.reconstruction_error(result.x_augmented)

        np.savez(
            out_path,
            y_true=y,
            y_pred=result.y_pred_dae,
            reconstruction_error=recon_err,
        )
        logger.info(
            "detection_engine: wrote %s (%d samples, c_detect range "
            "[%.4f, %.4f])",
            out_path, len(y), result.c_detect.min(), result.c_detect.max(),
        )
        return out_path

    def write_test_predictions(self, *args, **kwargs) -> Path:
        """Deprecated alias for :meth:`write_predictions`.

        Renamed when the method gained ``split`` support — the old name
        suggested test-only semantics that no longer hold. Existing
        callers continue to work; new code should call ``write_predictions``.
        """
        import warnings
        warnings.warn(
            "DetectionEngine.write_test_predictions is deprecated; "
            "use write_predictions(...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.write_predictions(*args, **kwargs)
