"""Online-capable per-alert explanation pipeline.

``AlertExplainer`` takes ``feat_names`` at construction time and never
mutates it (Y10 fix — previously the explainer carried mutable state
that could race when multiple callers invoked ``explain()`` with
different feat_names in a service context).

Single source of truth for top-k extraction (``module4_explanations.compute``)
and clinician NLG (``module4_explanations.nlg``) — pre-cleanup these
were duplicated across two files.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np

from .compute import (
    _normalise_shap_output,
    _top_features_dae,
    _top_features_shap,
)
from .config import SKIP_SHAP_MODELS, TRACK_A_MODELS, format_clinician_template
from .nlg import build_shap_context, clinician_nlg

logger = logging.getLogger(__name__)


class AlertExplainer:
    """Per-alert explanation engine.

    Load once at service startup, call ``explain()`` per sample. Both
    the constructor and ``explain()`` are deterministic given the same
    inputs and the same loaded model registry.

    Args:
        feat_names: ordered feature names used by all classifiers.
            Stored at construction; cannot change per call (Y10 fix).
            ``explain()`` will reject calls that pass a different list.

    Models in ``SKIP_SHAP_MODELS`` (defaults to non-XGBoost models) are
    still consulted for the vote / severity computation but their
    TreeExplainer is skipped because:
      1. RF dominates startup_ms (~1.5 s of the 2.0 s baseline)
      2. RF dominates treeshap_ms (~95 of the 108 ms baseline)
      3. Nothing downstream reads ``analyst.track_a.random_forest`` —
         the clinician summary uses XGBoost, the analyst waterfall is
         XGBoost-only, and the offline batch explainer is the source
         of all RF charts.
    """

    def __init__(self, feat_names: Sequence[str]) -> None:
        t0 = time.perf_counter()
        if not feat_names:
            raise ValueError(
                "AlertExplainer requires feat_names at construction time."
            )
        self.feat_names: tuple[str, ...] = tuple(feat_names)

        # Track A: extract classifiers + create TreeExplainers selectively.
        # Phase 2 final-training writes the bare classifier (NOT a full
        # Pipeline with the SMOTE wrapper) and signs it with the
        # Module 5 ECDSA key. The registry handles signed-pickle loading.
        from common.model_registry import (
            get_dae,
            get_track_a_classifiers,
            get_track_a_thresholds,
        )
        import shap

        registry_clfs = get_track_a_classifiers()
        registry_thresholds = get_track_a_thresholds()

        self.classifiers: dict = {}
        self.explainers: dict = {}
        self.thresholds: dict = {}
        for name in TRACK_A_MODELS:
            self.classifiers[name] = registry_clfs[name]
            if name not in SKIP_SHAP_MODELS:
                self.explainers[name] = shap.TreeExplainer(self.classifiers[name])
            self.thresholds[name] = registry_thresholds[name]

        self.dae = get_dae()

        self._startup_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("AlertExplainer loaded in %.1fms", self._startup_ms)

    def _severity(self, n_flagged: int) -> str:
        if n_flagged >= 4:
            return "CRITICAL"
        if n_flagged == 3:
            return "HIGH"
        if n_flagged == 2:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _sanitise(x: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf with zeros (OOD-05 fix)."""
        if np.isnan(x).any() or np.isinf(x).any():
            logger.warning("NaN/Inf in sample — replacing with zeros")
            x = np.where(np.isfinite(x), x, 0.0)
        return x

    def _validate_feat_names(self, feat_names: Sequence[str] | None) -> None:
        """Y10 guard: refuse calls that try to override feat_names."""
        if feat_names is None:
            return
        if tuple(feat_names) != self.feat_names:
            raise ValueError(
                "AlertExplainer is constructed with feat_names; do not override "
                "per-call. Construct a new instance if the feature schema changes."
            )

    def explain(
        self,
        x_sample: np.ndarray,
        feat_names: Sequence[str] | None = None,
    ) -> dict:
        """Generate per-alert explanation with component-level timing.

        Args:
            x_sample: Single sample, shape (n_features,).
            feat_names: kept as backward-compat parameter. If provided,
                must match the constructor's feat_names exactly —
                otherwise raises ValueError. Pass ``None`` to use the
                instance feat_names directly.
        """
        self._validate_feat_names(feat_names)
        feats = list(self.feat_names)

        x_2d = self._sanitise(x_sample.reshape(1, -1))
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        # ── Step 1: Model predictions ──
        t0 = time.perf_counter()
        votes: dict[str, dict] = {}
        for name, clf in self.classifiers.items():
            proba = float(clf.predict_proba(x_2d)[0, 1])
            pred = int(proba >= self.thresholds[name])
            votes[name] = {"prediction": pred, "confidence": round(proba, 4)}

        from detection_engine import DetectionEngine
        x_augmented = DetectionEngine().build_augmented(x_2d)
        dae_error_arr, dae_per_feature = self.dae.reconstruction_error_decomposed(
            x_augmented,
        )
        dae_error = float(dae_error_arr[0])
        dae_pred = int(dae_error > self.dae.threshold)
        votes["dae"] = {
            "prediction": dae_pred,
            "reconstruction_error": round(dae_error, 8),
        }
        timings["predict_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # ── Step 2: Severity ──
        n_flagged = sum(1 for v in votes.values() if v["prediction"] == 1)
        severity = self._severity(n_flagged)

        if severity == "LOW" and n_flagged <= 1:
            timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 3)
            return {
                "severity": severity,
                "n_models_flagged": n_flagged,
                "votes": votes,
                "explanation_level": "minimal",
                "clinician_summary": format_clinician_template(
                    "LOW", sample_index=None,
                ),
                "timings_ms": timings,
            }

        # ── Step 3: TreeSHAP ──
        t0 = time.perf_counter()
        shap_explanations: dict[str, dict] = {}
        for name, explainer in self.explainers.items():
            sv = explainer.shap_values(x_2d)
            sv = _normalise_shap_output(sv)
            sv_row = sv[0]
            shap_explanations[name] = {
                "top_features": _top_features_shap(sv_row, feats),
                "shap_values": sv_row.tolist(),
            }
        timings["treeshap_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # ── Step 4: DAE decomposition (reuses Step 1's forward pass) ──
        t0 = time.perf_counter()
        w_err = dae_per_feature[0]
        dae_explanation = {"top_features": _top_features_dae(w_err, feats)}
        timings["dae_decompose_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # ── Step 5: NLG ──
        t0 = time.perf_counter()
        primary_top = shap_explanations["xgboost"]["top_features"]
        clinician_summary = clinician_nlg(severity, primary_top)
        shap_context = build_shap_context(primary_top)
        timings["nlg_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # ── Step 6: Risk decomposition ──
        t0 = time.perf_counter()
        flagging_models = [name for name, v in votes.items() if v["prediction"] == 1]
        confidences = [
            votes[m].get("confidence", 0)
            for m in flagging_models if "confidence" in votes[m]
        ]
        risk_decomposition = {
            "flagging_models": flagging_models,
            "confidence_spread": {
                "min":  round(min(confidences), 4) if confidences else 0,
                "max":  round(max(confidences), 4) if confidences else 0,
                "mean": round(float(np.mean(confidences)), 4) if confidences else 0,
            },
            "dae_contributes": dae_pred == 1,
        }
        timings["risk_decompose_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 3)

        return {
            "severity": severity,
            "n_models_flagged": n_flagged,
            "votes": votes,
            "explanation_level": "full",
            "analyst": {
                "track_a": shap_explanations,
                "track_b": dae_explanation,
            },
            "clinician_summary": clinician_summary,
            "shap_context": shap_context,
            "risk_decomposition": risk_decomposition,
            "timings_ms": timings,
        }

    # ── Backward-compat shim helpers (delegate to compute module) ──

    def _top_shap(self, sv_row: np.ndarray, k: int = 3) -> list:
        return _top_features_shap(sv_row, list(self.feat_names), k=k)

    def _top_dae(self, werr_row: np.ndarray, k: int = 3) -> list:
        return _top_features_dae(werr_row, list(self.feat_names), k=k)

    def _clinician_nlg(self, severity: str, top_features: list) -> str:
        return clinician_nlg(severity, top_features)

    @staticmethod
    def build_shap_context(top_features: list) -> dict:
        return build_shap_context(top_features)


__all__ = ["AlertExplainer"]
