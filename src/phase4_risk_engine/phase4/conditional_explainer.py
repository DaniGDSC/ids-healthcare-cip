"""Conditional explainability — compute explanations only when needed.

Resolves the Performance-Efficiency-Explainability trilemma by
generating explanations on-demand rather than for every sample:

  ROUTINE/ADVISORY  → no explanation (saves 109ms/sample)
  URGENT            → lightweight explanation (attention weights only, ~1ms)
  EMERGENT/CRITICAL → full explanation (attention + SHAP top-K, ~110ms)

This preserves the attention mechanism's clinical value (temporal
importance weights) while avoiding the SHAP computation cost on
samples that don't need it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from .base import BaseDetector
from .clinical_impact import ClinicalSeverity

logger = logging.getLogger(__name__)


class ExplainabilityLevel:
    """Explanation depth for a given clinical severity."""

    NONE = "none"
    LIGHTWEIGHT = "attention_only"
    FULL = "attention_and_shap"


# Severity → explanation level mapping
_EXPLANATION_POLICY: Dict[int, str] = {
    ClinicalSeverity.ROUTINE: ExplainabilityLevel.NONE,
    ClinicalSeverity.ADVISORY: ExplainabilityLevel.NONE,
    ClinicalSeverity.URGENT: ExplainabilityLevel.LIGHTWEIGHT,
    ClinicalSeverity.EMERGENT: ExplainabilityLevel.FULL,
    ClinicalSeverity.CRITICAL: ExplainabilityLevel.FULL,
}


class SampleExplanation:
    """Explanation for a single sample."""

    __slots__ = ("level", "attention_weights", "top_features", "timestep_importance")

    def __init__(
        self,
        level: str,
        attention_weights: Optional[np.ndarray] = None,
        top_features: Optional[List[Dict[str, Any]]] = None,
        timestep_importance: Optional[List[float]] = None,
    ) -> None:
        self.level = level
        self.attention_weights = attention_weights
        self.top_features = top_features
        self.timestep_importance = timestep_importance

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"level": self.level}
        if self.timestep_importance is not None:
            result["timestep_importance"] = self.timestep_importance
        if self.top_features is not None:
            result["top_features"] = self.top_features
        return result


class ConditionalExplainer(BaseDetector):
    """Generate explanations conditionally based on clinical severity.

    Extracts attention weights from the model backbone for URGENT+
    severity, and optionally computes SHAP-like feature importance
    via gradient-based attribution for EMERGENT/CRITICAL.

    Args:
        model: Full classification model (backbone + head).
        feature_names: List of feature column names.
        top_k: Number of top features to report in explanations.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        feature_names: List[str],
        top_k: int = 5,
    ) -> None:
        self._model = model
        self._feature_names = feature_names
        self._top_k = top_k
        self._attention_model: Optional[tf.keras.Model] = None
        self._stats = {
            "none": 0,
            "lightweight": 0,
            "full": 0,
        }

    def explain(
        self,
        risk_results: List[Dict[str, Any]],
        X_windows: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Add conditional explanations to risk results.

        Args:
            risk_results: List of per-sample risk dicts (with clinical_severity).
            X_windows: Windowed input data, shape (N, timesteps, features).

        Returns:
            Same list with 'explanation' field added to each result.
        """
        for i, result in enumerate(risk_results):
            severity = result.get("clinical_severity", 1)
            level = _EXPLANATION_POLICY.get(severity, ExplainabilityLevel.NONE)

            if level == ExplainabilityLevel.NONE:
                result["explanation"] = {"level": "none"}
                self._stats["none"] += 1

            elif level == ExplainabilityLevel.LIGHTWEIGHT:
                explanation = self._explain_lightweight(X_windows, i)
                result["explanation"] = explanation.to_dict()
                self._stats["lightweight"] += 1

            elif level == ExplainabilityLevel.FULL:
                explanation = self._explain_full(X_windows, i)
                result["explanation"] = explanation.to_dict()
                self._stats["full"] += 1

        logger.info(
            "  Explanations: %d none, %d lightweight, %d full",
            self._stats["none"],
            self._stats["lightweight"],
            self._stats["full"],
        )
        return risk_results

    def _explain_lightweight(self, X_windows: np.ndarray, idx: int) -> SampleExplanation:
        """Extract attention weights only (~1ms)."""
        attn_model = self._get_attention_model()
        sample = X_windows[idx: idx + 1]
        weights = attn_model.predict(sample, verbose=0)

        if weights.ndim > 1:
            weights = weights.squeeze()

        return SampleExplanation(
            level=ExplainabilityLevel.LIGHTWEIGHT,
            timestep_importance=[round(float(w), 4) for w in weights],
        )

    def _explain_full(self, X_windows: np.ndarray, idx: int) -> SampleExplanation:
        """Attention weights + gradient-based feature importance (~5-10ms)."""
        # Attention weights
        attn_model = self._get_attention_model()
        sample = X_windows[idx: idx + 1]
        weights = attn_model.predict(sample, verbose=0)
        if weights.ndim > 1:
            weights = weights.squeeze()

        # Gradient-based feature importance (faster than SHAP)
        sample_tensor = tf.constant(sample, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(sample_tensor)
            pred = self._model(sample_tensor, training=False)
        grads = tape.gradient(pred, sample_tensor)

        if grads is not None:
            # Mean absolute gradient per feature across timesteps
            feat_importance = np.mean(np.abs(grads.numpy().squeeze()), axis=0)
            top_indices = np.argsort(feat_importance)[::-1][: self._top_k]
            top_features = [
                {
                    "feature": self._feature_names[j] if j < len(self._feature_names) else f"feature_{j}",
                    "importance": round(float(feat_importance[j]), 6),
                }
                for j in top_indices
            ]
        else:
            top_features = []

        return SampleExplanation(
            level=ExplainabilityLevel.FULL,
            timestep_importance=[round(float(w), 4) for w in weights],
            top_features=top_features,
        )

    def _get_attention_model(self) -> tf.keras.Model:
        """Build sub-model that outputs attention weights."""
        if self._attention_model is not None:
            return self._attention_model

        # Find the attention layer and extract its weights output
        for layer in self._model.layers:
            if layer.name == "attention":
                # BahdanauAttention outputs (context, weights)
                # Build a model that returns just the weights
                try:
                    attn_weights_output = layer.output[1]
                    self._attention_model = tf.keras.Model(
                        self._model.input, attn_weights_output, name="attention_weights_extractor",
                    )
                    return self._attention_model
                except (IndexError, TypeError):
                    pass

        # Fallback: return uniform weights
        logger.warning("Could not extract attention weights; using uniform fallback")
        n_timesteps = self._model.input_shape[1]

        class _UniformModel:
            def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
                return np.ones((len(x), n_timesteps)) / n_timesteps

        self._attention_model = _UniformModel()  # type: ignore[assignment]
        return self._attention_model

    def get_summary(self) -> Dict[str, Any]:
        """Return explainability summary."""
        total = sum(self._stats.values())
        return {
            "total_samples": total,
            "explanations_generated": self._stats["lightweight"] + self._stats["full"],
            "explanations_skipped": self._stats["none"],
            "breakdown": dict(self._stats),
            "savings_pct": round(self._stats["none"] / max(total, 1) * 100, 1),
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "top_k": self._top_k,
            "policy": {
                sev.name: level for sev_val, level in _EXPLANATION_POLICY.items()
                for sev in ClinicalSeverity if sev.value == sev_val
            },
        }
