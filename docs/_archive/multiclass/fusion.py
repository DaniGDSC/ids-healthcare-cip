"""Unified fusion entry point (Pre-redesign Task 3).

A single canonical interface that selects between the binary-cascade
fusion and the multi-class cascade fusion via a ``fusion_mode`` flag.
Defaults to ``"multiclass"`` per the redesigned cascade contract; the
legacy ``"binary"`` path is preserved for back-compat and benchmarking.

This module wraps the two existing implementations rather than
duplicating their logic:
  - binary mode    → ``module3_risk_scores.py::classify_fusion``
  - multiclass mode → ``multiclass_fusion.py::classify_fusion_with_diversity``

The wrapper enforces a uniform return shape so downstream consumers
(Module 4 explanations, Module 5 responses) do not need to branch on
the fusion mode internally — they call ``fuse(...)`` and receive the
same ``FusionResult`` dataclass either way.

Defense narrative
-----------------
- The thesis ships with multi-class as the canonical Track-A→DAE
  cascade because LOCO experiments on MedSec-25 demonstrated that the
  cascade contract ("trees handle known, DAE handles unknown") only
  holds when Track A is multi-class. The binary path is preserved for
  ablation comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from src.data_models import FusionClass, MULTICLASS_LABEL_ORDER_EHMS, normal_index

FusionMode = Literal["binary", "multiclass"]
DEFAULT_FUSION_MODE: FusionMode = "multiclass"


@dataclass
class FusionResult:
    """Uniform output shape for both fusion modes.

    Attributes:
        fusion_class: ndarray of ``FusionClass`` string values, shape (n,).
            For binary mode this only contains
            {KNOWN_ATTACK, CONFIRMED_ANOMALY, NOVEL_ANOMALY, BENIGN}; for
            multiclass mode DISAGREEMENT_ANOMALY may also appear.
        predicted_attack_class_idx: ndarray of int, shape (n,). Index
            into ``label_order`` when ``fusion_class == KNOWN_ATTACK``
            (or DISAGREEMENT_ANOMALY demoted from KNOWN_ATTACK), -1
            otherwise. For binary mode, this is always -1 (binary trees
            output a scalar P(attack), not a class).
        diversity: ndarray of float, shape (n,). Per-row ensemble
            disagreement std(P_xgb, P_rf, P_dt). For binary mode,
            populated when the per-model probas were passed; otherwise
            zeros.
        fusion_mode: which path produced this result, for tracing.
        label_order: the class names corresponding to indices in
            ``predicted_attack_class_idx``. None for binary mode.
    """

    fusion_class: np.ndarray
    predicted_attack_class_idx: np.ndarray
    diversity: np.ndarray
    fusion_mode: FusionMode
    label_order: tuple[str, ...] | None = None
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.fusion_class)

    @property
    def predicted_attack_class_name(self) -> np.ndarray:
        """Resolve indices → class names; '-' for rows without a class."""
        if self.label_order is None:
            return np.full(len(self), "-", dtype=object)
        names = np.full(len(self), "-", dtype=object)
        valid = self.predicted_attack_class_idx >= 0
        names[valid] = np.array(
            [self.label_order[i] for i in self.predicted_attack_class_idx[valid]],
            dtype=object,
        )
        return names


# ── Binary path ────────────────────────────────────────────────────────


def _fuse_binary(
    p_xgb: np.ndarray,
    dae_score: np.ndarray,
    *,
    p_rf: np.ndarray | None,
    p_dt: np.ndarray | None,
    a_high: float | None,
    a_low: float | None,
    b: float | None,
) -> FusionResult:
    from module3_risk_scoring.module3_risk_scores import classify_fusion

    fusion = classify_fusion(
        c_track_a=p_xgb,
        c_track_b=dae_score,
        a_high=a_high,
        a_low=a_low,
        b=b,
    )
    pred_idx = np.full(len(p_xgb), -1, dtype=np.int64)

    if p_rf is not None and p_dt is not None:
        diversity = np.std(np.stack([p_xgb, p_rf, p_dt], axis=0), axis=0).astype(np.float32)
    else:
        diversity = np.zeros(len(p_xgb), dtype=np.float32)

    return FusionResult(
        fusion_class=fusion,
        predicted_attack_class_idx=pred_idx,
        diversity=diversity,
        fusion_mode="binary",
        label_order=None,
    )


# ── Multiclass path ────────────────────────────────────────────────────


def _fuse_multiclass(
    softmax_per_model: tuple[np.ndarray, ...],
    dae_score: np.ndarray,
    *,
    label_order: Sequence[str],
    a_high: float | None,
    b: float,
    b_diversity: float,
    normal_idx: int,
    gate_normal_through_dae: bool,
) -> FusionResult:
    from module3_risk_scoring.multiclass_fusion import (
        classify_fusion_with_diversity,
        ensemble_softmax,
    )

    sm_ensemble = ensemble_softmax(*softmax_per_model, method="mean")
    p_attack_per_model = tuple(
        (1.0 - s[:, normal_idx]).astype(np.float32) for s in softmax_per_model
    )

    fusion, pred_idx, diversity = classify_fusion_with_diversity(
        softmax_a=sm_ensemble,
        dae_score=dae_score,
        p_attack_per_model=p_attack_per_model,
        label_order=label_order,
        a_high=a_high,
        b=b,
        b_diversity=b_diversity,
        normal_idx=normal_idx,
        gate_normal_through_dae=gate_normal_through_dae,
    )
    return FusionResult(
        fusion_class=fusion,
        predicted_attack_class_idx=pred_idx,
        diversity=diversity,
        fusion_mode="multiclass",
        label_order=tuple(label_order),
    )


# ── Public API ─────────────────────────────────────────────────────────


def fuse(
    *,
    fusion_mode: FusionMode = DEFAULT_FUSION_MODE,
    dae_score: np.ndarray,
    # binary inputs (used iff fusion_mode == "binary")
    p_xgb: np.ndarray | None = None,
    p_rf: np.ndarray | None = None,
    p_dt: np.ndarray | None = None,
    # multiclass inputs (used iff fusion_mode == "multiclass")
    softmax_per_model: tuple[np.ndarray, ...] | None = None,
    label_order: Sequence[str] | None = None,
    # thresholds (mode-specific defaults applied internally)
    a_high: float | None = None,
    a_low: float | None = None,
    b: float | None = None,
    b_diversity: float = 0.20,
    gate_normal_through_dae: bool = True,
) -> FusionResult:
    """Single canonical fusion call. Routes to binary or multiclass.

    Args:
        fusion_mode: "binary" or "multiclass". Default
            ``DEFAULT_FUSION_MODE = "multiclass"``.
        dae_score: ``(n,)`` DAE flag or score (binary or in [0, 1]).
        p_xgb, p_rf, p_dt: per-model P(attack) arrays for binary mode.
            ``p_xgb`` is required; ``p_rf``/``p_dt`` enable diversity.
        softmax_per_model: tuple of ``(n, K)`` softmax matrices, one per
            Track A model. Required for multiclass mode.
        label_order: class names matching softmax columns. Required for
            multiclass; defaults to ``MULTICLASS_LABEL_ORDER_EHMS`` if
            None.
        a_high, a_low, b: cascade thresholds. ``a_low`` is binary-only.
            ``b`` defaults to 0.70 (multiclass) or None (binary, where
            it falls back to spec default in ``classify_fusion``).
        b_diversity: multiclass-only — DISAGREEMENT_ANOMALY threshold.
        gate_normal_through_dae: multiclass-only — when True (default),
            confident-normal predictions are not trusted on their own.

    Returns:
        ``FusionResult`` — uniform shape regardless of mode.

    Raises:
        ValueError: missing required inputs for the chosen mode.
    """
    if fusion_mode == "binary":
        if p_xgb is None:
            raise ValueError("binary mode requires p_xgb")
        return _fuse_binary(
            p_xgb=p_xgb,
            dae_score=dae_score,
            p_rf=p_rf,
            p_dt=p_dt,
            a_high=a_high,
            a_low=a_low,
            b=b,
        )

    if fusion_mode == "multiclass":
        if softmax_per_model is None:
            raise ValueError("multiclass mode requires softmax_per_model")
        if label_order is None:
            label_order = MULTICLASS_LABEL_ORDER_EHMS
        return _fuse_multiclass(
            softmax_per_model=softmax_per_model,
            dae_score=dae_score,
            label_order=label_order,
            a_high=a_high,
            b=0.70 if b is None else b,
            b_diversity=b_diversity,
            normal_idx=normal_index(tuple(label_order)),
            gate_normal_through_dae=gate_normal_through_dae,
        )

    raise ValueError(f"unknown fusion_mode {fusion_mode!r} (expected 'binary' or 'multiclass')")


__all__ = ["fuse", "FusionResult", "FusionMode", "DEFAULT_FUSION_MODE"]
