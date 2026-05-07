"""Layer 3 v4.1 — enriched 9-stage triage classifier (XGB + DAE only).

Maps the per-alert Layer 2 outputs (calibrated XGBoost probability and
DAE percentile-rank score) to the 9-class ``AlertType`` typology + a
4-level ``Confidence`` indicator.

v5 architectural change
-----------------------
Track A is XGB-only at runtime. RandomForest and DecisionTree are
retained as offline reference baselines for the thesis comparison
(``analysis/`` scripts) but no longer participate in the runtime
classification path. The previous ``diversity_score`` signal —
``std(p_xgb, p_rf, p_dt)`` — is therefore retired from the predicates;
``DISAGREEMENT_ANOMALY`` is redefined as Track-A-vs-Track-B disagreement
(XGB in the borderline band AND DAE strongly anomalous), the cleaner
semantics that don't depend on three correlated ensemble members.

This module is purely a function over Layer 2 outputs — it does no I/O
and loads no models. Construct a ``Layer2Output`` (e.g. via
``module2_detection.layer2_detector.Layer2Detector.score_alert``), then
pass the per-alert scalars to :func:`classify_alert_v4`.

INVARIANT 1
-----------
``c_detect = max(p_xgb, dae_score)`` — the DAE can only ELEVATE
confidence, never reduce it. The classifier sanity-checks this before
returning.

Decision tree
-------------
The 9 stages are ordered by specificity. Each stage's predicate is
disjoint from the predicates above it (see
``test_layer3_v4_triage::test_stage_predicates_partition_input_space``).

  Stage 1 KNOWN_ATTACK            p_xgb >= 0.85 AND dae <  0.50
  Stage 2 KNOWN_ATTACK_UNCERTAIN  p_xgb >= 0.85 AND dae >= 0.50
  Stage 3 DISAGREEMENT_ANOMALY    0.40 <= p_xgb < 0.85 AND dae >= 0.95
  Stage 4 STRONG_NOVEL_ANOMALY    p_xgb < 0.40 AND dae >= 0.95
  Stage 5 NOVEL_ANOMALY           p_xgb < 0.40 AND 0.70 <= dae < 0.95
  Stage 6 CONFIRMED_ANOMALY       0.40 <= p_xgb < 0.85 AND 0.70 <= dae < 0.95
  Stage 7 SUSPICIOUS_PATTERN      0.40 <= p_xgb < 0.85 AND dae <  0.70
  Stage 8 BENIGN_WATCH            p_xgb < 0.40 AND 0.50 <= dae < 0.70
  Stage 9 BENIGN                  default
"""
from __future__ import annotations

from dataclasses import dataclass

from src.data_models import AlertType, Confidence


# Threshold constants — keep in sync with the docstring above.
P_XGB_HIGH = 0.85
P_XGB_LOW = 0.40
DAE_HIGH = 0.95          # 99th-percentile rank
DAE_MODERATE = 0.70      # 95th-percentile rank
DAE_WEAK = 0.50          # marginal — used to split Stage 1 / Stage 2

# v5: retired but kept for back-compat imports (some legacy callers
# still reference these constants). New stage predicates do not use
# diversity at all.
DIVERSITY_HIGH = 0.30
DIVERSITY_MODERATE = 0.15


@dataclass
class TriageDecisionV4:
    """Result of the v4.0 enriched triage classifier.

    Carries the source signals so audit logs can reproduce the routing
    decision without re-running the classifier. ``diversity_score`` is
    retained as an audit field (always ``0.0`` in v5 unless the caller
    explicitly passes one) but is no longer consumed by the predicates.
    """

    alert_type: AlertType
    confidence: Confidence
    rationale: str
    template_id: str
    c_detect: float

    p_xgb: float
    diversity_score: float
    dae_score: float
    threshold_level: str


def classify_alert_v4(
    p_xgb: float,
    dae_score: float,
    *,
    threshold_level: str = "below_threshold",
    p_rf: float | None = None,
    p_dt: float | None = None,
    diversity_score: float | None = None,
) -> TriageDecisionV4:
    """Apply the 9-stage v4 triage decision tree to a single alert.

    Args:
        p_xgb: Calibrated P(attack) from XGBoost (Track A).
        dae_score: DAE anomaly score in [0, 1] (Layer 2 ``dae_score``,
            percentile-rank calibrated; Track B).
        threshold_level: One of ``below_threshold`` / ``weak`` /
            ``moderate`` / ``strong`` (Layer 2 ``threshold_level``).
            Carried through for audit; not used by the predicates.
        p_rf, p_dt, diversity_score: **Deprecated as of v5.** Retained as
            optional kwargs so existing callers don't break, but the
            predicates no longer consume them. ``diversity_score`` is
            echoed onto the returned ``TriageDecisionV4`` for audit
            continuity; ``p_rf`` / ``p_dt`` are ignored.

    Returns:
        ``TriageDecisionV4`` with the matched stage's alert type,
        confidence, rationale, and the source signals.

    Raises:
        AssertionError: if INVARIANT 1 (c_detect >= p_xgb) is violated.
            This is a real safety check, not a debug assert — the
            classifier refuses to emit a decision that would let the
            DAE *reduce* confidence below the primary tree.
    """
    p_xgb = float(p_xgb)
    dae_score = float(dae_score)
    diversity_audit = float(diversity_score) if diversity_score is not None else 0.0

    c_detect = max(p_xgb, dae_score)
    if c_detect < p_xgb - 1e-9:
        raise AssertionError(
            "Layer 3 v4 INVARIANT 1 violated: "
            f"c_detect={c_detect} < p_xgb={p_xgb}"
        )

    def _decision(
        alert_type: AlertType,
        confidence: Confidence,
        rationale: str,
        template_id: str,
    ) -> TriageDecisionV4:
        return TriageDecisionV4(
            alert_type=alert_type,
            confidence=confidence,
            rationale=rationale,
            template_id=template_id,
            c_detect=c_detect,
            p_xgb=p_xgb,
            diversity_score=diversity_audit,
            dae_score=dae_score,
            threshold_level=threshold_level,
        )

    # Stage 1 — KNOWN_ATTACK (XGB confident, DAE confirms not novel)
    if p_xgb >= P_XGB_HIGH and dae_score < DAE_WEAK:
        return _decision(
            AlertType.KNOWN_ATTACK,
            Confidence.VERY_HIGH,
            "Track A high P + Track B agrees not novel",
            "known_attack_high_confidence",
        )

    # Stage 2 — KNOWN_ATTACK_UNCERTAIN (XGB confident BUT DAE flags anomaly)
    if p_xgb >= P_XGB_HIGH and dae_score >= DAE_WEAK:
        return _decision(
            AlertType.KNOWN_ATTACK_UNCERTAIN,
            Confidence.HIGH,
            "Track A high P but Track B sees anomaly — verify",
            "known_attack_with_uncertainty",
        )

    # Stage 3 — DISAGREEMENT_ANOMALY (Track A borderline, Track B strong)
    # v5 redefinition: the disagreement signal is now between the two
    # tracks (XGB and DAE) rather than within Track A. XGB sits in the
    # moderate-P band but DAE asserts strong novelty — the canonical
    # adversarial-input signature.
    if P_XGB_LOW <= p_xgb < P_XGB_HIGH and dae_score >= DAE_HIGH:
        return _decision(
            AlertType.DISAGREEMENT_ANOMALY,
            Confidence.HIGH,
            "Track A borderline + Track B strongly anomalous → potential adversarial input",
            "adversarial_pattern",
        )

    # Stage 4 — STRONG_NOVEL_ANOMALY (Track A silent, DAE strong)
    if p_xgb < P_XGB_LOW and dae_score >= DAE_HIGH:
        return _decision(
            AlertType.STRONG_NOVEL_ANOMALY,
            Confidence.HIGH,
            "DAE strongly indicates outside benign manifold",
            "novel_pattern_strong",
        )

    # Stage 5 — NOVEL_ANOMALY (Track A silent, DAE moderate)
    if p_xgb < P_XGB_LOW and DAE_MODERATE <= dae_score < DAE_HIGH:
        return _decision(
            AlertType.NOVEL_ANOMALY,
            Confidence.MEDIUM,
            "DAE moderate novelty signal, Track A silent",
            "novel_pattern_moderate",
        )

    # Stage 6 — CONFIRMED_ANOMALY (multi-signal corroboration)
    # v5: tightened to dae < DAE_HIGH so Stage 3 (which now also lives
    # in the same XGB band) can fire on the high-DAE side without overlap.
    if P_XGB_LOW <= p_xgb < P_XGB_HIGH and DAE_MODERATE <= dae_score < DAE_HIGH:
        return _decision(
            AlertType.CONFIRMED_ANOMALY,
            Confidence.HIGH,
            "Both Track A and Track B indicate suspicion",
            "multi_signal_corroborated",
        )

    # Stage 7 — SUSPICIOUS_PATTERN (Track A moderate, DAE benign)
    if P_XGB_LOW <= p_xgb < P_XGB_HIGH and dae_score < DAE_MODERATE:
        return _decision(
            AlertType.SUSPICIOUS_PATTERN,
            Confidence.MEDIUM,
            "Track A moderate, Track B agrees benign — review",
            "suspicious_pattern_review",
        )

    # Stage 8 — BENIGN_WATCH (marginal, audit only)
    if p_xgb < P_XGB_LOW and DAE_WEAK <= dae_score < DAE_MODERATE:
        return _decision(
            AlertType.BENIGN_WATCH,
            Confidence.LOW,
            "Marginal anomaly, no attack signature",
            "benign_watch",
        )

    # Stage 9 — BENIGN (default)
    return _decision(
        AlertType.BENIGN,
        Confidence.HIGH,
        "All filters indicate benign",
        "benign",
    )


__all__ = [
    "TriageDecisionV4",
    "classify_alert_v4",
    "P_XGB_HIGH",
    "P_XGB_LOW",
    "DAE_HIGH",
    "DAE_MODERATE",
    "DAE_WEAK",
    # v5: retired but exported for back-compat imports.
    "DIVERSITY_HIGH",
    "DIVERSITY_MODERATE",
]
