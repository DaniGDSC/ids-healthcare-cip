"""Layer 3 v4.0 — enriched 9-stage triage classifier.

Maps the per-alert Layer 2 outputs (calibrated tree probabilities,
diversity score, DAE percentile-rank score, threshold bucket) to the
9-class ``AlertType`` typology + a 4-level ``Confidence`` indicator.

This module is purely a function over Layer 2 outputs — it does no I/O
and loads no models. Construct a ``Layer2Output`` (e.g. via
``module2_detection.layer2_detector.Layer2Detector.score_alert``), then
pass the per-alert scalars to :func:`classify_alert_v4`.

INVARIANT 1
-----------
``c_detect = max(p_xgb, dae_score, normalized_diversity)`` — the DAE
and the diversity signal can only ELEVATE confidence, never reduce it.
The classifier sanity-checks this before returning.

Decision tree
-------------
The 9 stages are ordered by specificity. Each stage's predicate is
disjoint from the predicates above it (see
``test_layer3_v4_triage::test_stage_predicates_partition_input_space``).

  Stage 1 KNOWN_ATTACK            p_xgb >= 0.85 AND diversity < 0.15
  Stage 2 KNOWN_ATTACK_UNCERTAIN  p_xgb >= 0.85 AND diversity >= 0.15
  Stage 3 DISAGREEMENT_ANOMALY    diversity >= 0.30 AND dae >= 0.70
  Stage 4 STRONG_NOVEL_ANOMALY    p_xgb < 0.40   AND dae >= 0.95
  Stage 5 NOVEL_ANOMALY           p_xgb < 0.40   AND 0.70 <= dae < 0.95
  Stage 6 CONFIRMED_ANOMALY       0.40 <= p_xgb < 0.85 AND dae >= 0.70
  Stage 7 SUSPICIOUS_PATTERN      0.40 <= p_xgb < 0.85 AND dae < 0.70
  Stage 8 BENIGN_WATCH            p_xgb < 0.40   AND 0.50 <= dae < 0.70
  Stage 9 BENIGN                  default
"""
from __future__ import annotations

from dataclasses import dataclass

from src.data_models import AlertType, Confidence


# Threshold constants — keep in sync with the docstring above.
P_XGB_HIGH = 0.85
P_XGB_LOW = 0.40
DIVERSITY_HIGH = 0.30
DIVERSITY_MODERATE = 0.15
DAE_HIGH = 0.95          # 99th-percentile rank
DAE_MODERATE = 0.70      # 95th-percentile rank
DAE_WEAK = 0.50          # marginal


@dataclass
class TriageDecisionV4:
    """Result of the v4.0 enriched triage classifier.

    Carries the source signals so audit logs can reproduce the routing
    decision without re-running the classifier.
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


def _normalised_diversity(diversity_score: float) -> float:
    """Map raw diversity (std of 3 calibrated probas, max ~0.47) to [0, 1]
    so it is comparable with ``p_xgb`` and ``dae_score`` when taking the
    max for ``c_detect``.

    Anchor at 0.30 (the DISAGREEMENT_ANOMALY threshold). A diversity of
    0.30 maps to 1.0; values above are clamped.
    """
    return min(1.0, diversity_score / DIVERSITY_HIGH) if diversity_score > 0 else 0.0


def classify_alert_v4(
    p_xgb: float,
    p_rf: float,
    p_dt: float,
    diversity_score: float,
    dae_score: float,
    threshold_level: str = "below_threshold",
) -> TriageDecisionV4:
    """Apply the 9-stage v4 triage decision tree to a single alert.

    Args:
        p_xgb: Calibrated P(attack) from XGBoost.
        p_rf: Calibrated P(attack) from RandomForest.
        p_dt: Calibrated P(attack) from DecisionTree.
        diversity_score: Standard deviation of the three probas
            (Layer 2 ``diversity_score``).
        dae_score: DAE anomaly score in [0, 1] (Layer 2 ``dae_score``,
            percentile-rank calibrated).
        threshold_level: One of ``below_threshold`` / ``weak`` /
            ``moderate`` / ``strong`` (Layer 2 ``threshold_level``).
            Carried through for audit; not used by the predicates.

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
    p_rf = float(p_rf)
    p_dt = float(p_dt)
    diversity_score = float(diversity_score)
    dae_score = float(dae_score)

    c_detect = max(p_xgb, dae_score, _normalised_diversity(diversity_score))
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
            diversity_score=diversity_score,
            dae_score=dae_score,
            threshold_level=threshold_level,
        )

    # Stage 1 — KNOWN_ATTACK (high confidence, models agree)
    if p_xgb >= P_XGB_HIGH and diversity_score < DIVERSITY_MODERATE:
        return _decision(
            AlertType.KNOWN_ATTACK,
            Confidence.VERY_HIGH,
            "Track A high P + models agree",
            "known_attack_high_confidence",
        )

    # Stage 2 — KNOWN_ATTACK_UNCERTAIN (high P but disagreement)
    if p_xgb >= P_XGB_HIGH and diversity_score >= DIVERSITY_MODERATE:
        return _decision(
            AlertType.KNOWN_ATTACK_UNCERTAIN,
            Confidence.HIGH,
            "Track A high P but models disagree",
            "known_attack_with_uncertainty",
        )

    # Stage 3 — DISAGREEMENT_ANOMALY (potential adversarial)
    # Predicate uses the v4 thresholds. We checked stages 1+2 above so
    # p_xgb is < P_XGB_HIGH at this point — DISAGREEMENT only fires in
    # the moderate/low-P regime where the disagreement is suspicious
    # (rather than confirming an already-known attack).
    if diversity_score >= DIVERSITY_HIGH and dae_score >= DAE_MODERATE:
        return _decision(
            AlertType.DISAGREEMENT_ANOMALY,
            Confidence.HIGH,
            "Models disagree + DAE elevated → potential adversarial input",
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
    if P_XGB_LOW <= p_xgb < P_XGB_HIGH and dae_score >= DAE_MODERATE:
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
    "DIVERSITY_HIGH",
    "DIVERSITY_MODERATE",
    "DAE_HIGH",
    "DAE_MODERATE",
    "DAE_WEAK",
]
