"""Human-in-the-Loop feedback management — 7 methods.

Manages analyst feedback collection, quality tracking, auto-recalibration,
active learning prioritization, and disagreement resolution.

Methods:
  1. Feedback summary dashboard data
  2. Active learning (uncertainty-based prioritization)
  3. Feedback impact visualization
  4. Automatic recalibration trigger
  5. Per-analyst quality tracking
  6. Disagreement detection and resolution
  7. Retraining readiness assessment
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """Manages the human-in-the-loop feedback cycle.

    Args:
        database: Database instance for feedback queries.
        calibrator_fn: Callable to trigger recalibration (optional).
        recalibration_threshold: Feedback count before auto-recalibration.
        retraining_threshold: Feedback count before retraining is recommended.
    """

    def __init__(
        self,
        database: Any,
        calibrator_fn: Any = None,
        recalibration_threshold: int = 50,
        retraining_threshold: int = 500,
    ) -> None:
        self._db = database
        self._calibrator_fn = calibrator_fn
        self._recal_threshold = recalibration_threshold
        self._retrain_threshold = retraining_threshold
        self._last_recal_feedback_id = 0
        self._recalibration_count = 0

    # ═══════════════════════════════════════════════════════════════
    # Method 1: Feedback Summary
    # ═══════════════════════════════════════════════════════════════

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive feedback summary for dashboard."""
        if self._db is None:
            return {}

        summary = self._db.get_feedback_summary()
        summary["recalibrations_performed"] = self._recalibration_count
        summary["next_recalibration_at"] = (
            self._last_recal_feedback_id + self._recal_threshold
        )
        summary["retraining_ready"] = summary.get("total_feedback", 0) >= self._retrain_threshold
        summary["retraining_threshold"] = self._retrain_threshold

        return summary

    # ═══════════════════════════════════════════════════════════════
    # Method 2: Active Learning (Uncertainty Prioritization)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def compute_uncertainty(prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Compute model uncertainty for a prediction.

        Alerts near the detection boundary are most uncertain and
        most valuable for feedback. Returns uncertainty score + label.
        """
        score = prediction.get("anomaly_score", 0)
        percentile = prediction.get("percentile") or 0
        risk = prediction.get("risk_level", "NORMAL")

        # Distance from detection boundary (P85-P93 range)
        # Closer to boundary = higher uncertainty
        if 70 < percentile < 95:
            boundary_distance = min(abs(percentile - 85), abs(percentile - 93))
            uncertainty = max(0, 1.0 - boundary_distance / 15.0)
        else:
            uncertainty = 0.1  # Far from boundary = low uncertainty

        if uncertainty > 0.7:
            label = "HIGH"
            message = "Model uncertain — your feedback is especially valuable here"
        elif uncertainty > 0.4:
            label = "MEDIUM"
            message = "Model moderately confident — feedback helps refine detection"
        else:
            label = "LOW"
            message = "Model confident — feedback confirms system accuracy"

        return {
            "uncertainty": round(uncertainty, 2),
            "label": label,
            "message": message,
        }

    # ═══════════════════════════════════════════════════════════════
    # Method 3: Feedback Impact Visualization
    # ═══════════════════════════════════════════════════════════════

    def get_impact_summary(self) -> Dict[str, Any]:
        """Compute the impact of analyst feedback on system behavior."""
        if self._db is None:
            return {}

        summary = self._db.get_feedback_summary()
        total = summary.get("total_feedback", 0)
        fp = summary.get("marked_safe", 0)
        tp = summary.get("confirmed_attacks", 0)

        impact = {
            "total_feedback": total,
            "false_positives_identified": fp,
            "true_positives_confirmed": tp,
            "recalibrations_triggered": self._recalibration_count,
        }

        if fp > 0:
            impact["fpr_reduction_estimate"] = (
                f"Your {fp} 'Mark Safe' entries can reduce false alarm rate by ~{min(fp * 2, 30)}%"
            )
        if tp > 0:
            impact["detection_confidence"] = (
                f"Your {tp} 'Confirm Attack' entries validate detection accuracy"
            )
        if total >= self._retrain_threshold:
            impact["retraining_message"] = (
                f"{total} feedback entries available — sufficient for model retraining"
            )
        elif total > 0:
            remaining = self._retrain_threshold - total
            impact["retraining_message"] = (
                f"{remaining} more feedback entries needed before model retraining"
            )

        return impact

    # ═══════════════════════════════════════════════════════════════
    # Method 4: Automatic Recalibration Trigger
    # ═══════════════════════════════════════════════════════════════

    def check_and_recalibrate(self) -> Optional[Dict[str, Any]]:
        """Check if enough new feedback exists to trigger recalibration.

        Called periodically (e.g., every 100 predictions). If 50+ new
        feedback entries since last recalibration, triggers re-fit.

        Returns:
            Recalibration result dict if triggered, None otherwise.
        """
        if self._db is None or self._calibrator_fn is None:
            return None

        new_count = self._db.get_feedback_count_since(self._last_recal_feedback_id)
        if new_count < self._recal_threshold:
            return None

        logger.info(
            "Auto-recalibration triggered: %d new feedback entries (threshold: %d)",
            new_count, self._recal_threshold,
        )

        try:
            result = self._calibrator_fn()
            if result:
                self._recalibration_count += 1
                # Update watermark to current max feedback ID
                with self._db._lock:
                    row = self._db._conn.execute(
                        "SELECT MAX(id) as max_id FROM feedback"
                    ).fetchone()
                self._last_recal_feedback_id = row["max_id"] if row and row["max_id"] else 0
                logger.info("Recalibration #%d complete", self._recalibration_count)
                return {"recalibrated": True, "feedback_used": new_count}
        except Exception as exc:
            logger.error("Auto-recalibration failed: %s", exc)

        return None

    # ═══════════════════════════════════════════════════════════════
    # Method 5: Feedback Quality Tracking
    # ═══════════════════════════════════════════════════════════════

    def get_analyst_quality(self) -> List[Dict[str, Any]]:
        """Get per-analyst feedback quality metrics.

        Tracks: total reviews, TP/FP ratio, average confidence,
        consistency score (how much analyst agrees with model).
        """
        if self._db is None:
            return []

        analysts = self._db.get_feedback_by_analyst()
        for a in analysts:
            total = a.get("total", 1)
            tp = a.get("tp_count", 0)
            fp = a.get("fp_count", 0)

            # Agreement rate: how often analyst agrees with model
            # (TP on HIGH+ alerts = agrees, FP on HIGH+ = disagrees)
            a["agreement_rate"] = round(tp / max(total, 1), 2)

            # Quality label
            if total < 10:
                a["quality"] = "INSUFFICIENT"
                a["note"] = f"Need {10 - total} more reviews for quality assessment"
            elif a["agreement_rate"] > 0.8:
                a["quality"] = "HIGH"
                a["note"] = "Consistent with system — reliable reviewer"
            elif a["agreement_rate"] > 0.5:
                a["quality"] = "MEDIUM"
                a["note"] = "Moderate agreement — review controversial cases"
            else:
                a["quality"] = "REVIEW"
                a["note"] = "Low agreement — may indicate model blind spot or review needed"

        return analysts

    # ═══════════════════════════════════════════════════════════════
    # Method 6: Disagreement Detection
    # ═══════════════════════════════════════════════════════════════

    def get_disagreements(self) -> List[Dict[str, Any]]:
        """Find alerts where analysts disagree on ground truth.

        Returns list of conflicts with both analysts' verdicts.
        """
        if self._db is None:
            return []

        disagreements = self._db.get_feedback_disagreements()
        for d in disagreements:
            d["resolution_status"] = "PENDING"
            d["recommendation"] = (
                f"Alert #{d['alert_id']}: Analyst {d['analyst_a'][:6]} says "
                f"{'ATTACK' if d['verdict_a'] == 1 else 'SAFE'}, "
                f"Analyst {d['analyst_b'][:6]} says "
                f"{'ATTACK' if d['verdict_b'] == 1 else 'SAFE'}. "
                f"Senior analyst review required."
            )
        return disagreements

    # ═══════════════════════════════════════════════════════════════
    # Method 7: Retraining Readiness
    # ═══════════════════════════════════════════════════════════════

    def get_retraining_status(self) -> Dict[str, Any]:
        """Assess whether enough feedback exists for model retraining."""
        if self._db is None:
            return {"ready": False, "reason": "No database"}

        summary = self._db.get_feedback_summary()
        total = summary.get("total_feedback", 0)
        tp = summary.get("confirmed_attacks", 0)
        fp = summary.get("marked_safe", 0)

        status: Dict[str, Any] = {
            "total_feedback": total,
            "threshold": self._retrain_threshold,
            "progress_pct": round(min(total / max(self._retrain_threshold, 1), 1.0) * 100, 1),
            "confirmed_attacks": tp,
            "marked_safe": fp,
        }

        if total < self._retrain_threshold:
            status["ready"] = False
            status["reason"] = f"Need {self._retrain_threshold - total} more feedback entries"
        elif tp < 20:
            status["ready"] = False
            status["reason"] = f"Need more confirmed attacks (have {tp}, need ≥20)"
        elif fp < 10:
            status["ready"] = False
            status["reason"] = f"Need more false positive labels (have {fp}, need ≥10)"
        else:
            status["ready"] = True
            status["reason"] = "Sufficient feedback for retraining"
            status["command"] = "python -m src.phase2_5_fine_tuning.run_finetuned_training"

        return status
