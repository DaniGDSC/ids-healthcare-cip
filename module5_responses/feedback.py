"""Module 5 — FeedbackLoop for closed-loop threshold adjustment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FeedbackLoop:
    """Record TP/FP labels; suggest weight/threshold adjustments.

    Y9 follow-up: ``save_state`` / ``load_state`` provide minimal
    round-trip persistence so the closed loop survives process restarts.
    """

    def __init__(self):
        self.records: list[dict] = []

    def record(
        self,
        alert_id: str,
        ground_truth: str,
        predicted_tier: str,
        risk_score: float,
        actions: list,
    ) -> None:
        self.records.append(
            {
                "alert_id": alert_id,
                "ground_truth": ground_truth,
                "predicted_tier": predicted_tier,
                "risk_score": risk_score,
                "actions": actions,
                "is_tp": ground_truth == "attack"
                and predicted_tier in ("MEDIUM", "HIGH", "CRITICAL"),
                "is_fp": ground_truth == "benign"
                and predicted_tier in ("MEDIUM", "HIGH", "CRITICAL"),
                "is_fn": ground_truth == "attack" and predicted_tier == "LOW",
            }
        )

    def compute_adjustments(self, current_thresholds: dict | None = None) -> dict:
        """Return numeric threshold adjustments based on TP/FP/FN rates.

        Rules
        -----
        * FPR > 10 %  →  raise MEDIUM threshold by  0.05 × (FPR − 0.10) / 0.10
                         raise HIGH   threshold by  0.03 × (FPR − 0.10) / 0.10
        * FNR >  5 %  →  lower MEDIUM threshold by  0.05 × (FNR − 0.05) / 0.05
                         lower HIGH   threshold by  0.03 × (FNR − 0.05) / 0.05
        """
        if not self.records:
            return {}

        if current_thresholds is None:
            current_thresholds = {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}

        tp = sum(1 for r in self.records if r["is_tp"])
        fp = sum(1 for r in self.records if r["is_fp"])
        fn = sum(1 for r in self.records if r["is_fn"])
        total = len(self.records)

        fpr = fp / total if total > 0 else 0.0
        fnr = fn / total if total > 0 else 0.0

        suggested = dict(current_thresholds)
        adjustments = []

        if fpr > 0.10:
            delta_med = 0.05 * (fpr - 0.10) / 0.10
            delta_high = 0.03 * (fpr - 0.10) / 0.10
            suggested["MEDIUM"] += delta_med
            suggested["HIGH"] += delta_high
            suggested["CRITICAL"] += delta_high * 0.5
            adjustments.append(
                {
                    "metric": "fpr",
                    "current_value": round(fpr, 4),
                    "target": 0.10,
                    "direction": "raise",
                }
            )

        if fnr > 0.05:
            delta_med = 0.05 * (fnr - 0.05) / 0.05
            delta_high = 0.03 * (fnr - 0.05) / 0.05
            suggested["MEDIUM"] -= delta_med
            suggested["HIGH"] -= delta_high
            suggested["CRITICAL"] -= delta_high * 0.5
            adjustments.append(
                {
                    "metric": "fnr",
                    "current_value": round(fnr, 4),
                    "target": 0.05,
                    "direction": "lower",
                }
            )

        if fpr <= 0.10 and fnr <= 0.05:
            adjustments.append(
                {
                    "metric": "calibrated",
                    "current_value": None,
                    "target": None,
                    "direction": "none",
                }
            )

        suggested = {k: round(v, 4) for k, v in suggested.items()}

        fp_scores = [r["risk_score"] for r in self.records if r["is_fp"]]
        tp_scores = [r["risk_score"] for r in self.records if r["is_tp"]]

        return {
            "total_evaluated": total,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "mean_fp_risk_score": round(float(np.mean(fp_scores)), 4)
            if fp_scores
            else None,
            "mean_tp_risk_score": round(float(np.mean(tp_scores)), 4)
            if tp_scores
            else None,
            "current_thresholds": current_thresholds,
            "suggested_threshold_change": suggested,
            "adjustments": adjustments,
        }

    # ── persistence (Y9 follow-up) ─────────────────────────────────

    def save_state(self, path: Path) -> None:
        """Persist the recorded labels so closed-loop state survives restarts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    def load_state(self, path: Path) -> None:
        """Restore previously persisted labels (appends; does not clear)."""
        path = Path(path)
        if not path.exists():
            return
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(
                f"FeedbackLoop.load_state expected a JSON list at {path}, "
                f"got {type(loaded).__name__}"
            )
        self.records.extend(loaded)


__all__ = ["FeedbackLoop"]
