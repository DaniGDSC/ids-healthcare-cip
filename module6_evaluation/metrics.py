"""Module 6 evaluation metrics — Task 6.6."""
from __future__ import annotations

import pandas as pd


def compute_evaluation_metrics(responses: list) -> dict:
    """Compute inter-rater reliability, mean Likert, accuracy, time."""
    df = pd.DataFrame(responses)

    metrics = {
        "n_participants": df["participant_id"].nunique() if len(df) else 0,
        "n_alerts": df["alert_id"].nunique() if len(df) else 0,
        "n_responses": len(df),
    }

    for condition in ["with_xai", "without_xai"]:
        cond_df = df[df["condition"] == condition] if len(df) else df
        if len(cond_df) == 0:
            metrics[condition] = {
                "decision_accuracy": 0.0,
                "mean_decision_time_sec": 0.0,
                "mean_confidence": 0.0,
                "likert_trust": 0.0,
                "likert_usefulness": 0.0,
                "likert_comprehensibility": 0.0,
                "likert_actionability": 0.0,
            }
            continue
        metrics[condition] = {
            "decision_accuracy": round(float(cond_df["decision_correct"].mean()), 4),
            "mean_decision_time_sec": round(float(cond_df["decision_time_sec"].mean()), 1),
            "mean_confidence": round(float(cond_df["confidence"].mean()), 2),
            "likert_trust": round(float(cond_df["likert_trust"].mean()), 2),
            "likert_usefulness": round(float(cond_df["likert_usefulness"].mean()), 2),
            "likert_comprehensibility": round(float(cond_df["likert_comprehensibility"].mean()), 2),
            "likert_actionability": round(float(cond_df["likert_actionability"].mean()), 2),
        }

    metrics["per_role"] = {}
    if len(df) == 0:
        return metrics
    for role in df["participant_role"].unique():
        role_df = df[df["participant_role"] == role]
        with_xai = role_df[role_df["condition"] == "with_xai"]
        without_xai = role_df[role_df["condition"] == "without_xai"]
        metrics["per_role"][role] = {
            "with_xai_accuracy": round(float(with_xai["decision_correct"].mean()), 4)
            if len(with_xai) else 0.0,
            "without_xai_accuracy": round(float(without_xai["decision_correct"].mean()), 4)
            if len(without_xai) else 0.0,
        }

    return metrics


__all__ = ["compute_evaluation_metrics"]
