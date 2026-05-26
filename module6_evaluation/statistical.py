"""Module 6 statistical analysis — Task 6.7 (paired Wilcoxon + Cohen's d)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def statistical_analysis(responses: list) -> dict:
    """Paired Wilcoxon signed-rank test: with-XAI vs without-XAI."""
    from scipy.stats import wilcoxon

    df = pd.DataFrame(responses)
    results: dict = {}

    if len(df) == 0:
        return results

    for measure in [
        "decision_correct", "decision_time_sec", "confidence",
        "likert_trust", "likert_usefulness", "likert_comprehensibility",
        "likert_actionability",
    ]:
        paired_with = df[df["condition"] == "with_xai"].groupby("participant_id")[measure].mean()
        paired_without = df[df["condition"] == "without_xai"].groupby("participant_id")[measure].mean()

        common = paired_with.index.intersection(paired_without.index)
        if len(common) < 3:
            continue

        a = paired_with.loc[common].values
        b = paired_without.loc[common].values
        diff = a - b

        if np.all(diff == 0):
            results[measure] = {
                "statistic": 0,
                "p_value": 1.0,
                "significant": False,
                "mean_with": round(float(a.mean()), 4),
                "mean_without": round(float(b.mean()), 4),
            }
            continue

        stat, p_val = wilcoxon(a, b)

        pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
        cohens_d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0

        results[measure] = {
            "mean_with_xai": round(float(a.mean()), 4),
            "mean_without_xai": round(float(b.mean()), 4),
            "difference": round(float(a.mean() - b.mean()), 4),
            "wilcoxon_statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
            "cohens_d": round(float(cohens_d), 4),
            "effect_size": (
                "large" if abs(cohens_d) > 0.8
                else "medium" if abs(cohens_d) > 0.5
                else "small"
            ),
        }

    return results


__all__ = ["statistical_analysis"]
