"""Module 6 inter-rater reliability — Task 6D.5 (Krippendorff's alpha approx)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .alerts import ACTIONS


def compute_inter_rater_reliability(responses: list) -> dict:
    """Compute Krippendorff's alpha for action selection and Likert scores."""
    df = pd.DataFrame(responses)
    results: dict = {}

    if len(df) == 0:
        return results

    for measure in [
        "chosen_action", "likert_trust", "likert_usefulness",
        "likert_comprehensibility", "likert_actionability",
    ]:
        pivot = df.pivot_table(
            index="participant_id", columns="alert_id",
            values=measure, aggfunc="first",
        )

        if measure == "chosen_action":
            action_map = {a: i for i, a in enumerate(ACTIONS)}
            pivot = pivot.map(lambda x: action_map.get(x, -1) if isinstance(x, str) else x)

        matrix = pivot.values.astype(float)
        matrix = np.where(np.isnan(matrix), np.nan, matrix)

        n_coders, n_items = matrix.shape
        valid_pairs = 0
        observed_disagree = 0
        all_values = []

        for j in range(n_items):
            col = matrix[:, j]
            valid = col[~np.isnan(col)]
            all_values.extend(valid.tolist())
            for a in range(len(valid)):
                for b in range(a + 1, len(valid)):
                    valid_pairs += 1
                    if valid[a] != valid[b]:
                        observed_disagree += 1

        if valid_pairs == 0:
            results[measure] = {
                "alpha": 0.0, "n_coders": int(n_coders), "n_items": int(n_items),
            }
            continue

        Do = observed_disagree / valid_pairs

        # M6-E6: vectorised expected disagreement.
        vals = np.array(all_values)
        finite_vals = vals[~np.isnan(vals)]
        unique_vals = np.unique(finite_vals)
        n_total = len(finite_vals)
        counts = np.array([(finite_vals == v).sum() for v in unique_vals])
        probs = counts / n_total
        De = float(1.0 - np.dot(probs, probs))

        alpha = 1 - (Do / De) if De > 0 else 1.0

        results[measure] = {
            "alpha": round(float(alpha), 4),
            "n_coders": int(n_coders),
            "n_items": int(n_items),
            "interpretation": (
                "good" if alpha > 0.67
                else "moderate" if alpha > 0.33
                else "poor"
            ),
        }

    return results


__all__ = ["compute_inter_rater_reliability"]
