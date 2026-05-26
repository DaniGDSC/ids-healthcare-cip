"""Synthetic participant-response generator for thesis validation."""
from __future__ import annotations

import numpy as np


def generate_simulated_responses(alerts: list, n_participants: int = 15) -> list:
    """Generate simulated participant responses for thesis validation.

    In production, these come from the Streamlit evaluation app. For thesis
    development, we simulate realistic responses with XAI improving accuracy
    and confidence per the H2 hypothesis. Seeded ``RandomState(42)`` so the
    output is reproducible.
    """
    rng = np.random.RandomState(42)
    responses = []

    roles = (["analyst"] * 5 + ["clinician"] * 5 + ["administrator"] * 5)

    for p_idx in range(n_participants):
        role = roles[p_idx]
        for a_idx, alert in enumerate(alerts):
            has_xai = a_idx < 10

            base_accuracy = 0.70 if not has_xai else 0.88
            base_trust = 3.0 if not has_xai else 4.2
            base_time = 45 if not has_xai else 28

            if role == "analyst":
                base_accuracy += 0.05
                base_time -= 5
            elif role == "clinician":
                base_trust += 0.3

            correct_action = alert["correct_action"]
            chose_correctly = rng.random() < base_accuracy
            chosen_action = correct_action if chose_correctly else rng.choice(
                ["dismiss", "monitor", "investigate", "isolate"])

            responses.append({
                "participant_id": f"P{p_idx+1:02d}",
                "participant_role": role,
                "alert_id": alert["alert_id"],
                "condition": "with_xai" if has_xai else "without_xai",
                "chosen_action": chosen_action,
                "correct_action": correct_action,
                "decision_correct": chose_correctly,
                "decision_time_sec": max(5, int(base_time + rng.normal(0, 8))),
                "confidence": min(5, max(1, int(rng.normal(3.5 if has_xai else 2.8, 0.8) + 0.5))),
                "likert_trust": min(5, max(1, int(rng.normal(base_trust, 0.7) + 0.5))),
                "likert_usefulness": min(5, max(1, int(rng.normal(base_trust + 0.2, 0.6) + 0.5))),
                "likert_comprehensibility": min(5, max(1, int(rng.normal(base_trust - 0.1, 0.8) + 0.5))),
                "likert_actionability": min(5, max(1, int(rng.normal(base_trust + 0.1, 0.7) + 0.5))),
                "feedback": "",
                "reclassification": None,
            })

    return responses


__all__ = ["generate_simulated_responses"]
