"""Module 6 thematic feedback analysis — Task 6D.7."""
from __future__ import annotations

import pandas as pd


THEMES = {
    "wanted_more_detail": ["more detail", "more info", "explain more", "unclear"],
    "nlg_helpful": ["helpful", "clear", "understandable", "good explanation"],
    "too_technical": ["technical", "jargon", "confusing", "complex"],
    "shap_useful": ["shap", "feature", "contribution", "waterfall"],
    "trust_concern": ["trust", "confident", "unsure", "doubt"],
    "action_unclear": ["action", "what to do", "response", "next step"],
}


def analyze_feedback(responses: list) -> dict:
    """Thematic analysis of free-text feedback and reclassifications."""
    df = pd.DataFrame(responses)

    if len(df) == 0:
        return {
            "total_feedback_texts": 0,
            "reclassification_counts": {},
            "n_reclassifications": 0,
            "thematic_counts": {t: 0 for t in THEMES},
            "sample_feedback": [],
            "corrections_for_modules_3_5": [],
            "recommendations": [],
        }

    reclass = df[df["reclassification"].notna() & (df["reclassification"] != "")]
    reclass_counts = reclass["reclassification"].value_counts().to_dict() if len(reclass) > 0 else {}

    feedback_texts = df[df["feedback"].notna() & (df["feedback"] != "")]["feedback"].tolist()
    theme_counts = {theme: 0 for theme in THEMES}

    for text in feedback_texts:
        text_lower = text.lower()
        for theme, keywords in THEMES.items():
            if any(kw in text_lower for kw in keywords):
                theme_counts[theme] += 1

    corrections = []
    for _, row in reclass.iterrows():
        corrections.append({
            "alert_id": row["alert_id"],
            "original_tier": "inferred",
            "suggested_tier": row["reclassification"],
            "participant_role": row["participant_role"],
        })

    return {
        "total_feedback_texts": len(feedback_texts),
        "reclassification_counts": reclass_counts,
        "n_reclassifications": len(reclass),
        "thematic_counts": theme_counts,
        "sample_feedback": feedback_texts[:5],
        "corrections_for_modules_3_5": corrections[:10],
        "recommendations": [
            "If 'wanted_more_detail' > 3: expand NLG templates with feature-value context",
            "If 'too_technical' > 3: simplify analyst view terminology for clinician role",
            "If 'action_unclear' > 3: add explicit step-by-step response instructions",
        ],
    }


__all__ = ["analyze_feedback", "THEMES"]
