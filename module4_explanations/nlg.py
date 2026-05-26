"""Natural-language generation for stakeholder explanations.

Two entry points:
  - ``clinician_nlg(severity, top_features)`` — short narrative for the
    clinician summary (confidence-band aware).
  - ``generate_clinician_alert(...)`` — 6-step assembly used by the
    stakeholder router and the thesis worked examples.

The two share the same canonical CLINICIAN_TEMPLATES + FEATURE_CONCEPTS
from ``config.py``; previously the offline and online paths each
carried a near-duplicate template dict that could drift.
"""

from __future__ import annotations

import numpy as np

from .compute import _top_features_shap
from .config import (
    BIOMETRIC_FEATURES,
    FEATURE_CONCEPTS,
    NLG_TEMPLATES,
    format_clinician_template,
)
from .feature_groups import _feature_to_narrative


def clinician_nlg(severity: str, top_features: list) -> str:
    """Short clinician narrative for one alert.

    Confidence-band behavior (carried over from pre-cleanup):
      - When ``top_features[1]`` magnitude is ≥80% of ``top_features[0]``,
        cite the secondary indicator if it sits in a different feature
        category.
      - If any biometric feature appears in top-3 but the top-1 is
        non-biometric, append a biometric note.
    """
    if severity == "LOW":
        return format_clinician_template(
            "LOW", sample_index=None, narrative="", secondary_note="",
        )

    top1_feat = top_features[0]["feature"]
    top1_val = abs(top_features[0]["shap_value"])
    narrative_1, category_1 = _feature_to_narrative(top1_feat)

    secondary_note = ""
    if len(top_features) >= 2:
        top2_feat = top_features[1]["feature"]
        top2_val = abs(top_features[1]["shap_value"])
        ambiguity_ratio = top2_val / top1_val if top1_val > 0 else 0

        if ambiguity_ratio > 0.8:
            narrative_2, category_2 = _feature_to_narrative(top2_feat)
            if category_1 != category_2:
                secondary_note = (
                    f"A secondary indicator ({narrative_2}) also contributed. "
                )

        bio_feats = [
            f["feature"] for f in top_features if f["feature"] in BIOMETRIC_FEATURES
        ]
        if bio_feats and category_1 != "biometric":
            secondary_note += (
                f"Note: Biometric data ({', '.join(bio_feats)}) "
                "showed unusual values. "
            )

    return format_clinician_template(
        severity,
        sample_index=None,
        narrative=narrative_1,
        secondary_note=secondary_note,
    )


def build_shap_context(top_features: list) -> dict:
    """Assemble a spec-shaped shap_context dict from TreeSHAP output.

    Maps ranked top_features to research_spec.yaml §2.module_4 fields:
      - top_category: feature group with highest |SHAP| sum
      - top_features: feature names (rank preserved)
      - shap_direction: 'elevated' if top-1 SHAP > 0 else 'suppressed'
      - confidence_from_shap: HIGH (>0.3), MEDIUM (0.1-0.3), LOW (<0.1)
    """
    if not top_features:
        return {}

    category_abs: dict[str, float] = {}
    for tf in top_features:
        _, cat = _feature_to_narrative(tf["feature"])
        category_abs[cat] = category_abs.get(cat, 0.0) + abs(tf["shap_value"])
    top_category = max(category_abs, key=category_abs.get)

    top1 = top_features[0]
    top1_abs = abs(top1["shap_value"])
    if top1_abs > 0.3:
        confidence = "HIGH"
    elif top1_abs >= 0.1:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "top_category": top_category,
        "top_features": [tf["feature"] for tf in top_features],
        "shap_direction": "elevated" if top1["shap_value"] > 0 else "suppressed",
        "confidence_from_shap": confidence,
    }


# ── 6-step NLG (admin / analyst / clinician composite) ──────────────


def generate_clinician_alert(
    idx: int,
    sv_row: np.ndarray,
    feat_names: list,
    severity: str,
    confidence: float,
    consensus: str,
    risk_score: float = 0.0,
    risk_components: dict | None = None,
    d_clinical_tier_val: float = 0.0,
) -> str:
    """6-step NLG assembly for clinician-facing alert."""
    parts: list[str] = []

    # Step 1: Severity header
    parts.append(NLG_TEMPLATES["severity_header"].get(severity, severity))

    # Step 2: Detection sentence
    parts.append(
        NLG_TEMPLATES["detection_sentence"].format(
            consensus=consensus, confidence=confidence,
        )
    )

    # Step 3: Top-5 feature explanations
    abs_vals = np.abs(sv_row)
    top_idx = np.argsort(abs_vals)[-5:][::-1]
    for fi in top_idx:
        fname = feat_names[fi]
        concept = FEATURE_CONCEPTS.get(fname, {})
        label = concept.get("label", fname)
        cat = concept.get("category", "network")
        direction = concept.get(
            "direction_high" if sv_row[fi] > 0 else "direction_low",
            "abnormal value",
        )
        template_key = f"feature_explanation_{cat}"
        parts.append(
            NLG_TEMPLATES[template_key].format(
                label=label, direction=direction, shap_value=float(sv_row[fi]),
            )
        )

    # Step 4: Risk context
    if risk_components:
        parts.append(
            NLG_TEMPLATES["risk_context"].format(
                risk_score=risk_score, risk_level=severity, **risk_components,
            )
        )

    # Step 5: Clinical-tier note
    if d_clinical_tier_val > 0:
        n_abnormal = int(round(d_clinical_tier_val * 8))
        parts.append(
            NLG_TEMPLATES["acuity_note_abnormal"].format(n_abnormal=n_abnormal)
        )
    else:
        parts.append(NLG_TEMPLATES["acuity_note_normal"])

    # Step 6: Action recommendation
    parts.append(NLG_TEMPLATES["action_recommendation"].get(severity, ""))

    return "\n\n".join(p for p in parts if p)


# ── Stakeholder router ──────────────────────────────────────────────


def route_explanation(
    idx: int,
    stakeholder_role: str,
    sv_row: np.ndarray,
    feat_names: list,
    severity: str,
    confidence: float,
    consensus: str,
    risk_score: float,
    risk_components: dict,
    d_clinical_tier_val: float,
    dae_top_features: list,
) -> dict:
    """Route alert to correct stakeholder view."""
    if stakeholder_role == "clinician":
        return {
            "role": "clinician",
            "format": "text",
            "content": generate_clinician_alert(
                idx, sv_row, feat_names, severity, confidence, consensus,
                risk_score, risk_components, d_clinical_tier_val,
            ),
        }
    if stakeholder_role == "analyst":
        return {
            "role": "analyst",
            "format": "json",
            "content": {
                "sample_index": idx,
                "severity": severity,
                "consensus": consensus,
                "top_features_shap": _top_features_shap(sv_row, feat_names, k=5),
                "dae_top_features": dae_top_features,
                "risk_score": risk_score,
                "risk_components": risk_components,
                "charts": [
                    f"waterfall_xgboost_sample_{idx:04d}.png",
                    f"force_xgboost_sample_{idx:04d}.png",
                ],
            },
        }
    if stakeholder_role == "administrator":
        return {
            "role": "administrator",
            "format": "json",
            "content": {
                "sample_index": idx,
                "severity": severity,
                "risk_score": risk_score,
                "risk_level": severity,
                "action_required": severity in ("CRITICAL", "HIGH"),
                "global_charts": [
                    "global_importance_xgboost.png",
                    "beeswarm_xgboost.png",
                ],
            },
        }
    return {"role": stakeholder_role, "format": "text", "content": "Unknown role"}


__all__ = [
    "clinician_nlg",
    "build_shap_context",
    "generate_clinician_alert",
    "route_explanation",
]
