"""Adapter: pipeline evaluation-alert dict → src/ prototype inputs.

Bridges `evaluation_alerts.json` records (produced by
pipeline/module6_evaluation/module6_evaluation.py) into the 3-component
research prototype in src/ — currently src.risk_scorer.score_alert().

Fields missing from the pipeline artifact (patchable, event_context) fall
back to safe defaults per research_spec.yaml component_2.behavior_rules:
  - patchable defaults to True (treats unknown devices as LOW-risk
    assumption; CRITICAL+unpatchable is only applied when explicitly set)
  - event_context=None (no maintenance/vendor/history data in the
    curated artifact — pipeline-side changes would be required to
    populate this without a schema change)
"""
from __future__ import annotations

from typing import Any

from src.data_models import ScoredAlert
from src.risk_scorer import score_alert


def scored_from_eval_alert(alert_data: dict[str, Any]) -> ScoredAlert:
    """Run a pipeline evaluation-alert dict through src.risk_scorer.

    Args:
        alert_data: One record from evaluation_alerts.json. Required keys:
            risk_score, device_criticality. Optional: patchable,
            affected_system.

    Returns:
        ScoredAlert with adjusted_score, threshold, should_surface,
        risk_multiplier, and optional suppression_reason.
    """
    return score_alert(
        anomaly_score=float(alert_data.get("risk_score", 0.0)),
        device_context={
            "criticality": alert_data.get("device_criticality", "LOW"),
            "patchable": bool(alert_data.get("patchable", True)),
            "clinical_function": alert_data.get("affected_system", ""),
        },
        event_context=None,
    )
