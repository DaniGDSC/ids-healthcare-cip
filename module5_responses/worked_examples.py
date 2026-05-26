"""Module 5 — end-to-end worked examples (Task 5.7)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np

from .executor import ActionExecutor, NotificationService
from .policy import PolicyEngine

logger = logging.getLogger(__name__)


def run_worked_examples(
    risk_data: dict,
    attack_cats: np.ndarray,
    analyst_by_idx: dict,
    clinician_by_idx: dict,
) -> list:
    """Run 3 end-to-end scenarios: CRITICAL, HIGH, LOW."""
    R = risk_data["R"]
    levels = risk_data["risk_levels"]
    y_true = risk_data["y_true"]

    engine = PolicyEngine()
    executor = ActionExecutor()
    notifier = NotificationService()

    scenarios = []
    target_tiers = ["CRITICAL", "HIGH", "LOW"]
    for tier in target_tiers:
        mask = (levels == tier) & (y_true == 1)
        if not mask.any():
            mask = levels == tier
        if not mask.any():
            continue

        idx = int(np.where(mask)[0][np.argmax(R[mask])])
        cat = str(attack_cats[idx])
        gt = "attack" if y_true[idx] == 1 else "benign"
        a_pat = float(risk_data["d_clinical_tier"][idx])

        rec = engine.recommend(
            alert_tier=tier,
            device_tier="vital_monitoring",
            attack_category=cat,
            patient_acuity=a_pat,
        )

        ts = datetime(2026, 4, 3, 12, 0, 0) + timedelta(seconds=idx)
        exec_result = executor.execute(
            f"ALERT-{idx:05d}", idx, rec["actions"], rec, gt, ts,
        )

        clin_summary = clinician_by_idx.get(idx, {}).get("summary", "")
        analyst_feats = []
        if idx in analyst_by_idx:
            analyst_feats = (
                analyst_by_idx[idx]
                .get("models", {})
                .get("xgboost", {})
                .get("top_features", [])
            )

        notifications = notifier.notify(
            idx, tier, rec, clin_summary, analyst_feats, float(R[idx]),
        )

        scenario = {
            "scenario": f"{tier} alert — {cat} on vital_monitoring device",
            "sample_index": idx,
            "ground_truth": gt,
            "attack_category": cat,
            "risk_score": round(float(R[idx]), 4),
            "risk_level": tier,
            "components": {
                "C_detect": round(float(risk_data["c_detect"][idx]), 4),
                "D_crit": round(float(risk_data["d_crit"][idx]), 4),
                "S_data": round(float(risk_data["s_data"][idx]), 4),
                "D_clinical_tier": round(float(risk_data["d_clinical_tier"][idx]), 4),
            },
            "policy_recommendation": rec,
            "execution_result": exec_result,
            "notifications": notifications,
            "clinical_override": rec["clinical_override"],
        }
        scenarios.append(scenario)
        logger.info(
            "  %s: sample %d, R=%.4f, actions=%s, outcome=%s",
            tier, idx, float(R[idx]), rec["actions"], exec_result["outcome"],
        )

    return scenarios


__all__ = ["run_worked_examples"]
