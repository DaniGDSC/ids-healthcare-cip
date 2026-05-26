"""Evaluation alert curation — Task 6.2."""
from __future__ import annotations

import json
import logging
from collections import Counter

import numpy as np
import pandas as pd

from common.device_class import (
    DEVICE_CONTEXT,
    derive_device_class_row as _derive_device_class_row,
)

logger = logging.getLogger(__name__)

ACTIONS = ["dismiss", "monitor", "investigate", "isolate", "escalate"]


def _derive_device_class(sample_index: int, test_df: pd.DataFrame) -> str:
    """Per-row wrapper kept as a thin shim over ``common.device_class``."""
    return _derive_device_class_row(test_df.iloc[sample_index])


def _curate_split_paths(split: str) -> dict:
    """Resolve per-split inputs + output suffix for evaluation-alert curation."""
    from common import split_paths as sp
    return {
        "risk_npz":  sp.risk_scores(split),
        "parquet":   sp.parquet(split),
        "analyst":   sp.analyst_report(split),
        "clinician": sp.clinician_summaries(split),
        "examples":  sp.example_explanations(split),
        "suffix":    sp.suffix(split),
    }


def _ground_truth_action(tier: str, is_attack: bool) -> str:
    """Optimal action based on ground truth."""
    if not is_attack:
        return "dismiss"
    if tier in ("CRITICAL", "HIGH"):
        return "isolate"
    if tier == "MEDIUM":
        return "investigate"
    return "monitor"


def _build_eval_alert(idx, R, levels, y_true, attack_cats,
                      analyst_by_idx, clinician_by_idx, examples_by_idx,
                      test_df=None) -> dict:
    """Build a single evaluation alert with all context."""
    analyst = analyst_by_idx.get(idx, {})
    clinician = clinician_by_idx.get(idx, {})
    xgb_top = analyst.get("models", {}).get("xgboost", {}).get("top_features", [])
    dae_top = analyst.get("models", {}).get("dae", {}).get("top_features", [])

    device_cls = "other"
    if test_df is not None:
        device_cls = _derive_device_class(idx, test_df)
    ctx = DEVICE_CONTEXT.get(device_cls, DEVICE_CONTEXT["other"])

    return {
        "alert_id": f"EVAL-{idx:04d}",
        "sample_index": int(idx),
        "ground_truth": "attack" if y_true[idx] == 1 else "benign",
        "attack_category": str(attack_cats[idx]),
        "risk_score": round(float(R[idx]), 4),
        "risk_level": str(levels[idx]),
        "device_class": device_cls,
        "device_criticality": ctx["device_criticality"],
        "affected_system": ctx["affected_system"],
        "patient_care_impact": ctx["patient_care_impact"],
        "active_device": ctx["active_device"],
        "xai_explanation": {
            "xgboost_top_features": xgb_top,
            "dae_top_features": dae_top,
            "consensus": analyst.get("consensus", ""),
            "clinician_summary": clinician.get("summary", ""),
        },
        "correct_action": _ground_truth_action(str(levels[idx]), y_true[idx] == 1),
    }


def curate_evaluation_alerts(split: str = "test") -> list:
    """Select 20 diverse alerts spanning all tiers and attack types."""
    paths = _curate_split_paths(split)
    logger.info("Curating evaluation alert set (split=%s)...", split)

    risk_data = np.load(paths["risk_npz"], allow_pickle=True)
    R = risk_data["R"]
    levels = risk_data["risk_levels"]
    y_true = risk_data["y_true"]

    df = pd.read_parquet(paths["parquet"])
    attack_cats = df["Attack Category"].values

    with open(paths["analyst"]) as f:
        analyst_by_idx = {a["sample_index"]: a for a in json.load(f)}
    with open(paths["clinician"]) as f:
        clinician_by_idx = {s["sample_index"]: s for s in json.load(f)}

    try:
        with open(paths["examples"]) as f:
            examples_by_idx = {e["sample_index"]: e for e in json.load(f)}
    except FileNotFoundError:
        examples_by_idx = {}

    alerts = []
    used_idx: set = set()

    cats_str = attack_cats.astype(str)
    all_indices = np.arange(len(R))

    tier_targets = {
        "CRITICAL": {"attack": 2, "attack_cats": ["Spoofing", "Data Alteration"]},
        "HIGH":     {"attack": 2, "attack_cats": ["Spoofing", "Data Alteration"]},
        "MEDIUM":   {"attack": 2, "attack_cats": ["Spoofing", "Data Alteration"]},
        "LOW":      {"attack": 2, "attack_cats": ["Spoofing", "Data Alteration"]},
    }

    for tier, cfg in tier_targets.items():
        tier_mask = levels == tier
        for cat in cfg["attack_cats"]:
            cat_mask = cats_str == cat
            combined = tier_mask & cat_mask & (y_true == 1)
            candidates = np.where(combined)[0]
            candidates = [c for c in candidates if c not in used_idx]

            if len(candidates) > 0:
                idx = int(candidates[np.argmax(R[candidates])])
                used_idx.add(idx)
                alerts.append(_build_eval_alert(
                    idx, R, levels, y_true, attack_cats,
                    analyst_by_idx, clinician_by_idx, examples_by_idx,
                    test_df=df,
                ))

    for target_r in [0.20, 0.30, 0.45, 0.55]:
        benign_mask = (y_true == 0) & (~np.isin(all_indices, used_idx))
        candidates = np.where(benign_mask)[0]
        if len(candidates) == 0:
            continue
        idx = int(candidates[np.argmin(np.abs(R[candidates] - target_r))])
        used_idx.add(idx)
        alerts.append(_build_eval_alert(
            idx, R, levels, y_true, attack_cats,
            analyst_by_idx, clinician_by_idx, examples_by_idx,
        ))

    while len(alerts) < 20:
        remaining = np.where(~np.isin(all_indices, used_idx))[0]
        if len(remaining) == 0:
            break
        idx = int(remaining[np.argmax(R[remaining])])
        used_idx.add(idx)
        alerts.append(_build_eval_alert(
            idx, R, levels, y_true, attack_cats,
            analyst_by_idx, clinician_by_idx, examples_by_idx,
        ))

    logger.info("  Curated %d evaluation alerts", len(alerts))
    tier_counts = Counter(a["risk_level"] for a in alerts)
    logger.info("  By tier: %s", dict(tier_counts))

    return alerts[:20]


__all__ = [
    "ACTIONS",
    "_derive_device_class",
    "_curate_split_paths",
    "_ground_truth_action",
    "_build_eval_alert",
    "curate_evaluation_alerts",
]
