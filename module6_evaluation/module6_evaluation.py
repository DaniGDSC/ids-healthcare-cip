#!/usr/bin/env python3
"""Module 6 — Build Evaluation Artifacts (Tasks 6.1, 6.2, 6.6-6.8).

Curates evaluation alert set, computes evaluation metrics from
participant responses (or simulated data for thesis validation),
and generates thesis-ready figures.

Usage:
    python build_evaluation.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

# Project root on sys.path so absolute imports below resolve when this
# script is invoked directly (e.g. by run_all_modules.py via subprocess).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.risk_scorer import score_alert  # noqa: E402
from src.mve_generator import generate_mve, _generate_rule_based  # noqa: E402

# Shared device context and severity-derivation mappings live in
# common/context_mappings.py — single source of truth shared with
# Module 3 (risk scoring) per RQ1_pipeline.md §3.1.
from common.context_mappings import (  # noqa: E402
    DEVICE_CONTEXT,
    lookup_device_context,
    map_true_severity,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results/reports"
CHARTS_DIR = PROJECT_ROOT / "results/charts"

ACTIONS = ["dismiss", "monitor", "investigate", "isolate", "escalate"]

_DEVICE_TYPE_KEYWORDS = [
    ("infusion", "infusion_pump"), ("ventilator", "ventilator"),
    ("insulin", "insulin_pump"),
    ("patient monitor", "patient_monitor"), ("monitor", "patient_monitor"),
    ("ehr", "ehr_workstation"), ("pacs", "pacs_server"),
    ("pharmacy", "pharmacy_system"),
    ("workstation", "ehr_workstation"), ("server", "server"),
]


_BIO_FEATS = ('Temp', 'SpO2', 'Pulse_Rate', 'Heart_rate', 'Resp_Rate', 'ST')


def _derive_device_class(sample_index: int, test_df: pd.DataFrame) -> str:
    """Derive device class for one row.

    GAP-A7: prefers the parquet's `device_class` column (authoritative,
    written by Module 1) when present; falls back to the shared biometric
    heuristic in common.device_class otherwise. Both paths return the same
    five-class taxonomy (ventilator / patient_monitor / infusion_pump /
    ehr_workstation / other).
    """
    if "device_class" in test_df.columns:
        return str(test_df.iloc[sample_index]["device_class"])
    from common.device_class import derive_device_class_row
    return derive_device_class_row(test_df.iloc[sample_index])


def _build_group_a_display(row: pd.Series, alert_id: str, anomaly_score: float) -> str:
    """Builds the raw, uninterpretable alert text for Group A (baseline).
    
    Excludes all biometric fields. Injects safe placeholders for network markers 
    that were sanitized during WUSTL-EHMS Phase 1.
    """
    src_ip = row.get("SrcIP", "10.4.12.8")
    dst_ip = row.get("DstIP", "203.0.113.42")
    
    # In WUSTL-EHMS Phase 1, Sport is standard-scaled. Map negative/float to a generic port.
    sport = row.get("Sport", "443")
    try:
        if pd.isna(sport) or float(sport) < 0:
            sport = "54321"  # Ephemeral port mock
        else:
            sport = str(int(float(sport)))
    except ValueError:
        sport = "443"
    dport = row.get("Dport", "443")
    proto = row.get("Proto", "TCP")
    timestamp = row.get("Timestamp", "2024-01-15 10:23:45")
    
    return (
        f"Alert ID: {alert_id}\n"
        f"Anomaly score: {anomaly_score:.2f}\n"
        f"Prediction: Attack\n"
        f"Source: {src_ip}:{sport} → {dst_ip}:{dport}\n"
        f"Protocol: {proto}\n"
        f"Timestamp: {timestamp}"
    )


def _build_group_b_display(mve_output) -> str:
    """Assembles the 3-layer Minimum Viable Explanation into a readable dashboard string."""
    if not mve_output:
        return "WHY ANOMALOUS\nNot available\n\nCLINICAL SEVERITY\nNot available\n\nRECOMMENDED ACTION\nNot available"
    
    if isinstance(mve_output, dict):
        l1 = mve_output.get("layer_1", {})
        l2 = mve_output.get("layer_2", {})
        l3 = mve_output.get("layer_3", {})
    else:
        l1 = getattr(mve_output, "layer_1", {})
        l2 = getattr(mve_output, "layer_2", {})
        l3 = getattr(mve_output, "layer_3", {})
        
    t1 = " ".join(str(v) for v in l1.values() if v) if l1 else "Not available"
    t2 = " ".join(str(v) for v in l2.values() if v) if l2 else "Not available"
    t3 = " ".join(str(v) for v in l3.values() if v) if l3 else "Not available"
    
    if "DO NOT" in t3.upper():
        # Specifically prefix only instances of DO NOT for better UI emphasis
        t3 = t3.replace("DO NOT", "⚠️ DO NOT")
        
    return (
        f"WHY ANOMALOUS\n{t1}\n\n"
        f"CLINICAL SEVERITY\n{t2}\n\n"
        f"RECOMMENDED ACTION\n{t3}"
    )


# ═══════════════════════════════════════════════════════════════════════
# 6.2  Curate evaluation alert set
# ═══════════════════════════════════════════════════════════════════════

def curate_evaluation_alerts() -> list:
    """Select 20 diverse alerts spanning all tiers and attack types."""
    logger.info("Curating evaluation alert set...")

    # Per ARCHITECTURE.md Strategy 1: dashboard alerts come from the
    # frozen demo pool, NEVER from the test split. demo_scores.npz is
    # produced by module3_risk_scoring.module3_demo_scores; it shares
    # schema with risk_scores.npz but is sourced from demo_phase1.parquet.
    risk_data = np.load(PROJECT_ROOT / "results/reports/demo_scores.npz",
                        allow_pickle=True)
    R = risk_data["R"]
    levels = risk_data["risk_levels"]
    y_true = risk_data["y_true"]

    df = pd.read_parquet(PROJECT_ROOT / "data/processed/demo_phase1.parquet")
    attack_cats = df["Attack Category"].values

    with open(PROJECT_ROOT / "results/reports/analyst_report.json") as f:
        analyst_by_idx = {a["sample_index"]: a for a in json.load(f)}
    with open(PROJECT_ROOT / "results/reports/clinician_summaries.json") as f:
        clinician_by_idx = {s["sample_index"]: s for s in json.load(f)}

    # Load example explanations for full clinician NLG
    try:
        with open(PROJECT_ROOT / "results/reports/example_explanations.json") as f:
            examples_by_idx = {e["sample_index"]: e for e in json.load(f)}
    except FileNotFoundError:
        examples_by_idx = {}

    alerts = []
    used_idx: set = set()

    # M6-E2: pre-cast once — eliminates O(K×N) Python str() boxing in the loop
    cats_str = attack_cats.astype(str)
    # M6-E3: cache the index range so we don't reallocate np.arange per iteration
    all_indices = np.arange(len(R))

    def _process_and_append(idx):
        used_idx.add(idx)
        raw_row = df.iloc[idx]
        anomaly_score = float(R[idx])
        
        device_cls = _derive_device_class(idx, df)
        device_criticality = DEVICE_CONTEXT.get(device_cls, DEVICE_CONTEXT["other"]).get("device_criticality", "LOW")
        patchable = device_cls not in {"infusion_pump", "ventilator", "insulin_pump", "patient_monitor", "pacs_server"}
        
        device_context = {
            "criticality": device_criticality,
            "patchable": patchable,
            "similar_events_past_30d": 0,
            "is_maintenance_window": False,
            "is_known_vendor_ip": False,
            "device_type": device_cls
        }
        
        scored_alert = score_alert(anomaly_score, device_context, event_context=None)
        
        try:
            raw_dict = raw_row.to_dict()
        except AttributeError:
            raw_dict = dict(raw_row)
            
        try:
            mve_out = generate_mve(raw_dict, device_context, {}, None)
            if mve_out is None:
                mve_out = _generate_rule_based(raw_dict, device_context, {}, None, "T1")
        except Exception:
            mve_out = _generate_rule_based(raw_dict, device_context, {}, None, "T1")
            
        alert_id_str = f"EVAL-{idx:04d}"
        grp_a = _build_group_a_display(raw_row, alert_id_str, anomaly_score)
        grp_b = _build_group_b_display(mve_out)
            
        alerts.append(_build_eval_alert(
            idx, R, levels, y_true, attack_cats,
            analyst_by_idx, clinician_by_idx, examples_by_idx,
            test_df=df, raw_row=raw_row,
            device_criticality=device_criticality, patchable=patchable,
            scored_alert=scored_alert, mve_output=mve_out,
            group_a_display=grp_a, group_b_display=grp_b
        ))

    # Target: 4 per tier (CRITICAL, HIGH, MEDIUM, LOW) + 4 benign calibration
    tier_targets = {
        "CRITICAL": {"attack": 2, "attack_cats": ["Spoofing", "Data Alteration"]},
        "HIGH":     {"attack": 2, "attack_cats": ["Spoofing", "Data Alteration"]},
        "MEDIUM":   {"attack": 2, "attack_cats": ["Spoofing", "Data Alteration"]},
        "LOW":      {"attack": 2, "attack_cats": ["Spoofing", "Data Alteration"]},
    }

    for tier, cfg in tier_targets.items():
        tier_mask = levels == tier
        for cat in cfg["attack_cats"]:
            # M6-E2: vectorised string comparison via pre-cast array
            cat_mask = cats_str == cat
            combined = tier_mask & cat_mask & (y_true == 1)
            candidates = np.where(combined)[0]
            candidates = [c for c in candidates if c not in used_idx]

            if len(candidates) > 0:
                idx = int(candidates[np.argmax(R[candidates])])
                _process_and_append(idx)

    # Benign calibration: 4 benign at various risk levels
    for target_r in [0.20, 0.30, 0.45, 0.55]:
        # M6-E3: pass set directly to np.isin (no list() copy); reuse all_indices
        benign_mask = (y_true == 0) & (~np.isin(all_indices, used_idx))
        candidates = np.where(benign_mask)[0]
        if len(candidates) == 0:
            continue
        # Pick closest to target_r
        idx = int(candidates[np.argmin(np.abs(R[candidates] - target_r))])
        _process_and_append(idx)

    # Fill remaining to reach 20
    while len(alerts) < 20:
        # M6-E3: reuse all_indices, pass set to np.isin
        remaining = np.where(~np.isin(all_indices, used_idx))[0]
        if len(remaining) == 0:
            break
        idx = int(remaining[np.argmax(R[remaining])])
        _process_and_append(idx)

    logger.info("  Curated %d evaluation alerts", len(alerts))
    # M6-E4: Counter replaces manual get/+1 dict accumulation
    tier_counts = Counter(a["risk_level"] for a in alerts)
    logger.info("  By tier: %s", dict(tier_counts))

    return alerts[:20]


# NOTE: ``map_true_severity`` has moved to ``common.context_mappings``
# (RQ1_pipeline.md §3.1) and is re-exported via the top-of-file import.


def _build_eval_alert(idx, R, levels, y_true, attack_cats,
                      analyst_by_idx, clinician_by_idx, examples_by_idx,
                      test_df=None, raw_row=None, device_criticality=None, patchable=None,
                      scored_alert=None, mve_output=None, group_a_display=None,
                      group_b_display=None) -> dict:
    """Build a single evaluation alert with all context."""
    analyst = analyst_by_idx.get(idx, {})
    clinician = clinician_by_idx.get(idx, {})
    xgb_top = analyst.get("models", {}).get("xgboost", {}).get("top_features", [])
    dae_top = analyst.get("models", {}).get("dae", {}).get("top_features", [])

    # UX-X-01/X-02: Derive device context
    device_cls = "other"
    if test_df is not None:
        device_cls = _derive_device_class(idx, test_df)
    ctx = DEVICE_CONTEXT.get(device_cls, DEVICE_CONTEXT["other"])

    # Resolve new parameters
    if raw_row is None and test_df is not None:
        raw_row = test_df.iloc[idx]
        
    if device_criticality is None:
        device_criticality = ctx.get("device_criticality", "LOW")
        
    if patchable is None:
        # legacy/critical medical systems are unlikely to be patchable
        patchable = device_cls not in {"infusion_pump", "ventilator", "insulin_pump", "patient_monitor", "pacs_server"}

    attack_category = str(attack_cats[idx])

    alert_dict = {
        "alert_id": f"EVAL-{idx:04d}",
        "sample_index": int(idx),
        "ground_truth": "attack" if y_true[idx] == 1 else "benign",
        "attack_category": attack_category,
        "risk_score": round(float(R[idx]), 4),
        "risk_level": str(levels[idx]),
        "device_class": device_cls,
        "true_severity": map_true_severity(attack_category, device_cls),
        "device_criticality": device_criticality,
        "device_patchable": patchable,
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
    
    if group_a_display:
        alert_dict["group_a_display"] = group_a_display
        
    if group_b_display:
        alert_dict["group_b_display"] = group_b_display
    
    if scored_alert:
        alert_dict["adjusted_score"] = scored_alert.adjusted_score
        alert_dict["should_surface"] = scored_alert.should_surface
        alert_dict["threshold"] = scored_alert.threshold
        alert_dict["risk_multiplier"] = scored_alert.risk_multiplier
        alert_dict["fusion_class"] = scored_alert.fusion_class.value
        alert_dict["data_quality"] = scored_alert.data_quality.value

    if mve_output:
        alert_dict["mve_structured"] = {
            "layer_1": mve_output.layer_1 if isinstance(mve_output, dict) else getattr(mve_output, "layer_1", {}),
            "layer_2": mve_output.layer_2 if isinstance(mve_output, dict) else getattr(mve_output, "layer_2", {}),
            "layer_3": mve_output.layer_3 if isinstance(mve_output, dict) else getattr(mve_output, "layer_3", {})
        }
        # ARCHITECTURE.md Step [12] Mode A reproducibility — surface
        # the LLM mode + provider/model on the alert so M6's "Rule-based
        # fallback" badge can render and the audit log can replay.
        mode_used = (
            mve_output.get("mode_used") if isinstance(mve_output, dict)
            else getattr(mve_output, "mode_used", "B_rule")
        )
        alert_dict["mve_mode"] = mode_used or "B_rule"
        alert_dict["llm_provider"] = (
            mve_output.get("llm_provider") if isinstance(mve_output, dict)
            else getattr(mve_output, "llm_provider", None)
        )
        alert_dict["llm_model_version"] = (
            mve_output.get("llm_model_version") if isinstance(mve_output, dict)
            else getattr(mve_output, "llm_model_version", None)
        )

    # ARCHITECTURE.md Step [11] SHAP stability indicator. The dashboard
    # reads ``stability_score`` + ``is_stable`` to decide whether to
    # render a fragility warning. ``shap_source`` flags NOVEL alerts as
    # ``xgboost_low_confidence`` (DAE-driven; XGB SHAP is not faithful).
    shap_ctx = analyst.get("shap_context") or {}
    fusion = (scored_alert.fusion_class.value if scored_alert
              else alert_dict.get("fusion_class", ""))
    novel = fusion in {"NOVEL_ANOMALY", "STRONG_NOVEL_ANOMALY"}
    stability = float(shap_ctx.get("stability_score", 1.0))
    alert_dict["shap_stability_score"] = round(stability, 4)
    alert_dict["shap_is_stable"] = bool(stability >= 0.90)
    alert_dict["shap_source"] = (
        "xgboost_low_confidence" if novel else "xgboost"
    )

    # ARCHITECTURE.md Step [13] / INVARIANT 9 — shared anchor block.
    # Identical across the 3 role views; lives at the top of every
    # operator panel to prevent miscommunication during phone-based
    # incident handling.
    from datetime import datetime, timezone
    # ``active_device`` is sometimes a boolean flag (legacy schema)
    # and sometimes a string device label — coerce to a sensible
    # device_id by falling back to the class name unless the field
    # already carries a non-empty string.
    raw_active = alert_dict.get("active_device")
    if isinstance(raw_active, str) and raw_active:
        device_id_str = raw_active
    else:
        device_id_str = str(device_cls)
    alert_dict["shared_anchor"] = {
        "alert_id":         alert_dict["alert_id"],
        "risk_tier":        alert_dict["risk_level"],
        "device_id":        device_id_str,
        "one_line_summary": (
            f"{alert_dict['risk_level']} {attack_category} on {device_cls}"
        ),
        "timestamp":        datetime.now(tz=timezone.utc).isoformat(),
    }
    
    if raw_row is not None:
        try:
            alert_dict["raw_features"] = raw_row.to_dict()
        except AttributeError:
            alert_dict["raw_features"] = dict(raw_row)
            
    return alert_dict


def _ground_truth_action(tier: str, is_attack: bool) -> str:
    """Optimal action based on ground truth."""
    if not is_attack:
        return "dismiss"
    if tier in ("CRITICAL", "HIGH"):
        return "isolate"
    if tier == "MEDIUM":
        return "investigate"
    return "monitor"


# ═══════════════════════════════════════════════════════════════════════
# 6.6  Compute evaluation metrics (from participant responses)
# ═══════════════════════════════════════════════════════════════════════

def generate_simulated_responses(alerts: list, n_participants: int = 15) -> list:
    """Generate simulated participant responses for thesis validation.

    In production, these come from the Streamlit evaluation app.
    For thesis development, we simulate realistic responses.
    """
    rng = np.random.RandomState(42)
    responses = []

    roles = (["analyst"] * 5 + ["clinician"] * 5 + ["administrator"] * 5)

    for p_idx in range(n_participants):
        role = roles[p_idx]
        for a_idx, alert in enumerate(alerts):
            is_attack = alert["ground_truth"] == "attack"
            has_xai = a_idx < 10  # first 10 with XAI, last 10 without

            # Simulate: XAI improves accuracy and confidence
            base_accuracy = 0.70 if not has_xai else 0.88
            base_trust = 3.0 if not has_xai else 4.2
            base_time = 45 if not has_xai else 28

            # Role effects
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


def compute_evaluation_metrics(responses: list) -> dict:
    """Compute inter-rater reliability, mean Likert, accuracy, time."""
    df = pd.DataFrame(responses)

    metrics = {"n_participants": df["participant_id"].nunique(),
               "n_alerts": df["alert_id"].nunique(),
               "n_responses": len(df)}

    # Per-condition aggregates
    for condition in ["with_xai", "without_xai"]:
        cond_df = df[df["condition"] == condition]
        metrics[condition] = {
            "decision_accuracy": round(float(cond_df["decision_correct"].mean()), 4),
            "mean_decision_time_sec": round(float(cond_df["decision_time_sec"].mean()), 1),
            "mean_confidence": round(float(cond_df["confidence"].mean()), 2),
            "likert_trust": round(float(cond_df["likert_trust"].mean()), 2),
            "likert_usefulness": round(float(cond_df["likert_usefulness"].mean()), 2),
            "likert_comprehensibility": round(float(cond_df["likert_comprehensibility"].mean()), 2),
            "likert_actionability": round(float(cond_df["likert_actionability"].mean()), 2),
        }

    # Per-role breakdown
    metrics["per_role"] = {}
    for role in df["participant_role"].unique():
        role_df = df[df["participant_role"] == role]
        metrics["per_role"][role] = {
            "with_xai_accuracy": round(float(role_df[role_df["condition"] == "with_xai"]["decision_correct"].mean()), 4),
            "without_xai_accuracy": round(float(role_df[role_df["condition"] == "without_xai"]["decision_correct"].mean()), 4),
        }

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# 6.7  Statistical analysis
# ═══════════════════════════════════════════════════════════════════════

def statistical_analysis(responses: list) -> dict:
    """Paired Wilcoxon signed-rank test: with-XAI vs without-XAI."""
    from scipy.stats import wilcoxon

    df = pd.DataFrame(responses)
    results = {}

    # Pair by participant: mean score with XAI vs without for each participant
    for measure in ["decision_correct", "decision_time_sec", "confidence",
                    "likert_trust", "likert_usefulness", "likert_comprehensibility",
                    "likert_actionability"]:
        paired_with = df[df["condition"] == "with_xai"].groupby("participant_id")[measure].mean()
        paired_without = df[df["condition"] == "without_xai"].groupby("participant_id")[measure].mean()

        common = paired_with.index.intersection(paired_without.index)
        if len(common) < 3:
            continue

        a = paired_with.loc[common].values
        b = paired_without.loc[common].values
        diff = a - b

        if np.all(diff == 0):
            results[measure] = {"statistic": 0, "p_value": 1.0, "significant": False,
                                "mean_with": round(float(a.mean()), 4),
                                "mean_without": round(float(b.mean()), 4)}
            continue

        stat, p_val = wilcoxon(a, b)

        # Cohen's d
        pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
        cohens_d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0

        results[measure] = {
            "mean_with_xai": round(float(a.mean()), 4),
            "mean_without_xai": round(float(b.mean()), 4),
            "difference": round(float(a.mean() - b.mean()), 4),
            "wilcoxon_statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": p_val < 0.05,
            "cohens_d": round(float(cohens_d), 4),
            "effect_size": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small",
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# 6D.5  Inter-rater reliability (Krippendorff's alpha)
# ═══════════════════════════════════════════════════════════════════════

def compute_inter_rater_reliability(responses: list) -> dict:
    """Compute Krippendorff's alpha for action selection and Likert scores."""
    df = pd.DataFrame(responses)

    # Build reliability matrices: participants × alerts
    results = {}

    for measure in ["chosen_action", "likert_trust", "likert_usefulness",
                    "likert_comprehensibility", "likert_actionability"]:
        pivot = df.pivot_table(index="participant_id", columns="alert_id",
                               values=measure, aggfunc="first")

        if measure == "chosen_action":
            # Encode actions as integers
            action_map = {a: i for i, a in enumerate(ACTIONS)}
            pivot = pivot.map(lambda x: action_map.get(x, -1) if isinstance(x, str) else x)

        matrix = pivot.values.astype(float)
        # Replace NaN with np.nan for Krippendorff
        matrix = np.where(np.isnan(matrix), np.nan, matrix)

        # Simple Krippendorff's alpha approximation (without external lib)
        # Using observed vs expected disagreement
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
            results[measure] = {"alpha": 0.0, "n_coders": n_coders, "n_items": n_items}
            continue

        Do = observed_disagree / valid_pairs

        # M6-E6: vectorised expected disagreement — O(V×N + V) replacing O(V²×N).
        # De = 1 - sum(p_v²) = 1 - ||p||²  (complement of Herfindahl index).
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
            "interpretation": "good" if alpha > 0.67 else "moderate" if alpha > 0.33 else "poor",
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# 6D.7  Thematic feedback analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_feedback(responses: list) -> dict:
    """Thematic analysis of free-text feedback and reclassifications."""
    df = pd.DataFrame(responses)

    # Reclassification analysis
    reclass = df[df["reclassification"].notna() & (df["reclassification"] != "")]
    reclass_counts = reclass["reclassification"].value_counts().to_dict() if len(reclass) > 0 else {}

    # Keyword-based thematic extraction from feedback
    themes = {
        "wanted_more_detail": ["more detail", "more info", "explain more", "unclear"],
        "nlg_helpful": ["helpful", "clear", "understandable", "good explanation"],
        "too_technical": ["technical", "jargon", "confusing", "complex"],
        "shap_useful": ["shap", "feature", "contribution", "waterfall"],
        "trust_concern": ["trust", "confident", "unsure", "doubt"],
        "action_unclear": ["action", "what to do", "response", "next step"],
    }

    feedback_texts = df[df["feedback"].notna() & (df["feedback"] != "")]["feedback"].tolist()
    theme_counts = {theme: 0 for theme in themes}

    for text in feedback_texts:
        text_lower = text.lower()
        for theme, keywords in themes.items():
            if any(kw in text_lower for kw in keywords):
                theme_counts[theme] += 1

    # User corrections for Module 3-5 feedback
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


# ═══════════════════════════════════════════════════════════════════════
# 6.8  Thesis figures
# ═══════════════════════════════════════════════════════════════════════

def generate_thesis_figures(metrics: dict, stats: dict, responses: list) -> None:
    """Generate thesis-ready figures."""
    df = pd.DataFrame(responses)

    # Figure 1: Likert scores comparison (with vs without XAI)
    dimensions = ["trust", "usefulness", "comprehensibility", "actionability"]
    with_scores = [metrics["with_xai"][f"likert_{d}"] for d in dimensions]
    without_scores = [metrics["without_xai"][f"likert_{d}"] for d in dimensions]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dimensions))
    w = 0.35
    bars1 = ax.bar(x - w/2, without_scores, w, label="Without XAI", color="#95a5a6", alpha=0.8)
    bars2 = ax.bar(x + w/2, with_scores, w, label="With XAI", color="#3274A1", alpha=0.8)

    for bar, val in zip(bars1, without_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, with_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", fontsize=9)

    # Mark significant differences
    for i, d in enumerate(dimensions):
        key = f"likert_{d}"
        if key in stats and stats[key].get("significant"):
            ax.annotate("*", xy=(i, max(with_scores[i], without_scores[i]) + 0.3),
                       ha="center", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dimensions])
    ax.set_ylabel("Mean Likert Score (1-5)")
    ax.set_title("Explanation Quality: With vs Without XAI")
    ax.set_ylim(0, 5.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "likert_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: likert_comparison.png")

    # Figure 2: Decision accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    acc_with = metrics["with_xai"]["decision_accuracy"] * 100
    acc_without = metrics["without_xai"]["decision_accuracy"] * 100
    bars = ax.bar(["Without XAI", "With XAI"], [acc_without, acc_with],
                  color=["#95a5a6", "#3274A1"], alpha=0.8, width=0.5)
    for bar, val in zip(bars, [acc_without, acc_with]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Decision Accuracy (%)")
    ax.set_title("Decision Accuracy: With vs Without XAI Explanations")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: accuracy_comparison.png")

    # Figure 3: Decision time boxplots
    fig, ax = plt.subplots(figsize=(8, 6))
    data_with = df[df["condition"] == "with_xai"]["decision_time_sec"].values
    data_without = df[df["condition"] == "without_xai"]["decision_time_sec"].values
    bp = ax.boxplot([data_without, data_with],
                    tick_labels=["Without XAI", "With XAI"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#95a5a6")
    bp["boxes"][1].set_facecolor("#3274A1")
    for b in bp["boxes"]:
        b.set_alpha(0.7)
    ax.set_ylabel("Decision Time (seconds)")
    ax.set_title("Time-to-Decision: With vs Without XAI")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "decision_time_boxplot.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: decision_time_boxplot.png")

    # Figure 4: Per-role accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    roles = sorted(metrics["per_role"].keys())
    x = np.arange(len(roles))
    w = 0.35
    acc_with_role = [metrics["per_role"][r]["with_xai_accuracy"] * 100 for r in roles]
    acc_without_role = [metrics["per_role"][r]["without_xai_accuracy"] * 100 for r in roles]
    ax.bar(x - w/2, acc_without_role, w, label="Without XAI", color="#95a5a6", alpha=0.8)
    ax.bar(x + w/2, acc_with_role, w, label="With XAI", color="#3274A1", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in roles])
    ax.set_ylabel("Decision Accuracy (%)")
    ax.set_title("Decision Accuracy by Stakeholder Role")
    ax.set_ylim(0, 105)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "accuracy_by_role.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: accuracy_by_role.png")

    # Figure 5: Radar chart — 4 Likert dimensions × 3 roles (with XAI)
    _plot_radar_chart(df, stats)

    # Figure 6: Decision time by tier
    _plot_decision_time_by_tier(df)

    # Figure 7: Per-tier accuracy comparison
    _plot_accuracy_by_tier(df)

    # Figure 8: Effect size forest plot
    _plot_effect_sizes(stats)


def _plot_radar_chart(df: pd.DataFrame, stats: dict) -> None:
    """Radar chart: 4 Likert dimensions × 3 roles (with-XAI condition)."""
    dimensions = ["trust", "usefulness", "comprehensibility", "actionability"]
    labels = [d.capitalize() for d in dimensions]
    roles = sorted(df["participant_role"].unique())
    role_colors = {"analyst": "#3274A1", "clinician": "#2ecc71", "administrator": "#e67e22"}

    # Compute means per role (with XAI only)
    with_df = df[df["condition"] == "with_xai"]
    role_means = {}
    for role in roles:
        rdf = with_df[with_df["participant_role"] == role]
        role_means[role] = [float(rdf[f"likert_{d}"].mean()) for d in dimensions]

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for role in roles:
        values = role_means[role] + role_means[role][:1]
        ax.plot(angles, values, "o-", linewidth=2,
                color=role_colors.get(role, "#999"),
                label=role.capitalize())
        ax.fill(angles, values, alpha=0.15,
                color=role_colors.get(role, "#999"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
    ax.set_title("Likert Ratings by Role (With XAI)", y=1.08, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "radar_likert_by_role.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("  Chart: radar_likert_by_role.png")


def _plot_decision_time_by_tier(df: pd.DataFrame) -> None:
    """Boxplot of decision time grouped by alert tier and A/B condition."""
    # Merge alert tier info
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = ["without_xai", "with_xai"]
    colors = ["#95a5a6", "#3274A1"]
    positions = []
    data_groups = []
    tick_labels = []

    for i, cond in enumerate(conditions):
        cond_df = df[df["condition"] == cond]
        for j, role in enumerate(sorted(df["participant_role"].unique())):
            role_df = cond_df[cond_df["participant_role"] == role]
            data_groups.append(role_df["decision_time_sec"].values)
            positions.append(j * 3 + i)
            if i == 0:
                tick_labels.append(role.capitalize())

    bp = ax.boxplot(data_groups, positions=positions, widths=0.8,
                     patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % 2])
        patch.set_alpha(0.7)

    # Set ticks at center of each group
    ax.set_xticks([j * 3 + 0.5 for j in range(len(tick_labels))])
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Decision Time (seconds)")
    ax.set_title("Decision Time by Role and Condition")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#95a5a6", alpha=0.7, label="Without XAI"),
        Patch(facecolor="#3274A1", alpha=0.7, label="With XAI"),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "decision_time_by_role.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: decision_time_by_role.png")


def _plot_accuracy_by_tier(df: pd.DataFrame) -> None:
    """Per-condition accuracy broken down by correct_action (proxy for tier).

    M6-E8: groupby computes all (condition × action) means in one pass,
    replacing nested O(conditions × actions × N) double-filter.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    action_order = ["dismiss", "monitor", "investigate", "isolate", "escalate"]
    action_labels = [a.capitalize() for a in action_order]

    x = np.arange(len(action_order))
    w = 0.35

    # Single groupby pass — (condition, correct_action) → mean decision_correct
    acc_table = (
        df.groupby(["condition", "correct_action"])["decision_correct"]
        .mean()
        .unstack(level="correct_action", fill_value=0.0)
    )

    for i, (cond, color, label) in enumerate([
        ("without_xai", "#95a5a6", "Without XAI"),
        ("with_xai", "#3274A1", "With XAI"),
    ]):
        cond_row = acc_table.loc[cond] if cond in acc_table.index else None
        accs = [
            float(cond_row[a]) * 100 if (cond_row is not None and a in cond_row) else 0.0
            for a in action_order
        ]
        offset = -w / 2 + i * w
        bars = ax.bar(x + offset, accs, w, label=label, color=color, alpha=0.8)
        for bar, val in zip(bars, accs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(action_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Decision Accuracy by Correct Action (Tier Proxy)")
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "accuracy_by_tier.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: accuracy_by_tier.png")


def _plot_effect_sizes(stats: dict) -> None:
    """Forest plot of Cohen's d effect sizes for all measures."""
    measures = []
    effects = []
    ci_lo = []
    ci_hi = []

    for measure, result in stats.items():
        if "cohens_d" not in result:
            continue
        d = result["cohens_d"]
        # Approximate 95% CI for Cohen's d: d ± 1.96 * SE(d)
        # SE(d) ≈ sqrt(2/n + d^2/(2*n)) for paired, n≈15
        n = 15
        se = np.sqrt(2 / n + d ** 2 / (2 * n))
        measures.append(measure.replace("likert_", "").replace("_", " ").title())
        effects.append(d)
        ci_lo.append(d - 1.96 * se)
        ci_hi.append(d + 1.96 * se)

    if not measures:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(measures) * 0.6)))
    y = np.arange(len(measures))

    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in effects]
    ax.barh(y, effects, color=colors, alpha=0.7, height=0.5)
    ax.errorbar(effects, y, xerr=[
        [e - lo for e, lo in zip(effects, ci_lo)],
        [hi - e for e, hi in zip(effects, ci_hi)],
    ], fmt="none", ecolor="black", capsize=3)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axvline(-0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axvline(0.8, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axvline(-0.8, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(measures)
    ax.set_xlabel("Cohen's d (With XAI − Without XAI)")
    ax.set_title("Effect Sizes: XAI Impact on Evaluation Measures")
    ax.text(0.55, -0.6, "medium", fontsize=7, color="gray", ha="center")
    ax.text(0.85, -0.6, "large", fontsize=7, color="gray", ha="center")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "effect_size_forest.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: effect_size_forest.png")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sep = "=" * 72
    t0 = time.perf_counter()

    logger.info(sep)
    logger.info("MODULE 6 — BUILD EVALUATION ARTIFACTS")
    logger.info(sep)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # 6.2 Curate evaluation alerts
    alerts = curate_evaluation_alerts()
    alerts_path = OUTPUT_DIR / "evaluation_alerts.json"
    alerts_path.write_text(json.dumps(alerts, indent=2), encoding="utf-8")
    logger.info("6.2 Saved: evaluation_alerts.json (%d alerts)", len(alerts))

    # Generate simulated responses (replace with real data from evaluation_app.py)
    logger.info("")
    logger.info("Generating simulated participant responses (for thesis validation)...")
    responses = generate_simulated_responses(alerts)
    resp_path = OUTPUT_DIR / "participant_responses.json"
    resp_path.write_text(json.dumps(responses, indent=2), encoding="utf-8")
    logger.info("  Saved: participant_responses.json (%d responses)", len(responses))

    # 6.6 Compute metrics
    logger.info("")
    logger.info("── 6.6 Evaluation Metrics ──")
    metrics = compute_evaluation_metrics(responses)
    logger.info("  With XAI:    accuracy=%.1f%%, trust=%.1f, time=%.0fs",
                metrics["with_xai"]["decision_accuracy"] * 100,
                metrics["with_xai"]["likert_trust"],
                metrics["with_xai"]["mean_decision_time_sec"])
    logger.info("  Without XAI: accuracy=%.1f%%, trust=%.1f, time=%.0fs",
                metrics["without_xai"]["decision_accuracy"] * 100,
                metrics["without_xai"]["likert_trust"],
                metrics["without_xai"]["mean_decision_time_sec"])

    # 6.7 Statistical analysis
    logger.info("")
    logger.info("── 6.7 Statistical Analysis ──")
    stats = statistical_analysis(responses)
    for measure, result in stats.items():
        sig = "***" if result.get("significant") else "n.s."
        logger.info("  %s: Δ=%.3f, p=%.4f %s (d=%.2f, %s)",
                    measure, result.get("difference", 0), result.get("p_value", 1),
                    sig, result.get("cohens_d", 0), result.get("effect_size", ""))

    # 6D.5 Inter-rater reliability
    logger.info("")
    logger.info("── 6D.5 Inter-Rater Reliability ──")
    irr = compute_inter_rater_reliability(responses)
    for measure, result in irr.items():
        logger.info("  %s: alpha=%.4f (%s)", measure, result["alpha"], result.get("interpretation", ""))

    # 6D.7 Thematic feedback analysis
    logger.info("")
    logger.info("── 6D.7 Feedback Analysis ──")
    feedback = analyze_feedback(responses)
    logger.info("  Feedback texts: %d, Reclassifications: %d",
                feedback["total_feedback_texts"], feedback["n_reclassifications"])
    logger.info("  Themes: %s", feedback["thematic_counts"])

    # Save all results
    eval_results = {
        "metrics": metrics,
        "statistical_tests": stats,
        "inter_rater_reliability": irr,
        "feedback_analysis": feedback,
    }
    (OUTPUT_DIR / "evaluation_results.json").write_text(
        json.dumps(eval_results, indent=2, default=str), encoding="utf-8")
    logger.info("  Saved: evaluation_results.json")

    # 6D.8 Feedback recommendations
    (OUTPUT_DIR / "feedback_recommendations.json").write_text(
        json.dumps({
            "corrections": feedback["corrections_for_modules_3_5"],
            "thematic_recommendations": feedback["recommendations"],
            "reclassification_summary": feedback["reclassification_counts"],
        }, indent=2, default=str), encoding="utf-8")
    logger.info("  Saved: feedback_recommendations.json")

    # 6.8 Thesis figures
    logger.info("")
    logger.info("── 6.8 Thesis Figures ──")
    generate_thesis_figures(metrics, stats, responses)

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("EVALUATION BUILD COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    logger.info("  Protocol   : data/phase2/evaluation/evaluation_protocol.md")
    logger.info("  Alerts     : evaluation_alerts.json (%d)", len(alerts))
    logger.info("  Responses  : participant_responses.json (%d)", len(responses))
    logger.info("  Results    : evaluation_results.json")
    logger.info("  Charts     : %s", CHARTS_DIR)
    logger.info(sep)


if __name__ == "__main__":
    main()
