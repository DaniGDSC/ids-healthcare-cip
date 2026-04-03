#!/usr/bin/env python3
"""Module 5 — Closed-Loop Response Recommendation Engine (RQ3/RO3).

Translates risk-scored detections and explanations into proportional,
adaptive response recommendations with:
  1. Adaptive mitigation selection (magnitude + device + attack-aware)
  2. Device-constrained responses (safety-critical device protection)
  3. Attack-category-aware escalation routing
  4. FDA-style audit trail with simulated outcome tracking
  5. Closed-loop effectiveness analysis

Usage:
    python generate_responses.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results/reports"
CHARTS_DIR = PROJECT_ROOT / "results/charts"

BIOMETRIC_FEATURES = frozenset({
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
    "Heart_rate", "Resp_Rate", "ST",
})

# ── Mitigation action catalogue ────────────────────────────────────────

MITIGATION_ACTIONS = {
    "log_event": {
        "severity_floor": "LOW",
        "cost": 0.1,
        "description": "Log event to SIEM for audit trail",
        "reversible": True,
    },
    "enhanced_monitoring": {
        "severity_floor": "LOW",
        "cost": 0.2,
        "description": "Enable enhanced logging and monitoring on device",
        "reversible": True,
    },
    "re_authenticate": {
        "severity_floor": "MEDIUM",
        "cost": 0.3,
        "description": "Force device re-authentication and credential verification",
        "reversible": True,
    },
    "restrict_traffic": {
        "severity_floor": "MEDIUM",
        "cost": 0.5,
        "description": "Restrict device to essential clinical traffic only (whitelist mode)",
        "reversible": True,
    },
    "isolate_device": {
        "severity_floor": "HIGH",
        "cost": 0.8,
        "description": "Isolate device from network segment via VLAN quarantine",
        "reversible": True,
    },
    "forensic_snapshot": {
        "severity_floor": "HIGH",
        "cost": 0.4,
        "description": "Capture full packet capture and device state for forensics",
        "reversible": True,
    },
    "escalate_clinical": {
        "severity_floor": "HIGH",
        "cost": 0.7,
        "description": "Escalate to clinical staff — verify patient vitals independently",
        "reversible": False,
    },
    "escalate_incident": {
        "severity_floor": "CRITICAL",
        "cost": 1.0,
        "description": "Initiate full incident response — page CISO + on-call physician",
        "reversible": False,
    },
}

# ── Attack-category-aware escalation routing ───────────────────────────

ESCALATION_ROUTING = {
    "Spoofing": {
        "primary": "IT Security",
        "secondary": "Biomedical Engineering",
        "tertiary": None,
        "rationale": "Spoofing targets device identity — biomed must verify physical device integrity",
        "attack_specific_actions": ["re_authenticate", "restrict_traffic"],
    },
    "Data Alteration": {
        "primary": "IT Security",
        "secondary": "Charge Nurse",
        "tertiary": "On-call Physician",
        "rationale": "Data alteration may corrupt biometric readings — clinical verification required",
        "attack_specific_actions": ["isolate_device", "forensic_snapshot", "escalate_clinical"],
    },
    "normal": {
        "primary": None,
        "secondary": None,
        "tertiary": None,
        "rationale": "No attack detected",
        "attack_specific_actions": [],
    },
}
DEFAULT_ROUTING = {
    "primary": "IT Security",
    "secondary": "Incident Commander",
    "tertiary": None,
    "rationale": "Unknown attack type — follow general incident response protocol",
    "attack_specific_actions": ["restrict_traffic", "forensic_snapshot"],
}

# ── Device constraint tiers ────────────────────────────────────────────

DEVICE_TIERS = {
    "life_sustaining": {
        "max_action": "restrict_traffic",  # NEVER full isolate
        "fallback_required": True,
        "clinical_escalation_mandatory": True,
        "examples": "infusion pump, ventilator",
    },
    "vital_monitoring": {
        "max_action": "isolate_device",  # can isolate WITH fallback note
        "fallback_required": True,
        "clinical_escalation_mandatory": False,
        "examples": "ECG monitor, pulse oximeter",
    },
    "diagnostic": {
        "max_action": "isolate_device",
        "fallback_required": False,
        "clinical_escalation_mandatory": False,
        "examples": "blood pressure monitor, thermometer",
    },
    "auxiliary": {
        "max_action": "isolate_device",
        "fallback_required": False,
        "clinical_escalation_mandatory": False,
        "examples": "environmental sensor, room monitor",
    },
}
DEFAULT_DEVICE_TIER = "vital_monitoring"

# ── Base response protocol ─────────────────────────────────────────────

BASE_PROTOCOL = {
    "CRITICAL": {
        "priority": 1,
        "base_actions": ["isolate_device", "escalate_incident", "forensic_snapshot", "escalate_clinical"],
        "max_response_min": 5,
    },
    "HIGH": {
        "priority": 2,
        "base_actions": ["isolate_device", "forensic_snapshot", "enhanced_monitoring"],
        "max_response_min": 15,
    },
    "MEDIUM": {
        "priority": 3,
        "base_actions": ["restrict_traffic", "enhanced_monitoring"],
        "max_response_min": 60,
    },
    "LOW": {
        "priority": 4,
        "base_actions": ["log_event", "enhanced_monitoring"],
        "max_response_min": 480,
    },
    "NORMAL": {
        "priority": 5,
        "base_actions": ["log_event"],
        "max_response_min": 0,
    },
}


# ── Data loading ────────────────────────────────────────────────────────

def load_risk_scores() -> dict:
    """Load Module 3 risk scores."""
    data = np.load(PROJECT_ROOT / "results/reports/risk_scores.npz",
                   allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_explanations() -> tuple:
    """Load Module 4 analyst reports and clinician summaries."""
    with open(PROJECT_ROOT / "results/reports/analyst_report.json") as f:
        analyst = {a["sample_index"]: a for a in json.load(f)}
    with open(PROJECT_ROOT / "results/reports/clinician_summaries.json") as f:
        clinician = {s["sample_index"]: s for s in json.load(f)}
    return analyst, clinician


def load_attack_categories() -> np.ndarray:
    df = pd.read_parquet(PROJECT_ROOT / "data/processed/test_phase1.parquet",
                         columns=["Attack Category"])
    return df["Attack Category"].values


# ── Adaptive response selection ────────────────────────────────────────

def select_adaptive_response(
    risk_level: str,
    risk_score: float,
    attack_category: str,
    device_tier: str = DEFAULT_DEVICE_TIER,
    biometric_in_top_features: bool = False,
) -> dict:
    """Select proportional response adapting to context beyond risk level."""
    base = BASE_PROTOCOL.get(risk_level, BASE_PROTOCOL["NORMAL"])
    actions = list(base["base_actions"])
    rationale_parts = [f"Base response for {risk_level} risk level"]

    # 1. Magnitude scaling
    if risk_score >= 0.70 and risk_level != "CRITICAL":
        # Escalate: add next-tier action
        if "isolate_device" not in actions:
            actions.append("isolate_device")
        if "forensic_snapshot" not in actions:
            actions.append("forensic_snapshot")
        rationale_parts.append(f"Escalated: R={risk_score:.2f} exceeds 0.70 magnitude threshold")
    elif risk_score < 0.30 and risk_level in ("MEDIUM", "HIGH"):
        # Demote: replace isolate with restrict
        if "isolate_device" in actions:
            actions.remove("isolate_device")
            if "restrict_traffic" not in actions:
                actions.append("restrict_traffic")
            rationale_parts.append(f"Demoted: R={risk_score:.2f} below 0.30, restrict instead of isolate")

    # 2. Attack-category-specific actions
    routing = ESCALATION_ROUTING.get(attack_category, DEFAULT_ROUTING)
    for action in routing["attack_specific_actions"]:
        if action not in actions:
            actions.append(action)
    if routing["attack_specific_actions"]:
        rationale_parts.append(f"Attack-specific ({attack_category}): added {routing['attack_specific_actions']}")

    # 3. Device constraints
    tier_info = DEVICE_TIERS.get(device_tier, DEVICE_TIERS["vital_monitoring"])
    max_action_cost = MITIGATION_ACTIONS[tier_info["max_action"]]["cost"]
    constrained_actions = []
    device_note = None
    for a in actions:
        if MITIGATION_ACTIONS[a]["cost"] <= max_action_cost:
            constrained_actions.append(a)
        else:
            # Downgrade to max allowed
            if tier_info["max_action"] not in constrained_actions:
                constrained_actions.append(tier_info["max_action"])
            device_note = (
                f"Device constraint ({device_tier}): {a} downgraded to "
                f"{tier_info['max_action']} — {tier_info['examples']}"
            )
    if device_note:
        rationale_parts.append(device_note)
    if tier_info["fallback_required"] and "isolate_device" in constrained_actions:
        rationale_parts.append("Fallback monitoring required before isolation")
    actions = constrained_actions

    # 4. Clinical escalation for biometric-involved alerts
    if biometric_in_top_features and "escalate_clinical" not in actions:
        actions.append("escalate_clinical")
        rationale_parts.append("Biometric features in top SHAP contributors — clinical escalation added")

    # Ensure log_event always present
    if "log_event" not in actions:
        actions.insert(0, "log_event")

    # Sort by cost (least disruptive first)
    actions = sorted(set(actions), key=lambda a: MITIGATION_ACTIONS[a]["cost"])

    return {
        "actions": actions,
        "action_descriptions": [MITIGATION_ACTIONS[a]["description"] for a in actions],
        "escalation_chain": {
            "primary": routing["primary"],
            "secondary": routing["secondary"],
            "tertiary": routing["tertiary"],
        },
        "escalation_rationale": routing["rationale"],
        "max_response_min": base["max_response_min"],
        "priority": base["priority"],
        "rationale": "; ".join(rationale_parts),
        "device_tier": device_tier,
        "device_constraint_applied": device_note is not None,
    }


# ── Audit trail ────────────────────────────────────────────────────────

def build_audit_record(
    idx: int,
    risk_score: float,
    risk_level: str,
    attack_category: str,
    ground_truth: str,
    response: dict,
    explanation_summary: str,
) -> dict:
    """Build FDA-style audit record with simulated outcome."""
    timestamp = datetime(2026, 4, 3, 12, 0, 0) + timedelta(seconds=idx)

    # Simulate outcome based on ground truth + action taken
    has_isolate = "isolate_device" in response["actions"] or "restrict_traffic" in response["actions"]
    is_true_attack = ground_truth == "attack"

    if is_true_attack and has_isolate:
        sim_outcome = "threat_contained"
        sim_effective = True
        sim_tte_sec = int(response["max_response_min"] * 60 * 0.6)  # 60% of SLA
    elif is_true_attack and not has_isolate:
        sim_outcome = "threat_logged_not_mitigated"
        sim_effective = False
        sim_tte_sec = None
    elif not is_true_attack and has_isolate:
        sim_outcome = "false_positive_isolated"
        sim_effective = False
        sim_tte_sec = int(response["max_response_min"] * 60 * 0.3)
    else:
        sim_outcome = "benign_logged"
        sim_effective = True
        sim_tte_sec = None

    record_data = json.dumps({
        "idx": idx, "risk_score": risk_score, "risk_level": risk_level,
        "actions": response["actions"], "outcome": sim_outcome,
    }, sort_keys=True)
    integrity_hash = hashlib.sha256(record_data.encode()).hexdigest()[:16]

    return {
        "alert_id": f"ALERT-{idx:05d}",
        "timestamp": timestamp.isoformat(),
        "device_tier": response["device_tier"],
        "attack_category": attack_category,
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "recommended_actions": response["actions"],
        "action_rationale": response["rationale"],
        "escalation_chain": response["escalation_chain"],
        "explanation_summary": explanation_summary[:200] if explanation_summary else "",
        "simulated_outcome": {
            "outcome": sim_outcome,
            "action_effective": sim_effective,
            "time_to_effectiveness_sec": sim_tte_sec,
            "ground_truth": ground_truth,
        },
        "integrity_hash": integrity_hash,
    }


# ── Effectiveness analysis ─────────────────────────────────────────────

def compute_effectiveness(audit_records: list) -> dict:
    """Compute action effectiveness metrics from simulated outcomes."""
    action_stats = {}
    outcome_counts = {}

    for rec in audit_records:
        outcome = rec["simulated_outcome"]["outcome"]
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        gt = rec["simulated_outcome"]["ground_truth"]

        for action in rec["recommended_actions"]:
            if action not in action_stats:
                action_stats[action] = {"true_attacks": 0, "false_positives": 0, "total": 0}
            action_stats[action]["total"] += 1
            if gt == "attack":
                action_stats[action]["true_attacks"] += 1
            else:
                action_stats[action]["false_positives"] += 1

    # Precision per action
    for action, stats in action_stats.items():
        t = stats["total"]
        stats["precision"] = round(stats["true_attacks"] / t, 4) if t > 0 else 0
        stats["false_positive_rate"] = round(stats["false_positives"] / t, 4) if t > 0 else 0

    # Proportionality: costly actions should have higher precision
    costly_actions = sorted(action_stats.keys(),
                            key=lambda a: MITIGATION_ACTIONS.get(a, {}).get("cost", 0),
                            reverse=True)
    proportionality = []
    for a in costly_actions:
        proportionality.append({
            "action": a,
            "cost": MITIGATION_ACTIONS.get(a, {}).get("cost", 0),
            "precision": action_stats[a]["precision"],
            "total": action_stats[a]["total"],
        })

    # Over/under response rates
    over_response = sum(1 for r in audit_records
                        if r["simulated_outcome"]["outcome"] == "false_positive_isolated")
    under_response = sum(1 for r in audit_records
                         if r["simulated_outcome"]["outcome"] == "threat_logged_not_mitigated")

    return {
        "outcome_distribution": outcome_counts,
        "per_action_stats": action_stats,
        "proportionality_analysis": proportionality,
        "over_response_count": over_response,
        "under_response_count": under_response,
        "over_response_rate": round(over_response / len(audit_records), 4) if audit_records else 0,
        "under_response_rate": round(under_response / len(audit_records), 4) if audit_records else 0,
    }


# ── Build all records ──────────────────────────────────────────────────

def build_all_records(
    risk_data: dict,
    attack_cats: np.ndarray,
    analyst_by_idx: dict,
    clinician_by_idx: dict,
) -> tuple:
    """Build adaptive response records + audit trail for all non-NORMAL alerts."""
    R = risk_data["R"]
    levels = risk_data["risk_levels"]
    y_true = risk_data["y_true"]

    records = []
    audit_trail = []

    for idx in range(len(R)):
        level = str(levels[idx])
        if level == "NORMAL":
            continue

        cat = str(attack_cats[idx]) if attack_cats is not None else "unknown"
        gt = "attack" if y_true[idx] == 1 else "benign"

        # Check if biometric features are in SHAP top-3
        bio_in_top = False
        if idx in analyst_by_idx:
            xgb_top = analyst_by_idx[idx].get("models", {}).get("xgboost", {}).get("top_features", [])
            bio_in_top = any(f["feature"] in BIOMETRIC_FEATURES for f in xgb_top)

        # Adaptive response selection
        response = select_adaptive_response(
            risk_level=level,
            risk_score=float(R[idx]),
            attack_category=cat,
            biometric_in_top_features=bio_in_top,
        )

        # Clinician summary for explanation pairing
        clin_summary = ""
        if idx in clinician_by_idx:
            clin_summary = clinician_by_idx[idx]["summary"]

        # Build record
        record = {
            "sample_index": int(idx),
            "ground_truth": gt,
            "attack_category": cat,
            "risk_score": round(float(R[idx]), 4),
            "risk_level": level,
            "risk_components": {
                "C_detect": round(float(risk_data["c_detect"][idx]), 4),
                "C_track_a": round(float(risk_data["c_track_a"][idx]), 4),
                "C_track_b": round(float(risk_data["c_track_b"][idx]), 4),
                "D_crit": round(float(risk_data["d_crit"][idx]), 4),
                "S_data": round(float(risk_data["s_data"][idx]), 4),
                "A_patient": round(float(risk_data["a_patient"][idx]), 4),
            },
            "response": response,
            "explanation": {
                "clinician_summary": clin_summary,
                "analyst_available": idx in analyst_by_idx,
            },
        }
        records.append(record)

        # Audit record
        audit = build_audit_record(
            idx, float(R[idx]), level, cat, gt, response, clin_summary,
        )
        audit_trail.append(audit)

    return records, audit_trail


# ── Statistics ─────────────────────────────────────────────────────────

def compute_response_stats(records: list) -> dict:
    """Aggregate response statistics."""
    level_counts = {}
    action_counts = {}
    tp_by_level = {}
    fp_by_level = {}

    for rec in records:
        level = rec["risk_level"]
        level_counts[level] = level_counts.get(level, 0) + 1
        is_attack = rec["ground_truth"] == "attack"
        if is_attack:
            tp_by_level[level] = tp_by_level.get(level, 0) + 1
        else:
            fp_by_level[level] = fp_by_level.get(level, 0) + 1

        for a in rec["response"]["actions"]:
            action_counts[a] = action_counts.get(a, 0) + 1

    precision_by_level = {}
    for level in level_counts:
        tp = tp_by_level.get(level, 0)
        total = tp + fp_by_level.get(level, 0)
        precision_by_level[level] = round(tp / total, 4) if total > 0 else 0.0

    return {
        "total_alerts": len(records),
        "alerts_by_level": level_counts,
        "actions_triggered": action_counts,
        "true_positives_by_level": tp_by_level,
        "false_positives_by_level": fp_by_level,
        "precision_by_level": precision_by_level,
    }


# ── Visualizations ─────────────────────────────────────────────────────

def plot_response_distribution(records: list) -> None:
    """Bar chart of response actions by risk level."""
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    all_actions = sorted(MITIGATION_ACTIONS.keys(), key=lambda a: MITIGATION_ACTIONS[a]["cost"])
    colors_list = plt.cm.Set2(np.linspace(0, 1, len(all_actions)))

    action_by_level = {l: {a: 0 for a in all_actions} for l in levels}
    for rec in records:
        level = rec["risk_level"]
        if level in action_by_level:
            for a in rec["response"]["actions"]:
                if a in action_by_level[level]:
                    action_by_level[level][a] += 1

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(levels))
    width = 0.8 / len(all_actions)

    for i, action in enumerate(all_actions):
        vals = [action_by_level[l][action] for l in levels]
        if max(vals) > 0:
            ax.bar(x + i * width, vals, width, label=action.replace("_", " "),
                   color=colors_list[i], alpha=0.85)

    ax.set_xticks(x + width * len(all_actions) / 2)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Number of Alerts")
    ax.set_title("Adaptive Response Actions by Risk Level")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "response_actions_by_level.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: response_actions_by_level.png")


def plot_precision_by_level(stats: dict) -> None:
    """Precision (true attack rate) per risk level."""
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    tp = [stats["true_positives_by_level"].get(l, 0) for l in levels]
    fp = [stats["false_positives_by_level"].get(l, 0) for l in levels]
    prec = [stats["precision_by_level"].get(l, 0) for l in levels]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(levels))
    w = 0.35
    ax.bar(x - w / 2, tp, w, label="True Attacks", color="#e74c3c", alpha=0.8)
    ax.bar(x + w / 2, fp, w, label="False Positives", color="#95a5a6", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, prec, "ko-", linewidth=2, markersize=8, label="Precision")
    ax2.set_ylabel("Precision")
    ax2.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Count")
    ax.set_title("Alert Precision by Risk Level")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "precision_by_level.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: precision_by_level.png")


def plot_escalation_funnel(stats: dict) -> None:
    """Horizontal funnel of alert volumes per tier."""
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts = [stats["alerts_by_level"].get(l, 0) for l in levels]
    colors_map = {"LOW": "#2ecc71", "MEDIUM": "#f1c40f", "HIGH": "#e74c3c", "CRITICAL": "#8e44ad"}
    sla = [480, 60, 15, 5]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(levels, counts, color=[colors_map[l] for l in levels],
                   alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, level, count, s in zip(bars, levels, counts, sla):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"n={count} | SLA ≤{s}min", va="center", fontsize=9)
    ax.set_xlabel("Number of Alerts")
    ax.set_title("Response Escalation Funnel")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "response_escalation_funnel.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: response_escalation_funnel.png")


def plot_effectiveness_by_action(effectiveness: dict) -> None:
    """Precision per mitigation action (higher cost should have higher precision)."""
    prop = effectiveness["proportionality_analysis"]
    prop = [p for p in prop if p["total"] > 0]
    if not prop:
        return

    names = [p["action"].replace("_", "\n") for p in prop]
    precs = [p["precision"] for p in prop]
    costs = [p["cost"] for p in prop]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, precs, color=plt.cm.RdYlGn_r([c for c in costs]), alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Precision (true attack rate)")
    ax.set_title("Response Proportionality — Costly Actions Should Have Higher Precision")
    ax.set_ylim(0, 1.05)
    for bar, p in zip(bars, prop):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={p['total']}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "effectiveness_by_action.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: effectiveness_by_action.png")


def plot_response_sankey(audit_records: list) -> None:
    """Simulated flow: risk level → primary action → outcome."""
    # Count flows
    flows = {}
    for rec in audit_records:
        level = rec["risk_level"]
        # Pick the highest-cost action as the "primary"
        actions = rec["recommended_actions"]
        costs = [(a, MITIGATION_ACTIONS.get(a, {}).get("cost", 0)) for a in actions]
        primary = max(costs, key=lambda x: x[1])[0] if costs else "log_event"
        outcome = rec["simulated_outcome"]["outcome"]
        key = (level, primary, outcome)
        flows[key] = flows.get(key, 0) + 1

    # Build a grouped bar chart as Sankey proxy (matplotlib has no native Sankey for categorical)
    outcomes = sorted(set(k[2] for k in flows))
    outcome_colors = {
        "threat_contained": "#2ecc71",
        "benign_logged": "#3498db",
        "false_positive_isolated": "#e67e22",
        "threat_logged_not_mitigated": "#e74c3c",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    x = np.arange(len(levels))
    width = 0.8 / max(len(outcomes), 1)

    for i, outcome in enumerate(outcomes):
        vals = []
        for level in levels:
            count = sum(v for (l, a, o), v in flows.items() if l == level and o == outcome)
            vals.append(count)
        ax.bar(x + i * width, vals, width,
               label=outcome.replace("_", " "),
               color=outcome_colors.get(outcome, "#999"),
               alpha=0.85)

    ax.set_xticks(x + width * len(outcomes) / 2)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Count")
    ax.set_title("Risk Level → Simulated Outcome Flow")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "response_sankey.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: response_sankey.png")


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sep = "=" * 72
    t0 = time.perf_counter()

    logger.info(sep)
    logger.info("MODULE 5 — CLOSED-LOOP RESPONSE ENGINE (RQ3/RO3)")
    logger.info(sep)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load inputs
    risk_data = load_risk_scores()
    analyst_by_idx, clinician_by_idx = load_explanations()
    attack_cats = load_attack_categories()

    n_samples = len(risk_data["R"])
    logger.info("Loaded: %d samples, %d analyst alerts, %d clinician summaries",
                n_samples, len(analyst_by_idx), len(clinician_by_idx))

    # Build adaptive records + audit trail
    logger.info("Building adaptive response records...")
    records, audit_trail = build_all_records(
        risk_data, attack_cats, analyst_by_idx, clinician_by_idx,
    )
    logger.info("  Generated %d alert-response records", len(records))

    # Statistics
    stats = compute_response_stats(records)
    logger.info("")
    logger.info("── Response Statistics ──")
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        n = stats["alerts_by_level"].get(level, 0)
        prec = stats["precision_by_level"].get(level, 0)
        tp = stats["true_positives_by_level"].get(level, 0)
        fp = stats["false_positives_by_level"].get(level, 0)
        logger.info("  %-10s %4d alerts (TP=%d, FP=%d, prec=%.2f)",
                    level, n, tp, fp, prec)
    logger.info("  Actions: %s", stats["actions_triggered"])

    # Effectiveness analysis
    logger.info("")
    logger.info("── Effectiveness Analysis ──")
    effectiveness = compute_effectiveness(audit_trail)
    logger.info("  Outcomes: %s", effectiveness["outcome_distribution"])
    logger.info("  Over-response (FP isolated): %d (%.1f%%)",
                effectiveness["over_response_count"],
                effectiveness["over_response_rate"] * 100)
    logger.info("  Under-response (attack only logged): %d (%.1f%%)",
                effectiveness["under_response_count"],
                effectiveness["under_response_rate"] * 100)

    # Save outputs
    logger.info("")
    logger.info("Saving outputs...")

    (OUTPUT_DIR / "alert_responses.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    logger.info("  Saved: alert_responses.json (%d records)", len(records))

    (OUTPUT_DIR / "audit_trail.json").write_text(
        json.dumps(audit_trail, indent=2), encoding="utf-8")
    logger.info("  Saved: audit_trail.json (%d records)", len(audit_trail))

    (OUTPUT_DIR / "effectiveness_analysis.json").write_text(
        json.dumps(effectiveness, indent=2), encoding="utf-8")
    logger.info("  Saved: effectiveness_analysis.json")

    report = {
        "module": "Module 5 — Closed-Loop Response Engine (RQ3/RO3)",
        "total_samples": n_samples,
        "total_alerts": len(records),
        "statistics": stats,
        "effectiveness": effectiveness,
        "mitigation_catalogue": {k: v["description"] for k, v in MITIGATION_ACTIONS.items()},
        "escalation_routing": {k: {kk: vv for kk, vv in v.items()
                                   if kk != "attack_specific_actions"}
                               for k, v in ESCALATION_ROUTING.items()},
        "device_constraints": DEVICE_TIERS,
    }
    (OUTPUT_DIR / "response_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    logger.info("  Saved: response_report.json")

    # CSV
    rows = []
    for rec in records:
        rows.append({
            "sample_index": rec["sample_index"],
            "ground_truth": rec["ground_truth"],
            "attack_category": rec["attack_category"],
            "risk_score": rec["risk_score"],
            "risk_level": rec["risk_level"],
            "actions": "|".join(rec["response"]["actions"]),
            "max_response_min": rec["response"]["max_response_min"],
            "escalation_primary": rec["response"]["escalation_chain"]["primary"],
            "device_constraint": rec["response"]["device_constraint_applied"],
            "rationale": rec["response"]["rationale"][:100],
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "alert_responses_detail.csv", index=False)
    logger.info("  Saved: alert_responses_detail.csv")

    # Visualizations
    logger.info("Generating charts...")
    plot_response_distribution(records)
    plot_precision_by_level(stats)
    plot_escalation_funnel(stats)
    plot_effectiveness_by_action(effectiveness)
    plot_response_sankey(audit_trail)

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("CLOSED-LOOP RESPONSE ENGINE COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    logger.info("  Alerts         : %d", len(records))
    logger.info("  Audit records  : %d", len(audit_trail))
    logger.info("  Over-response  : %.1f%%", effectiveness["over_response_rate"] * 100)
    logger.info("  Under-response : %.1f%%", effectiveness["under_response_rate"] * 100)
    logger.info("  Output         : %s", OUTPUT_DIR)
    logger.info(sep)


if __name__ == "__main__":
    main()
