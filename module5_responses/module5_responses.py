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
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results/reports"
CHARTS_DIR = PROJECT_ROOT / "results/charts"

from common.alert_response_schema import (  # noqa: E402
    AlertResponsesEnvelope,
    InputFile,
    Provenance,
)
from common.phi import BIOMETRIC_COLUMNS as BIOMETRIC_FEATURES  # noqa: E402

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
        "attack_specific_actions": [
            "isolate_device",
            "forensic_snapshot",
            "escalate_clinical",
        ],
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
        "base_actions": [
            "isolate_device",
            "escalate_incident",
            "forensic_snapshot",
            "escalate_clinical",
        ],
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


def _paths(split: str) -> dict:
    """Resolve per-split input + output paths.

    Test = paper-clean (the default; preserves legacy filename
    `alert_responses.json` for backward compatibility with the dashboard's
    fallback loader and downstream tooling).
    Demo = operator-clean (suffixed `_demo` everywhere).
    """
    if split == "test":
        scores_npz = "risk_scores.npz"
        parquet = "test_phase1.parquet"
        suffix = ""               # legacy: no suffix on test outputs
    elif split == "demo":
        scores_npz = "demo_scores.npz"
        parquet = "demo_phase1.parquet"
        suffix = "_demo"
    else:
        raise ValueError(f"unknown split: {split!r} (expected 'test' or 'demo')")

    return {
        "split": split,
        "scores_npz": PROJECT_ROOT / "results/reports" / scores_npz,
        "parquet": PROJECT_ROOT / "data/processed" / parquet,
        "analyst_json": PROJECT_ROOT / "results/reports" / f"analyst_report{suffix}.json",
        "clinician_json": PROJECT_ROOT / "results/reports" / f"clinician_summaries{suffix}.json",
        "out_alert_responses": OUTPUT_DIR / f"alert_responses{suffix}.json",
        "out_audit_trail": OUTPUT_DIR / f"audit_trail{suffix}.json",
        "out_effectiveness": OUTPUT_DIR / f"effectiveness_analysis{suffix}.json",
        "out_response_report": OUTPUT_DIR / f"response_report{suffix}.json",
        "out_detail_csv": OUTPUT_DIR / f"alert_responses_detail{suffix}.csv",
        "suffix": suffix,
    }


def load_risk_scores(scores_npz_path: Path | None = None) -> dict:
    """Load Module 3 risk scores from the configured split's npz."""
    path = scores_npz_path or (PROJECT_ROOT / "results/reports/risk_scores.npz")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_explanations(
    analyst_json_path: Path | None = None,
    clinician_json_path: Path | None = None,
) -> tuple:
    """Load Module 4 analyst reports and clinician summaries.

    Both files are OPTIONAL — when running against the demo split before
    Module 4 has produced demo-specific explanations, falls back to empty
    dicts. Downstream record builders gracefully handle this case (records
    are marked ``analyst_available: false``).
    """
    a_path = analyst_json_path or (PROJECT_ROOT / "results/reports/analyst_report.json")
    c_path = clinician_json_path or (PROJECT_ROOT / "results/reports/clinician_summaries.json")
    analyst: dict = {}
    clinician: dict = {}
    if a_path.exists():
        with open(a_path) as f:
            analyst = {a["sample_index"]: a for a in json.load(f)}
    else:
        logger.warning("analyst report missing at %s — proceeding with empty dict", a_path)
    if c_path.exists():
        with open(c_path) as f:
            clinician = {s["sample_index"]: s for s in json.load(f)}
    else:
        logger.warning("clinician summaries missing at %s — proceeding with empty dict", c_path)
    return analyst, clinician


def load_attack_categories(parquet_path: Path | None = None) -> np.ndarray:
    """Load Attack Category column from the configured split's parquet."""
    path = parquet_path or (PROJECT_ROOT / "data/processed/test_phase1.parquet")
    df = pd.read_parquet(path, columns=["Attack Category"])
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
        rationale_parts.append(
            f"Escalated: R={risk_score:.2f} exceeds 0.70 magnitude threshold"
        )
    elif risk_score < 0.30 and risk_level in ("MEDIUM", "HIGH"):
        # Demote: replace isolate with restrict
        if "isolate_device" in actions:
            actions.remove("isolate_device")
            if "restrict_traffic" not in actions:
                actions.append("restrict_traffic")
            rationale_parts.append(
                f"Demoted: R={risk_score:.2f} below 0.30, restrict instead of isolate"
            )

    # 2. Attack-category-specific actions
    routing = ESCALATION_ROUTING.get(attack_category, DEFAULT_ROUTING)
    for action in routing["attack_specific_actions"]:
        if action not in actions:
            actions.append(action)
    if routing["attack_specific_actions"]:
        rationale_parts.append(
            f"Attack-specific ({attack_category}): added {routing['attack_specific_actions']}"
        )

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
        rationale_parts.append(
            "Biometric features in top SHAP contributors — clinical escalation added"
        )

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
    has_isolate = (
        "isolate_device" in response["actions"]
        or "restrict_traffic" in response["actions"]
    )
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

    # M5-2: f-string avoids dict construction + json.dumps + encode per record.
    # 16 hex chars = 64 bits of collision resistance — same as before.
    integrity_hash = hashlib.sha256(
        f"{idx}:{risk_score:.4f}:{risk_level}:{sim_outcome}".encode()
    ).hexdigest()[:16]

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
    # M5-4: defaultdict removes per-action guard; outcome_counts reused for
    # over/under response so the record list is scanned only once.
    action_stats: dict = defaultdict(
        lambda: {"true_attacks": 0, "false_positives": 0, "total": 0}
    )
    outcome_counts: dict = defaultdict(int)

    for rec in audit_records:
        outcome = rec["simulated_outcome"]["outcome"]
        outcome_counts[outcome] += 1
        gt = rec["simulated_outcome"]["ground_truth"]
        is_attack = gt == "attack"

        for action in rec["recommended_actions"]:
            s = action_stats[action]
            s["total"] += 1
            if is_attack:
                s["true_attacks"] += 1
            else:
                s["false_positives"] += 1

    # Precision per action
    for action, stats in action_stats.items():
        t = stats["total"]
        stats["precision"] = round(stats["true_attacks"] / t, 4) if t > 0 else 0
        stats["false_positive_rate"] = (
            round(stats["false_positives"] / t, 4) if t > 0 else 0
        )

    # Proportionality: costly actions should have higher precision
    costly_actions = sorted(
        action_stats.keys(),
        key=lambda a: MITIGATION_ACTIONS.get(a, {}).get("cost", 0),
        reverse=True,
    )
    proportionality = [
        {
            "action": a,
            "cost": MITIGATION_ACTIONS.get(a, {}).get("cost", 0),
            "precision": action_stats[a]["precision"],
            "total": action_stats[a]["total"],
        }
        for a in costly_actions
    ]

    # M5-4: reuse outcome_counts — no second scan of audit_records
    over_response = outcome_counts["false_positive_isolated"]
    under_response = outcome_counts["threat_logged_not_mitigated"]

    return {
        "outcome_distribution": dict(outcome_counts),
        "per_action_stats": dict(action_stats),
        "proportionality_analysis": proportionality,
        "over_response_count": over_response,
        "under_response_count": under_response,
        "over_response_rate": round(over_response / len(audit_records), 4)
        if audit_records
        else 0,
        "under_response_rate": round(under_response / len(audit_records), 4)
        if audit_records
        else 0,
    }


# ── Build all records ──────────────────────────────────────────────────


def build_all_records(
    risk_data: dict,
    attack_cats: np.ndarray,
    analyst_by_idx: dict,
    clinician_by_idx: dict,
    parquet_path: Path,
) -> tuple:
    """Build adaptive response records + audit trail for all non-NORMAL alerts.

    Now also generates a 3-layer MVE per record via src.mve_generator and
    attaches it under ``explanation.mve``. Provider chain is the default
    OpenAI → Anthropic → rule-based, with a tripwire that flips to
    force-rule-based after MVE_LLM_FAIL_STREAK_MAX consecutive LLM failures
    so a quota outage doesn't waste a 1-2 second API attempt per record.
    """
    # Imported here (not at module top) so module5_responses keeps its
    # fast import cost when only build_audit_record / compute_response_stats
    # are needed by the dashboard's runtime.
    from common.device_class import (
        device_context_for_idx,
        synthesize_raw_alert,
    )
    from src.mve_generator import generate_mve

    R = risk_data["R"]
    levels = risk_data["risk_levels"]
    y_true = risk_data["y_true"]

    # M5-3: pre-cast numpy string arrays to Python lists once — avoids per-row
    # numpy-to-Python boxing; pre-filter NORMAL indices so the loop body only
    # runs for actionable alerts.
    levels_list = levels.tolist()
    cats_list = attack_cats.tolist() if attack_cats is not None else None
    active_indices = [i for i, lv in enumerate(levels_list) if lv != "NORMAL"]

    # Per-row biometric activity drives the device-class heuristic that
    # mve_generator needs. Loaded once for the whole batch.
    test_df = pd.read_parquet(parquet_path)

    records = []
    audit_trail = []

    # LLM-quota tripwire: after MVE_LLM_FAIL_STREAK_MAX consecutive failures
    # (OpenAI key absent or quota exhausted), stop attempting the API. The
    # rule-based fallback still runs for every record, so each alert always
    # gets an MVE — we just skip the wasted handshake on the dead provider.
    MVE_LLM_FAIL_STREAK_MAX = 5
    llm_fail_streak = 0
    force_rule_based = False
    provider_counts: dict[str, int] = {"openai": 0, "anthropic": 0, "rule_based": 0}

    for idx in active_indices:
        level = levels_list[idx]
        cat = str(cats_list[idx]) if cats_list is not None else "unknown"
        gt = "attack" if y_true[idx] == 1 else "benign"

        # Check if biometric features are in SHAP top-3
        bio_in_top = False
        if idx in analyst_by_idx:
            xgb_top = (
                analyst_by_idx[idx]
                .get("models", {})
                .get("xgboost", {})
                .get("top_features", [])
            )
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

        # MVE generation — see module docstring for the rationale and the
        # tripwire logic. patchable defaults to True; the offline pipeline
        # has no per-device patchability registry.
        device_ctx_full = device_context_for_idx(idx, test_df)
        # mve_generator._normalize_device_type does not recognise the
        # "other" sentinel (returns "" → is_unknown → forces CRITICAL safe
        # default). Map to "system" so the rule-based path picks the
        # MEDIUM-criticality template that matches DEVICE_CONTEXT["other"].
        mve_device_type = device_ctx_full["device_class"]
        if mve_device_type == "other":
            mve_device_type = "system"
        mve_device_ctx = {
            "device_type": mve_device_type,
            "criticality": device_ctx_full["device_criticality"],
            "clinical_function": device_ctx_full["affected_system"],
            "patchable": True,
        }
        raw_alert = synthesize_raw_alert(idx, cat, float(R[idx]))
        try:
            mve_out = generate_mve(
                raw_alert=raw_alert,
                device_context=mve_device_ctx,
                baseline={"baseline_days": 90},
                user_context=None,
                shap_context=None,
                event_context=None,
                force_rule_based=force_rule_based,
                risk_level=level,
            )
            mve_dict = mve_out.to_dict()
            mve_payload = {
                "layer_1": mve_dict["layer_1"],
                "layer_2": mve_dict["layer_2"],
                "layer_3": mve_dict["layer_3"],
                "why_anomalous": mve_dict["layer_1_why_anomalous"],
                "alert_involves_clinical_system": mve_dict[
                    "alert_involves_clinical_system"
                ],
                "total_word_count": mve_dict["total_word_count"],
                "provider": mve_out.provider,
            }
            provider_counts[mve_out.provider] = (
                provider_counts.get(mve_out.provider, 0) + 1
            )
            # Tripwire bookkeeping. Reset streak on success; trip when streak
            # hits the threshold so the next records short-circuit to rules.
            if not force_rule_based:
                if mve_out.provider == "rule_based":
                    llm_fail_streak += 1
                    if llm_fail_streak >= MVE_LLM_FAIL_STREAK_MAX:
                        force_rule_based = True
                        logger.warning(
                            "MVE LLM tripwire: %d consecutive rule-based "
                            "fallbacks — forcing rule-based for the rest of "
                            "the batch (idx=%d)",
                            llm_fail_streak,
                            idx,
                        )
                else:
                    llm_fail_streak = 0
        except Exception as exc:
            logger.warning(
                "MVE generation failed for sample %d: %s — proceeding without MVE",
                idx,
                exc,
            )
            mve_payload = None

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
                "D_clinical_tier": round(float(risk_data["d_clinical_tier"][idx]), 4),
            },
            "response": response,
            "explanation": {
                "clinician_summary": clin_summary,
                "analyst_available": idx in analyst_by_idx,
                "mve": mve_payload,
            },
        }
        records.append(record)

        # Audit record
        audit = build_audit_record(
            idx,
            float(R[idx]),
            level,
            cat,
            gt,
            response,
            clin_summary,
        )
        audit_trail.append(audit)

    logger.info(
        "  MVE provider mix: openai=%d, anthropic=%d, rule_based=%d (tripwire=%s)",
        provider_counts.get("openai", 0),
        provider_counts.get("anthropic", 0),
        provider_counts.get("rule_based", 0),
        "fired" if force_rule_based else "not-fired",
    )

    return records, audit_trail


# ── Statistics ─────────────────────────────────────────────────────────


def compute_response_stats(records: list) -> dict:
    """Aggregate response statistics."""
    # M5-5: Counter + .update() — C-level accumulation, no per-key guards
    level_counts: Counter = Counter()
    action_counts: Counter = Counter()
    tp_by_level: Counter = Counter()
    fp_by_level: Counter = Counter()

    for rec in records:
        level = rec["risk_level"]
        level_counts[level] += 1
        if rec["ground_truth"] == "attack":
            tp_by_level[level] += 1
        else:
            fp_by_level[level] += 1
        action_counts.update(rec["response"]["actions"])

    precision_by_level = {}
    for level in level_counts:
        tp = tp_by_level.get(level, 0)
        total = tp + fp_by_level.get(level, 0)
        precision_by_level[level] = round(tp / total, 4) if total > 0 else 0.0

    return {
        "total_alerts": len(records),
        "alerts_by_level": dict(level_counts),
        "actions_triggered": dict(action_counts),
        "true_positives_by_level": dict(tp_by_level),
        "false_positives_by_level": dict(fp_by_level),
        "precision_by_level": precision_by_level,
    }


# ── Visualizations ─────────────────────────────────────────────────────


def plot_response_distribution(records: list) -> None:
    """Bar chart of response actions by risk level."""
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    all_actions = sorted(
        MITIGATION_ACTIONS.keys(), key=lambda a: MITIGATION_ACTIONS[a]["cost"]
    )
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
            ax.bar(
                x + i * width,
                vals,
                width,
                label=action.replace("_", " "),
                color=colors_list[i],
                alpha=0.85,
            )

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
    colors_map = {
        "LOW": "#2ecc71",
        "MEDIUM": "#f1c40f",
        "HIGH": "#e74c3c",
        "CRITICAL": "#8e44ad",
    }
    sla = [480, 60, 15, 5]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(
        levels,
        counts,
        color=[colors_map[l] for l in levels],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, level, count, s in zip(bars, levels, counts, sla):
        ax.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            f"n={count} | SLA ≤{s}min",
            va="center",
            fontsize=9,
        )
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
    bars = ax.bar(
        names,
        precs,
        color=plt.cm.RdYlGn_r([c for c in costs]),
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Precision (true attack rate)")
    ax.set_title(
        "Response Proportionality — Costly Actions Should Have Higher Precision"
    )
    ax.set_ylim(0, 1.05)
    for bar, p in zip(bars, prop):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={p['total']}",
            ha="center",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "effectiveness_by_action.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: effectiveness_by_action.png")


def plot_response_sankey(audit_records: list) -> None:
    """Simulated flow: risk level → primary action → outcome."""
    # Count flows
    flows: dict = defaultdict(int)
    for rec in audit_records:
        level = rec["risk_level"]
        # Pick the highest-cost action as the "primary"
        actions = rec["recommended_actions"]
        costs = [(a, MITIGATION_ACTIONS.get(a, {}).get("cost", 0)) for a in actions]
        primary = max(costs, key=lambda x: x[1])[0] if costs else "log_event"
        outcome = rec["simulated_outcome"]["outcome"]
        flows[(level, primary, outcome)] += 1

    # M5-7: pre-aggregate to (level, outcome) → count in one pass — O(|flows|).
    # Replaces O(|flows| × L × O) nested scan inside the bar-chart loop.
    level_outcome: dict = defaultdict(int)
    for (lv, _a, oc), v in flows.items():
        level_outcome[(lv, oc)] += v

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
        vals = [level_outcome[(level, outcome)] for level in levels]
        ax.bar(
            x + i * width,
            vals,
            width,
            label=outcome.replace("_", " "),
            color=outcome_colors.get(outcome, "#999"),
            alpha=0.85,
        )

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
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m module5_responses.module5_responses",
        description="Module 5 — closed-loop response engine. Operates on the "
                    "selected frozen split (test=paper-clean, demo=operator-clean).",
    )
    parser.add_argument(
        "--split",
        choices=["test", "demo", "both"],
        default="test",
        help="Frozen split to process. 'test' writes paper-clean artifacts (legacy "
             "`alert_responses.json`); 'demo' writes operator-clean artifacts with "
             "`_demo` suffix; 'both' processes test then demo sequentially.",
    )
    args = parser.parse_args()

    splits_to_run = ["test", "demo"] if args.split == "both" else [args.split]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sep = "=" * 72
    t0 = time.perf_counter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits_to_run:
        _run_one_split(split, sep)
    logger.info("Module 5 complete (%.1fs, splits=%s)",
                time.perf_counter() - t0, splits_to_run)


def _build_provenance(
    paths: dict,
    risk_data: dict,
    n_alerts: int,
    n_normal: int,
    filter_applied: str = "non_normal",
) -> Provenance:
    """Capture mtime/sha256 of every input + git rev + run timestamp.

    The Dashboard reads this back to detect when an upstream artefact
    (risk_scores.npz, analyst_report.json, ...) has been regenerated
    after this responses file was built — i.e. when the numbers shown
    are stale relative to the source pipeline outputs.
    """
    def _stat(p: Path) -> InputFile | None:
        if not p.exists():
            return None
        b = p.read_bytes()
        return InputFile(
            path=str(p.relative_to(PROJECT_ROOT)),
            mtime_iso=datetime.fromtimestamp(p.stat().st_mtime, UTC).isoformat(),
            sha256=hashlib.sha256(b).hexdigest(),
            size_bytes=len(b),
        )

    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        rev = None

    return Provenance(
        split=paths["split"],
        generated_at=datetime.now(UTC).isoformat(),
        module5_git_rev=rev,
        n_input_samples=len(risk_data["R"]),
        n_alerts_emitted=n_alerts,
        n_normal_excluded=n_normal,
        filter_applied=filter_applied,
        inputs={
            "risk_scores_npz": _stat(paths["scores_npz"]),
            "parquet": _stat(paths["parquet"]),
            "analyst_json": _stat(paths["analyst_json"]),
            "clinician_json": _stat(paths["clinician_json"]),
        },
    )


def _assert_no_score_drift(
    records: list,
    risk_data: dict,
    tol: float = 1e-4,
) -> None:
    """Fail-loud if any record field diverges from the source npz.

    Catches the highest-probability silent failure mode: Module 3
    rerun without Module 5 rerun, leaving record[i]['risk_score']
    pointing at an obsolete R[i]. Called once at end-of-build so a
    drift surfaces in the same run that introduced it.
    """
    component_map = [
        ("C_detect", "c_detect"),
        ("C_track_a", "c_track_a"),
        ("C_track_b", "c_track_b"),
        ("D_crit", "d_crit"),
        ("S_data", "s_data"),
        ("D_clinical_tier", "d_clinical_tier"),
    ]
    for rec in records:
        idx = rec["sample_index"]
        expected_R = round(float(risk_data["R"][idx]), 4)
        if abs(rec["risk_score"] - expected_R) > tol:
            raise ValueError(
                f"Score drift at sample_index={idx}: "
                f"record.risk_score={rec['risk_score']} vs "
                f"npz.R={expected_R}"
            )
        expected_level = str(risk_data["risk_levels"][idx])
        if rec["risk_level"] != expected_level:
            raise ValueError(
                f"Risk-level drift at sample_index={idx}: "
                f"record.risk_level={rec['risk_level']} vs "
                f"npz.risk_levels={expected_level}"
            )
        for rec_key, npz_key in component_map:
            expected = round(float(risk_data[npz_key][idx]), 4)
            actual = rec["risk_components"][rec_key]
            if abs(actual - expected) > tol:
                raise ValueError(
                    f"Component drift at sample_index={idx} {rec_key}: "
                    f"record={actual} vs npz={expected}"
                )


def _run_one_split(split: str, sep: str) -> None:
    paths = _paths(split)
    logger.info(sep)
    logger.info("MODULE 5 — CLOSED-LOOP RESPONSE ENGINE (RQ3/RO3) — split=%s", split)
    logger.info(sep)

    # Load inputs
    risk_data = load_risk_scores(paths["scores_npz"])
    analyst_by_idx, clinician_by_idx = load_explanations(
        paths["analyst_json"], paths["clinician_json"]
    )
    attack_cats = load_attack_categories(paths["parquet"])

    n_samples = len(risk_data["R"])
    logger.info(
        "Loaded: %d samples, %d analyst alerts, %d clinician summaries",
        n_samples,
        len(analyst_by_idx),
        len(clinician_by_idx),
    )

    # Build adaptive records + audit trail
    logger.info("Building adaptive response records...")
    records, audit_trail = build_all_records(
        risk_data,
        attack_cats,
        analyst_by_idx,
        clinician_by_idx,
        paths["parquet"],
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
        logger.info(
            "  %-10s %4d alerts (TP=%d, FP=%d, prec=%.2f)", level, n, tp, fp, prec
        )
    logger.info("  Actions: %s", stats["actions_triggered"])

    # Effectiveness analysis
    logger.info("")
    logger.info("── Effectiveness Analysis ──")
    effectiveness = compute_effectiveness(audit_trail)
    logger.info("  Outcomes: %s", effectiveness["outcome_distribution"])
    logger.info(
        "  Over-response (FP isolated): %d (%.1f%%)",
        effectiveness["over_response_count"],
        effectiveness["over_response_rate"] * 100,
    )
    logger.info(
        "  Under-response (attack only logged): %d (%.1f%%)",
        effectiveness["under_response_count"],
        effectiveness["under_response_rate"] * 100,
    )

    # P0-2: drift check — fails loud if record fields diverge from npz.
    # Done before persistence so a drift surfaces here, not at Dashboard
    # render time. Runs in microseconds at this dataset size.
    _assert_no_score_drift(records, risk_data)

    # Save outputs
    logger.info("")
    logger.info("Saving outputs...")

    # P0-1 + P0-3: wrap records in a provenance-bearing envelope and
    # validate via pydantic. Build failure on schema/provenance
    # mismatch is preferable to a runtime KeyError in the Streamlit
    # dashboard.
    n_normal = sum(
        1 for lv in risk_data["risk_levels"].tolist() if lv == "NORMAL"
    )
    provenance = _build_provenance(
        paths,
        risk_data,
        n_alerts=len(records),
        n_normal=n_normal,
        filter_applied="non_normal",
    )
    envelope = AlertResponsesEnvelope(
        _provenance=provenance, records=records
    )
    paths["out_alert_responses"].write_text(
        envelope.model_dump_json(by_alias=True, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "  Saved: %s (%d records, envelope schema v1)",
        paths["out_alert_responses"].name, len(records),
    )

    paths["out_audit_trail"].write_text(
        json.dumps(audit_trail, indent=2), encoding="utf-8"
    )
    logger.info("  Saved: %s (%d records)",
                paths["out_audit_trail"].name, len(audit_trail))

    paths["out_effectiveness"].write_text(
        json.dumps(effectiveness, indent=2), encoding="utf-8"
    )
    logger.info("  Saved: %s", paths["out_effectiveness"].name)

    report = {
        "module": "Module 5 — Closed-Loop Response Engine (RQ3/RO3)",
        "total_samples": n_samples,
        "total_alerts": len(records),
        "statistics": stats,
        "effectiveness": effectiveness,
        "mitigation_catalogue": {
            k: v["description"] for k, v in MITIGATION_ACTIONS.items()
        },
        "escalation_routing": {
            k: {kk: vv for kk, vv in v.items() if kk != "attack_specific_actions"}
            for k, v in ESCALATION_ROUTING.items()
        },
        "device_constraints": DEVICE_TIERS,
    }
    paths["out_response_report"].write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info("  Saved: %s", paths["out_response_report"].name)

    # CSV
    rows = []
    for rec in records:
        rows.append(
            {
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
            }
        )
    pd.DataFrame(rows).to_csv(paths["out_detail_csv"], index=False)
    logger.info("  Saved: %s", paths["out_detail_csv"].name)

    # Visualizations (test-only; demo skips charts to avoid clobbering
    # paper figures and to keep demo runs fast).
    if split == "test":
        logger.info("Generating charts...")
        plot_response_distribution(records)
        plot_precision_by_level(stats)
        plot_escalation_funnel(stats)
        plot_effectiveness_by_action(effectiveness)
        plot_response_sankey(audit_trail)

    logger.info("")
    logger.info(sep)
    logger.info("SPLIT %s COMPLETE", split.upper())
    logger.info(sep)
    logger.info("  Alerts         : %d", len(records))
    logger.info("  Audit records  : %d", len(audit_trail))
    logger.info("  Over-response  : %.1f%%", effectiveness["over_response_rate"] * 100)
    logger.info("  Under-response : %.1f%%", effectiveness["under_response_rate"] * 100)
    logger.info("  Output         : %s", paths["out_alert_responses"])
    logger.info(sep)


if __name__ == "__main__":
    main()
