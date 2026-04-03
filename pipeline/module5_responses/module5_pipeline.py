#!/usr/bin/env python3
"""Module 5 — Response Pipeline Integration (Tasks 5.1–5.8).

Wraps the closed-loop response engine (generate_responses.py) into
the class-based structure required by the thesis spec:
  5.1  Export standalone response_policy.json
  5.2  PolicyEngine class
  5.3  Clinical safety override with confirmation request
  5.4  Simulated ActionExecutor with audit trail
  5.5  NotificationService per stakeholder
  5.6  Immutable audit logger (JSONL)
  5.7  End-to-end worked examples (CRITICAL/HIGH/LOW)
  5.8  Feedback loop stub

Usage:
    python run_response_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data/phase2/responses"

BIOMETRIC_FEATURES = frozenset({
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
    "Heart_rate", "Resp_Rate", "ST",
})


# ═══════════════════════════════════════════════════════════════════════
# 5.1  Response Policy Config
# ═══════════════════════════════════════════════════════════════════════

RESPONSE_POLICY = {
    "version": "1.0",
    "description": "Maps (alert_tier, device_tier, patient_acuity_level) to response action sets",
    "action_catalogue": {
        "log_event":           {"cost": 0.1, "reversible": True,  "requires_approval": False},
        "enhanced_monitoring": {"cost": 0.2, "reversible": True,  "requires_approval": False},
        "re_authenticate":     {"cost": 0.3, "reversible": True,  "requires_approval": False},
        "restrict_traffic":    {"cost": 0.5, "reversible": True,  "requires_approval": False},
        "isolate_device":      {"cost": 0.8, "reversible": True,  "requires_approval": True},
        "forensic_snapshot":   {"cost": 0.4, "reversible": True,  "requires_approval": False},
        "escalate_clinical":   {"cost": 0.7, "reversible": False, "requires_approval": False},
        "escalate_incident":   {"cost": 1.0, "reversible": False, "requires_approval": False},
    },
    "tier_policies": {
        "CRITICAL": {
            "default_actions": ["log_event", "isolate_device", "forensic_snapshot", "escalate_incident", "escalate_clinical"],
            "max_response_min": 5,
            "auto_execute": True,
        },
        "HIGH": {
            "default_actions": ["log_event", "isolate_device", "forensic_snapshot", "enhanced_monitoring"],
            "max_response_min": 15,
            "auto_execute": True,
        },
        "MEDIUM": {
            "default_actions": ["log_event", "restrict_traffic", "enhanced_monitoring"],
            "max_response_min": 60,
            "auto_execute": False,
        },
        "LOW": {
            "default_actions": ["log_event", "enhanced_monitoring"],
            "max_response_min": 480,
            "auto_execute": False,
        },
    },
    "device_constraints": {
        "life_sustaining":  {"max_action_cost": 0.5, "isolation_blocked": True,  "clinical_approval_required": True},
        "vital_monitoring": {"max_action_cost": 0.8, "isolation_blocked": False, "clinical_approval_required": True},
        "diagnostic":       {"max_action_cost": 0.8, "isolation_blocked": False, "clinical_approval_required": False},
        "auxiliary":        {"max_action_cost": 0.8, "isolation_blocked": False, "clinical_approval_required": False},
    },
    "acuity_overrides": {
        "elevated_acuity_threshold": 0.25,
        "action_on_elevated": "Add escalate_clinical if not present; require clinical confirmation before isolation",
    },
    "attack_routing": {
        "Spoofing":        {"add_actions": ["re_authenticate"], "primary_notify": "IT Security", "secondary_notify": "Biomedical Engineering"},
        "Data Alteration": {"add_actions": ["forensic_snapshot", "escalate_clinical"], "primary_notify": "IT Security", "secondary_notify": "Charge Nurse"},
    },
}


def export_response_policy() -> None:
    """Task 5.1: Export standalone response policy config."""
    path = OUTPUT_DIR / "response_policy.json"
    path.write_text(json.dumps(RESPONSE_POLICY, indent=2), encoding="utf-8")
    logger.info("5.1 Saved: response_policy.json")


# ═══════════════════════════════════════════════════════════════════════
# 5.2  PolicyEngine
# ═══════════════════════════════════════════════════════════════════════

class PolicyEngine:
    """Rule-based engine: reads policy config, returns recommended actions."""

    def __init__(self, policy: dict = RESPONSE_POLICY):
        self.policy = policy
        self.catalogue = policy["action_catalogue"]

    def recommend(
        self,
        alert_tier: str,
        device_tier: str = "vital_monitoring",
        attack_category: str = "unknown",
        patient_acuity: float = 0.0,
    ) -> dict:
        tier_policy = self.policy["tier_policies"].get(alert_tier, self.policy["tier_policies"]["LOW"])
        actions = list(tier_policy["default_actions"])

        # Attack-specific actions
        routing = self.policy["attack_routing"].get(attack_category, {})
        for a in routing.get("add_actions", []):
            if a not in actions:
                actions.append(a)

        # Device constraints
        constraint = self.policy["device_constraints"].get(device_tier, {})
        max_cost = constraint.get("max_action_cost", 1.0)
        if constraint.get("isolation_blocked") and "isolate_device" in actions:
            actions.remove("isolate_device")
            if "restrict_traffic" not in actions:
                actions.append("restrict_traffic")

        actions = [a for a in actions if self.catalogue.get(a, {}).get("cost", 0) <= max_cost
                   or a in ("log_event", "escalate_clinical")]

        # Clinical safety override (Task 5.3)
        override = clinical_safety_check(
            alert_tier, device_tier, patient_acuity, actions,
        )

        # Sort by cost
        actions = sorted(set(actions), key=lambda a: self.catalogue.get(a, {}).get("cost", 0))

        return {
            "actions": actions,
            "max_response_min": tier_policy["max_response_min"],
            "auto_execute": tier_policy["auto_execute"],
            "primary_notify": routing.get("primary_notify", "IT Security"),
            "secondary_notify": routing.get("secondary_notify"),
            "clinical_override": override,
            "requires_approval": any(
                self.catalogue.get(a, {}).get("requires_approval", False) for a in actions
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# 5.3  Clinical Safety Override
# ═══════════════════════════════════════════════════════════════════════

def clinical_safety_check(
    alert_tier: str,
    device_tier: str,
    patient_acuity: float,
    actions: list,
) -> dict:
    """Check if device is safety-critical AND patient acuity elevated → override."""
    override = {
        "triggered": False,
        "reason": None,
        "original_actions": list(actions),
        "clinical_confirmation_required": False,
    }

    is_critical_device = device_tier in ("life_sustaining", "vital_monitoring")
    acuity_elevated = patient_acuity >= RESPONSE_POLICY["acuity_overrides"]["elevated_acuity_threshold"]

    if is_critical_device and acuity_elevated:
        override["triggered"] = True
        override["clinical_confirmation_required"] = True

        if "isolate_device" in actions:
            override["reason"] = (
                f"Device tier '{device_tier}' with elevated patient acuity "
                f"({patient_acuity:.2f}) — isolation downgraded to restrict_traffic. "
                "Clinical confirmation required before any disruptive action."
            )
            actions.remove("isolate_device")
            if "restrict_traffic" not in actions:
                actions.append("restrict_traffic")
        else:
            override["reason"] = (
                f"Device tier '{device_tier}' with elevated acuity ({patient_acuity:.2f}) — "
                "clinical confirmation required."
            )

        if "escalate_clinical" not in actions:
            actions.append("escalate_clinical")

    return override


# ═══════════════════════════════════════════════════════════════════════
# 5.4  ActionExecutor (simulated)
# ═══════════════════════════════════════════════════════════════════════

class ActionExecutor:
    """Simulated executor: logs actions to audit trail instead of real changes."""

    def __init__(self):
        self.execution_log = []

    def execute(
        self,
        alert_id: str,
        sample_index: int,
        actions: list,
        recommendation: dict,
        ground_truth: str,
        timestamp: datetime,
    ) -> dict:
        has_mitigation = any(a in actions for a in
                            ("isolate_device", "restrict_traffic", "re_authenticate"))
        is_attack = ground_truth == "attack"

        if is_attack and has_mitigation:
            outcome = "threat_contained"
            effective = True
        elif is_attack and not has_mitigation:
            outcome = "threat_logged_not_mitigated"
            effective = False
        elif not is_attack and has_mitigation:
            outcome = "false_positive_isolated"
            effective = False
        else:
            outcome = "benign_logged"
            effective = True

        record = {
            "alert_id": alert_id,
            "sample_index": sample_index,
            "timestamp": timestamp.isoformat(),
            "actions_executed": actions,
            "auto_executed": recommendation.get("auto_execute", False),
            "clinical_override": recommendation.get("clinical_override", {}).get("triggered", False),
            "requires_approval": recommendation.get("requires_approval", False),
            "outcome": outcome,
            "effective": effective,
            "ground_truth": ground_truth,
        }
        self.execution_log.append(record)
        return record


# ═══════════════════════════════════════════════════════════════════════
# 5.5  NotificationService
# ═══════════════════════════════════════════════════════════════════════

class NotificationService:
    """Generate structured alert messages per stakeholder."""

    def __init__(self):
        self.notifications = []

    def notify(
        self,
        sample_index: int,
        alert_tier: str,
        recommendation: dict,
        clinician_summary: str,
        analyst_top_features: list,
        risk_score: float,
    ) -> list:
        msgs = []

        # Security analyst notification
        msgs.append({
            "recipient": recommendation["primary_notify"],
            "channel": "SIEM + Dashboard",
            "priority": alert_tier,
            "message": (
                f"[{alert_tier}] Alert #{sample_index}: "
                f"Risk={risk_score:.2f}. Actions: {', '.join(recommendation['actions'])}. "
                f"Top features: {', '.join(f['feature'] for f in analyst_top_features[:3])}."
            ),
        })

        # Clinical notification (if escalation required)
        if "escalate_clinical" in recommendation["actions"]:
            msgs.append({
                "recipient": "Clinical Staff",
                "channel": "Page / Dashboard",
                "priority": alert_tier,
                "message": clinician_summary[:300] if clinician_summary else "Clinical review requested.",
            })

        # Secondary notification
        if recommendation.get("secondary_notify"):
            msgs.append({
                "recipient": recommendation["secondary_notify"],
                "channel": "Email / Ticket",
                "priority": alert_tier,
                "message": f"[{alert_tier}] Sample #{sample_index}: {', '.join(recommendation['actions'])}",
            })

        self.notifications.extend(msgs)
        return msgs


# ═══════════════════════════════════════════════════════════════════════
# 5.6  Audit Logger (JSONL)
# ═══════════════════════════════════════════════════════════════════════

class AuditLogger:
    """Immutable append-only JSONL audit log with integrity hashes."""

    def __init__(self, path: Path):
        self.path = path
        self.prev_hash = "0" * 64

    def log(self, record: dict) -> None:
        record["prev_hash"] = self.prev_hash
        payload = json.dumps(record, sort_keys=True)
        record["integrity_hash"] = hashlib.sha256(payload.encode()).hexdigest()
        self.prev_hash = record["integrity_hash"]

        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ═══════════════════════════════════════════════════════════════════════
# 5.8  Feedback Loop Stub
# ═══════════════════════════════════════════════════════════════════════

class FeedbackLoop:
    """Record TP/FP labels; suggest weight/threshold adjustments."""

    def __init__(self):
        self.records = []

    def record(self, alert_id: str, ground_truth: str, predicted_tier: str,
               risk_score: float, actions: list) -> None:
        self.records.append({
            "alert_id": alert_id,
            "ground_truth": ground_truth,
            "predicted_tier": predicted_tier,
            "risk_score": risk_score,
            "actions": actions,
            "is_tp": ground_truth == "attack" and predicted_tier in ("MEDIUM", "HIGH", "CRITICAL"),
            "is_fp": ground_truth == "benign" and predicted_tier in ("MEDIUM", "HIGH", "CRITICAL"),
            "is_fn": ground_truth == "attack" and predicted_tier == "LOW",
        })

    def compute_adjustments(self) -> dict:
        """Suggest weight/threshold adjustments based on TP/FP/FN rates."""
        if not self.records:
            return {}

        tp = sum(1 for r in self.records if r["is_tp"])
        fp = sum(1 for r in self.records if r["is_fp"])
        fn = sum(1 for r in self.records if r["is_fn"])
        total = len(self.records)

        fp_rate = fp / total if total > 0 else 0
        fn_rate = fn / total if total > 0 else 0

        suggestions = []
        if fp_rate > 0.10:
            suggestions.append(
                f"High FP rate ({fp_rate:.1%}): consider raising MEDIUM threshold from 0.40 to 0.45"
            )
        if fn_rate > 0.05:
            suggestions.append(
                f"FN rate ({fn_rate:.1%}): consider increasing w1 (C_detect weight) to capture more attacks"
            )
        if fp_rate < 0.02 and fn_rate < 0.01:
            suggestions.append("System well-calibrated. No adjustments recommended.")

        # Risk score distribution for FP vs TP
        fp_scores = [r["risk_score"] for r in self.records if r["is_fp"]]
        tp_scores = [r["risk_score"] for r in self.records if r["is_tp"]]

        return {
            "total_evaluated": total,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "fp_rate": round(fp_rate, 4),
            "fn_rate": round(fn_rate, 4),
            "mean_fp_risk_score": round(float(np.mean(fp_scores)), 4) if fp_scores else None,
            "mean_tp_risk_score": round(float(np.mean(tp_scores)), 4) if tp_scores else None,
            "suggestions": suggestions,
        }


# ═══════════════════════════════════════════════════════════════════════
# 5.7  End-to-End Worked Examples
# ═══════════════════════════════════════════════════════════════════════

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

    # Find one sample per tier
    target_tiers = ["CRITICAL", "HIGH", "LOW"]
    for tier in target_tiers:
        mask = (levels == tier) & (y_true == 1)  # prefer true attacks
        if not mask.any():
            mask = levels == tier
        if not mask.any():
            continue

        idx = int(np.where(mask)[0][np.argmax(R[mask])])
        cat = str(attack_cats[idx])
        gt = "attack" if y_true[idx] == 1 else "benign"
        a_pat = float(risk_data["a_patient"][idx])

        # Step 1: Policy recommendation
        rec = engine.recommend(
            alert_tier=tier,
            device_tier="vital_monitoring",
            attack_category=cat,
            patient_acuity=a_pat,
        )

        # Step 2: Execute (simulated)
        ts = datetime(2026, 4, 3, 12, 0, 0) + timedelta(seconds=idx)
        exec_result = executor.execute(
            f"ALERT-{idx:05d}", idx, rec["actions"], rec, gt, ts,
        )

        # Step 3: Notify
        clin_summary = clinician_by_idx.get(idx, {}).get("summary", "")
        analyst_feats = []
        if idx in analyst_by_idx:
            analyst_feats = analyst_by_idx[idx].get("models", {}).get("xgboost", {}).get("top_features", [])

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
                "A_patient": round(float(risk_data["a_patient"][idx]), 4),
            },
            "policy_recommendation": rec,
            "execution_result": exec_result,
            "notifications": notifications,
            "clinical_override": rec["clinical_override"],
        }
        scenarios.append(scenario)
        logger.info("  %s: sample %d, R=%.4f, actions=%s, outcome=%s",
                    tier, idx, float(R[idx]), rec["actions"], exec_result["outcome"])

    return scenarios


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
    logger.info("MODULE 5 — RESPONSE PIPELINE INTEGRATION (Tasks 5.1-5.8)")
    logger.info(sep)

    # Load data
    risk_data = {k: v for k, v in
                 np.load(PROJECT_ROOT / "data/phase2/risk_scores/risk_scores.npz",
                         allow_pickle=True).items()}
    with open(PROJECT_ROOT / "data/phase2/explanations/analyst_report.json") as f:
        analyst_by_idx = {a["sample_index"]: a for a in json.load(f)}
    with open(PROJECT_ROOT / "data/phase2/explanations/clinician_summaries.json") as f:
        clinician_by_idx = {s["sample_index"]: s for s in json.load(f)}
    attack_cats = pd.read_parquet(
        PROJECT_ROOT / "data/processed/test_phase1.parquet",
        columns=["Attack Category"],
    )["Attack Category"].values

    n_samples = len(risk_data["R"])
    logger.info("Loaded: %d samples", n_samples)

    # 5.1 Export policy config
    export_response_policy()

    # 5.7 End-to-end worked examples
    logger.info("")
    logger.info("── 5.7 End-to-End Worked Examples ──")
    scenarios = run_worked_examples(risk_data, attack_cats, analyst_by_idx, clinician_by_idx)
    (OUTPUT_DIR / "worked_examples.json").write_text(
        json.dumps(scenarios, indent=2, default=str), encoding="utf-8")
    logger.info("  Saved: worked_examples.json (%d scenarios)", len(scenarios))

    # 5.6 + 5.8 Full pipeline run with audit logger + feedback loop
    logger.info("")
    logger.info("── 5.6/5.8 Full Pipeline Run (audit + feedback) ──")

    engine = PolicyEngine()
    executor = ActionExecutor()
    notifier = NotificationService()
    audit = AuditLogger(OUTPUT_DIR / "audit_log.jsonl")
    feedback = FeedbackLoop()

    R = risk_data["R"]
    levels = risk_data["risk_levels"]
    y_true = risk_data["y_true"]

    alert_count = 0
    for idx in range(n_samples):
        tier = str(levels[idx])
        if tier == "LOW" and R[idx] < 0.25:
            continue  # skip lowest-risk LOW alerts for efficiency

        cat = str(attack_cats[idx])
        gt = "attack" if y_true[idx] == 1 else "benign"
        a_pat = float(risk_data["a_patient"][idx])

        rec = engine.recommend(tier, "vital_monitoring", cat, a_pat)
        ts = datetime(2026, 4, 3, 12, 0, 0) + timedelta(seconds=idx)
        alert_id = f"ALERT-{idx:05d}"

        exec_result = executor.execute(alert_id, idx, rec["actions"], rec, gt, ts)
        audit.log(exec_result)
        feedback.record(alert_id, gt, tier, float(R[idx]), rec["actions"])
        alert_count += 1

    logger.info("  Processed %d alerts through pipeline", alert_count)
    logger.info("  Audit log: %s (%d records)", OUTPUT_DIR / "audit_log.jsonl", alert_count)

    # 5.8 Feedback analysis
    adjustments = feedback.compute_adjustments()
    (OUTPUT_DIR / "feedback_analysis.json").write_text(
        json.dumps(adjustments, indent=2), encoding="utf-8")
    logger.info("")
    logger.info("── 5.8 Feedback Loop Analysis ──")
    logger.info("  TP=%d, FP=%d, FN=%d", adjustments["true_positives"],
                adjustments["false_positives"], adjustments["false_negatives"])
    logger.info("  FP rate: %.1f%%, FN rate: %.1f%%",
                adjustments["fp_rate"] * 100, adjustments["fn_rate"] * 100)
    for s in adjustments["suggestions"]:
        logger.info("  Suggestion: %s", s)
    logger.info("  Saved: feedback_analysis.json")

    # Notification stats
    logger.info("")
    logger.info("  Notifications generated: %d", len(notifier.notifications))

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("RESPONSE PIPELINE COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    logger.info("  5.1 response_policy.json")
    logger.info("  5.6 audit_log.jsonl (%d records)", alert_count)
    logger.info("  5.7 worked_examples.json (%d scenarios)", len(scenarios))
    logger.info("  5.8 feedback_analysis.json")
    logger.info("  Output: %s", OUTPUT_DIR)
    logger.info(sep)


if __name__ == "__main__":
    main()
