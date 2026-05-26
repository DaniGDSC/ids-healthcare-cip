"""ActionExecutor (simulated) + NotificationService for Module 5."""
from __future__ import annotations

from datetime import datetime


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
        has_mitigation = any(
            a in actions
            for a in ("isolate_device", "restrict_traffic", "re_authenticate")
        )
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
            "clinical_override": recommendation.get("clinical_override", {}).get(
                "triggered", False
            ),
            "requires_approval": recommendation.get("requires_approval", False),
            "outcome": outcome,
            "effective": effective,
            "ground_truth": ground_truth,
        }
        self.execution_log.append(record)
        return record


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

        msgs.append(
            {
                "recipient": recommendation["primary_notify"],
                "channel": "SIEM + Dashboard",
                "priority": alert_tier,
                "message": (
                    f"[{alert_tier}] Alert #{sample_index}: "
                    f"Risk={risk_score:.2f}. Actions: {', '.join(recommendation['actions'])}. "
                    f"Top features: {', '.join(f['feature'] for f in analyst_top_features[:3])}."
                ),
            }
        )

        if "escalate_clinical" in recommendation["actions"]:
            msgs.append(
                {
                    "recipient": "Clinical Staff",
                    "channel": "Page / Dashboard",
                    "priority": alert_tier,
                    "message": clinician_summary[:300]
                    if clinician_summary
                    else "Clinical review requested.",
                }
            )

        if recommendation.get("secondary_notify"):
            msgs.append(
                {
                    "recipient": recommendation["secondary_notify"],
                    "channel": "Email / Ticket",
                    "priority": alert_tier,
                    "message": f"[{alert_tier}] Sample #{sample_index}: {', '.join(recommendation['actions'])}",
                }
            )

        self.notifications.extend(msgs)
        return msgs


__all__ = ["ActionExecutor", "NotificationService"]
