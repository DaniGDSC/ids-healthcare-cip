"""Effectiveness + per-level statistics over Module 5 audit records."""
from __future__ import annotations

from collections import Counter, defaultdict

from .config import ACTION_CATALOGUE


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

    for stats in action_stats.values():
        t = stats["total"]
        stats["precision"] = round(stats["true_attacks"] / t, 4) if t > 0 else 0
        stats["false_positive_rate"] = (
            round(stats["false_positives"] / t, 4) if t > 0 else 0
        )

    costly_actions = sorted(
        action_stats.keys(),
        key=lambda a: ACTION_CATALOGUE.get(a, {}).get("cost", 0),
        reverse=True,
    )
    proportionality = [
        {
            "action": a,
            "cost": ACTION_CATALOGUE.get(a, {}).get("cost", 0),
            "precision": action_stats[a]["precision"],
            "total": action_stats[a]["total"],
        }
        for a in costly_actions
    ]

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


def compute_response_stats(records: list) -> dict:
    """Aggregate response statistics."""
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


__all__ = ["compute_effectiveness", "compute_response_stats"]
