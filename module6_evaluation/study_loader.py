"""Load and serve user study alert scenarios."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AlertScenario:
    alert_id: str
    alert_type: str
    group_a_display: str
    group_b_display: str
    correct_severity: str
    correct_action: str
    ground_truth_label: str
    # Optional list of additional actions that count as acceptable for accuracy
    # scoring. Empty default = strict accuracy (only correct_action counts).
    reasonable_alternatives: list[str] = field(default_factory=list)


import json
import random

def load_study_alerts(participant_id: str = "default_seed") -> list[AlertScenario]:
    """Load 20 alert scenarios from JSON evaluation pipeline output."""
    path = PROJECT_ROOT / "results/reports/evaluation_alerts.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = []
    for alert in data:
        scenarios.append(AlertScenario(
            alert_id=alert["alert_id"],
            alert_type=alert.get("attack_category", "unknown"),
            group_a_display=alert["group_a_display"],
            group_b_display=alert["group_b_display"],
            correct_severity=alert["risk_level"],
            correct_action=alert.get("correct_action", ""),
            ground_truth_label=alert["ground_truth"],
            reasonable_alternatives=list(alert.get("reasonable_alternatives", [])),
        ))

    # Sort dynamically using participant ID as a seed to ensure exact presentation
    # reproduction per user while distributing biases evenly across population.
    pid_seed = int(hashlib.md5(participant_id.encode()).hexdigest(), 16)
    rng = random.Random(pid_seed)
    rng.shuffle(scenarios)

    return scenarios


def assign_ab_condition(participant_id: str, alert_index: int,
                        n_alerts: int = 20) -> bool:
    """
    Returns True = show MVE (Group B), False = hide MVE (Group A).
    Counterbalanced: first half and second half swap based on participant.
    """
    pid_num = int(hashlib.md5(participant_id.encode()).hexdigest(), 16)
    half = n_alerts // 2
    if pid_num % 2 == 0:
        return alert_index < half   # even PID: MVE first
    else:
        return alert_index >= half  # odd PID: no-MVE first
