"""Load and serve user study alert scenarios."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

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


def load_study_alerts() -> list[AlertScenario]:
    """Load 20 alert scenarios from fixture file."""
    path = PROJECT_ROOT / "tests/fixtures/user_study_alert_scenarios.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    scoring = data["scoring_key"]["correct_answers"]
    scenarios = []

    for alert in data["alerts"]:
        aid = alert["alert_id"]
        scenarios.append(AlertScenario(
            alert_id=aid,
            alert_type=alert["type"],
            group_a_display=alert["group_a"]["display"],
            group_b_display=alert["group_b"]["display"],
            correct_severity=scoring[aid]["severity"],
            correct_action=scoring[aid]["action"],
            ground_truth_label=alert["ground_truth"]["label"],
        ))

    # Sort by presentation order
    order = data["scoring_key"]["presentation_order"]["order"]
    order_map = {aid: i for i, aid in enumerate(order)}
    scenarios.sort(key=lambda s: order_map.get(s.alert_id, 99))

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
