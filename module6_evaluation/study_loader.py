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


# Frozen parity map for the 10 enrolled study participants. Their
# survey responses were collected under the original MD5-seeded
# counterbalancing; replacing the hash for them would invalidate the
# already-collected analysis. New participants (any PID outside this
# table) use SHA-256 to match the rest of the project's reproducibility
# seeds — keeps the documented hash family uniform going forward.
_FROZEN_PID_PARITY: dict[str, int] = {
    "P01": 1, "P02": 1, "P03": 1, "P04": 1, "P05": 0,
    "P06": 0, "P07": 0, "P08": 0, "P09": 0, "P10": 0,
}


def assign_ab_condition(participant_id: str, alert_index: int,
                        n_alerts: int = 20) -> bool:
    """
    Returns True = show MVE (Group B), False = hide MVE (Group A).
    Counterbalanced: first half and second half swap based on participant.

    Hash family: SHA-256 of the participant ID, matching the rest of the
    project's reproducibility seeds (e.g. ``forms.assign_ab_conditions``).
    The 10 enrolled study participants (P01..P10) use a frozen lookup
    table because their survey responses were already collected under
    the original MD5 algorithm; switching their hash would invalidate
    the analysis.
    """
    if participant_id in _FROZEN_PID_PARITY:
        pid_parity = _FROZEN_PID_PARITY[participant_id]
    else:
        pid_parity = int(hashlib.sha256(participant_id.encode()).hexdigest(), 16) % 2
    half = n_alerts // 2
    if pid_parity == 0:
        return alert_index < half   # even parity: MVE first
    else:
        return alert_index >= half  # odd parity: no-MVE first
