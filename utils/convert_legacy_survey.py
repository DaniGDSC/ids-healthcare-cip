#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_DIR = PROJECT_ROOT / "survey" / "result"
OUTPUT_DIR = PROJECT_ROOT / "results" / "reports"
SCENARIOS_PATH = PROJECT_ROOT / "tests" / "fixtures" / "user_study_alert_scenarios.yaml"

LEVEL = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
PAIR_RE = re.compile(r"responses_(P\d+)_group([AB])\.json$")


def load_scenarios() -> dict[str, dict[str, Any]]:
    with open(SCENARIOS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    scenarios: dict[str, dict[str, Any]] = {}
    for alert in data.get("alerts", []):
        alert_id = str(alert["alert_id"])
        gt = alert["ground_truth"]
        scenarios[alert_id] = {
            "alert_type": alert.get("type"),
            "alert_index": int(alert.get("order", 0)) - 1 if alert.get("order") is not None else None,
            "correct_severity": gt["severity"],
            "correct_action": gt["correct_action"],
            "ground_truth_label": gt["label"],
        }
    return scenarios


def severity_score(chosen: str, correct: str) -> tuple[bool, float, bool]:
    chosen_norm = str(chosen).upper()
    correct_norm = str(correct).upper()
    diff = abs(LEVEL.get(chosen_norm, -1) - LEVEL.get(correct_norm, -1))
    score = 1.0 if diff == 0 else (0.5 if diff == 1 else 0.0)
    return chosen_norm == correct_norm, score, diff == 3


def convert_response(
    *,
    participant_id: str,
    group_name: str,
    row: dict[str, Any],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    chosen_severity = str(row.get("chosen_severity", "")).upper()
    correct_severity = str(scenario["correct_severity"]).upper()
    sev_correct, sev_score, catastrophic = severity_score(chosen_severity, correct_severity)

    chosen_action = str(row.get("chosen_action", "")).strip().lower()
    correct_action = str(scenario["correct_action"]).strip().lower()
    action_correct = chosen_action == correct_action
    composite_score = (sev_score + (1.0 if action_correct else 0.0)) / 2

    return {
        "participant_id": participant_id,
        "participant_role": "Legacy Survey Participant",
        "alert_id": str(row["alert_id"]),
        "alert_type": scenario["alert_type"],
        "alert_index": scenario["alert_index"],
        "condition": "with_mve" if group_name == "B" else "without_mve",
        "chosen_severity": chosen_severity,
        "correct_severity": correct_severity,
        "severity_correct": sev_correct,
        "severity_score": sev_score,
        "catastrophic_miss": catastrophic,
        "chosen_action": chosen_action,
        "correct_action": correct_action,
        "action_correct": action_correct,
        "composite_score": composite_score,
        "confidence": row.get("confidence"),
        "decision_time_sec": row.get("decision_time_sec"),
        "ground_truth_label": scenario["ground_truth_label"],
        "reasoning_note": row.get("reasoning_note", ""),
    }


def extract_participant_files() -> dict[str, dict[str, Path]]:
    pairs: dict[str, dict[str, Path]] = {}
    for path in sorted(LEGACY_DIR.glob("responses_P*_group*.json")):
        match = PAIR_RE.match(path.name)
        if not match:
            continue
        participant_id, group_name = match.groups()
        pairs.setdefault(participant_id, {})[group_name] = path
    return pairs


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a top-level JSON list")
    return [row for row in data if isinstance(row, dict)]


def build_proxy_row(participant_id: str, grouped_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    # Prefer the more explanation-rich Group B proxy if present; otherwise Group A.
    proxy_source = None
    for group_name in ("B", "A"):
        rows = grouped_rows.get(group_name, [])
        proxy = next((row for row in rows if row.get("alert_id") == "PROXY"), None)
        if proxy is not None:
            proxy_source = proxy
            break

    if proxy_source is None:
        return None

    return {
        "participant_id": participant_id,
        "q21_clinical_clarity": proxy_source.get("q21_clinical_clarity"),
        "q21_note": proxy_source.get("q21_note", ""),
        "q22_management_justification": proxy_source.get("q22_management_justification"),
        "q22_note": proxy_source.get("q22_note", ""),
    }


def convert_participant(
    participant_id: str,
    files: dict[str, Path],
    scenarios: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_rows = {group_name: load_json_list(path) for group_name, path in files.items()}

    converted: list[dict[str, Any]] = []
    for group_name in ("A", "B"):
        for row in grouped_rows.get(group_name, []):
            alert_id = row.get("alert_id")
            if alert_id == "PROXY":
                continue
            if alert_id not in scenarios:
                raise KeyError(f"Unknown alert_id {alert_id!r} in participant {participant_id} group {group_name}")
            converted.append(
                convert_response(
                    participant_id=participant_id,
                    group_name=group_name,
                    row=row,
                    scenario=scenarios[alert_id],
                )
            )

    converted.sort(key=lambda row: (row.get("alert_index", 999), row["condition"]))

    proxy_row = build_proxy_row(participant_id, grouped_rows)
    if proxy_row is not None:
        converted.append(proxy_row)

    return converted


def main() -> None:
    scenarios = load_scenarios()
    participant_files = extract_participant_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not participant_files:
        print(f"No legacy survey files found in {LEGACY_DIR}")
        return

    written = 0
    for participant_id, files in sorted(participant_files.items()):
        if "A" not in files or "B" not in files:
            print(f"Skipping {participant_id}: missing paired Group A/B files")
            continue

        converted = convert_participant(participant_id, files, scenarios)
        out_path = OUTPUT_DIR / f"study_responses_{participant_id}.json"
        out_path.write_text(json.dumps(converted, indent=2), encoding="utf-8")
        written += 1
        print(f"Wrote {out_path.relative_to(PROJECT_ROOT)} ({len(converted)} rows)")

    print(f"Converted {written} participants into {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
