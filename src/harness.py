"""Component 3: Alert Simulation Harness.

Replays the 50-alert labeled dataset through Components 2 and 1,
then runs all acceptance and negative tests.
Testing infrastructure only — not part of the production system.

Workflow (per research_spec.yaml component_3.workflow):
  1. Load alert_dataset from tests/fixtures/sample_alerts.yaml
  2. For each alert:
     a. Run through Risk-Adaptive Scoring Engine (Component 2)
     b. Compute baseline (static threshold) result for M6
     c. If should_surface: run through MVE Generator (Component 1)
     d. Store ScoredAlert + MVEOutput
  3. Run all acceptance_tests against collected outputs
  4. Run all negative_tests against collected outputs
  5. Generate alignment_report.yaml
  6. Print summary to stdout and return TestReport
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import yaml

from src import sanitize_for_log
from src.data_models import (
    AlertGroundTruth,
    AlertRecord,
    MVEOutput,
    TestReport,
)
from src.mve_generator import generate_mve
from src.risk_scorer import score_alert, score_alert_static

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SAMPLE_ALERTS_PATH = FIXTURES_DIR / "sample_alerts.yaml"


# ── Fixture loading ──────────────────────────────────────────────────────

def load_dataset(path: Optional[Path] = None) -> List[AlertRecord]:
    """Load alert dataset from YAML fixture and build AlertRecord objects.

    Args:
        path: Path to sample_alerts.yaml. Defaults to the canonical fixture.

    Returns:
        List of AlertRecord, one per alert in the fixture.
    """
    fixture_path = path or SAMPLE_ALERTS_PATH
    with open(fixture_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    records = []
    for a in data.get("alerts", []):
        gt = AlertGroundTruth(
            alert_id=a["alert_id"],
            true_severity=a["ground_truth_severity"],
            true_clinical_system=a["true_clinical_system"],
            true_label=a["ground_truth_label"],
            device_patchable=bool(a["device_context"]["patchable"]),
            device_criticality=a["device_context"]["criticality"],
        )
        records.append(AlertRecord(
            alert_id=a["alert_id"],
            raw_alert=a["raw_alert"],
            device_context=a["device_context"],
            behavioral_baseline=a["behavioral_baseline"],
            user_context=a.get("user_context"),
            ground_truth=gt,
            anomaly_score=float(a["anomaly_score"]),
            event_context=a.get("event_context"),
        ))

    logger.info(
        "Loaded %d alerts from %s",
        len(records),
        sanitize_for_log(fixture_path),
    )
    return records


# ── Pipeline execution ───────────────────────────────────────────────────

def _build_system_logs(records: List[AlertRecord]) -> List[dict[str, Any]]:
    """Build system action log for negative test_no_device_discovery_attempted.

    Records what the harness actually did — loads inventory from fixture,
    runs scoring, generates MVE. Never calls scan/discover/fingerprint.
    """
    logs: list[dict[str, Any]] = []
    for r in records:
        logs.append({"action": "score_alert", "alert_id": r.alert_id})
        if r.mve is not None:
            logs.append({"action": "generate_mve", "alert_id": r.alert_id})
    return logs


def _build_system_actions(records: List[AlertRecord]) -> List[dict[str, Any]]:
    """Build recommendation list for negative test_no_automated_blocking.

    All actions are tagged type='recommendation'. The harness never calls
    any enforcement function — it mirrors module5's ActionExecutor
    (simulated-only) design.
    """
    actions: list[dict[str, Any]] = []
    for r in records:
        if r.mve is not None:
            actions.append({
                "type": "recommendation",
                "alert_id": r.alert_id,
                "content": r.mve.layer_3.get("immediate_action", ""),
            })
    return actions


def run_simulation(
    dataset: Optional[List[AlertRecord]] = None,
    fixture_path: Optional[Path] = None,
) -> TestReport:
    """Run all 50 alerts through Components 2 and 1, then all tests.

    Args:
        dataset: Pre-loaded list of AlertRecord. If None, loads from fixture.
        fixture_path: Override path to sample_alerts.yaml (for testing).

    Returns:
        TestReport with metrics, negative_tests, and alignment results.
    """
    from tests.acceptance_tests import run_acceptance_tests
    from tests.negative_tests import run_negative_tests

    if dataset is None:
        dataset = load_dataset(fixture_path)

    baseline_results: List[dict[str, Any]] = []   # static threshold for M6
    adaptive_results: List[dict[str, Any]] = []   # adaptive for M6
    surfaced_records: List[AlertRecord] = []

    # ── Step 2: Run each alert through pipeline ──────────────────────────
    for record in dataset:
        # 2a. Risk-Adaptive Scoring (Component 2)
        scored = score_alert(
            anomaly_score=record.anomaly_score,
            device_context=record.device_context,
            event_context=record.event_context,
        )
        record.scored = scored

        # 2b. Static baseline for M6 comparison
        baseline_results.append(
            score_alert_static(record.anomaly_score)
        )
        adaptive_results.append({"surfaced": scored.should_surface})

        # 2c. MVE Generator (Component 1) — only for surfaced alerts
        if scored.should_surface:
            mve = generate_mve(
                raw_alert=record.raw_alert,
                device_context=record.device_context,
                baseline=record.behavioral_baseline,
                user_context=record.user_context,
            )
            record.mve = mve
            surfaced_records.append(record)

    logger.info(
        "Pipeline: %d alerts processed, %d surfaced, %d suppressed",
        len(dataset),
        len(surfaced_records),
        len(dataset) - len(surfaced_records),
    )

    # ── Step 3: Extract lists for acceptance tests ───────────────────────
    mve_outputs: List[MVEOutput] = [
        r.mve for r in surfaced_records if r.mve is not None
    ]
    surfaced_gts: List[AlertGroundTruth] = [r.ground_truth for r in surfaced_records]
    all_gts: List[AlertGroundTruth] = [r.ground_truth for r in dataset]

    # ── Step 4: Build test inputs for negative tests ─────────────────────
    mve_dicts: list[dict[str, Any]] = [
        r.mve.to_dict(alert_id=r.alert_id)
        for r in surfaced_records
        if r.mve is not None
    ]
    system_logs = _build_system_logs(dataset)
    system_actions = _build_system_actions(dataset)

    # ── Step 5: Run acceptance tests ─────────────────────────────────────
    metric_results = run_acceptance_tests(
        mve_outputs=mve_outputs,
        ground_truths=surfaced_gts,
        baseline_results=baseline_results,
        adaptive_results=adaptive_results,
    )
    # M6 needs all-alerts ground truths (not just surfaced)
    # Patch M6 to use full dataset GTs
    for m in metric_results:
        if m["metric_id"] == "M6":
            from tests.acceptance_tests import test_false_positive_rate
            try:
                val = test_false_positive_rate(
                    baseline_results, adaptive_results, all_gts
                )
                m["result_value"] = round(val, 4)
                if val >= m["target"]:
                    m["pass_fail"] = "PASS"  # noqa: S105 — status string, not credential
                elif val >= m["minimum"]:
                    m["pass_fail"] = "WARN"  # noqa: S105
                else:
                    m["pass_fail"] = "FAIL"  # noqa: S105
                m["detail"] = ""
            except AssertionError as exc:
                m["pass_fail"] = "FAIL"  # noqa: S105
                m["detail"] = str(exc)

    # ── Step 6: Run negative tests ────────────────────────────────────────
    negative_results = run_negative_tests(mve_dicts, system_logs, system_actions)

    # ── Step 7: Build alignment ───────────────────────────────────────────
    alignment = _build_alignment(metric_results)

    return TestReport(
        metrics=metric_results,
        negative_tests=negative_results,
        alignment=alignment,
    )


# ── Alignment mapping ────────────────────────────────────────────────────

_CLAIM_MAP: list[dict[str, Any]] = [
    {
        "claim_id": "C1",
        "claim_text": "explainable anomaly narratives that translate network "
                      "detections into clinically contextualized alerts",
        "supported_by": ["M2", "M8"],
    },
    {
        "claim_id": "C2",
        "claim_text": "risk-adaptive thresholds that auto-adjust detection sensitivity",
        "supported_by": ["M7", "M6"],
    },
    {
        "claim_id": "C3",
        "claim_text": "clinical-constraint-aware response recommendations",
        "supported_by": ["M3", "M4"],
    },
    {
        "claim_id": "C7",
        "claim_text": "MVE structural completeness",
        "supported_by": ["M1", "M1b"],
    },
    {
        "claim_id": "C8",
        "claim_text": "alert fatigue reduction",
        "supported_by": ["M6"],
    },
    {
        "claim_id": "C4",
        "claim_text": "enabling correct triage from non-specialist operators",
        "supported_by": [],
        "verdict": "NOT_TESTED — requires A/B user study with n≥20 IT generalists (Phase 2)",
    },
    {
        "claim_id": "C5",
        "claim_text": "reducing dwell time for non-ransomware intrusions",
        "supported_by": [],
        "verdict": "NOT_TESTED — requires longitudinal field deployment (Phase 3)",
    },
]


def _build_alignment(metric_results: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Map metrics to research claims and compute verdicts."""
    metric_by_id: dict[str, dict[str, Any]] = {
        m["metric_id"]: m for m in metric_results
    }
    alignment: list[dict[str, Any]] = []

    for claim in _CLAIM_MAP:
        if not claim["supported_by"]:
            # Untested claims (C4, C5)
            alignment.append({
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "supported_by": [],
                "all_tests_pass": False,
                "verdict": claim.get("verdict", "NOT_TESTED"),
            })
            continue

        supporting = [
            metric_by_id[mid]
            for mid in claim["supported_by"]
            if mid in metric_by_id
        ]
        all_pass = all(m["pass_fail"] in ("PASS", "WARN") for m in supporting)
        all_target = all(m["pass_fail"] == "PASS" for m in supporting)

        if all_target:
            verdict = "SUPPORTED"
        elif all_pass:
            verdict = "PARTIAL"
        else:
            verdict = "NOT_SUPPORTED"

        alignment.append({
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "supported_by": claim["supported_by"],
            "all_tests_pass": all_pass,
            "verdict": verdict,
        })

    return alignment
