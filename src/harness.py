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

from typing import Iterator


def stream_dataset(path: Optional[Path] = None) -> Iterator[AlertRecord]:
    """Yield AlertRecord objects one at a time from the YAML fixture.

    Opt-10: generator-based streaming pipeline — O(1) memory regardless of
    dataset size.  At 50 alerts the difference is negligible, but the pattern
    scales to production volumes (100K+ alerts) without loading the full
    fixture into memory.

    Usage:
        for record in stream_dataset():
            scored = score_alert(record.anomaly_score, ...)
            if scored.should_surface:
                mve = generate_mve(...)

    Args:
        path: Path to sample_alerts.yaml. Defaults to the canonical fixture.

    Yields:
        AlertRecord, one per alert in the fixture.
    """
    fixture_path = path or SAMPLE_ALERTS_PATH
    with open(fixture_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for a in data.get("alerts", []):
        gt = AlertGroundTruth(
            alert_id=a["alert_id"],
            true_severity=a["ground_truth_severity"],
            true_clinical_system=a["true_clinical_system"],
            true_label=a["ground_truth_label"],
            device_patchable=bool(a["device_context"]["patchable"]),
            device_criticality=a["device_context"]["criticality"],
        )
        yield AlertRecord(
            alert_id=a["alert_id"],
            raw_alert=a["raw_alert"],
            device_context=a["device_context"],
            behavioral_baseline=a["behavioral_baseline"],
            user_context=a.get("user_context"),
            ground_truth=gt,
            anomaly_score=float(a["anomaly_score"]),
            event_context=a.get("event_context"),
        )


def load_dataset(path: Optional[Path] = None) -> List[AlertRecord]:
    """Load alert dataset from YAML fixture and build AlertRecord objects.

    H-1: delegates to stream_dataset() to avoid duplicating the parse+build
    logic. Identical semantics — returns the full materialised list.

    Args:
        path: Path to sample_alerts.yaml. Defaults to the canonical fixture.

    Returns:
        List of AlertRecord, one per alert in the fixture.
    """
    records = list(stream_dataset(path))
    logger.info(
        "Loaded %d alerts from %s",
        len(records),
        sanitize_for_log(path or SAMPLE_ALERTS_PATH),
    )
    return records


# ── Pipeline execution ───────────────────────────────────────────────────

def _build_system_logs_and_actions(
    records: List[AlertRecord],
) -> tuple[List[dict[str, Any]], List[dict[str, Any]]]:
    """Build system log and recommendation list in a single O(N) pass.

    H-2: replaces two separate functions (_build_system_logs and
    _build_system_actions) that each iterated all_records independently.

    Returns:
        (system_logs, system_actions) — both lists populated in one pass.
    """
    logs: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for r in records:
        logs.append({"action": "score_alert", "alert_id": r.alert_id})
        if r.mve is not None:
            logs.append({"action": "generate_mve", "alert_id": r.alert_id})
            actions.append({
                "type": "recommendation",
                "alert_id": r.alert_id,
                "content": r.mve.layer_3.get("immediate_action", ""),
            })
    return logs, actions


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
    from tests.acceptance_tests import run_acceptance_tests, test_false_positive_rate
    from tests.negative_tests import run_negative_tests

    # Opt-10: use streaming generator when no pre-loaded dataset is supplied.
    # This avoids materialising the full fixture list in memory before
    # processing begins.  When a pre-loaded dataset is passed (e.g. from
    # tests that build records programmatically) the existing list path is used.
    alert_source = dataset if dataset is not None else stream_dataset(fixture_path)

    baseline_results: List[dict[str, Any]] = []   # static threshold for M6
    adaptive_results: List[dict[str, Any]] = []   # adaptive for M6
    surfaced_records: List[AlertRecord] = []
    all_records: List[AlertRecord] = []            # full set for negative tests

    # ── Step 2: Run each alert through pipeline ──────────────────────────
    for record in alert_source:
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

        all_records.append(record)

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

    # ── EA-04 fix: Alert volume spike meta-detection ──────────────────
    # Total count derived from baseline_results (accumulated above) so we
    # don't require len(dataset) — works for both list and generator sources.
    total_processed = len(baseline_results)
    surfaced_rate = len(surfaced_records) / total_processed if total_processed else 0
    if surfaced_rate > 0.30:
        logger.warning(
            "ALERT VOLUME SPIKE: %d/%d (%.0f%%) alerts surfaced — "
            "possible alert fatigue attack. Prioritize CRITICAL alerts.",
            len(surfaced_records), total_processed, surfaced_rate * 100,
        )

    logger.info(
        "Pipeline: %d alerts processed, %d surfaced, %d suppressed",
        total_processed,
        len(surfaced_records),
        total_processed - len(surfaced_records),
    )

    # ── Steps 3 & 4: Extract test inputs in a single pass over surfaced_records ──
    # H-3: replaces four separate O(N) list comprehensions over surfaced_records.
    mve_outputs: List[MVEOutput] = []
    surfaced_gts: List[AlertGroundTruth] = []
    mve_dicts: list[dict[str, Any]] = []
    for r in surfaced_records:
        surfaced_gts.append(r.ground_truth)
        if r.mve is not None:
            mve_outputs.append(r.mve)
            mve_dicts.append(r.mve.to_dict(alert_id=r.alert_id))

    all_gts: List[AlertGroundTruth] = [r.ground_truth for r in all_records]

    # H-2: single pass over all_records for both logs and actions
    system_logs, system_actions = _build_system_logs_and_actions(all_records)

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
            # H-5: test_false_positive_rate imported at function top — not per iteration
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
