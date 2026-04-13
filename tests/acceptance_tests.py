"""Acceptance tests for XAI-IDS-Healthcare prototype.

Tests M1–M8 from research_claims.yaml, converted from pseudo-code to
runnable Python.  Implemented BEFORE the components (per CLAUDE.md step 3).

All functions return a float metric value and raise AssertionError if
below the minimum threshold.  A WARN is printed when below target but
above minimum.

Usage:
    results = run_acceptance_tests(mve_outputs, ground_truths,
                                   baseline_results, adaptive_results)
"""
from __future__ import annotations

import re
from typing import Any, List

from src.data_models import AlertGroundTruth, MVEOutput

# ── Constants ────────────────────────────────────────────────────────────

L1_REQUIRED = ["baseline_behavior", "deviation_description", "confidence_indicator"]
L2_REQUIRED = ["affected_system", "patient_care_impact", "phi_exposure", "severity_label",
               "severity_rationale"]
L3_REQUIRED = ["immediate_action", "clinical_constraint", "escalation_path", "timeframe"]

VALID_SEVERITY = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
SEVERITY_LEVEL = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}

MAX_TOTAL_WORDS = 150
MAX_L1_WORDS = 60

VAGUE_PATTERNS = re.compile(
    r"\b(investigate further|monitor closely|consider isolating|"
    r"contact appropriate|as needed|if necessary|look into)\b",
    re.IGNORECASE,
)
SPECIFIC_ACTION_PATTERNS = re.compile(
    r"\b(block|isolate|disable|apply|force|rate-limit|quarantine|"
    r"NAC|firewall|policy|restrict|force MFA|force re-auth)\b",
    re.IGNORECASE,
)
PATIENT_IMPACT_PATTERNS = re.compile(
    r"\b(patient|care|infusion|vital|medication|clinical|treatment|"
    r"monitoring|PHI|drug delivery|nursing)\b",
    re.IGNORECASE,
)


# ── M1: MVE Completeness ─────────────────────────────────────────────────

def test_mve_completeness(outputs: List[MVEOutput]) -> float:
    """Claim C7: MVE contains 3 layers with all required elements.

    Checks that every output has all required fields in each layer,
    and that all fields are non-empty strings.

    Pass:  >= 85% (minimum), target 95%.
    Fail:  <  85%.

    Args:
        outputs: List of MVEOutput objects from generate_mve().

    Returns:
        Fraction of complete outputs [0.0, 1.0].
    """
    if not outputs:
        raise AssertionError("No MVE outputs provided")

    complete = 0
    incomplete_ids = []
    for i, mve in enumerate(outputs):
        l1_ok = all(bool(mve.layer_1.get(k, "").strip()) for k in L1_REQUIRED)
        l2_ok = all(bool(mve.layer_2.get(k, "").strip()) for k in L2_REQUIRED)
        l3_ok = all(bool(mve.layer_3.get(k, "").strip()) for k in L3_REQUIRED)
        if l1_ok and l2_ok and l3_ok:
            complete += 1
        else:
            incomplete_ids.append(i)

    rate = complete / len(outputs)
    assert rate >= 0.85, (
        f"M1 MVE completeness {rate:.1%} below minimum 85% "
        f"(incomplete indices: {incomplete_ids[:5]})"
    )
    if rate < 0.95:
        print(f"  WARN M1: completeness {rate:.1%} below target 95%")
    return rate


# ── M2: Clinical Relevance ───────────────────────────────────────────────

def test_clinical_relevance(
    outputs: List[MVEOutput],
    ground_truths: List[AlertGroundTruth],
) -> float:
    """Claim C1: Layer 2 correctly identifies clinical system and mentions
    patient-care impact.

    Scoring per alert:
      1.0 — affected_system contains true_clinical_system AND patient_care_impact
            matches patient-impact regex
      0.5 — affected_system correct but patient_care_impact is generic
      0.0 — affected_system wrong

    Pass:  >= 75% (minimum), target 90%.

    Args:
        outputs: List of MVEOutput from generate_mve().
        ground_truths: Matching list of AlertGroundTruth from fixtures.

    Returns:
        Mean clinical relevance score [0.0, 1.0].
    """
    if len(outputs) != len(ground_truths):
        raise AssertionError(
            f"outputs ({len(outputs)}) and ground_truths ({len(ground_truths)}) "
            "must be the same length"
        )

    scores = []
    for mve, gt in zip(outputs, ground_truths):
        affected = mve.layer_2.get("affected_system", "").lower()
        system_correct = gt.true_clinical_system.lower() in affected
        has_patient_impact = bool(
            PATIENT_IMPACT_PATTERNS.search(mve.layer_2.get("patient_care_impact", ""))
        )
        if system_correct and has_patient_impact:
            scores.append(1.0)
        elif system_correct:
            scores.append(0.5)
        else:
            scores.append(0.0)

    rate = sum(scores) / len(scores)
    assert rate >= 0.75, f"M2 Clinical relevance {rate:.1%} below minimum 75%"
    if rate < 0.90:
        print(f"  WARN M2: clinical relevance {rate:.1%} below target 90%")
    return rate


# ── M3: Actionability ────────────────────────────────────────────────────

def test_actionability(outputs: List[MVEOutput]) -> float:
    """Claim C3: Layer 3 provides specific executable action, not vague advice.

    Pass: immediate_action contains at least one specific keyword AND
          does not contain vague phrases.
    Fail: vague-only, or missing immediate_action.

    Pass:  >= 70% (minimum), target 85%.

    Args:
        outputs: List of MVEOutput objects.

    Returns:
        Fraction of specific actions [0.0, 1.0].
    """
    if not outputs:
        raise AssertionError("No MVE outputs provided")

    specific = 0
    for mve in outputs:
        action = mve.layer_3.get("immediate_action", "")
        is_vague = bool(VAGUE_PATTERNS.search(action))
        has_specific = bool(SPECIFIC_ACTION_PATTERNS.search(action))
        if has_specific and not is_vague:
            specific += 1

    rate = specific / len(outputs)
    assert rate >= 0.70, f"M3 Actionability {rate:.1%} below minimum 70%"
    if rate < 0.85:
        print(f"  WARN M3: actionability {rate:.1%} below target 85%")
    return rate


# ── M4: Clinical Constraint Awareness ────────────────────────────────────

def test_clinical_constraint_awareness(outputs: List[MVEOutput]) -> float:
    """Claim C3: Layer 3 includes 'DO NOT' statement for clinical-system alerts.

    Only alerts where alert_involves_clinical_system=True are tested.
    The clinical_constraint field must contain 'DO NOT' (case-insensitive).

    Pass:  >= 80% (minimum), target 90%.

    Args:
        outputs: List of MVEOutput objects.

    Returns:
        Fraction of clinical alerts with DO NOT constraint [0.0, 1.0].
    """
    clinical_alerts = [m for m in outputs if m.alert_involves_clinical_system]
    if not clinical_alerts:
        return 1.0   # no clinical alerts in dataset

    has_constraint = 0
    for mve in clinical_alerts:
        constraint = mve.layer_3.get("clinical_constraint", "")
        if re.search(r"DO NOT|do not|Do not", constraint):
            has_constraint += 1

    rate = has_constraint / len(clinical_alerts)
    assert rate >= 0.80, (
        f"M4 Clinical constraint awareness {rate:.1%} below minimum 80% "
        f"({len(clinical_alerts)} clinical alerts tested)"
    )
    if rate < 0.90:
        print(f"  WARN M4: clinical constraint {rate:.1%} below target 90%")
    return rate


# ── M8: Severity Label Accuracy ──────────────────────────────────────────

def test_severity_label_accuracy(
    outputs: List[MVEOutput],
    ground_truths: List[AlertGroundTruth],
) -> float:
    """Claim C1: System assigns correct CRITICAL/HIGH/MEDIUM/LOW severity.

    Hard fail: any CRITICAL <-> LOW mismatch (|level_diff| == 3).
    Soft fail: exact match rate < 70%.

    Pass:  >= 70% exact match, 0 CRITICAL<->LOW mismatches.
    Target: 80% exact match.

    Args:
        outputs: List of MVEOutput objects.
        ground_truths: Matching list of AlertGroundTruth from fixtures.

    Returns:
        Exact match fraction [0.0, 1.0].
    """
    if len(outputs) != len(ground_truths):
        raise AssertionError("outputs and ground_truths must be the same length")

    exact = 0
    catastrophic = 0
    for mve, gt in zip(outputs, ground_truths):
        predicted = mve.layer_2.get("severity_label", "").upper()
        actual = gt.true_severity.upper()
        assert predicted in VALID_SEVERITY, (
            f"M8 Invalid severity label: '{predicted}' for alert {gt.alert_id}"
        )
        if predicted == actual:
            exact += 1
        diff = abs(SEVERITY_LEVEL.get(predicted, -1) - SEVERITY_LEVEL.get(actual, -1))
        if diff == 3:
            catastrophic += 1

    exact_rate = exact / len(outputs)
    assert catastrophic == 0, (
        f"M8 HARD FAIL: {catastrophic} CRITICAL<->LOW mismatch(es) — "
        "catastrophic severity error"
    )
    assert exact_rate >= 0.70, (
        f"M8 Severity accuracy {exact_rate:.1%} below minimum 70%"
    )
    if exact_rate < 0.80:
        print(f"  WARN M8: severity accuracy {exact_rate:.1%} below target 80%")
    return exact_rate


# ── M1b: Layer 1 Length Constraint ───────────────────────────────────────

def test_layer1_length_constraint(outputs: List[MVEOutput]) -> float:
    """MVE spec: Layer 1 <= 60 words, total output <= 150 words.

    Checks both constraints per output.

    Pass:  >= 90% within limits (minimum), target 95%.

    Args:
        outputs: List of MVEOutput objects.

    Returns:
        Fraction of outputs within word limits [0.0, 1.0].
    """
    if not outputs:
        raise AssertionError("No MVE outputs provided")

    within_limit = 0
    violations = []
    for i, mve in enumerate(outputs):
        l1_words = sum(
            len(mve.layer_1.get(f, "").split()) for f in L1_REQUIRED
        )
        total = mve.total_word_count
        if l1_words <= MAX_L1_WORDS and total <= MAX_TOTAL_WORDS:
            within_limit += 1
        else:
            violations.append((i, l1_words, total))

    rate = within_limit / len(outputs)
    assert rate >= 0.90, (
        f"M1b Length compliance {rate:.1%} below minimum 90% "
        f"(violations: {violations[:3]})"
    )
    if rate < 0.95:
        print(f"  WARN M1b: length compliance {rate:.1%} below target 95%")
    return rate


# ── M7: Risk-Adaptive Threshold Behavior ────────────────────────────────

def test_risk_adaptive_threshold() -> bool:
    """Claim C2: Unpatchable CRITICAL devices have lower (more sensitive)
    alert thresholds than patchable LOW devices.

    Binary pass/fail — 100% required.

    Returns:
        True if threshold ordering is correct.
    """
    from src.risk_scorer import get_threshold

    crit_unpatchable = get_threshold(device_criticality="CRITICAL", patchable=False)
    low_patchable = get_threshold(device_criticality="LOW", patchable=True)
    high_unpatchable = get_threshold(device_criticality="HIGH", patchable=False)
    medium_patchable = get_threshold(device_criticality="MEDIUM", patchable=True)

    assert crit_unpatchable < low_patchable, (
        f"M7 FAIL: CRITICAL-unpatchable threshold ({crit_unpatchable}) must be "
        f"LOWER than LOW-patchable ({low_patchable})"
    )
    assert high_unpatchable < low_patchable, (
        f"M7 FAIL: HIGH-unpatchable threshold ({high_unpatchable}) must be "
        f"lower than LOW-patchable ({low_patchable})"
    )
    assert crit_unpatchable <= medium_patchable, (
        f"M7 FAIL: CRITICAL-unpatchable ({crit_unpatchable}) should be <= "
        f"MEDIUM-patchable ({medium_patchable})"
    )

    # Verify 30% reduction requirement for CRITICAL + unpatchable
    from src.risk_scorer import DEFAULT_THRESHOLD
    actual_reduction = (DEFAULT_THRESHOLD - crit_unpatchable) / DEFAULT_THRESHOLD
    assert actual_reduction >= 0.30, (
        f"M7 FAIL: CRITICAL+unpatchable threshold reduction is {actual_reduction:.1%} "
        f"(required >= 30%)"
    )
    return True


# ── M6: False Positive Rate Reduction ───────────────────────────────────

def test_false_positive_rate(
    baseline_results: List[dict[str, Any]],
    adaptive_results: List[dict[str, Any]],
    ground_truths: List[AlertGroundTruth],
) -> float:
    """Claim C8: Risk-adaptive thresholds reduce false positive rate vs.
    static-threshold baseline by >= 20%.

    FP rate = (non-true-positive alerts surfaced) / (total surfaced).
    Includes both 'false_positive' and 'legitimate_rare' labels as FP.

    Pass:  >= 20% FP reduction (minimum), target 40%.

    Args:
        baseline_results: List of {'surfaced': bool} from score_alert_static().
        adaptive_results: List of {'surfaced': bool} from score_alert().
        ground_truths: Matching list of AlertGroundTruth from fixtures.

    Returns:
        FP reduction fraction [0.0, 1.0].
    """
    def calc_fp_rate(results: List[dict[str, Any]], gts: List[AlertGroundTruth]) -> float:
        surfaced = [
            (r, g) for r, g in zip(results, gts)
            if r.get("surfaced", False)
        ]
        if not surfaced:
            return 0.0
        fp = sum(1 for r, g in surfaced if g.true_label != "true_positive")
        return fp / len(surfaced)

    baseline_fpr = calc_fp_rate(baseline_results, ground_truths)
    adaptive_fpr = calc_fp_rate(adaptive_results, ground_truths)

    if baseline_fpr == 0.0:
        print("  INFO M6: no false positives in baseline — trivially 0% FPR")
        return 1.0

    reduction = (baseline_fpr - adaptive_fpr) / baseline_fpr
    assert reduction >= 0.20, (
        f"M6 FP reduction {reduction:.1%} below minimum 20% "
        f"(baseline_fpr={baseline_fpr:.1%}, adaptive_fpr={adaptive_fpr:.1%})"
    )
    if reduction < 0.40:
        print(
            f"  WARN M6: FP reduction {reduction:.1%} below target 40% "
            f"(baseline={baseline_fpr:.1%}, adaptive={adaptive_fpr:.1%})"
        )
    return reduction


# ── Runner ───────────────────────────────────────────────────────────────

def run_acceptance_tests(
    mve_outputs: List[MVEOutput],
    ground_truths: List[AlertGroundTruth],
    baseline_results: List[dict[str, Any]],
    adaptive_results: List[dict[str, Any]],
) -> List[dict[str, Any]]:
    """Run all 8 acceptance tests and return metric results.

    Args:
        mve_outputs: MVEOutput for each alert that was surfaced.
        ground_truths: Matching ground truth for surfaced alerts.
        baseline_results: Static-threshold surfacing results for all alerts.
        adaptive_results: Adaptive-threshold surfacing results for all alerts.

    Returns:
        List of metric dicts:
        {metric_id, metric_name, result_value, target, minimum, pass_fail, detail}
    """
    METRICS = [
        ("M1",  "test_mve_completeness",            0.95, 0.85),
        ("M2",  "test_clinical_relevance",           0.90, 0.75),
        ("M3",  "test_actionability",                0.85, 0.70),
        ("M4",  "test_clinical_constraint_awareness",0.90, 0.80),
        ("M8",  "test_severity_label_accuracy",      0.80, 0.70),
        ("M1b", "test_layer1_length_constraint",     0.95, 0.90),
        ("M7",  "test_risk_adaptive_threshold",      1.00, 1.00),
        ("M6",  "test_false_positive_rate",          0.40, 0.20),
    ]

    results = []
    for metric_id, metric_name, target, minimum in METRICS:
        detail = ""
        try:
            if metric_id == "M1":
                val = float(test_mve_completeness(mve_outputs))
            elif metric_id == "M2":
                val = float(test_clinical_relevance(mve_outputs, ground_truths))
            elif metric_id == "M3":
                val = float(test_actionability(mve_outputs))
            elif metric_id == "M4":
                val = float(test_clinical_constraint_awareness(mve_outputs))
            elif metric_id == "M8":
                val = float(test_severity_label_accuracy(mve_outputs, ground_truths))
            elif metric_id == "M1b":
                val = float(test_layer1_length_constraint(mve_outputs))
            elif metric_id == "M7":
                test_risk_adaptive_threshold()
                val = 1.0
            elif metric_id == "M6":
                val = float(test_false_positive_rate(
                    baseline_results, adaptive_results,
                    # M6 uses all-alerts ground truth, not just surfaced
                    ground_truths,
                ))
            else:
                val = 0.0
                detail = "unknown metric"

            if val >= target:
                pf = "PASS"
            elif val >= minimum:
                pf = "WARN"
            else:
                pf = "FAIL"

        except AssertionError as exc:
            val = 0.0
            pf = "FAIL"
            detail = str(exc)
        except Exception as exc:
            val = 0.0
            pf = "FAIL"
            detail = f"ERROR: {exc}"

        results.append({
            "metric_id": metric_id,
            "metric_name": metric_name,
            "result_value": round(val, 4),
            "target": target,
            "minimum": minimum,
            "pass_fail": pf,
            "detail": detail,
        })

    return results
