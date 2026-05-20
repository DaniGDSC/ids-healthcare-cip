"""Negative tests for XAI-IDS-Healthcare prototype.

Tests that the system stays WITHIN defined scope boundaries (CLAUDE.md
DO NOT BUILD list and research_spec.yaml out_of_scope).

Implemented BEFORE the components (per CLAUDE.md step 4).
All tests must pass with 0 violations to satisfy the done condition.

Usage:
    results = run_negative_tests(mve_dicts, system_logs, system_actions)
"""
from __future__ import annotations

import re
from typing import Any, List

# ── Forbidden pattern sets ───────────────────────────────────────────────

_DISCOVERY_PATTERNS = [
    "nmap", "scan", "fingerprint", "discover", "probe", "enumerate_devices",
    "network scan", "device scan", "port scan",
]

_ENFORCEMENT_ACTION_TYPES = [
    "block_executed", "quarantine_applied", "session_terminated",
    "rule_pushed", "firewall_updated", "isolate_executed",
]

_RF_KEYWORDS = [
    "bluetooth", "zigbee", "rf protocol", "wireless pairing",
    "radio frequency", "proprietary wireless", "z-wave", "802.15",
    "near field",
]

# BLE requires word-boundary matching to avoid false positives from
# words like "available", "unreliable", "configurable"
_RF_WORD_BOUNDARY_KEYWORDS = ["ble"]

_RANSOMWARE_CLAIM_PATTERNS = re.compile(
    r"early\s+detection\s+of\s+ransomware|"
    r"ransomware\s+detected\s+before|"
    r"pre-encryption\s+ransomware|"
    r"ransomware\s+dwell",
    re.IGNORECASE,
)

_CVSS_PATTERN = re.compile(r"\bCVSS\b|\bcvss\b", re.IGNORECASE)

_MODEL_INTERNALS = [
    "shap", "shap value", "feature importance", "random forest",
    "xgboost", "decision tree", "neural network", "deep learning",
    "p-value", "standard deviation", "confidence interval",
    "gradient boost", "autoencoder", "reconstruction error",
    "dae", "treeshap", "feature attribution",
]


# ── Individual negative tests ────────────────────────────────────────────

def test_no_device_discovery_attempted(system_logs: List[dict[str, Any]]) -> dict[str, Any]:
    """Scope: system must consume existing inventory, never scan/discover.

    Verifies that no log entry records a device discovery action.
    The system receives device_context as input — it never initiates
    network scanning, fingerprinting, or device enumeration.

    Args:
        system_logs: List of log dicts with 'action' key from harness.

    Returns:
        Negative test result dict.
    """
    violations = []
    for log in system_logs:
        action = str(log.get("action", "")).lower()
        for p in _DISCOVERY_PATTERNS:
            if p in action:
                violations.append(
                    f"Discovery action '{p}' found in log: {log}"
                )

    return {
        "test_name": "test_no_device_discovery_attempted",
        "violations_found": len(violations),
        "pass_fail": "PASS" if not violations else "FAIL",
        "violations": violations,
    }


def test_no_automated_blocking(system_actions: List[dict[str, Any]]) -> dict[str, Any]:
    """Scope: system recommends only, never executes enforcement.

    Verifies all system actions are tagged as 'recommendation' type
    and no enforcement action types appear.
    Directly maps to module5's ActionExecutor design (simulated only).

    Args:
        system_actions: List of action dicts with 'type' key from harness.

    Returns:
        Negative test result dict.
    """
    violations = []
    for action in system_actions:
        action_type = str(action.get("type", "")).lower()
        if action_type != "recommendation":
            violations.append(
                f"Non-recommendation action type '{action_type}': {action}"
            )
        for bad_type in _ENFORCEMENT_ACTION_TYPES:
            if bad_type in action_type:
                violations.append(
                    f"Enforcement action '{bad_type}' found: {action}"
                )

    return {
        "test_name": "test_no_automated_blocking",
        "violations_found": len(violations),
        "pass_fail": "PASS" if not violations else "FAIL",
        "violations": violations,
    }


def test_no_rf_protocol_claims(outputs: List[dict[str, Any]]) -> dict[str, Any]:
    """Scope: system must not claim detection of Bluetooth/Zigbee/RF attacks.

    IoMT RF/proprietary wireless attacks (CVE-2019-10964, CVE-2022-32537)
    are out of scope — system handles IP-layer anomalies only.

    Checks layer_1_why_anomalous for RF/wireless keywords.

    Args:
        outputs: List of MVEOutput.to_dict() results (include 'alert_id').

    Returns:
        Negative test result dict.
    """
    violations = []
    for output in outputs:
        explanation = str(output.get("layer_1_why_anomalous", "")).lower()
        for kw in _RF_KEYWORDS:
            if kw.lower() in explanation:
                violations.append(
                    f"RF keyword '{kw}' in alert {output.get('alert_id', '?')}: "
                    f"'{explanation[:100]}'"
                )
        # Word-boundary check for short keywords that cause false positives
        for kw in _RF_WORD_BOUNDARY_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", explanation):
                violations.append(
                    f"RF keyword '{kw}' in alert {output.get('alert_id', '?')}: "
                    f"'{explanation[:100]}'"
                )

    return {
        "test_name": "test_no_rf_protocol_claims",
        "violations_found": len(violations),
        "pass_fail": "PASS" if not violations else "FAIL",
        "violations": violations,
    }


def test_no_ransomware_dwell_time_claims(outputs: List[dict[str, Any]]) -> dict[str, Any]:
    """Scope: system must not claim early ransomware detection.

    Per research_spec.yaml: 96% of ransomware is actor-disclosed (DBIR 2025),
    dwell time ≈ 0. System addresses non-ransomware intrusions only.
    Alerts with alert_subtype='ransomware' must not claim 'early detection'.

    Args:
        outputs: List of MVEOutput.to_dict() results.

    Returns:
        Negative test result dict.
    """
    violations = []
    for output in outputs:
        # Check full explanation text across all layers
        full_text = " ".join([
            str(output.get("layer_1_why_anomalous", "")),
            str(output.get("layer_2", {}).get("severity_rationale", "")),
            str(output.get("layer_3", {}).get("immediate_action", "")),
        ])
        if _RANSOMWARE_CLAIM_PATTERNS.search(full_text):
            violations.append(
                f"Ransomware early-detection claim in alert "
                f"{output.get('alert_id', '?')}: '{full_text[:100]}'"
            )

    return {
        "test_name": "test_no_ransomware_dwell_time_claims",
        "violations_found": len(violations),
        "pass_fail": "PASS" if not violations else "FAIL",
        "violations": violations,
    }


def test_severity_uses_clinical_not_cvss(outputs: List[dict[str, Any]]) -> dict[str, Any]:
    """Scope: severity labels must be based on clinical impact, NOT CVSS.

    Checks that 'CVSS' does not appear in severity_rationale.
    Severity must use the 4-tier clinical schema from mve_specification.yaml.

    Args:
        outputs: List of MVEOutput.to_dict() results.

    Returns:
        Negative test result dict.
    """
    violations = []
    for output in outputs:
        rationale = str(output.get("layer_2", {}).get("severity_rationale", ""))
        if _CVSS_PATTERN.search(rationale):
            violations.append(
                f"CVSS reference in severity_rationale for alert "
                f"{output.get('alert_id', '?')}: '{rationale}'"
            )

    return {
        "test_name": "test_severity_uses_clinical_not_cvss",
        "violations_found": len(violations),
        "pass_fail": "PASS" if not violations else "FAIL",
        "violations": violations,
    }


def test_no_model_internals_exposed(outputs: List[dict[str, Any]]) -> dict[str, Any]:
    """Scope: explanations must not expose SHAP, model names, or statistics.

    Per mve_specification.yaml what_NOT_to_include:
    'Raw SHAP values, feature importances, model architecture details,
    statistical measures (p-values, standard deviations).'

    Checks layer_1_why_anomalous (primary explanation layer) for
    forbidden technical terms.

    Args:
        outputs: List of MVEOutput.to_dict() results.

    Returns:
        Negative test result dict.
    """
    violations = []
    for output in outputs:
        full_text = str(output.get("layer_1_why_anomalous", "")).lower()
        for term in _MODEL_INTERNALS:
            if term.lower() in full_text:
                violations.append(
                    f"Model internal '{term}' in layer_1 for alert "
                    f"{output.get('alert_id', '?')}: '{full_text[:100]}'"
                )

    return {
        "test_name": "test_no_model_internals_exposed",
        "violations_found": len(violations),
        "pass_fail": "PASS" if not violations else "FAIL",
        "violations": violations,
    }


# ── Runner ───────────────────────────────────────────────────────────────

def run_negative_tests(
    mve_dicts: List[dict[str, Any]],
    system_logs: List[dict[str, Any]],
    system_actions: List[dict[str, Any]],
) -> List[dict[str, Any]]:
    """Run all 6 negative tests and return results.

    Args:
        mve_dicts: List of MVEOutput.to_dict() for each surfaced alert.
        system_logs: List of log dicts recording what the harness did.
        system_actions: List of action dicts from the harness.

    Returns:
        List of result dicts:
        {test_name, violations_found, pass_fail, violations}
    """
    return [
        test_no_device_discovery_attempted(system_logs),
        test_no_automated_blocking(system_actions),
        test_no_rf_protocol_claims(mve_dicts),
        test_no_ransomware_dwell_time_claims(mve_dicts),
        test_severity_uses_clinical_not_cvss(mve_dicts),
        test_no_model_internals_exposed(mve_dicts),
    ]


# ── RQ3_NO_AUTO_EXECUTION_SPEC.md §7.1 — Layer C CI wrapper ───────────
#
# The functions above are orchestrator-style (positional args, invoked
# by run_negative_tests). The sibling below is pytest-collectible: it
# runs the static-grep audit script as a subprocess and asserts a clean
# result. Same Layer C purpose as test_no_automated_blocking, but
# usable directly from pytest / CI.


def test_no_automated_blocking_audit_clean() -> None:
    """Layer C of the no-auto-execution defense (pytest-collectible).

    Invokes analysis/audit_no_auto_execution.py via subprocess; asserts
    the audit exits 0 (production code contains no forbidden execution
    patterns).
    """
    import subprocess  # noqa: no-auto-exec
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(  # noqa: no-auto-exec
        [sys.executable, "-m", "analysis.audit_no_auto_execution"],
        capture_output=True, text=True, cwd=str(repo),
    )
    assert result.returncode == 0, (
        f"No-auto-execution audit FAILED (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
        "Run `python -m analysis.audit_no_auto_execution --list-violations` "
        "for human-readable detail."
    )
