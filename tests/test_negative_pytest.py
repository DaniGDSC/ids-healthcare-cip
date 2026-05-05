"""Pytest-idiomatic wrappers around `tests/negative_tests.py`.

The negative-test functions return dicts because `run_tests.py` consumes
them as part of an alignment-report pipeline. Pytest prefers `assert`
over return values; these wrappers translate.

Closes GAP-A13. The `outputs` / `system_logs` / `system_actions` fixtures
are defined in `tests/conftest.py`.
"""
from __future__ import annotations

from typing import Any, List

from tests.negative_tests import (
    test_no_automated_blocking as _no_blocking,
    test_no_device_discovery_attempted as _no_discovery,
    test_no_model_internals_exposed as _no_model_internals,
    test_no_ransomware_dwell_time_claims as _no_ransomware,
    test_no_rf_protocol_claims as _no_rf,
    test_severity_uses_clinical_not_cvss as _no_cvss,
)


def _assert_zero_violations(result: dict[str, Any]) -> None:
    assert result.get("violations_found", 0) == 0, (
        f"{result.get('test_name')}: {result.get('violations_found')} violations: "
        f"{result.get('violations')}"
    )


def test_neg_no_device_discovery(system_logs: List[dict[str, Any]]) -> None:
    _assert_zero_violations(_no_discovery(system_logs))


def test_neg_no_automated_blocking(system_actions: List[dict[str, Any]]) -> None:
    _assert_zero_violations(_no_blocking(system_actions))


def test_neg_no_rf_protocol_claims(outputs: List[dict[str, Any]]) -> None:
    _assert_zero_violations(_no_rf(outputs))


def test_neg_no_ransomware_dwell_claims(outputs: List[dict[str, Any]]) -> None:
    _assert_zero_violations(_no_ransomware(outputs))


def test_neg_severity_uses_clinical_not_cvss(outputs: List[dict[str, Any]]) -> None:
    _assert_zero_violations(_no_cvss(outputs))


def test_neg_no_model_internals_exposed(outputs: List[dict[str, Any]]) -> None:
    _assert_zero_violations(_no_model_internals(outputs))
