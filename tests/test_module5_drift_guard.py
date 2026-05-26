"""Module 5 drift guard — _assert_no_score_drift catches divergent records."""
from __future__ import annotations

import numpy as np
import pytest

from module5_responses.pipeline import _assert_no_score_drift


def _risk_data(n=3):
    return {
        "R": np.array([0.10, 0.50, 0.90]),
        "risk_levels": np.array(["LOW", "MEDIUM", "CRITICAL"]),
        "c_detect": np.array([0.0, 0.4, 0.9]),
        "c_track_a": np.array([0.0, 0.3, 0.8]),
        "c_track_b": np.array([0.0, 0.5, 0.7]),
        "d_crit": np.array([0.1, 0.4, 0.9]),
        "s_data": np.array([0.0, 0.3, 0.6]),
        "d_clinical_tier": np.array([0.2, 0.5, 0.95]),
    }


def _record(idx, R_val, level, components):
    return {
        "sample_index": idx,
        "risk_score": round(float(R_val), 4),
        "risk_level": level,
        "risk_components": components,
    }


def _good_components(rd, idx):
    return {
        "C_detect": round(float(rd["c_detect"][idx]), 4),
        "C_track_a": round(float(rd["c_track_a"][idx]), 4),
        "C_track_b": round(float(rd["c_track_b"][idx]), 4),
        "D_crit": round(float(rd["d_crit"][idx]), 4),
        "S_data": round(float(rd["s_data"][idx]), 4),
        "D_clinical_tier": round(float(rd["d_clinical_tier"][idx]), 4),
    }


def test_drift_guard_passes_for_matching():
    rd = _risk_data()
    records = [
        _record(i, rd["R"][i], rd["risk_levels"][i], _good_components(rd, i))
        for i in range(3)
    ]
    _assert_no_score_drift(records, rd)  # no raise


def test_drift_guard_raises_on_R_mismatch():
    rd = _risk_data()
    records = [_record(0, 0.99, "LOW", _good_components(rd, 0))]
    with pytest.raises(ValueError, match="Score drift"):
        _assert_no_score_drift(records, rd)


def test_drift_guard_raises_on_level_mismatch():
    rd = _risk_data()
    records = [_record(0, rd["R"][0], "CRITICAL", _good_components(rd, 0))]
    with pytest.raises(ValueError, match="Risk-level drift"):
        _assert_no_score_drift(records, rd)


def test_drift_guard_raises_on_component_mismatch():
    rd = _risk_data()
    components = _good_components(rd, 0)
    components["C_detect"] = 0.999
    records = [_record(0, rd["R"][0], "LOW", components)]
    with pytest.raises(ValueError, match="Component drift"):
        _assert_no_score_drift(records, rd)


def test_drift_guard_tolerates_small_rounding():
    rd = _risk_data()
    components = _good_components(rd, 0)
    components["C_detect"] += 1e-5  # below tol=1e-4
    records = [_record(0, rd["R"][0], "LOW", components)]
    _assert_no_score_drift(records, rd)  # no raise
