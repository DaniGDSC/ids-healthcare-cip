"""Pre-redesign Task 1: verify calibrated probas wired into Module 3 fusion.

Contract:
    ``load_xgboost_proba()`` prefers ``xgboost_test_proba_calibrated.npy``
    when it exists; falls back to raw ``y_proba`` from
    ``xgboost_test_predictions.npz`` when the calibrated file is missing.
    Threshold is always loaded from the report (calibration does not move
    the F2-tuned operating point).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from module3_risk_scoring.module3_risk_scores import load_xgboost_proba


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch) -> Path:
    """Build an isolated models/ dir and patch PROJECT_ROOT to point at it."""
    md = tmp_path / "results" / "models"
    md.mkdir(parents=True)

    # Raw .npz with y_proba
    raw = np.array([0.10, 0.50, 0.92, 0.03], dtype=np.float32)
    np.savez(md / "xgboost_test_predictions.npz", y_proba=raw,
             y_true=np.array([0, 1, 1, 0]),
             y_pred=np.array([0, 1, 1, 0]))
    # Optimal threshold report
    (md / "xgboost_final_report.json").write_text(
        json.dumps({"optimal_threshold": 0.05})
    )

    import module3_risk_scoring.module3_risk_scores as m
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    return md


def test_falls_back_to_raw_when_no_calibrated(models_dir: Path) -> None:
    """No calibrated file → raw probas returned."""
    y_proba, thr = load_xgboost_proba(prefer_calibrated=True)
    expected = np.array([0.10, 0.50, 0.92, 0.03], dtype=np.float32)
    np.testing.assert_array_equal(y_proba, expected)
    assert thr == 0.05


def test_prefers_calibrated_when_present(models_dir: Path) -> None:
    """Calibrated file exists → calibrated probas returned, threshold unchanged."""
    cal = np.array([0.05, 0.42, 0.97, 0.02], dtype=np.float32)
    np.save(models_dir / "xgboost_test_proba_calibrated.npy", cal)

    y_proba, thr = load_xgboost_proba(prefer_calibrated=True)
    np.testing.assert_array_equal(y_proba, cal)
    assert thr == 0.05  # threshold from report, unchanged by calibration


def test_explicit_opt_out_uses_raw(models_dir: Path) -> None:
    """prefer_calibrated=False forces raw even when calibrated file present."""
    cal = np.array([0.05, 0.42, 0.97, 0.02], dtype=np.float32)
    np.save(models_dir / "xgboost_test_proba_calibrated.npy", cal)

    y_proba, _ = load_xgboost_proba(prefer_calibrated=False)
    expected_raw = np.array([0.10, 0.50, 0.92, 0.03], dtype=np.float32)
    np.testing.assert_array_equal(y_proba, expected_raw)


def test_calibration_changes_alert_distribution(models_dir: Path) -> None:
    """When calibrated probas differ from raw, the *set* of surfaced rows shifts.

    Counts can coincidentally match (a row drops out of "surfaced" while
    another enters); the contract is that the row-level decision changes
    on at least one row when the calibrator emits different values.
    """
    # Calibrated values where row 0 flips IN (0.10→0.04 vs thr=0.05) and
    # row 3 flips IN (0.03→0.06 vs thr=0.05). Surfacing identity differs.
    cal = np.array([0.04, 0.55, 0.88, 0.06], dtype=np.float32)
    np.save(models_dir / "xgboost_test_proba_calibrated.npy", cal)

    y_cal, thr = load_xgboost_proba(prefer_calibrated=True)
    y_raw, _ = load_xgboost_proba(prefer_calibrated=False)
    surfaced_cal = y_cal >= thr      # boolean per row
    surfaced_raw = y_raw >= thr

    assert not np.array_equal(surfaced_cal, surfaced_raw), (
        f"calibration should change at least one row's surfacing decision; "
        f"cal_mask={surfaced_cal.tolist()}, raw_mask={surfaced_raw.tolist()}"
    )
