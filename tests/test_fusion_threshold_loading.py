"""Tests for the fusion-threshold calibration JSON + loader path.

Covers Phase 3.5 of RQ1_pipeline.md (Stage 5B a_high calibration). The
loader is what wires the calibrated threshold into classify_fusion at
runtime; if it breaks, fusion silently reverts to the pre-calibration
default and the sensitivity headline-target miss returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from module3_risk_scoring.module3_risk_scores import (
    classify_fusion,
    load_fusion_thresholds,
)
from src.data_models import FusionClass, P_XGB_HIGH_CONF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_JSON = PROJECT_ROOT / "results/models/_fusion_thresholds.json"


def test_thresholds_json_schema():
    """The calibration artefact must exist and have schema v1.0 fields."""
    assert THRESHOLDS_JSON.exists(), (
        f"{THRESHOLDS_JSON.relative_to(PROJECT_ROOT)} missing — run "
        "analysis/calibrate_fusion_threshold.py."
    )
    payload = json.loads(THRESHOLDS_JSON.read_text())
    assert payload["schema_version"] == "1.0"
    for key in ("picked", "selection_rule", "tuning_split",
                "tuning_split_sha256", "tuning_metrics_at_picked",
                "sweep_table"):
        assert key in payload, f"missing key: {key}"
    picked = payload["picked"]
    for key in ("a_high", "a_low", "b"):
        assert key in picked, f"picked missing {key}"
        assert isinstance(picked[key], (int, float))

    # Four-class invariant: a_high > a_low. Otherwise the CONFIRM band
    # collapses to empty and the fusion semantics break.
    assert picked["a_high"] > picked["a_low"], (
        f"a_high ({picked['a_high']}) must be > a_low ({picked['a_low']})"
    )


def test_load_returns_picked_values():
    """load_fusion_thresholds() must return exactly what the JSON says."""
    payload = json.loads(THRESHOLDS_JSON.read_text())
    picked = payload["picked"]
    loaded = load_fusion_thresholds()
    assert loaded["a_high"] == picked["a_high"]
    assert loaded["a_low"] == picked["a_low"]
    assert loaded["b"] == picked["b"]


def test_classify_fusion_uses_loaded_defaults():
    """classify_fusion() with no kwargs must use the loaded a_high.

    Constructs a sample that sits in the loaded band but would fall in
    BENIGN under the pre-calibration P_XGB_HIGH_CONF=0.85 default, to
    confirm the wiring (and surface silent regressions if anything
    re-routes to the constant).
    """
    th = load_fusion_thresholds()
    a_high = th["a_high"]

    # A c_track_a just above the calibrated a_high but below 0.85, with
    # c_track_b low enough that NEITHER CONFIRMED nor NOVEL fires under
    # the old rules — only the lowered a_high gates whether KNOWN_ATTACK
    # fires for this row.
    c_track_a = np.array([a_high + 0.01], dtype=float)
    c_track_b = np.array([0.10], dtype=float)

    out_loaded = classify_fusion(c_track_a, c_track_b)
    out_old = classify_fusion(c_track_a, c_track_b,
                              a_high=P_XGB_HIGH_CONF, a_low=0.40, b=0.70)
    assert out_loaded[0] == FusionClass.KNOWN_ATTACK.value
    assert out_old[0] == FusionClass.BENIGN.value


def test_loader_falls_back_when_file_missing(tmp_path, monkeypatch):
    """When _fusion_thresholds.json is absent, the loader must fall back
    to (P_XGB_HIGH_CONF, 0.40, 0.70) — preserves the pre-calibration
    contract for tests that monkeypatch PROJECT_ROOT or for fresh clones
    where the calibration step has not yet been run."""
    import module3_risk_scoring.module3_risk_scores as m3
    monkeypatch.setattr(m3, "PROJECT_ROOT", tmp_path)
    # Sanity: the file does not exist under tmp_path.
    assert not (tmp_path / "results/models/_fusion_thresholds.json").exists()

    fallback = m3.load_fusion_thresholds()
    assert fallback == {
        "a_high": float(P_XGB_HIGH_CONF),
        "a_low": 0.40,
        "b": 0.70,
    }


def test_picked_threshold_meets_acceptance_targets_on_test():
    """End-to-end gate: applying the picked thresholds to the production
    test split must yield sens > 0.90 AND spec > 0.95 — same gate as
    tests/acceptance_tests.py::test_rq1_targets_met, but expressed in
    terms of the calibrated thresholds for traceability."""
    th = load_fusion_thresholds()
    npz = np.load(
        PROJECT_ROOT / "results/reports/risk_scores.npz",
        allow_pickle=False,
    )
    y_true = npz["y_true"]
    ta = npz["c_track_a"]
    tb = npz["c_track_b"]
    a_high, a_low, b = th["a_high"], th["a_low"], th["b"]
    high = ta >= a_high
    confirm = (ta >= a_low) & (ta < a_high) & (tb >= b)
    novel = (ta < a_low) & (tb >= b)
    y_pred = (high | confirm | novel).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    assert sens > 0.90, f"sensitivity = {sens:.4f} <= 0.90 on test"
    assert spec > 0.95, f"specificity = {spec:.4f} <= 0.95 on test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
