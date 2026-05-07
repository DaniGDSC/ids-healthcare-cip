"""Layer 1 v4.0 artifact tests (R2, R3, R4, R5).

Covers the gaps closed by this Layer 1 v4.0 batch:
  R2 — configs/per_class_thresholds.yaml mirrors src/risk_scorer.py
  R3 — results/models/dae_thresholds.json (p80, p95, p99) is well-formed
  R4 — results/models/dae_calibration.json yields a monotone percentile
       lookup that maps raw errors → [0, 1] correctly
  R5 — DAEDetector.anomalous_dims_z returns batch-z-score-flagged dims

Existing Layer-1-equivalent infrastructure (M0/M1/M2 training, isotonic
Track A calibration, stratified calibration/holdout split, curated
20-alert stress set) is verified through other tests already.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── R2: per_class_thresholds.yaml parity ─────────────────────────────────

def test_per_class_thresholds_yaml_exists_and_parses() -> None:
    path = PROJECT_ROOT / "configs" / "per_class_thresholds.yaml"
    assert path.exists(), f"missing {path}"
    body = yaml.safe_load(path.read_text())
    assert "per_device_thresholds" in body
    assert "unknown_fallback" in body
    assert isinstance(body["per_device_thresholds"], dict)


def test_per_class_thresholds_yaml_matches_risk_scorer() -> None:
    """The YAML config is a declaration of the values living in
    src/risk_scorer.py. The two MUST agree — drift here corrupts the
    Track A surfacing gate.
    """
    from src.risk_scorer import (
        _TRACK_A_SURFACING_BY_DEVICE,
        _TRACK_A_SURFACING_DEFAULT,
    )

    path = PROJECT_ROOT / "configs" / "per_class_thresholds.yaml"
    body = yaml.safe_load(path.read_text())
    yaml_thr = body["per_device_thresholds"]

    for device_class, py_value in _TRACK_A_SURFACING_BY_DEVICE.items():
        assert device_class in yaml_thr, (
            f"{device_class} is in src/risk_scorer.py but missing from "
            f"configs/per_class_thresholds.yaml"
        )
        assert yaml_thr[device_class] == pytest.approx(py_value), (
            f"threshold drift for {device_class}: yaml={yaml_thr[device_class]}"
            f" vs python={py_value}"
        )

    extras = set(yaml_thr) - set(_TRACK_A_SURFACING_BY_DEVICE)
    assert not extras, (
        f"YAML has device classes not in risk_scorer.py: {sorted(extras)}"
    )

    assert body["unknown_fallback"] == pytest.approx(_TRACK_A_SURFACING_DEFAULT)


# ── R3 + R4: DAE artifacts ───────────────────────────────────────────────

DAE_THR = PROJECT_ROOT / "results" / "models" / "dae_thresholds.json"
DAE_CAL = PROJECT_ROOT / "results" / "models" / "dae_calibration.json"
DAE_DET = PROJECT_ROOT / "results" / "models" / "dae_detector.json"

_artifacts_present = DAE_THR.exists() and DAE_CAL.exists() and DAE_DET.exists()
artifact_required = pytest.mark.skipif(
    not _artifacts_present,
    reason="DAE artifacts missing; run module2_detection.build_dae_v4_artifacts",
)


@artifact_required
def test_dae_thresholds_well_formed() -> None:
    body = json.loads(DAE_THR.read_text())
    thresholds = body["thresholds"]
    assert thresholds["screening_threshold"] < thresholds["confirmation_threshold"]
    assert thresholds["confirmation_threshold"] < thresholds["high_confidence_threshold"]
    assert thresholds["training_min_error"] <= thresholds["screening_threshold"]
    assert thresholds["high_confidence_threshold"] <= thresholds["training_max_error"]
    assert thresholds["training_size"] > 0
    assert body["format"] == "layer1_v4.dae_thresholds"
    assert body["source_detector_sha256"]


@artifact_required
def test_dae_thresholds_match_train_error_percentiles() -> None:
    """Thresholds must be byte-equivalent to the percentiles of the
    train_errors array stored in the source DAE sidecar — this is what
    'derived purely from the trained detector' means.
    """
    detector = json.loads(DAE_DET.read_text())
    body = json.loads(DAE_THR.read_text())
    train_errors = np.asarray(detector["train_errors"], dtype=np.float64)
    thresholds = body["thresholds"]
    assert thresholds["screening_threshold"] == pytest.approx(
        float(np.percentile(train_errors, 80))
    )
    assert thresholds["confirmation_threshold"] == pytest.approx(
        float(np.percentile(train_errors, 95))
    )
    assert thresholds["high_confidence_threshold"] == pytest.approx(
        float(np.percentile(train_errors, 99))
    )


@artifact_required
def test_dae_calibration_lookup_is_monotone() -> None:
    body = json.loads(DAE_CAL.read_text())
    lookup = np.asarray(body["percentile_lookup"], dtype=np.float64)
    assert lookup.size == body["n_lookup_points"]
    diffs = np.diff(lookup)
    assert (diffs >= 0).all(), "percentile_lookup must be non-decreasing"
    assert lookup[0] >= 0


@artifact_required
def test_dae_calibration_score_in_zero_one_range() -> None:
    """A min training error → score ≈ 0; a max training error → score ≈ 1.
    A value below all training errors clamps to 0; above all clamps to 1.
    """
    body = json.loads(DAE_CAL.read_text())
    lookup = np.asarray(body["percentile_lookup"], dtype=np.float64)
    n = lookup.size

    def score(raw: float) -> float:
        return float(np.searchsorted(lookup, raw)) / n

    assert score(lookup[0]) == pytest.approx(0.0, abs=1e-9)
    assert score(lookup[-1]) == pytest.approx(1.0 - 1.0 / n, abs=1.0 / n)
    assert score(lookup[0] - 1.0) == 0.0
    assert score(lookup[-1] + 1.0) == 1.0


@artifact_required
def test_dae_artifacts_share_source_sha() -> None:
    thr = json.loads(DAE_THR.read_text())
    cal = json.loads(DAE_CAL.read_text())
    assert thr["source_detector_sha256"] == cal["source_detector_sha256"]


# ── R5: anomalous_dims_z ─────────────────────────────────────────────────

@artifact_required
def test_anomalous_dims_z_matches_manual_z_score() -> None:
    """The flagged dims for each row must be exactly those whose
    per-feature error exceeds the batch-mean by ``z_threshold`` σ —
    verified against a manual computation on the same matrix.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from module2_detection.models.DAE import DAEDetector

    weights = PROJECT_ROOT / "results" / "models" / "dae_model.weights.h5"
    if not weights.exists():
        pytest.skip("DAE weights file missing; cannot exercise the model")

    dae = DAEDetector.from_artefacts(DAE_DET, weights)
    n_features = int(dae._feat_weights.shape[0])

    rng = np.random.default_rng(42)
    centre = ((np.asarray(dae._feat_min) + np.asarray(dae._clip_hi)) / 2.0)
    X = np.tile(centre, (10, 1)) + rng.normal(0, 0.05, size=(10, n_features))

    z_threshold = 2.0
    per_feat, anom = dae.anomalous_dims_z(
        X.astype(np.float32), z_threshold=z_threshold,
    )
    assert per_feat.shape == (10, n_features)

    mu = per_feat.mean(axis=0)
    sigma = per_feat.std(axis=0)
    expected_z = (per_feat - mu) / (sigma + 1e-8)
    for i in range(10):
        expected = np.where(expected_z[i] > z_threshold)[0].tolist()
        assert anom[i] == expected, (
            f"row {i}: anomalous dims {anom[i]} != expected {expected}"
        )


@artifact_required
def test_anomalous_dims_z_handles_single_sample_gracefully() -> None:
    """For n=1 the within-batch z-score is undefined; method must
    return an empty list, not crash.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from module2_detection.models.DAE import DAEDetector

    weights = PROJECT_ROOT / "results" / "models" / "dae_model.weights.h5"
    if not weights.exists():
        pytest.skip("DAE weights file missing; cannot exercise the model")

    dae = DAEDetector.from_artefacts(DAE_DET, weights)
    n_features = int(dae._feat_weights.shape[0])
    X = np.zeros((1, n_features), dtype=np.float32)
    per_feat, anom = dae.anomalous_dims_z(X)
    assert per_feat.shape == (1, n_features)
    assert anom == [[]]
