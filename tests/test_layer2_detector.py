"""Layer 2 redesign — per-alert detector contract tests.

Pins the canonical Layer 2 output shape and exercises the per-alert
path (Step 1 + Step 2a + Step 2b) against the artefacts already on
disk under ``results/models/``.

If these artefacts are missing in CI, the tests are skipped — the
training pipeline must run before the per-alert path can be exercised
end-to-end. Local development covers the happy path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Skip the whole module if heavy ML deps (TensorFlow for DAE) are unavailable.
pytest.importorskip("tensorflow")

from module2_detection.layer2_detector import (
    Layer2Detector,
    Layer2Output,
    THRESHOLD_LEVELS,
    PER_DIM_PERCENTILE,
)


MODELS_DIR = PROJECT_ROOT / "results/models"
TEST_PARQUET = PROJECT_ROOT / "data/processed/test_phase1.parquet"

REQUIRED_ARTEFACTS = (
    MODELS_DIR / "xgboost_final_pipeline.pkl",
    MODELS_DIR / "random_forest_final_pipeline.pkl",
    MODELS_DIR / "decision_tree_final_pipeline.pkl",
    MODELS_DIR / "dae_detector.json",
    MODELS_DIR / "dae_model.weights.h5",
)


def _artefacts_present() -> bool:
    return all(p.exists() for p in REQUIRED_ARTEFACTS) and TEST_PARQUET.exists()


pytestmark = pytest.mark.skipif(
    not _artefacts_present(),
    reason="Layer 2 detector requires trained Track A + DAE artefacts on disk",
)


@pytest.fixture(scope="module")
def detector() -> Layer2Detector:
    return Layer2Detector(prefer_calibrated=True)


@pytest.fixture(scope="module")
def sample_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull 5 rows from the on-disk test parquet for a smoke test.

    Returns (X, y, attack_cats) with X already-scaled (Phase 1 output).
    """
    df = pd.read_parquet(TEST_PARQUET)
    drop = [c for c in (
        "Label", "Attack Category", "row_id", "device_class", "attack_category",
    ) if c in df.columns]
    X = df.drop(columns=drop).head(5).values.astype(np.float32)
    y = df["Label"].head(5).values
    cats = df["Attack Category"].head(5).values
    return X, y, cats


def test_layer2output_field_shape(detector: Layer2Detector,
                                   sample_features) -> None:
    """All fields the architecture diagram specifies are populated."""
    X, _, _ = sample_features
    out = detector.score_alert(X[0])
    assert isinstance(out, Layer2Output)

    # All fields the diagram lists as "COMBINED OUTPUT TO LAYER 3"
    assert isinstance(out.p_xgb, float)
    assert isinstance(out.p_rf, float)
    assert isinstance(out.p_dt, float)
    assert isinstance(out.c_track_a, float)
    assert isinstance(out.diversity_score, float)
    assert isinstance(out.dae_score, float)
    assert isinstance(out.c_track_b, float)
    assert isinstance(out.device_class_threshold, float)
    assert isinstance(out.data_quality_flag, str)
    assert isinstance(out.nan_rate, float)
    assert isinstance(out.threshold_level, str)
    assert isinstance(out.anomalous_dims, list)


def test_probabilities_in_unit_interval(detector: Layer2Detector,
                                         sample_features) -> None:
    X, _, _ = sample_features
    for i in range(len(X)):
        out = detector.score_alert(X[i])
        for name, val in [("p_xgb", out.p_xgb),
                           ("p_rf", out.p_rf),
                           ("p_dt", out.p_dt),
                           ("dae_score", out.dae_score),
                           ("c_track_a", out.c_track_a)]:
            assert 0.0 <= val <= 1.0, f"{name}={val} out of [0,1]"


def test_diversity_is_nonneg_and_bounded(detector: Layer2Detector,
                                          sample_features) -> None:
    X, _, _ = sample_features
    out = detector.score_alert(X[0])
    # std of three values in [0,1]: bounded by ~0.47, can't be negative.
    assert 0.0 <= out.diversity_score <= 0.47


def test_c_track_a_equals_max_of_three(detector: Layer2Detector,
                                        sample_features) -> None:
    X, _, _ = sample_features
    out = detector.score_alert(X[0])
    assert out.c_track_a == max(out.p_xgb, out.p_rf, out.p_dt)


def test_device_threshold_resolves_per_class(detector: Layer2Detector,
                                              sample_features) -> None:
    X, _, _ = sample_features
    out_pump = detector.score_alert(X[0], device_class="infusion_pump")
    out_ehr = detector.score_alert(X[0], device_class="ehr_workstation")
    out_unknown = detector.score_alert(X[0], device_class=None)
    assert out_pump.device_class_threshold == 0.03
    assert out_ehr.device_class_threshold == 0.10
    assert out_unknown.device_class_threshold == 0.05


def test_data_quality_ok_for_clean_input(detector: Layer2Detector,
                                          sample_features) -> None:
    X, _, _ = sample_features
    out = detector.score_alert(X[0])
    assert out.data_quality_flag == "OK"
    assert out.nan_rate == 0.0


def test_data_quality_flagged_for_nan_injection(
    detector: Layer2Detector, sample_features,
) -> None:
    """EA-06 mitigation: NaN-injected rows route to a non-OK quality flag."""
    X, _, _ = sample_features
    poisoned = X[0].copy()
    poisoned[:5] = np.nan          # 5/25 = 20% NaN cells → DEGRADED band
    out = detector.score_alert(poisoned)
    assert out.data_quality_flag in ("DEGRADED", "FAILED")
    assert out.nan_rate >= 0.05


def test_threshold_level_is_one_of_vocabulary(detector: Layer2Detector,
                                                 sample_features) -> None:
    """Task 4: threshold_level emits one of the four documented buckets."""
    X, _, _ = sample_features
    for i in range(len(X)):
        out = detector.score_alert(X[i])
        assert out.threshold_level in THRESHOLD_LEVELS, (
            f"row {i}: threshold_level={out.threshold_level!r} not in "
            f"{THRESHOLD_LEVELS}"
        )


def test_threshold_level_monotone_with_error(detector: Layer2Detector) -> None:
    """Larger reconstruction error → higher (or equal) bucket level."""
    rank = {lv: i for i, lv in enumerate(THRESHOLD_LEVELS)}
    benign_x = np.zeros(25, dtype=np.float32)            # ~bulk of training
    poisoned_x = np.full(25, 1e3, dtype=np.float32)      # far off-manifold
    out_benign = detector.score_alert(benign_x)
    out_poisoned = detector.score_alert(poisoned_x)
    assert rank[out_poisoned.threshold_level] >= rank[out_benign.threshold_level]


def test_multi_thresholds_property_ordered(detector: Layer2Detector) -> None:
    """multi_thresholds property exposes p80 ≤ p95 ≤ p99."""
    th = detector.multi_thresholds
    assert th["p80"] <= th["p95"] <= th["p99"]


def test_per_dim_errors_shape_and_nonneg(detector: Layer2Detector,
                                           sample_features) -> None:
    """per_dim_errors is (n_features,) non-negative; sum ≈ recon error."""
    X, _, _ = sample_features
    out = detector.score_alert(X[0])
    assert out.per_dim_errors is not None
    assert out.per_dim_errors.ndim == 1
    assert out.per_dim_errors.shape[0] == len(detector.cascade_feature_names)
    assert (out.per_dim_errors >= 0).all()
    # `per_sample == per_feature_weighted.sum(axis=1)` per the DAE doc;
    # our recon_err includes an OOD penalty, so check inequality.
    assert out.dae_score_raw_error >= float(out.per_dim_errors.sum()) - 1e-6


def test_anomalous_dims_indices_valid(detector: Layer2Detector,
                                       sample_features) -> None:
    """anomalous_dims indices index into the (28-dim) cascade feature list."""
    X, _, _ = sample_features
    out = detector.score_alert(X[0])
    n_dims = len(detector.cascade_feature_names)
    for i in out.anomalous_dims:
        assert 0 <= i < n_dims
    assert len(out.anomalous_dim_names) == len(out.anomalous_dims)
    for idx, name in zip(out.anomalous_dims, out.anomalous_dim_names):
        assert detector.cascade_feature_names[idx] == name


def test_anomalous_dims_populated_for_off_manifold_input(
    detector: Layer2Detector,
) -> None:
    """A wildly off-manifold row triggers a non-empty anomalous_dims set."""
    poisoned = np.full(25, 1e3, dtype=np.float32)
    out = detector.score_alert(poisoned)
    assert len(out.anomalous_dims) > 0, (
        "an extreme off-manifold input should flag at least one anomalous dim"
    )


def test_per_dim_thresholds_cover_all_features(detector: Layer2Detector) -> None:
    """per_dim_thresholds aligns 1:1 with cascade_feature_names."""
    th = detector.per_dim_thresholds
    assert th.shape == (len(detector.cascade_feature_names),)
    assert (th >= 0).all()


def test_as_dict_serialisable(detector: Layer2Detector, sample_features) -> None:
    """Layer 3 / persistence path: as_dict() yields a JSON-friendly mapping."""
    import json
    X, _, _ = sample_features
    d = detector.score_alert(X[0]).as_dict()
    json.dumps(d)  # must not raise
    assert "p_xgb" in d
    assert "diversity_score" in d
    assert "data_quality_flag" in d


def test_calibration_status_reports_truth(detector: Layer2Detector) -> None:
    """The detector tells which models actually used a calibrator."""
    status = detector.calibration_status
    assert set(status.keys()) == {"xgboost", "random_forest", "decision_tree"}
    for k, v in status.items():
        assert isinstance(v, bool)
