"""Layer 2 v4.0 verification tests.

Targets the deltas added when bringing the existing Layer 2 detector up
to v4.0:

  * INVARIANT 1 — ``c_detect = max(c_track_a, c_track_b)`` and
    ``c_detect >= p_xgb`` for every alert (DAE only elevates).
  * R4 — the DAE ``dae_score`` is calibrated by percentile-rank against
    the benign training distribution when
    ``results/models/dae_calibration.json`` is present, falling back to
    the legacy linear-threshold scaling when it is not.
  * Multi-threshold loading — the detector picks up p80/p95/p99 from
    ``results/models/dae_thresholds.json`` when present (canonical
    Layer 1 v4 artifact).
  * EA-06 — NaN cells are replaced with the per-feature BENIGN_MEDIAN,
    not with 0.0.
  * Latency — single-alert ``score_alert`` stays well under the 500 ms
    Layer 2 budget on a CPU-only machine.

The whole module is skipped if the trained artefacts are not on disk
(matching ``test_layer2_detector.py``'s skip rule).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("tensorflow")

from module2_detection.layer2_detector import Layer2Detector  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "results/models"
TEST_PARQUET = PROJECT_ROOT / "data/processed/test_phase1.parquet"
DAE_CAL = MODELS_DIR / "dae_calibration.json"
DAE_THR = MODELS_DIR / "dae_thresholds.json"

# Phase B: XGB-only production runtime. RF/DT are baselines, optional.
REQUIRED = (
    MODELS_DIR / "xgboost_final_pipeline.pkl",
    MODELS_DIR / "dae_detector.json",
    MODELS_DIR / "dae_model.weights.h5",
)


pytestmark = pytest.mark.skipif(
    not (all(p.exists() for p in REQUIRED) and TEST_PARQUET.exists()),
    reason="Layer 2 v4 tests require trained XGBoost + DAE artefacts on disk",
)


@pytest.fixture(scope="module")
def detector() -> Layer2Detector:
    return Layer2Detector(prefer_calibrated=True)


@pytest.fixture(scope="module")
def sample_features() -> np.ndarray:
    df = pd.read_parquet(TEST_PARQUET)
    drop = [c for c in (
        "Label", "Attack Category", "row_id", "device_class", "attack_category",
    ) if c in df.columns]
    return df.drop(columns=drop).head(20).values.astype(np.float32)


# ── INVARIANT 1 ──────────────────────────────────────────────────────────

def test_c_detect_field_present_and_equals_max(
    detector: Layer2Detector, sample_features: np.ndarray,
) -> None:
    out = detector.score_alert(sample_features[0])
    assert hasattr(out, "c_detect")
    assert out.c_detect == pytest.approx(max(out.c_track_a, out.c_track_b))


def test_invariant_1_c_detect_geq_p_xgb_across_batch(
    detector: Layer2Detector, sample_features: np.ndarray,
) -> None:
    """INVARIANT 1: DAE must only elevate, never reduce — so c_detect
    is at least as large as the primary tree's calibrated probability
    on every alert.
    """
    for i in range(len(sample_features)):
        out = detector.score_alert(sample_features[i])
        assert out.c_detect >= out.p_xgb - 1e-9, (
            f"row {i}: c_detect={out.c_detect} < p_xgb={out.p_xgb}"
        )
        assert out.c_detect >= out.c_track_a - 1e-9


def test_invariant_1_holds_for_off_manifold_input(
    detector: Layer2Detector,
) -> None:
    """A wildly off-manifold input drives the DAE high; INVARIANT 1
    must still hold (and c_detect should be DAE-driven here).
    """
    poisoned = np.full(25, 1e3, dtype=np.float32)
    out = detector.score_alert(poisoned)
    assert out.c_detect >= out.p_xgb - 1e-9
    assert out.c_detect == pytest.approx(max(out.c_track_a, out.c_track_b))


# ── R4: percentile-rank DAE score calibration ───────────────────────────

@pytest.mark.skipif(not DAE_CAL.exists(), reason="dae_calibration.json missing")
def test_dae_score_uses_percentile_rank_when_calibration_present(
    detector: Layer2Detector, sample_features: np.ndarray,
) -> None:
    out = detector.score_alert(sample_features[0])
    assert out.dae_score_calibration == "percentile_rank"


@pytest.mark.skipif(not DAE_CAL.exists(), reason="dae_calibration.json missing")
def test_dae_score_matches_searchsorted_against_canonical_lookup(
    detector: Layer2Detector, sample_features: np.ndarray,
) -> None:
    """The dae_score the detector emits is exactly the rank of the raw
    reconstruction error in the persisted percentile_lookup — i.e. the
    detector is using the canonical Layer 1 v4 calibration artefact, not
    a re-derived approximation.
    """
    body = json.loads(DAE_CAL.read_text())
    lookup = np.asarray(body["percentile_lookup"], dtype=np.float64)
    for i in range(min(5, len(sample_features))):
        out = detector.score_alert(sample_features[i])
        expected = float(np.searchsorted(lookup, out.dae_score_raw_error)) / lookup.size
        assert out.dae_score == pytest.approx(np.clip(expected, 0.0, 1.0))


@pytest.mark.skipif(not DAE_CAL.exists(), reason="dae_calibration.json missing")
def test_dae_score_in_unit_interval_for_extreme_inputs(
    detector: Layer2Detector,
) -> None:
    """Both an off-manifold input (rank → 1) and a zero vector (rank → 0)
    stay strictly in [0, 1].
    """
    out_lo = detector.score_alert(np.zeros(25, dtype=np.float32))
    out_hi = detector.score_alert(np.full(25, 1e3, dtype=np.float32))
    assert 0.0 <= out_lo.dae_score <= 1.0
    assert 0.0 <= out_hi.dae_score <= 1.0
    # The off-manifold input should rank well above an in-distribution
    # zero vector — preserves "DAE flags novelty" semantics.
    assert out_hi.dae_score >= out_lo.dae_score


# ── Multi-threshold loaded from canonical JSON ───────────────────────────

@pytest.mark.skipif(not DAE_THR.exists(), reason="dae_thresholds.json missing")
def test_multi_thresholds_match_canonical_json(
    detector: Layer2Detector,
) -> None:
    body = json.loads(DAE_THR.read_text())
    t = body["thresholds"]
    th = detector.multi_thresholds
    assert th["p80"] == pytest.approx(float(t["screening_threshold"]))
    assert th["p95"] == pytest.approx(float(t["confirmation_threshold"]))
    assert th["p99"] == pytest.approx(float(t["high_confidence_threshold"]))


# ── EA-06: BENIGN_MEDIAN replacement ─────────────────────────────────────

def test_ea06_nan_replaced_with_benign_median_not_zero() -> None:
    """The sanitizer must replace NaN with the per-feature benign median
    — replacement with 0.0 would create an artificial outlier in the
    joint feature/prediction space exploitable by an attacker.
    """
    from src.preprocessing import (
        FEATURE_NAMES_25,
        load_benign_medians,
        sanitize_features,
    )

    medians = load_benign_medians()
    feats = list(FEATURE_NAMES_25)

    # Pick a feature whose benign median is meaningfully non-zero so a
    # zero-replacement bug would be detectable.
    target_idx, target_name, target_median = next(
        ((i, f, m) for i, (f, m) in enumerate(
            ((f, medians.get(f, 0.0)) for f in feats),
        ) if abs(m) > 1e-6),
        (None, None, None),
    )
    if target_idx is None:
        pytest.skip("All medians are ~0; cannot distinguish zero-fill from median-fill.")

    x = np.zeros(25, dtype=np.float64)
    x[target_idx] = float("nan")

    x_clean, flag, nan_rate = sanitize_features(x)
    assert flag == "OK"  # 1/25 = 4% < 5% DEGRADED threshold
    assert x_clean[target_idx] == pytest.approx(float(target_median))
    assert x_clean[target_idx] != 0.0


# ── Latency budget ───────────────────────────────────────────────────────

def test_score_alert_p95_latency_under_budget(
    detector: Layer2Detector, sample_features: np.ndarray,
) -> None:
    """Layer 2's per-alert path must stay well under the 500 ms total
    budget on this CPU-only test runner. We exercise 50 calls to get a
    stable P95.
    """
    timings_ms: list[float] = []
    # Warm up so the first-call TF graph compile doesn't dominate P95.
    for _ in range(3):
        detector.score_alert(sample_features[0])
    for i in range(50):
        x = sample_features[i % len(sample_features)]
        t0 = time.perf_counter()
        detector.score_alert(x)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    p95 = float(np.percentile(timings_ms, 95))
    assert p95 < 500.0, (
        f"Layer 2 score_alert P95={p95:.1f}ms exceeds 500ms budget; "
        f"min={min(timings_ms):.1f}, median={np.median(timings_ms):.1f}, "
        f"max={max(timings_ms):.1f}"
    )
