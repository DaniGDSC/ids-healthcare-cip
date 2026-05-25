"""Regression tests for the cascaded DAE-gating optimisation in
:class:`detection_engine.engine.DetectionEngine`.

Contract under test
-------------------
After the refactor, :meth:`DetectionEngine.predict` runs the DAE only
on rows where ``c_track_a < TAU_SKIP_DAE`` (XGBoost did NOT confidently
flag). Gated rows receive ``c_track_b = 0`` and ``y_pred_dae = 0`` —
``c_detect = max(c_track_a, c_track_b)`` collapses to ``c_track_a``
for those rows, which is identical to the un-gated path as long as the
gate threshold leaves enough margin above any plausible DAE score.

The ``_force_full_dae=True`` kwarg bypasses the gate and is used by
:meth:`write_test_predictions` so downstream evaluation artefacts
(AUC, PSI, threshold sweeps in ``drift_detection`` and
``dynamic_threshold_sim``) still receive a per-row reconstruction
error over the full test split.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.dae_input import TRACK_A_FOR_DAE
from detection_engine import DetectionEngine


# ── Fixture helpers ─────────────────────────────────────────────────────


def _load_test_split():
    """Load the phase1 test split via the same loader the engine uses.

    Skips the whole module if the parquet or model artefacts are not on
    disk — these tests are integration-style and need the real trained
    pipeline.
    """
    try:
        from module2_detection.module2_train_models import load_data
        _X_train, X_test, _y_train, y_test, _feat_names = load_data()
    except FileNotFoundError as e:
        pytest.skip(f"Phase 1 parquet not available: {e}")
    return X_test, y_test


@pytest.fixture(scope="module")
def engine_and_split():
    engine = DetectionEngine()
    X_test, y_test = _load_test_split()
    try:
        engine._load()
    except FileNotFoundError as e:
        pytest.skip(f"Model artefacts not available: {e}")
    return engine, X_test, y_test


# ── Tests ───────────────────────────────────────────────────────────────


def test_gated_rows_have_zero_c_track_b(engine_and_split):
    """Rows with c_track_a >= TAU_SKIP_DAE must skip the DAE entirely."""
    engine, X_test, _y = engine_and_split
    result = engine.predict(X_test)

    gated = result.c_track_a >= DetectionEngine.TAU_SKIP_DAE
    if not gated.any():
        pytest.skip(
            "No test rows have c_track_a >= TAU_SKIP_DAE; gating "
            "behaviour cannot be exercised on this artefact."
        )

    assert np.all(result.c_track_b[gated] == 0.0), (
        "Gated rows must have c_track_b == 0 (DAE skipped). "
        f"Saw nonzero on {(result.c_track_b[gated] != 0).sum()} rows."
    )
    assert np.all(result.y_pred_dae[gated] == 0), (
        "Gated rows must have y_pred_dae == 0 (DAE skipped)."
    )


def test_cascade_contract_vs_full_dae(engine_and_split):
    """Cascade vs full-DAE: enforce the three contract claims.

    1. Ungated rows are bit-identical (same code path).
    2. Gated rows have ``c_detect == c_track_a`` (because c_track_b=0
       and fusion is ``max(c_a, c_b)``).
    3. Cascade ``c_detect`` is never *higher* than full-DAE ``c_detect``
       — gating can only lose DAE elevations, never invent them.

    The cascade intentionally drops DAE elevations on rows XGBoost is
    already confident about (user-stated design: "DAE checks what
    XGBoost missed"). This test documents the expected magnitude of
    that loss so a future threshold tweak surfaces in CI.
    """
    engine, X_test, _y = engine_and_split
    cascade = engine.predict(X_test)
    full = engine.predict(X_test, _force_full_dae=True)

    gated = cascade.c_track_a >= DetectionEngine.TAU_SKIP_DAE
    ungated = ~gated

    # Claim 1: ungated rows go through the same code path.
    np.testing.assert_array_equal(
        cascade.c_track_b[ungated], full.c_track_b[ungated],
        err_msg="Ungated rows diverged between cascade and full-DAE paths.",
    )

    # Claim 2: gated rows' c_detect collapses to c_track_a.
    if gated.any():
        np.testing.assert_array_equal(
            cascade.c_detect[gated], cascade.c_track_a[gated],
            err_msg="Gated c_detect must equal c_track_a (since c_b=0).",
        )

    # Claim 3: cascade never overshoots full-DAE — it can only drop
    # elevations, not invent them.
    assert np.all(cascade.c_detect <= full.c_detect + 1e-6), (
        "Cascade c_detect exceeded full-DAE c_detect — fusion invariant"
        " violated."
    )

    # Diagnostic: record the worst elevation we dropped on gated rows
    # so the magnitude is visible in test output.
    if gated.any():
        max_lost = float((full.c_detect[gated] - cascade.c_detect[gated]).max())
        # Soft budget: if this grows past ~0.10, the gate may be too
        # aggressive for this dataset and TAU_SKIP_DAE should rise.
        assert max_lost <= 0.20, (
            f"Cascade dropped a DAE elevation of {max_lost:.4f} on a "
            f"gated row — exceeds 0.20 budget. Consider raising "
            f"TAU_SKIP_DAE (currently {DetectionEngine.TAU_SKIP_DAE})."
        )


def test_x_augmented_shape_and_population(engine_and_split):
    """x_augmented must be fully built for every row, gated or not."""
    engine, X_test, _y = engine_and_split
    result = engine.predict(X_test)

    n, n_raw = X_test.shape
    expected_cols = n_raw + len(TRACK_A_FOR_DAE)
    assert result.x_augmented.shape == (n, expected_cols), (
        f"x_augmented shape {result.x_augmented.shape} != "
        f"({n}, {expected_cols})"
    )

    # Last column(s) should be the XGBoost probas. x_augmented stores
    # them as float32 (engine cast); track_a_probas keeps sklearn's raw
    # float64, so the comparison is at float32 precision, not bitwise.
    xgb_col = result.x_augmented[:, n_raw + list(TRACK_A_FOR_DAE).index("xgboost")]
    np.testing.assert_allclose(
        xgb_col,
        result.track_a_probas["xgboost"].astype(np.float32),
        rtol=0, atol=0,
        err_msg="x_augmented xgboost column does not match track_a_probas (float32 cast).",
    )


def test_write_test_predictions_full_coverage(tmp_path, engine_and_split):
    """write_test_predictions must export per-row DAE scores for every test row."""
    engine, _X, _y = engine_and_split
    out = tmp_path / "dae_test_predictions.npz"
    engine.write_test_predictions(out_path=out)

    data = np.load(out)
    assert set(data.files) == {"y_true", "y_pred", "reconstruction_error"}, (
        f"npz keys changed: {data.files}"
    )
    n = len(data["y_true"])
    assert len(data["y_pred"]) == n
    assert len(data["reconstruction_error"]) == n

    # Full DAE coverage: reconstruction_error must have some non-trivial
    # values (not all zeros from a gated path).
    assert (data["reconstruction_error"] > 0).sum() == n, (
        "reconstruction_error has zeros — write_test_predictions "
        "appears to be using the cascade path instead of full DAE."
    )


def test_telemetry_logs_gated_count(engine_and_split, caplog):
    """predict() must log per-call gating telemetry."""
    engine, X_test, _y = engine_and_split
    with caplog.at_level(logging.INFO, logger="detection_engine.engine"):
        engine.predict(X_test[:64])

    matching = [r for r in caplog.records if "rows scored by DAE" in r.getMessage()]
    assert matching, (
        "Expected an INFO log line containing 'rows scored by DAE' from "
        f"detection_engine.engine; got messages: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
