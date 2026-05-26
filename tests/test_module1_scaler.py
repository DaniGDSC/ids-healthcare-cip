"""RobustScalerTransformer tests — fit-on-train + JSON sidecar.

Critical invariants:
  - fit ONLY on train data; transform-without-fit raises
  - JSON sidecar round-trip produces byte-identical transform
  - NO pickle path: legacy .pkl request rewritten to .json + .pkl deleted
  - Unknown scaler method rejected at construction
  - serialised attributes are allowlist-restricted (per method)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from module1_preprocessing.phase1.scaler import RobustScalerTransformer


@pytest.fixture
def X_train():
    return np.array([[1, 100], [2, 200], [3, 300], [4, 400], [5, 500]], dtype=float)


@pytest.fixture
def X_test():
    return np.array([[1.5, 150], [3.5, 350]], dtype=float)


# ── Fit / transform contract ──────────────────────────────────────────


def test_transform_without_fit_raises(X_test):
    scaler = RobustScalerTransformer()
    with pytest.raises(RuntimeError, match="Scaler not fitted"):
        scaler.transform(X_test)


def test_save_without_fit_raises(tmp_path):
    scaler = RobustScalerTransformer()
    with pytest.raises(RuntimeError, match="not fitted"):
        scaler.save(tmp_path / "s.json")


def test_fit_then_transform_changes_distribution(X_train):
    scaler = RobustScalerTransformer().fit(X_train)
    out = scaler.transform(X_train)
    # Median of robust-scaled train data → 0
    assert abs(np.median(out, axis=0)).max() < 1e-9


def test_scale_both_fits_train_only(X_train, X_test):
    scaler = RobustScalerTransformer()
    train_s, test_s = scaler.scale_both(X_train, X_test)
    # Transform same train → same result (deterministic)
    train_s2 = scaler.transform(X_train)
    np.testing.assert_array_equal(train_s, train_s2)
    # Test set scaled with TRAIN params (not refit)
    expected_test = scaler._scaler.transform(X_test)
    np.testing.assert_array_equal(test_s, expected_test)


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        RobustScalerTransformer(method="bogus")


# ── Sidecar round-trip (RCE-free guarantee) ──────────────────────────


def test_sidecar_round_trip_byte_identical_transform(X_train, X_test, tmp_path):
    """from_json must reproduce the exact same transform."""
    scaler1 = RobustScalerTransformer().fit(X_train)
    expected = scaler1.transform(X_test)
    sidecar = tmp_path / "s.json"
    scaler1.save(sidecar)

    scaler2 = RobustScalerTransformer.from_json(sidecar)
    actual = scaler2.transform(X_test)
    np.testing.assert_array_equal(expected, actual)


def test_save_rejects_pkl_path_writes_json(X_train, tmp_path):
    """Legacy .pkl filename → .json on disk + .pkl never created."""
    scaler = RobustScalerTransformer().fit(X_train)
    pkl = tmp_path / "scaler.pkl"
    scaler.save(pkl)
    assert (tmp_path / "scaler.json").exists()
    assert not pkl.exists()


def test_save_deletes_existing_legacy_pkl(X_train, tmp_path):
    """Existing .pkl on disk must be removed at save time."""
    pkl = tmp_path / "scaler.pkl"
    pkl.write_bytes(b"legacy executable pickle bytes")
    scaler = RobustScalerTransformer().fit(X_train)
    scaler.save(pkl)
    assert not pkl.exists()


def test_load_rejects_wrong_format(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"format": "not-a-scaler"}')
    with pytest.raises(ValueError, match="not a phase1.scaler.v1 sidecar"):
        RobustScalerTransformer.from_json(p)


def test_load_rejects_unknown_method_in_sidecar(tmp_path):
    import json
    p = tmp_path / "x.json"
    p.write_text(json.dumps({
        "format": "phase1.scaler.v1",
        "method": "bogus",
        "params": {},
    }))
    with pytest.raises(ValueError, match="Unknown scaler method"):
        RobustScalerTransformer.from_json(p)


def test_load_rejects_missing_required_attr(tmp_path):
    """If sidecar is missing a required scaler attribute, load fails."""
    import json
    p = tmp_path / "x.json"
    # Method=robust requires center_, scale_, n_features_in_; provide only 2.
    p.write_text(json.dumps({
        "format": "phase1.scaler.v1",
        "method": "robust",
        "params": {"center_": [1.0], "scale_": [1.0]},  # missing n_features_in_
    }))
    with pytest.raises(ValueError, match="missing required parameter"):
        RobustScalerTransformer.from_json(p)
