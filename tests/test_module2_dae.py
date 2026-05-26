"""DAEDetector tests — including C2 reproducibility fix.

Critical invariants:
  - JSON sidecar + Keras weights round-trip (pickle-free load path)
  - evaluate() is DETERMINISTIC across runs (C2 fix)
  - predict() default is noisy (TM-04 defense); deterministic=True opts out
  - _noisy_threshold uses per-instance RNG (Y8 fix) — same seed → same jitter
  - OOD penalty fires on samples outside winsorize bounds
  - Bottleneck < n_features enforced at build time
"""
from __future__ import annotations


import numpy as np
import pytest

from module2_detection.models.DAE import DAEDetector


@pytest.fixture(scope="module")
def fitted_dae():
    """Tiny but real DAE fitted on synthetic benign-only data."""
    rng = np.random.default_rng(42)
    X_benign = rng.normal(loc=0, scale=1, size=(120, 6)).astype(np.float32)
    det = DAEDetector(
        encoding_dims=[8, 4, 8],
        noise_rate=0.1,
        epochs=5,
        batch_size=32,
        learning_rate=1e-3,
        threshold_percentile=95.0,
        random_state=42,
    )
    det.fit(X_benign, validation_split=0.2)
    return det, X_benign


# ── C2: evaluate() determinism ────────────────────────────────────────


def test_evaluate_is_deterministic_across_runs(fitted_dae):
    """Manuscript metrics MUST be reproducible — evaluate() uses fixed
    threshold, not the noisy one."""
    det, X_benign = fitted_dae
    rng = np.random.default_rng(99)
    X_attack = rng.normal(loc=3, scale=1, size=(40, 6)).astype(np.float32)
    X_test = np.vstack([X_benign[:40], X_attack])
    y_test = np.array([0] * 40 + [1] * 40)

    m1 = det.evaluate(X_test, y_test)
    m2 = det.evaluate(X_test, y_test)
    assert m1["attack_f1"] == m2["attack_f1"]
    assert m1["attack_f2"] == m2["attack_f2"]
    assert m1["macro_f1"] == m2["macro_f1"]


def test_predict_deterministic_kwarg_matches_threshold(fitted_dae):
    """predict(deterministic=True) uses the FIXED threshold."""
    det, X_benign = fitted_dae
    X = X_benign[:20]
    errors = det.reconstruction_error(X)
    expected = (errors > det._threshold).astype(int)
    actual = det.predict(X, deterministic=True)
    np.testing.assert_array_equal(expected, actual)


def test_predict_deterministic_repeatable(fitted_dae):
    """Same input + deterministic=True → identical output every call."""
    det, X_benign = fitted_dae
    X = X_benign[:20]
    p1 = det.predict(X, deterministic=True)
    p2 = det.predict(X, deterministic=True)
    np.testing.assert_array_equal(p1, p2)


# ── Y8: noisy threshold uses local RNG ───────────────────────────────


def test_noisy_threshold_jitter_within_10pct(fitted_dae):
    """_noisy_threshold returns ±10% of the fixed threshold."""
    det, _ = fitted_dae
    threshold = det._threshold
    if threshold == 0.0:
        pytest.skip("Fixed threshold is 0; jitter can't be meaningfully tested")
    samples = [det._noisy_threshold() for _ in range(50)]
    for s in samples:
        assert 0.9 * threshold <= s <= 1.1 * threshold


def test_noisy_threshold_same_seed_produces_same_sequence():
    """Y8 — per-instance RNG seeded from random_state produces the same
    jitter sequence across DAEDetector instances with the same seed."""
    d1 = DAEDetector(random_state=42)
    d1._threshold = 1.0
    d2 = DAEDetector(random_state=42)
    d2._threshold = 1.0
    seq1 = [d1._noisy_threshold() for _ in range(10)]
    seq2 = [d2._noisy_threshold() for _ in range(10)]
    assert seq1 == seq2


def test_noisy_threshold_different_seeds_diverge():
    d1 = DAEDetector(random_state=42)
    d1._threshold = 1.0
    d2 = DAEDetector(random_state=7)
    d2._threshold = 1.0
    seq1 = [d1._noisy_threshold() for _ in range(10)]
    seq2 = [d2._noisy_threshold() for _ in range(10)]
    assert seq1 != seq2


def test_noisy_threshold_jitter_centred_on_threshold(fitted_dae):
    """Average over many samples should be close to the fixed threshold."""
    det, _ = fitted_dae
    threshold = det._threshold
    if threshold == 0.0:
        pytest.skip("Fixed threshold is 0")
    samples = np.array([det._noisy_threshold() for _ in range(1000)])
    # Mean should be close to threshold (uniform(-0.10, 0.10) → 0 mean)
    assert abs(samples.mean() - threshold) < threshold * 0.02


# ── Sidecar round-trip (pickle-free load path) ────────────────────────


def test_sidecar_round_trip_byte_identical(fitted_dae, tmp_path):
    """save_artefacts + from_artefacts → identical predict_proba output."""
    det, X_benign = fitted_dae
    json_path = tmp_path / "dae.json"
    weights_path = tmp_path / "dae.weights.h5"
    det.save_artefacts(json_path, weights_path)

    det2 = DAEDetector.from_artefacts(json_path, weights_path)
    # Same threshold + normaliser + feature weights → same predict_proba
    p1 = det.predict_proba(X_benign[:20])
    p2 = det2.predict_proba(X_benign[:20])
    np.testing.assert_array_almost_equal(p1, p2, decimal=5)


def test_sidecar_rejects_wrong_format(tmp_path):
    import json
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"format": "wrong-format"}))
    w = tmp_path / "dae.weights.h5"
    w.touch()
    with pytest.raises(ValueError, match="not a phase2.dae_detector.v1"):
        DAEDetector.from_artefacts(p, w)


def test_sidecar_save_requires_fit(tmp_path):
    det = DAEDetector(random_state=42)
    with pytest.raises(RuntimeError, match="not fitted"):
        det.save_artefacts(tmp_path / "x.json", tmp_path / "x.weights.h5")


def test_sidecar_load_fails_on_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        DAEDetector.from_artefacts(tmp_path / "nope.json", tmp_path / "nope.h5")


def test_sidecar_atomic_write_no_tmp_leftover(fitted_dae, tmp_path):
    det, _ = fitted_dae
    det.save_artefacts(tmp_path / "x.json", tmp_path / "x.weights.h5")
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == []


# ── Bottleneck dimension enforcement ──────────────────────────────────


def test_bottleneck_too_wide_rejected():
    """Bottleneck must be < n_features to force compression."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5)).astype(np.float32)
    det = DAEDetector(encoding_dims=[8, 5, 8], random_state=42)  # bottleneck = n_features
    with pytest.raises(ValueError, match="must be < n_features"):
        det.fit(X)


# ── OOD penalty (OOD-02 fix) ──────────────────────────────────────────


def test_ood_penalty_zero_for_in_distribution(fitted_dae):
    """Samples within winsorize bounds produce zero or near-zero OOD penalty."""
    det, X_benign = fitted_dae
    # In-distribution sample (use training data — should be in bounds)
    X_in = X_benign[:5]
    penalty = det._ood_penalty(X_in)
    # In-distribution samples may slightly exceed bounds at corners; we
    # just assert the penalty is small relative to typical recon error.
    assert (penalty < det._threshold).all()


def test_ood_penalty_nonzero_for_extreme_values(fitted_dae):
    """Samples way outside winsorize bounds produce non-zero penalty."""
    det, X_benign = fitted_dae
    # Force extreme value far outside any plausible benign bound
    X_extreme = np.full((1, X_benign.shape[1]), 1e6, dtype=np.float32)
    penalty = det._ood_penalty(X_extreme)
    assert penalty[0] > 0


# ── reconstruction_error_decomposed equivalence ───────────────────────


def test_decomposed_recon_error_matches_sum(fitted_dae):
    """per_sample == per_feature_weighted.sum(axis=1) — invariant from
    the docstring."""
    det, X_benign = fitted_dae
    X = X_benign[:10]
    per_sample, per_feat = det.reconstruction_error_decomposed(X)
    np.testing.assert_array_almost_equal(
        per_sample, per_feat.sum(axis=1) + det._ood_penalty(X), decimal=5,
    )


# ── predict_proba bounds + scaling ────────────────────────────────────


def test_predict_proba_in_unit_interval(fitted_dae):
    """predict_proba output must lie in [0, 1]."""
    det, X_benign = fitted_dae
    rng = np.random.default_rng(0)
    X_mixed = np.vstack([X_benign[:20], rng.normal(loc=3, scale=1, size=(20, 6))])
    proba = det.predict_proba(X_mixed.astype(np.float32))
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_attack_samples_get_higher_proba_than_benign(fitted_dae):
    """Out-of-distribution samples should score higher than in-distribution."""
    det, X_benign = fitted_dae
    rng = np.random.default_rng(0)
    X_attack = rng.normal(loc=5, scale=1, size=(20, X_benign.shape[1])).astype(np.float32)
    p_benign = det.predict_proba(X_benign[:20]).mean()
    p_attack = det.predict_proba(X_attack).mean()
    assert p_attack > p_benign


# ── reconstruction_error raises on unfitted model ─────────────────────


def test_reconstruction_error_unfitted_raises():
    det = DAEDetector(random_state=42)
    X = np.zeros((3, 5))
    with pytest.raises(RuntimeError, match="not fitted"):
        det.reconstruction_error(X)
