"""dae_training tests — augmented-feature build path + OOF probas load.

The full ``train_dae()`` flow needs real Phase 1 parquets + bootstrap
baseline + per-detector OOF .npy files, so we test the unit-level pieces
(``_load_oof_probas`` + ``build_training_input``) directly with synthetic
fixtures.
"""
from __future__ import annotations


import numpy as np
import pytest

from module2_detection import dae_training as dt


@pytest.fixture
def mock_oof_dir(tmp_path, monkeypatch):
    """Stage fake OOF .npy files under a temp MODELS_DIR for the loader."""
    fake_dir = tmp_path / "models"
    fake_dir.mkdir()
    monkeypatch.setattr(dt, "MODELS_DIR", fake_dir)
    return fake_dir


def test_load_oof_probas_returns_per_detector_columns(mock_oof_dir):
    """Each Track A detector contributes one column; benign_mask filters rows."""
    from common.dae_input import TRACK_A_FOR_DAE

    n_train = 30
    benign_mask = np.array([True, False] * (n_train // 2))
    for name in TRACK_A_FOR_DAE:
        np.save(mock_oof_dir / f"{name}_oof_proba.npy",
                np.linspace(0, 1, n_train).astype(np.float32))

    out = dt._load_oof_probas(benign_mask)
    assert out.shape == (benign_mask.sum(), len(TRACK_A_FOR_DAE))
    assert out.dtype == np.float32


def test_load_oof_probas_missing_file_raises(mock_oof_dir):
    """Missing OOF file → clear error pointing at the right train step."""
    benign_mask = np.array([True] * 5)
    with pytest.raises(FileNotFoundError, match="train Track A first"):
        dt._load_oof_probas(benign_mask)


def test_load_oof_probas_no_thread_pool_executor():
    """Y7: ThreadPoolExecutor was removed — confirm it's not imported."""
    import ast
    src = open(dt.__file__).read()
    tree = ast.parse(src)
    bad_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for n in node.names:
                if "ThreadPool" in n.name:
                    bad_imports.append(n.name)
    assert not bad_imports, f"ThreadPool unexpectedly imported: {bad_imports}"


def test_build_training_input_shape(mock_oof_dir, monkeypatch):
    """X_benign_aug = [X_benign cols ... | Track A probas]."""
    from common.dae_input import TRACK_A_FOR_DAE
    n_train = 50
    n_feat = 4
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(n_train, n_feat)).astype(np.float32)
    y_train = np.array([0] * 30 + [1] * 20)
    feat_names = [f"f{i}" for i in range(n_feat)]

    for name in TRACK_A_FOR_DAE:
        np.save(mock_oof_dir / f"{name}_oof_proba.npy",
                rng.uniform(0, 1, n_train).astype(np.float32))

    X_aug, aug_names, benign_mask = dt.build_training_input(
        X_train, y_train, feat_names,
    )
    assert X_aug.shape == (30, n_feat + len(TRACK_A_FOR_DAE))  # 30 benign rows
    assert len(aug_names) == n_feat + len(TRACK_A_FOR_DAE)
    assert int(benign_mask.sum()) == 30
    assert X_aug.dtype == np.float32


def test_build_training_input_preserves_raw_features(mock_oof_dir):
    """First n_feat columns of X_aug must equal X_benign."""
    from common.dae_input import TRACK_A_FOR_DAE
    n_train = 20
    n_feat = 3
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(n_train, n_feat)).astype(np.float32)
    y_train = np.array([0] * 15 + [1] * 5)

    for name in TRACK_A_FOR_DAE:
        np.save(mock_oof_dir / f"{name}_oof_proba.npy",
                rng.uniform(0, 1, n_train).astype(np.float32))

    X_aug, _, benign_mask = dt.build_training_input(
        X_train, y_train, [f"f{i}" for i in range(n_feat)],
    )
    np.testing.assert_array_equal(X_aug[:, :n_feat], X_train[benign_mask])


def test_train_dae_module_default_random_state():
    """Module-level default seed should be the canonical 42."""
    assert dt.RANDOM_STATE == 42
