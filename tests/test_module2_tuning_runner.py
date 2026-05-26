"""tuning/_runner.run_track_a_tuning — shared runner end-to-end test.

Uses the smallest viable detector (DecisionTree with n_iter=2, 3 CV folds)
on a synthetic balanced fixture to verify the runner produces all 4
artefacts with the expected schema.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from module2_detection.models.DecisionTree import DecisionTreeDetector
from module2_detection.tuning._runner import run_track_a_tuning


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_synthetic_parquet(path: Path, n: int = 80, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n_half = n // 2
    f1 = np.concatenate([rng.normal(-1.5, 0.5, n_half), rng.normal(1.5, 0.5, n_half)])
    f2 = np.concatenate([rng.normal(-1.5, 0.5, n_half), rng.normal(1.5, 0.5, n_half)])
    df = pd.DataFrame({
        "f1": f1.astype(np.float32),
        "f2": f2.astype(np.float32),
        "Label": [0] * n_half + [1] * n_half,
        "Attack Category": ["normal"] * n_half + ["recon"] * n_half,
        "row_id": np.arange(n, dtype=np.int64),
        "device_class": ["patient_monitor"] * n,
    })
    df.to_parquet(path, index=False)


@pytest.fixture
def runner_workspace(tmp_path):
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    _make_synthetic_parquet(train, n=80, seed=0)
    _make_synthetic_parquet(test, n=40, seed=1)
    out_dir = tmp_path / "out"
    return tmp_path, train, test, out_dir


# ── Happy path: 4 artefacts written ───────────────────────────────────


def test_runner_writes_all_four_artefacts(runner_workspace, monkeypatch):
    """signed pkl + report.json + best_params.json + test_predictions.npz."""
    _ws, train, test, out_dir = runner_workspace

    # Patch PROJECT_ROOT in the runner so relative-path defaults resolve
    # to tmp_path. We do this by passing absolute paths via argv instead.
    from module2_detection.tuning import _runner
    monkeypatch.setattr(_runner, "PROJECT_ROOT", Path("/"))
    out_dir.mkdir()

    run_track_a_tuning(
        detector_class=DecisionTreeDetector,
        output_subdir="dt_test",  # ignored — --output-dir overrides
        report_filename="dt_report.json",
        description="test",
        default_n_iter=2,
        argv=[
            "--train-parquet", str(train),
            "--test-parquet", str(test),
            "--output-dir", str(out_dir.relative_to(Path("/"))),
            "--n-iter", "2",
            "--cv-folds", "3",
            "--random-state", "42",
        ],
    )

    assert (out_dir / "best_pipeline.pkl").exists()
    assert (out_dir / "dt_report.json").exists()
    assert (out_dir / "best_params.json").exists()
    assert (out_dir / "test_predictions.npz").exists()


def test_runner_report_includes_random_state(runner_workspace, monkeypatch):
    _ws, train, test, out_dir = runner_workspace

    from module2_detection.tuning import _runner
    monkeypatch.setattr(_runner, "PROJECT_ROOT", Path("/"))
    out_dir.mkdir()

    run_track_a_tuning(
        detector_class=DecisionTreeDetector,
        output_subdir="dt_test",
        report_filename="dt_report.json",
        description="test",
        default_n_iter=2,
        argv=[
            "--train-parquet", str(train),
            "--test-parquet", str(test),
            "--output-dir", str(out_dir.relative_to(Path("/"))),
            "--n-iter", "2", "--cv-folds", "3", "--random-state", "7",
        ],
    )

    report = json.loads((out_dir / "dt_report.json").read_text())
    assert report["data"]["random_state"] == 7
    assert report["data"]["n_features"] == 2
    assert report["data"]["train_samples"] == 80
    assert report["data"]["test_samples"] == 40


def test_runner_signed_classifier_loadable(runner_workspace, monkeypatch):
    """The pickled classifier must be loadable via the signed-pickle path."""
    _ws, train, test, out_dir = runner_workspace

    from module2_detection.tuning import _runner
    monkeypatch.setattr(_runner, "PROJECT_ROOT", Path("/"))
    out_dir.mkdir()

    run_track_a_tuning(
        detector_class=DecisionTreeDetector,
        output_subdir="dt_test",
        report_filename="dt_report.json",
        description="test",
        default_n_iter=2,
        argv=[
            "--train-parquet", str(train),
            "--test-parquet", str(test),
            "--output-dir", str(out_dir.relative_to(Path("/"))),
            "--n-iter", "2", "--cv-folds", "3",
        ],
    )

    from common import loads_signed
    pipeline_path = out_dir / "best_pipeline.pkl"
    classifier = loads_signed(pipeline_path)
    # SMOTE wrapper was stripped — should be the bare classifier
    assert hasattr(classifier, "predict_proba")
    assert "smote" not in classifier.__class__.__name__.lower()


def test_runner_predictions_npz_shape(runner_workspace, monkeypatch):
    _ws, train, test, out_dir = runner_workspace

    from module2_detection.tuning import _runner
    monkeypatch.setattr(_runner, "PROJECT_ROOT", Path("/"))
    out_dir.mkdir()

    run_track_a_tuning(
        detector_class=DecisionTreeDetector,
        output_subdir="dt_test",
        report_filename="dt_report.json",
        description="test",
        default_n_iter=2,
        argv=[
            "--train-parquet", str(train),
            "--test-parquet", str(test),
            "--output-dir", str(out_dir.relative_to(Path("/"))),
            "--n-iter", "2", "--cv-folds", "3",
        ],
    )

    pred = np.load(out_dir / "test_predictions.npz")
    assert "y_true" in pred and "y_pred" in pred and "y_proba" in pred
    assert pred["y_true"].shape == (40,)
    assert pred["y_pred"].shape == (40,)
    assert pred["y_proba"].shape == (40,)


def test_runner_best_params_no_classifier_prefix(runner_workspace, monkeypatch):
    """best_params.json carries the raw search-space keys (with prefix)."""
    _ws, train, test, out_dir = runner_workspace

    from module2_detection.tuning import _runner
    monkeypatch.setattr(_runner, "PROJECT_ROOT", Path("/"))
    out_dir.mkdir()

    run_track_a_tuning(
        detector_class=DecisionTreeDetector,
        output_subdir="dt_test",
        report_filename="dt_report.json",
        description="test",
        default_n_iter=2,
        argv=[
            "--train-parquet", str(train),
            "--test-parquet", str(test),
            "--output-dir", str(out_dir.relative_to(Path("/"))),
            "--n-iter", "2", "--cv-folds", "3",
        ],
    )

    best = json.loads((out_dir / "best_params.json").read_text())
    # Best params come from RandomizedSearchCV which uses the prefixed names.
    assert all(k.startswith("classifier__") for k in best)
