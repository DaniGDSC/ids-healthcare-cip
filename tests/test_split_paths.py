"""Smoke tests for the split-aware path consolidation.

Covers:
  - Single source of truth: :mod:`common.split_paths` resolves every
    artefact every module references for both ``test`` and ``demo`` splits.
  - Strict validation: unknown/typo split values raise ``ValueError``
    instead of silently falling back to test (the silent-fallback was
    the original bug class that motivated this refactor).
  - End-to-end demo wiring: when all five thin-wrapper ``_split_paths``
    facades agree with :mod:`common.split_paths`, the demo pipeline
    files exist on disk in the shapes Module 5 and the dashboard expect.
  - Renamed engine method exposes both the new and deprecated name.

These tests run against on-disk artefacts produced by the most recent
pipeline run. Each test skips (not fails) when a required input is
missing so a fresh clone with no demo run does not red CI.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import Split
from common import split_paths as sp


# ─────────────────────────────────────────────────────────────────────
# 1. common.split_paths — single source of truth
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("split,expected_suffix", [
    ("test", ""),
    ("demo", "_demo"),
    (Split.TEST, ""),
    (Split.DEMO, "_demo"),
])
def test_suffix_returns_canonical_value(split, expected_suffix):
    """:func:`common.split_paths.suffix` returns ``""`` for test and
    ``"_demo"`` for demo, accepting both raw strings and Split enum."""
    assert sp.suffix(split) == expected_suffix


@pytest.mark.parametrize("bad", ["tset", "DEMO", "Demo", "production", "", " ", "test ", "demo\n"])
def test_split_paths_rejects_unknown_split(bad):
    """Strict validation: any value that isn't a Split enum member must
    raise ValueError, NOT silently fall back to test. This catches the
    typo class that previous ``dict.get(split, '')`` masked."""
    with pytest.raises(ValueError):
        sp.suffix(bad)
    with pytest.raises(ValueError):
        sp.parquet(bad)
    with pytest.raises(ValueError):
        sp.dae_predictions(bad)


# ─────────────────────────────────────────────────────────────────────
# 2. Thin-wrapper facades agree with common.split_paths
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("split", ["test", "demo"])
def test_module_facades_delegate_to_common(split):
    """The five module-local ``_split_paths`` thin wrappers must produce
    paths identical to what :mod:`common.split_paths` returns. If they
    drift, splitting a new artefact will silently misalign."""
    from module3_risk_scoring.module3_risk_scores import _split_paths as m3
    from module4_explanations.module4_explanations import _split_paths as m4
    from module6_evaluation.module6_evaluation import _curate_split_paths as m6
    from tools.diagnostics.dynamic_threshold_sim import _split_paths as dts
    from tools.diagnostics.drift_detection import _split_paths as drf

    assert m3(split)["parquet"] == sp.parquet(split)
    assert m3(split)["out_npz"] == sp.risk_scores(split)

    assert m4(split)["parquet"] == sp.parquet(split)
    assert m4(split)["dae_preds"] == sp.dae_predictions(split)
    assert m4(split)["xgboost_preds"] == sp.model_predictions("xgboost", split)
    assert m4(split)["suffix"] == sp.suffix(split)

    assert m6(split)["risk_npz"] == sp.risk_scores(split)
    assert m6(split)["analyst"] == sp.analyst_report(split)
    assert m6(split)["clinician"] == sp.clinician_summaries(split)

    assert dts(split)["dae_preds"] == sp.dae_predictions(split)
    assert drf(split)["dae_preds"] == sp.dae_predictions(split)


# ─────────────────────────────────────────────────────────────────────
# 3. Demo split end-to-end disk wiring
# ─────────────────────────────────────────────────────────────────────


_DEMO_INPUTS = [
    "parquet",
    "model_predictions:xgboost",
    "model_predictions:random_forest",
    "model_predictions:decision_tree",
    "dae_predictions",
    "risk_scores",
    "analyst_report",
    "clinician_summaries",
    "alert_responses",
    "audit_trail",
    "evaluation_alerts",
]


@pytest.mark.parametrize("artifact_key", _DEMO_INPUTS)
def test_demo_artifact_exists_on_disk(artifact_key):
    """Every artefact the demo dashboard reads must exist after a full
    demo pipeline run. Missing files indicate the producer chain (M2
    predict-only → M3 → M4 thin → M5 → M6 curate-only) wasn't run end-
    to-end and the dashboard would degrade silently."""
    if artifact_key.startswith("model_predictions:"):
        model = artifact_key.split(":", 1)[1]
        path = sp.model_predictions(model, Split.DEMO)
    else:
        path = getattr(sp, artifact_key)(Split.DEMO)
    if not path.exists():
        pytest.skip(
            f"{path.name} not on disk — run the demo pipeline first "
            f"(see ONBOARDING / Day 1 acceptance commands)."
        )
    # File exists; assert it's non-empty (a 0-byte json is also broken)
    assert path.stat().st_size > 0, f"{path.name} exists but is empty"


# ─────────────────────────────────────────────────────────────────────
# 4. detection_engine API: renamed method + deprecated alias
# ─────────────────────────────────────────────────────────────────────


def test_write_predictions_renamed_with_deprecated_alias():
    """``write_test_predictions`` was misnamed after gaining split support.
    The new name is ``write_predictions``; the old name remains as a
    deprecation alias so existing call sites (and the cascade test) keep
    working until the next major bump."""
    from detection_engine import DetectionEngine

    assert hasattr(DetectionEngine, "write_predictions"), \
        "DetectionEngine.write_predictions missing — the rename regressed"
    assert hasattr(DetectionEngine, "write_test_predictions"), \
        "DetectionEngine.write_test_predictions alias missing — backward compat broken"

    # The alias should emit DeprecationWarning when called. We construct
    # the engine lazily and intercept the warning without actually
    # running predictions (which would need full model artefacts loaded).
    engine = DetectionEngine.__new__(DetectionEngine)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            # This will raise something else (no models loaded), but the
            # deprecation warning fires before that. We only assert on
            # the warning, not on the call's success.
            engine.write_test_predictions()
        except Exception:
            pass
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations, (
            "Calling write_test_predictions should emit DeprecationWarning"
        )
        msg = str(deprecations[0].message)
        assert "write_predictions" in msg, (
            "Deprecation message should point users to write_predictions"
        )
