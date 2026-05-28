"""Single source of truth for split-aware artifact paths.

Before this module existed, ``_split_paths(split)`` was duplicated in
five different files (module3, module4, module6_evaluation,
dynamic_threshold_sim, drift_detection). Each definition carried its
own dict-shape and its own split-validation logic, which made adding a
new artefact (or a new split) a five-file change with five chances to
drift apart.

This module centralises both the split enumeration and the canonical
file paths. Producers (M3, M4, M6, diagnostic scripts) call the per-
artefact helpers; the dashboard consumer (module6_app) calls
:func:`suffix` to render legacy filename templates.

Naming rationale:
  - ``test`` = paper-clean: legacy unsuffixed filenames so the thesis's
    headline artefacts remain compatible with downstream tooling.
  - ``demo`` = operator-clean: suffix ``_demo`` everywhere so operator
    interactions can't accidentally feed back into the paper-clean
    metrics.

Two artefacts are intentionally NOT split-aware here:
  - ``risk_scores.npz`` (test) / ``demo_scores.npz`` (demo) — pre-dates
    this consolidation; legacy naming preserved via :func:`risk_scores`.
  - DAE artefacts (``dae_detector.json`` + ``dae_model.weights.h5``) —
    single trained model shared across splits, not a per-split output.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Split(str, Enum):
    """Frozen splits the project supports."""
    TEST = "test"
    DEMO = "demo"


def _coerce(split: str | Split) -> Split:
    """Validate and coerce a split argument to :class:`Split`.

    Raises ``ValueError`` (via the enum constructor) for any value that
    isn't one of the enumerated splits — catches typos like ``"tset"``
    or ``"DEMO"`` that a permissive ``dict.get(split, fallback)`` would
    have silently coerced to test.
    """
    return Split(split)


def suffix(split: str | Split) -> str:
    """Filename suffix for a split (``""`` for test, ``"_demo"`` for demo).

    Use when constructing chart filenames or one-off artefact paths the
    helpers below don't already cover.
    """
    return "" if _coerce(split) is Split.TEST else "_demo"


# ── Inputs ─────────────────────────────────────────────────────────────

def parquet(split: str | Split) -> Path:
    """Phase-1 frozen parquet for the split (the ground-truth dataset)."""
    return PROJECT_ROOT / f"data/processed/{_coerce(split).value}_phase1.parquet"


def model_predictions(model: str, split: str | Split) -> Path:
    """Track A model predictions npz (xgboost / random_forest / decision_tree)."""
    return PROJECT_ROOT / f"results/models/{model}_{_coerce(split).value}_predictions.npz"


def dae_predictions(split: str | Split) -> Path:
    """DAE per-row reconstruction-error npz emitted by the detection engine."""
    return PROJECT_ROOT / f"results/models/dae_{_coerce(split).value}_predictions.npz"


def risk_scores(split: str | Split) -> Path:
    """Module 3 risk-score npz.

    Test split uses the legacy filename ``risk_scores.npz`` (unsuffixed,
    paper-clean); demo uses ``demo_scores.npz``. The asymmetry is a
    consequence of risk_scores predating the suffix convention.
    """
    s = _coerce(split)
    if s is Split.TEST:
        return PROJECT_ROOT / "results/reports/risk_scores.npz"
    return PROJECT_ROOT / "results/reports/demo_scores.npz"


# ── Module 4 outputs ───────────────────────────────────────────────────

def analyst_report(split: str | Split) -> Path:
    return PROJECT_ROOT / f"results/reports/analyst_report{suffix(split)}.json"


def clinician_summaries(split: str | Split) -> Path:
    return PROJECT_ROOT / f"results/reports/clinician_summaries{suffix(split)}.json"


def example_explanations(split: str | Split) -> Path:
    """Example explanations used by the thesis study mode.

    Demo-side may not exist (thin mode skips this); callers should
    handle a missing path with a graceful fallback.
    """
    return PROJECT_ROOT / f"results/reports/example_explanations{suffix(split)}.json"


# ── Module 5 / 6 outputs ───────────────────────────────────────────────

def alert_responses(split: str | Split) -> Path:
    return PROJECT_ROOT / f"results/reports/alert_responses{suffix(split)}.json"


def audit_trail(split: str | Split) -> Path:
    return PROJECT_ROOT / f"results/reports/audit_trail{suffix(split)}.json"


def evaluation_alerts(split: str | Split) -> Path:
    """20-alert curated set used by the dashboard for device-context enrichment."""
    return PROJECT_ROOT / f"results/reports/evaluation_alerts{suffix(split)}.json"


# ── Diagnostic outputs (B-phase studies) ───────────────────────────────

def dynamic_threshold_results(split: str | Split) -> Path:
    return PROJECT_ROOT / f"results/reports/dynamic_threshold_results{suffix(split)}.json"


def drift_detection_results(split: str | Split) -> Path:
    return PROJECT_ROOT / f"results/reports/drift_detection_results{suffix(split)}.json"
