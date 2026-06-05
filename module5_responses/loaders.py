"""Module 5 data loaders — risk scores, explanations, attack categories.

This module intentionally preserves the legacy dict-shaped return values
that older Module 5 callers expect, but routes path resolution and risk
score loading through the canonical shared helpers so schema and split
logic do not drift.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results/reports"
CHARTS_DIR = PROJECT_ROOT / "results/charts"


def _paths(split: str) -> dict:
    """Resolve per-split input + output paths.

    Test = paper-clean (the default; preserves legacy filename
    ``alert_responses.json`` for backward compatibility with the
    dashboard's fallback loader and downstream tooling).
    Demo = operator-clean (suffixed ``_demo`` everywhere).
    """
    from common import split_paths as sp

    suffix = sp.suffix(split)

    return {
        "split": split,
        "scores_npz": sp.risk_scores(split),
        "parquet": sp.parquet(split),
        "analyst_json": PROJECT_ROOT / "results/reports" / f"analyst_report{suffix}.json",
        "clinician_json": PROJECT_ROOT / "results/reports" / f"clinician_summaries{suffix}.json",
        "out_alert_responses": OUTPUT_DIR / f"alert_responses{suffix}.json",
        "out_audit_trail": OUTPUT_DIR / f"audit_trail{suffix}.json",
        "out_effectiveness": OUTPUT_DIR / f"effectiveness_analysis{suffix}.json",
        "out_response_report": OUTPUT_DIR / f"response_report{suffix}.json",
        "out_detail_csv": OUTPUT_DIR / f"alert_responses_detail{suffix}.csv",
        "suffix": suffix,
    }


def load_risk_scores(scores_npz_path: Path | None = None) -> dict:
    """Load Module 3 risk scores through the verified shared loader.

    Returns a plain dict view for backward compatibility with existing
    Module 5 callers that expect ``risk_data["R"]`` style access.
    """
    from common.risk_scores_loader import load_risk_scores as _verified_load

    path = scores_npz_path or (PROJECT_ROOT / "results/reports/risk_scores.npz")
    artefact = _verified_load(path)
    return {
        "R": artefact.R,
        "c_detect": artefact.c_detect,
        "c_track_a": artefact.c_track_a,
        "c_track_b": artefact.c_track_b,
        "d_crit": artefact.d_crit,
        "s_data": artefact.s_data,
        "d_clinical_tier": artefact.d_clinical_tier,
        "y_true": artefact.y_true,
        "risk_level_codes": artefact.risk_level_codes,
        "risk_levels": artefact.risk_levels,
        "schema_version": artefact.schema_version,
        "formula_version": artefact.formula_version,
    }


def load_explanations(
    analyst_json_path: Path | None = None,
    clinician_json_path: Path | None = None,
) -> tuple:
    """Load Module 4 analyst reports and clinician summaries.

    Both files are OPTIONAL — when running against the demo split before
    Module 4 has produced demo-specific explanations, falls back to empty
    dicts. Downstream record builders gracefully handle this case (records
    are marked ``analyst_available: false``).
    """
    a_path = analyst_json_path or (PROJECT_ROOT / "results/reports/analyst_report.json")
    c_path = clinician_json_path or (PROJECT_ROOT / "results/reports/clinician_summaries.json")
    analyst: dict = {}
    clinician: dict = {}
    if a_path.exists():
        with open(a_path) as f:
            analyst = {a["sample_index"]: a for a in json.load(f)}
    else:
        logger.warning("analyst report missing at %s — proceeding with empty dict", a_path)
    if c_path.exists():
        with open(c_path) as f:
            clinician = {s["sample_index"]: s for s in json.load(f)}
    else:
        logger.warning("clinician summaries missing at %s — proceeding with empty dict", c_path)
    return analyst, clinician


def load_attack_categories(parquet_path: Path | None = None) -> np.ndarray:
    """Load Attack Category column from the configured split's parquet."""
    path = parquet_path or (PROJECT_ROOT / "data/processed/test_phase1.parquet")
    df = pd.read_parquet(path, columns=["Attack Category"])
    return df["Attack Category"].values


__all__ = [
    "_paths",
    "load_risk_scores",
    "load_explanations",
    "load_attack_categories",
    "PROJECT_ROOT",
    "OUTPUT_DIR",
    "CHARTS_DIR",
]
