"""Module 5 data loaders — risk scores, explanations, attack categories."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
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
    if split == "test":
        scores_npz = "risk_scores.npz"
        parquet = "test_phase1.parquet"
        suffix = ""
    elif split == "demo":
        scores_npz = "demo_scores.npz"
        parquet = "demo_phase1.parquet"
        suffix = "_demo"
    else:
        raise ValueError(f"unknown split: {split!r} (expected 'test' or 'demo')")

    return {
        "split": split,
        "scores_npz": PROJECT_ROOT / "results/reports" / scores_npz,
        "parquet": PROJECT_ROOT / "data/processed" / parquet,
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
    """Load Module 3 risk scores from the configured split's npz."""
    path = scores_npz_path or (PROJECT_ROOT / "results/reports/risk_scores.npz")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


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
