"""I/O for Module 4 — data loading, artefact saving, config exports.

Single-responsibility module: filesystem reads and writes only. All
compute lives in ``compute.py`` / ``stakeholder.py`` / ``nlg.py`` /
``validation.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FEATURE_CONCEPTS, NLG_TEMPLATES

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results/reports"
CHARTS_DIR = PROJECT_ROOT / "results/charts"


# ── Path resolution per split ────────────────────────────────────────


def _split_paths(split: str) -> dict:
    """Resolve per-split inputs + output suffix.

    Test = paper-clean (legacy unsuffixed filenames);
    demo = operator-clean (suffix '_demo').
    """
    from common import split_paths as sp

    return {
        "parquet": sp.parquet(split),
        "xgboost_preds": sp.model_predictions("xgboost", split),
        "dae_preds": sp.dae_predictions(split),
        "suffix": sp.suffix(split),
    }


# ── Loaders ─────────────────────────────────────────────────────────


def load_test_data(parquet_path: Path | None = None) -> tuple:
    """Load a split parquet and return ``(X, y, attack_cats, feat_names)``."""
    path = parquet_path or (PROJECT_ROOT / "data/processed/test_phase1.parquet")
    df = pd.read_parquet(path)
    drop_cols = ["Label", "Attack Category", "row_id", "device_class"]
    feat_names = [c for c in df.columns if c not in drop_cols]
    X_test = df[feat_names].values.astype(np.float32)
    y_test = df["Label"].values
    attack_cats = (
        df["Attack Category"].values if "Attack Category" in df.columns else None
    )
    return X_test, y_test, attack_cats, feat_names


def load_predictions(npz_path: Path) -> dict:
    """Load pre-computed predictions from npz."""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


# ── Atomic JSON write helpers ───────────────────────────────────────


def write_json_sync(path: Path, data) -> None:
    """Atomic JSON write (sync; used by async wrapper and direct callers)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


async def _write_json_async(path: Path, data) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, write_json_sync, path, data)


def write_json_batch(outputs: dict[Path, Any]) -> None:
    """Write multiple JSON files concurrently."""

    async def _run():
        await asyncio.gather(
            *[_write_json_async(path, data) for path, data in outputs.items()]
        )

    asyncio.run(_run())


# ── Numpy-aware JSON encoding (Y6 fix) ──────────────────────────────


class NumpyJSONEncoder(json.JSONEncoder):
    """Encode numpy scalars/arrays as native JSON types.

    Replaces the recursive ``_clean`` helper that used to live in
    ``module4_online_explainer``. Caller passes ``cls=NumpyJSONEncoder``
    to ``json.dumps`` / ``json.dump`` once instead of pre-walking the
    payload by hand.
    """

    def default(self, obj):  # noqa: D401 — interface
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Atomic JSON write with optional numpy support ───────────────────


def write_json_strict(path: Path, data, *, allow_numpy: bool = False) -> None:
    """Atomic JSON write with strict-fail on non-serialisable values.

    Args:
        allow_numpy: when True, numpy scalars/arrays are converted via
            ``NumpyJSONEncoder``. Default False — matches the Module 0/1/2
            strict-JSON discipline (producer bugs surface as TypeError).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if allow_numpy:
            payload = json.dumps(data, indent=2, cls=NumpyJSONEncoder)
        else:
            payload = json.dumps(data, indent=2)
    except TypeError as exc:
        raise TypeError(
            f"{path.name} contains a non-JSON-serialisable value "
            f"(detail: {exc}). Fix the producer."
        ) from exc
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


# ── SHAP / DAE artefact savers ──────────────────────────────────────


def save_shap_values(
    model_name: str,
    sv: np.ndarray,
    expected: float,
    feat_names: list,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Save SHAP values to npz."""
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shap_values_{model_name}.npz"
    np.savez(
        path,
        shap_values=sv,
        expected_value=np.array(expected),
        feature_names=np.array(feat_names),
    )
    logger.info("  Saved: %s", path)
    return path


def save_global_importance(
    model_name: str,
    importance: list,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Save global importance to JSON."""
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"global_importance_{model_name}.json"
    write_json_strict(path, {"model": model_name, "features": importance})
    logger.info("  Saved: %s", path)
    return path


def save_dae_errors(
    sq_err: np.ndarray,
    weighted_err: np.ndarray,
    feat_weights: np.ndarray,
    feat_names: list,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Save DAE per-feature errors to npz."""
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dae_feature_errors.npz"
    np.savez(
        path,
        per_feature_error=sq_err,
        weighted_per_feature_error=weighted_err,
        feature_weights=feat_weights,
        feature_names=np.array(feat_names),
    )
    logger.info("  Saved: %s", path)
    return path


# ── Config JSON exports (Tasks 4.4, 4.6) ────────────────────────────


def export_feature_concepts(*, output_dir: Path | None = None) -> Path:
    """Export feature-to-concept mapping as standalone JSON."""
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_concepts.json"
    write_json_strict(path, FEATURE_CONCEPTS)
    logger.info("  Saved: feature_concepts.json")
    return path


def export_nlg_templates(*, output_dir: Path | None = None) -> Path:
    """Export NLG template library as JSON."""
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "nlg_templates.json"
    write_json_strict(path, NLG_TEMPLATES)
    logger.info("  Saved: nlg_templates.json")
    return path


__all__ = [
    "PROJECT_ROOT",
    "OUTPUT_DIR",
    "CHARTS_DIR",
    "_split_paths",
    "load_test_data",
    "load_predictions",
    "write_json_sync",
    "write_json_batch",
    "write_json_strict",
    "NumpyJSONEncoder",
    "save_shap_values",
    "save_global_importance",
    "save_dae_errors",
    "export_feature_concepts",
    "export_nlg_templates",
]
