"""Layer 1 v4.0 (R3 + R4): build multi-threshold + percentile-rank
calibration artifacts from the already-trained DAE detector.

The DAE training pipeline (``module2_detection/run_dae.py``) emits
``results/models/dae_detector.json`` whose ``train_errors`` field
contains the per-sample reconstruction errors on the benign training
set. Both v4.0 enhancements derive purely from those errors — there is
no retraining and no model state is mutated.

Outputs
-------
``results/models/dae_thresholds.json``
    Three operational thresholds for cascade tuning:
      * ``screening_threshold``         = p80 of benign training errors
      * ``confirmation_threshold``      = p95 (matches the legacy
                                              single threshold the
                                              detector was fitted at)
      * ``high_confidence_threshold``   = p99
    Plus min, mean, max for sanity checks.

``results/models/dae_calibration.json``
    Percentile-rank lookup so callers can map a raw reconstruction
    error to a [0, 1] anomaly score that is comparable across
    environments. The lookup is a sorted array of training errors at
    every percent (0..100); a caller does:

        score = searchsorted(sorted_errors, error) / len(sorted_errors)

    and gets a calibrated rank in [0, 1].

Both files include a SHA256 digest of the source ``dae_detector.json``
so downstream consumers can verify that calibration matches the model
they are using (Invariant 4 — audit trail).

Run
---
    python -m module2_detection.build_dae_v4_artifacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_train_errors(detector_json: Path) -> tuple[np.ndarray, Dict[str, Any]]:
    body = json.loads(detector_json.read_text())
    errors = body.get("train_errors")
    if errors is None:
        raise RuntimeError(
            f"{detector_json} has no 'train_errors' field — "
            "retrain the DAE before building v4 artifacts."
        )
    return np.asarray(errors, dtype=np.float64), body


def build_thresholds(train_errors: np.ndarray) -> Dict[str, float]:
    """Return p80 / p95 / p99 thresholds + sanity stats."""
    return {
        "screening_threshold": float(np.percentile(train_errors, 80)),
        "confirmation_threshold": float(np.percentile(train_errors, 95)),
        "high_confidence_threshold": float(np.percentile(train_errors, 99)),
        "training_min_error": float(np.min(train_errors)),
        "training_mean_error": float(np.mean(train_errors)),
        "training_max_error": float(np.max(train_errors)),
        "training_size": int(train_errors.size),
    }


def build_calibration(
    train_errors: np.ndarray,
    n_lookup: int = 1001,
) -> Dict[str, Any]:
    """Sorted training errors + percentile lookup for rank-based scoring.

    ``percentile_lookup`` is dense (1001 points by default → 0.1%
    resolution). Inference does:

        rank_in_0_1 = np.searchsorted(percentile_lookup, raw_error) / len(percentile_lookup)
    """
    sorted_errors = np.sort(train_errors)
    pct_pts = np.linspace(0.0, 100.0, n_lookup)
    percentile_lookup = np.percentile(sorted_errors, pct_pts).tolist()

    coarse_marks = {f"p{int(p)}": float(np.percentile(sorted_errors, p))
                    for p in (0, 5, 10, 25, 50, 75, 90, 95, 99, 100)}

    return {
        "method": "percentile_rank",
        "rationale": (
            "Maps raw reconstruction error → [0, 1] percentile rank "
            "relative to the benign training error distribution. Score "
            "interpretation is consistent across environments because "
            "it does not depend on raw error magnitude."
        ),
        "n_lookup_points": int(n_lookup),
        "percentile_lookup": percentile_lookup,
        "coarse_marks": coarse_marks,
        "training_size": int(sorted_errors.size),
    }


def _atomic_write_json(path: Path, body: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Layer 1 v4.0 DAE artifacts (R3 + R4)",
    )
    parser.add_argument(
        "--detector-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "models" / "dae_detector.json",
        help="Source DAE sidecar (default: results/models/dae_detector.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "models",
        help="Output directory (default: results/models).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.detector_json.exists():
        logger.error("DAE detector sidecar not found: %s", args.detector_json)
        return 2

    train_errors, body = _load_train_errors(args.detector_json)
    src_sha = _sha256_file(args.detector_json)
    now_iso = datetime.now(timezone.utc).isoformat()

    thresholds = build_thresholds(train_errors)
    thresholds_doc = {
        "format": "layer1_v4.dae_thresholds",
        "format_version": 1,
        "source_detector_path": str(args.detector_json.relative_to(PROJECT_ROOT)),
        "source_detector_sha256": src_sha,
        "generated_at_utc": now_iso,
        "fitted_threshold_percentile": body.get("hyperparameters", {}).get(
            "threshold_percentile",
            None,
        ),
        "thresholds": thresholds,
    }
    out_thr = args.out_dir / "dae_thresholds.json"
    _atomic_write_json(out_thr, thresholds_doc)

    calibration = build_calibration(train_errors)
    calibration_doc = {
        "format": "layer1_v4.dae_calibration",
        "format_version": 1,
        "source_detector_path": str(args.detector_json.relative_to(PROJECT_ROOT)),
        "source_detector_sha256": src_sha,
        "generated_at_utc": now_iso,
        **calibration,
    }
    out_cal = args.out_dir / "dae_calibration.json"
    _atomic_write_json(out_cal, calibration_doc)

    logger.info(
        "wrote %s (p80=%.6g, p95=%.6g, p99=%.6g)",
        out_thr.name,
        thresholds["screening_threshold"],
        thresholds["confirmation_threshold"],
        thresholds["high_confidence_threshold"],
    )
    logger.info(
        "wrote %s (n=%d lookup points, training_size=%d)",
        out_cal.name,
        calibration["n_lookup_points"],
        calibration["training_size"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
