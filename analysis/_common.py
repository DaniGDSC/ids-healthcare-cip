"""Shared helpers for thesis RQ computation: provenance, logging, hashing."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
LOG_PATH = RESULTS_DIR / "computation_log.txt"
RANDOM_SEED = 42
SCHEMA_VERSION = "1.0"
CODE_VERSION = "thesis-v1.0"


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """Return current git HEAD commit (or 'unknown')."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_provenance(
    input_files: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct standard provenance metadata block."""
    prov: dict[str, Any] = {
        "generated_at": now_iso(),
        "git_commit": git_commit(),
        "random_seed": RANDOM_SEED,
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "python_version": sys.version.split()[0],
    }
    if input_files:
        prov["input_files"] = input_files
    if extra:
        prov.update(extra)
    return prov


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def log(section: str, message: str) -> None:
    """Append a timestamped entry to computation_log.txt."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {section}: {message}\n"
    with open(LOG_PATH, "a") as f:
        f.write(line)


def section_begin(section: str, message: str = "") -> float:
    """Log section start and return start time (monotonic seconds)."""
    import time
    log(section, f"BEGIN {message}".strip())
    return time.monotonic()


def section_end(section: str, start: float, message: str = "") -> None:
    import time
    dt = time.monotonic() - start
    log(section, f"END ({dt:.1f}s) {message}".strip())


def file_hashes() -> dict[str, str]:
    """Compute SHA256 of canonical input files used across RQs."""
    paths = {
        "test_split": REPO_ROOT / "data" / "processed" / "test_phase1.parquet",
        "demo_split": REPO_ROOT / "data" / "processed" / "demo_phase1.parquet",
        "xgboost_model": REPO_ROOT / "results" / "models" / "xgboost_final_pipeline.pkl",
        "dae_detector_json": REPO_ROOT / "results" / "models" / "dae_detector.json",
        "dae_weights": REPO_ROOT / "results" / "models" / "dae_model.weights.h5",
        "composite_weights_yaml": REPO_ROOT / "configs" / "composite_risk_weights.yaml",
        "adaptive_thresholds_yaml": REPO_ROOT / "configs" / "risk_adaptive_thresholds.yaml",
    }
    out: dict[str, str] = {}
    for label, p in paths.items():
        if p.exists():
            out[label] = "sha256:" + sha256_file(p)
        else:
            out[label] = "MISSING"
    return out
