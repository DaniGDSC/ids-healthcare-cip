#!/usr/bin/env python3
"""Automated multi-phase pipeline runner for IDS Healthcare CIP.

Reads ``config/orchestration.yaml``, executes each phase in order as a
subprocess, handles retries, enforces timeouts, and writes a structured
run report to ``results/pipeline_run_<timestamp>.json``.

Usage
-----
    # Run all phases
    python run_pipeline.py

    # Run only specific phases (by name)
    python run_pipeline.py --phases phase0 phase1

    # Dry-run: print what would be executed without running anything
    python run_pipeline.py --dry-run

    # Start from a specific phase (skip earlier phases)
    python run_pipeline.py --from-phase phase1

    # Stop after a specific phase
    python run_pipeline.py --to-phase phase1
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
ORCHESTRATION_YAML = PROJECT_ROOT / "config" / "orchestration.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"

logger = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PhaseResult:
    """Holds the outcome of a single phase execution."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.status: str = "pending"     # pending | success | failed | skipped | timeout
        self.attempts: int = 0
        self.elapsed: float = 0.0
        self.return_code: Optional[int] = None
        self.error: Optional[str] = None
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "attempts": self.attempts,
            "elapsed_seconds": round(self.elapsed, 3),
            "return_code": self.return_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
        force=True,
    )


def _load_orchestration() -> Dict[str, Any]:
    if not ORCHESTRATION_YAML.exists():
        raise FileNotFoundError(f"Orchestration config not found: {ORCHESTRATION_YAML}")
    with open(ORCHESTRATION_YAML, "r") as f:
        raw = yaml.safe_load(f)
    return raw.get("orchestration", raw)


def _save_run_report(results: List[PhaseResult], total_elapsed: float) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = RESULTS_DIR / f"pipeline_run_{ts}.json"

    statuses = [r.status for r in results]
    overall = (
        "success" if all(s == "success" for s in statuses)
        else "partial" if any(s == "success" for s in statuses)
        else "failed"
    )

    report = {
        "pipeline_run": ts,
        "overall_status": overall,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "phases": [r.to_dict() for r in results],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report_path


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------

def _run_phase(
    phase_cfg: Dict[str, Any],
    dry_run: bool = False,
) -> PhaseResult:
    """Execute a single phase with retries and timeout."""
    name = phase_cfg["name"]
    command = phase_cfg["command"]
    timeout = phase_cfg.get("timeout_seconds", 3600)
    retries = phase_cfg.get("retries", 1)
    retry_delay = phase_cfg.get("retry_delay_seconds", 60)

    result = PhaseResult(name)

    if dry_run:
        logger.info("[DRY-RUN] Would run: %s (timeout=%ds, retries=%d)", command, timeout, retries)
        result.status = "skipped"
        return result

    logger.info("=" * 72)
    logger.info("STARTING  %s", name.upper())
    logger.info("Command : %s", command)
    logger.info("Timeout : %d s  |  Retries: %d", timeout, retries)
    logger.info("=" * 72)

    result.started_at = _now_iso()
    phase_start = time.perf_counter()

    for attempt in range(1, retries + 1):
        result.attempts = attempt
        attempt_start = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(PROJECT_ROOT),
                timeout=timeout,
                check=False,   # We handle return codes ourselves
            )
            result.return_code = proc.returncode
            attempt_elapsed = time.perf_counter() - attempt_start

            if proc.returncode == 0:
                result.status = "success"
                logger.info(
                    "✓ %s completed in %.1f s (attempt %d/%d)",
                    name, attempt_elapsed, attempt, retries,
                )
                break
            else:
                logger.warning(
                    "✗ %s exited with code %d (attempt %d/%d, %.1f s)",
                    name, proc.returncode, attempt, retries, attempt_elapsed,
                )
                result.error = f"Non-zero exit code: {proc.returncode}"
                if attempt < retries:
                    logger.info("  Retrying in %d s …", retry_delay)
                    time.sleep(retry_delay)

        except subprocess.TimeoutExpired:
            attempt_elapsed = time.perf_counter() - attempt_start
            logger.error(
                "✗ %s timed out after %d s (attempt %d/%d)",
                name, timeout, attempt, retries,
            )
            result.status = "timeout"
            result.error = f"Timed out after {timeout}s"
            if attempt < retries:
                logger.info("  Retrying in %d s …", retry_delay)
                time.sleep(retry_delay)

    else:
        # All attempts exhausted without success
        if result.status != "timeout":
            result.status = "failed"

    result.elapsed = time.perf_counter() - phase_start
    result.finished_at = _now_iso()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated multi-phase IDS Healthcare CIP pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phases", nargs="+", metavar="PHASE",
        help="Only run these named phases (e.g. --phases phase0 phase1).",
    )
    parser.add_argument(
        "--from-phase", metavar="PHASE",
        help="Skip all phases before this one.",
    )
    parser.add_argument(
        "--to-phase", metavar="PHASE",
        help="Stop after this phase (inclusive).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing anything.",
    )
    parser.add_argument(
        "--log-file", default="logs/pipeline.log",
        help="Log file path (default: logs/pipeline.log).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(PROJECT_ROOT / args.log_file)

    logger.info("IDS Healthcare CIP — Automated Pipeline")
    logger.info("Project root : %s", PROJECT_ROOT)
    logger.info("Config       : %s", ORCHESTRATION_YAML)
    logger.info("Dry run      : %s", args.dry_run)

    orch = _load_orchestration()
    all_phases: List[Dict[str, Any]] = orch.get("phases", [])

    if not all_phases:
        logger.error("No phases defined in orchestration.yaml")
        sys.exit(1)

    # --- Phase filtering ---
    phases_to_run = all_phases

    if args.phases:
        requested = set(args.phases)
        phases_to_run = [p for p in all_phases if p["name"] in requested]
        unknown = requested - {p["name"] for p in all_phases}
        if unknown:
            logger.error("Unknown phase(s): %s", unknown)
            sys.exit(1)

    if args.from_phase:
        names = [p["name"] for p in phases_to_run]
        if args.from_phase not in names:
            logger.error("--from-phase '%s' not found in phase list", args.from_phase)
            sys.exit(1)
        idx = names.index(args.from_phase)
        phases_to_run = phases_to_run[idx:]

    if args.to_phase:
        names = [p["name"] for p in phases_to_run]
        if args.to_phase not in names:
            logger.error("--to-phase '%s' not found in phase list", args.to_phase)
            sys.exit(1)
        idx = names.index(args.to_phase)
        phases_to_run = phases_to_run[: idx + 1]

    logger.info("Phases to run: %s", [p["name"] for p in phases_to_run])

    # --- Execution ---
    results: List[PhaseResult] = []
    pipeline_start = time.perf_counter()

    for phase_cfg in phases_to_run:
        result = _run_phase(phase_cfg, dry_run=args.dry_run)
        results.append(result)

        # Stop-on-failure (non-zero and not a dry-run)
        if result.status in ("failed", "timeout") and not args.dry_run:
            logger.error(
                "Pipeline halted: %s failed after %d attempt(s). "
                "Fix the error and re-run with --from-phase %s to resume.",
                result.name, result.attempts, result.name,
            )
            break

    total_elapsed = time.perf_counter() - pipeline_start

    # --- Summary ---
    logger.info("")
    logger.info("=" * 72)
    logger.info("PIPELINE RUN SUMMARY")
    logger.info("=" * 72)
    for r in results:
        icon = {"success": "✓", "failed": "✗", "timeout": "⏱", "skipped": "○", "pending": "?"}.get(r.status, "?")
        logger.info("  %s  %-12s  %-9s  %.1f s", icon, r.name, r.status, r.elapsed)
    logger.info("-" * 72)
    logger.info("  Total elapsed: %.1f s", total_elapsed)

    # --- Save run report ---
    if not args.dry_run:
        report_path = _save_run_report(results, total_elapsed)
        logger.info("  Run report   : %s", report_path)
    logger.info("=" * 72)

    # --- Exit code ---
    if any(r.status in ("failed", "timeout") for r in results):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
