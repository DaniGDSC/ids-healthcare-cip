"""Prefect orchestration for IDS Healthcare CIP pipeline.

Features
 - Per-phase retries and timeouts
 - Resume from checkpoints (skip completed phases)
 - Centralized configuration via config/orchestration.yaml

Usage
  python scripts/prefect_pipeline.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Any
import yaml

from prefect import flow, task, get_run_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "orchestration.yaml"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("orchestration", {})


@task
def run_phase(command: str, checkpoint: Path, timeout_seconds: int, resume: bool) -> None:
    logger = get_run_logger()

    if resume and checkpoint.exists():
        logger.info(f"Checkpoint found, skipping: {checkpoint}")
        return

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Running phase command: {command} (timeout={timeout_seconds}s)")

    try:
        subprocess.run(command, shell=True, check=True, timeout=timeout_seconds, cwd=PROJECT_ROOT)
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout after {timeout_seconds}s for command: {command}")
        raise
    except subprocess.CalledProcessError as exc:
        logger.error(f"Command failed (exit {exc.returncode}): {command}")
        raise

    checkpoint.touch()
    logger.info(f"Checkpoint written: {checkpoint}")


@flow(name="ids-cip-pipeline")
def orchestrate() -> None:
    cfg = load_config()
    resume = cfg.get("resume", True)
    checkpoints_dir = PROJECT_ROOT / cfg.get("checkpoints_dir", "results/checkpoints")
    phases = cfg.get("phases", [])

    for phase in phases:
        name = phase["name"]
        command = phase["command"]
        timeout_seconds = int(phase.get("timeout_seconds", 3600))
        retries = int(phase.get("retries", 0))
        retry_delay_seconds = int(phase.get("retry_delay_seconds", 60))

        checkpoint = checkpoints_dir / f"{name}.done"

        # Prefect task with dynamic retry settings
        run_phase.with_options(
            name=f"run-{name}",
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
        )(command=command, checkpoint=checkpoint, timeout_seconds=timeout_seconds, resume=resume)


if __name__ == "__main__":
    orchestrate()
