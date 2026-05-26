"""Module 5 closed-loop batch CLI (real-split run over test/demo)."""
from __future__ import annotations

import argparse
import logging
import time

from .loaders import CHARTS_DIR, OUTPUT_DIR
from .pipeline import run_one_split

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m module5_responses.module5_responses",
        description="Module 5 — closed-loop response engine. Operates on the "
                    "selected frozen split (test=paper-clean, demo=operator-clean).",
    )
    parser.add_argument(
        "--split",
        choices=["test", "demo", "both"],
        default="test",
        help="Frozen split to process. 'test' writes paper-clean artifacts "
             "(legacy `alert_responses.json`); 'demo' writes operator-clean "
             "artifacts with `_demo` suffix; 'both' processes test then demo "
             "sequentially.",
    )
    args = parser.parse_args()

    splits_to_run = ["test", "demo"] if args.split == "both" else [args.split]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sep = "=" * 72
    t0 = time.perf_counter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits_to_run:
        run_one_split(split, sep)
    logger.info(
        "Module 5 complete (%.1fs, splits=%s)",
        time.perf_counter() - t0, splits_to_run,
    )


__all__ = ["main"]
