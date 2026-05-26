"""One-time CLI to bootstrap the signed integrity baseline.

Usage:
    python -m module0_analysis.bootstrap_integrity \\
        [--config module0_analysis/config.yaml]

This MUST be run once per dataset, by an operator, on a known-good
file. ``DataLoader.load()`` refuses to run without an existing baseline,
and the baseline file is signed with the Module 5 ECDSA P-256 key, so
deleting or rewriting it does not whitewash a tampered dataset.

If a baseline already exists for the configured dataset, this command
refuses to overwrite it. To intentionally re-baseline (e.g. after a
sanctioned dataset update), the operator must delete the entry from
``dataset_integrity.json`` first — that delete is itself logged via
the host's filesystem audit and produces a visible signal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Phase0Config
from .security import IntegrityError, IntegrityVerifier, PathValidator

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "module0_analysis/config.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m module0_analysis.bootstrap_integrity",
        description=(
            "Establish the signed SHA-256 baseline for the Phase 0 dataset. "
            "Run once per known-good dataset."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to phase0 config.yaml (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = Phase0Config.from_yaml(args.config)
    workspace = Path(__file__).resolve().parents[1]
    validator = PathValidator(workspace)
    dataset_path = validator.validate_input_path(cfg.data_path)

    verifier = IntegrityVerifier(
        metadata_dir=workspace / "module0_analysis",
    )
    try:
        digest = verifier.bootstrap(dataset_path)
    except IntegrityError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(f"OK: baselined {dataset_path.name} → sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
