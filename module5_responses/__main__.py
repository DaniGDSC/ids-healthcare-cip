"""Module 5 CLI dispatcher.

Usage:
    python -m module5_responses                        # real-split batch run
    python -m module5_responses --split test|demo|both
    python -m module5_responses worked-examples        # pipeline_cli (Tasks 5.1-5.8)
    python -m module5_responses verify-audit-log [...] # AuditLogger.verify
    python -m module5_responses rotate-audit-log [...] # AuditLogger.rotate_and_purge
"""
from __future__ import annotations

import sys


def _strip_subcommand(name: str) -> None:
    """Remove the leading subcommand token so downstream argparse sees clean argv."""
    sys.argv = [sys.argv[0]] + sys.argv[2:]


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worked-examples":
        _strip_subcommand("worked-examples")
        from .pipeline_cli import main as worked_main
        worked_main()
        return

    if len(sys.argv) > 1 and sys.argv[1] in {"verify-audit-log", "rotate-audit-log"}:
        sub = sys.argv[1]
        _strip_subcommand(sub)
        # Translate to pipeline_cli's flag-based interface so the existing
        # CLI logic is reused unchanged.
        sys.argv.insert(1, f"--{sub}")
        from .pipeline_cli import cli_entry
        cli_entry()
        return

    # Default: real-split closed-loop batch.
    from .responses_cli import main as responses_main
    responses_main()


if __name__ == "__main__":
    main()
