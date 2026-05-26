"""Module 6 CLI dispatcher.

Usage::

    python -m module6_evaluation                  # offline batch (curate + stats + figures)
    python -m module6_evaluation evaluate         # alias for default
    python -m module6_evaluation curate-only      # 6.2 only, no stats
    python -m module6_evaluation --split test     # explicit split
    streamlit run module6_evaluation/module6_app.py  # interactive dashboard
"""
from __future__ import annotations

import sys


def main() -> None:
    # Strip a leading "evaluate" subcommand so argparse only sees flags.
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    elif len(sys.argv) > 1 and sys.argv[1] == "curate-only":
        sys.argv = [sys.argv[0], "--curate-only"] + sys.argv[2:]

    from .pipeline import main as _run
    _run()


if __name__ == "__main__":
    main()
