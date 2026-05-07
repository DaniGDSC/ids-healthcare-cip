"""Scope-A test-split-role assignment.

The 4-way data-split work proposed in v5.1 was scope-narrowed to a
**label-only intervention** so the change can land safely a week before
defense:

* Train and val partitions are unchanged. No model retraining.
* The existing 4 896-row test parquet is unchanged. No schema edit.
* Each row in test_phase1 gets a frozen role assignment recorded in a
  sidecar JSON: ``demo_pool`` (the rows the dashboard / Phase-2 study
  see) or ``test_paper`` (the rows ``run_tests.py`` M-metrics evaluate).

The demo set is exactly the unique ``sample_index`` values currently in
``results/reports/evaluation_alerts.json`` — i.e. the 10 underlying
test rows that have already been curated for the dashboard. Everything
else in test is paper-metrics territory and must never be exposed to
the operator UI.

Two roles
---------

* ``demo_pool``  — frozen rows visible to dashboard / study mode /
  user-facing artefacts.
* ``test_paper`` — frozen rows used only for paper-metrics computation
  (M1-M8 in ``run_tests.py``). Never to be surfaced in
  ``evaluation_alerts.json``.

Run the CLI to (re)generate the sidecar:

.. code-block:: bash

    python -m module1_preprocessing.phase1.finalize_test_split

Read the assignment from anywhere:

.. code-block:: python

    from module1_preprocessing.phase1.finalize_test_split import (
        is_demo_pool, demo_row_ids,
    )
    if is_demo_pool(row_id):
        ...

The sidecar lives at ``data/processed/test_split_assignment.json``
and is **deterministic** — re-running the CLI on the same inputs
produces a byte-identical sidecar (modulo the ``generated_at``
timestamp which is regenerated each run).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module1_preprocessing.phase1._sidecar_io import (  # noqa: E402
    atomic_write_json,
    load_sidecar,
)

logger = logging.getLogger(__name__)


SIDECAR_PATH = PROJECT_ROOT / "data" / "processed" / "test_split_assignment.json"
TEST_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "test_phase1.parquet"
EVAL_ALERTS_PATH = PROJECT_ROOT / "results" / "reports" / "evaluation_alerts.json"

SIDECAR_FORMAT = "phase1.test_split_assignment.v1"


class SplitAssignmentMissing(RuntimeError):
    """Raised when a consumer asks about split membership but the
    sidecar has never been generated. Don't silently fall back —
    make the operator run :func:`finalize`."""


# ── Read API ──────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def load_test_split_assignment() -> Dict[str, Any]:
    """Return the parsed sidecar.

    Raises :class:`SplitAssignmentMissing` if the sidecar is absent.
    """
    try:
        return load_sidecar(
            SIDECAR_PATH,
            expected_format=SIDECAR_FORMAT,
            artefact_label="test split assignment",
        )
    except FileNotFoundError as exc:
        raise SplitAssignmentMissing(
            f"{SIDECAR_PATH} not found. Run "
            "`python -m module1_preprocessing.phase1.finalize_test_split` "
            "to generate the test/demo role assignment."
        ) from exc


@lru_cache(maxsize=1)
def _demo_row_ids_set() -> FrozenSet[int]:
    return frozenset(int(r) for r in load_test_split_assignment()["demo_pool"]["row_ids"])


def is_demo_pool(row_id: int) -> bool:
    """True iff the test row at ``row_id`` is part of the frozen demo
    pool (visible to dashboard / Phase-2 study)."""
    return int(row_id) in _demo_row_ids_set()


def is_test_paper(row_id: int) -> bool:
    """True iff the test row at ``row_id`` is part of the frozen
    paper-metrics subset (M-metrics only)."""
    return not is_demo_pool(row_id)


def demo_row_ids() -> List[int]:
    """Sorted list of demo-pool row_ids."""
    return sorted(_demo_row_ids_set())


def assert_no_paper_rows_in_eval_alerts() -> None:
    """Assert ``evaluation_alerts.json`` contains only demo_pool rows.

    Used by the leakage-guard test to catch any regression that would
    expose paper-metrics rows on the operator UI.
    """
    with EVAL_ALERTS_PATH.open(encoding="utf-8") as f:
        eval_alerts = json.load(f)
    demo = _demo_row_ids_set()
    leaks = [
        a for a in eval_alerts
        if int(a["sample_index"]) not in demo
    ]
    if leaks:
        raise AssertionError(
            f"{len(leaks)} alert(s) in {EVAL_ALERTS_PATH.name} reference "
            f"row_ids outside the demo_pool: "
            f"{[a['alert_id'] for a in leaks[:5]]}…"
        )


# ── Write API (CLI) ───────────────────────────────────────────────────


def _build_assignment(test_df: pd.DataFrame, eval_alerts: list) -> Dict[str, Any]:
    """Construct the sidecar payload.

    The demo pool is the **deduplicated** set of ``sample_index`` values
    currently in ``evaluation_alerts.json``. The test_paper subset is
    everything else in the test parquet.
    """
    n_test = int(len(test_df))
    all_row_ids = set(int(r) for r in test_df["row_id"].tolist())

    demo_indices = sorted({int(a["sample_index"]) for a in eval_alerts})
    bad = [r for r in demo_indices if r not in all_row_ids]
    if bad:
        raise ValueError(
            f"{len(bad)} sample_index value(s) in evaluation_alerts.json "
            f"are not present as row_ids in test_phase1: {bad[:5]}…"
        )

    demo_set = set(demo_indices)
    paper_indices = sorted(all_row_ids - demo_set)

    # Class-distribution accounting for both subsets (audit / sanity).
    cat_col = "Attack Category"
    label_col = "Label"

    def _dist(rows: list[int]) -> Dict[str, Any]:
        sub = test_df[test_df["row_id"].isin(rows)]
        # ``pd.Series(...)`` wrap is a no-op at runtime but lets pyright
        # see ``value_counts`` (the column slice's static type widens to
        # ``ndarray | Series`` under some pandas-stubs versions).
        labels = pd.Series(sub[label_col]).value_counts().to_dict()
        cats = pd.Series(sub[cat_col]).value_counts().to_dict()
        return {
            "n": int(len(sub)),
            "label_counts": {str(int(k)): int(v) for k, v in labels.items()},
            "attack_category_counts": {str(k): int(v) for k, v in cats.items()},
        }

    payload: Dict[str, Any] = {
        "format": SIDECAR_FORMAT,
        "split_strategy": "scope_a_split_flag_v1",
        "rationale": (
            "Scope-A: freeze the existing curated demo subset "
            "(unique sample_indices in evaluation_alerts.json) and "
            "label everything else in test_phase1 as paper-metrics. "
            "No retraining; no parquet schema change."
        ),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": {
            "test_parquet": str(TEST_PARQUET_PATH.relative_to(PROJECT_ROOT)),
            "test_n_rows": n_test,
            "demo_seed": str(EVAL_ALERTS_PATH.relative_to(PROJECT_ROOT)),
            "demo_seed_n_alerts": int(len(eval_alerts)),
            "demo_seed_n_unique_sample_indices": int(len(demo_indices)),
        },
        "demo_pool": {
            "row_ids": demo_indices,
            **_dist(demo_indices),
        },
        "test_paper": {
            # Don't enumerate ~4 886 row_ids in the JSON — derive from
            # the complement at read time. We just pin the count and
            # the class distribution for audit.
            "n": int(len(paper_indices)),
            "fraction_of_test": round(len(paper_indices) / n_test, 6),
            **{k: v for k, v in _dist(paper_indices).items() if k != "n"},
        },
        "invariants": [
            "demo_pool ∩ test_paper = ∅",
            "demo_pool ∪ test_paper = test_phase1 row_ids",
            "every sample_index in evaluation_alerts.json must be in demo_pool",
        ],
    }
    return payload


def finalize() -> Path:
    """(Re)generate the sidecar. Returns the path written."""
    if not TEST_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Missing {TEST_PARQUET_PATH}")
    if not EVAL_ALERTS_PATH.exists():
        raise FileNotFoundError(f"Missing {EVAL_ALERTS_PATH}")

    test_df = pd.read_parquet(TEST_PARQUET_PATH)
    if "row_id" not in test_df.columns:
        raise ValueError(
            f"{TEST_PARQUET_PATH} has no `row_id` column; "
            "Scope-A split assignment relies on positional row_id."
        )

    with EVAL_ALERTS_PATH.open(encoding="utf-8") as f:
        eval_alerts = json.load(f)

    payload = _build_assignment(test_df, eval_alerts)
    atomic_write_json(SIDECAR_PATH, payload)

    # Invalidate read-cache so a follow-up call in the same process
    # picks up the new sidecar.
    load_test_split_assignment.cache_clear()
    _demo_row_ids_set.cache_clear()

    logger.info(
        "Wrote %s — demo_pool=%d rows, test_paper=%d rows",
        SIDECAR_PATH.relative_to(PROJECT_ROOT),
        payload["demo_pool"]["n"],
        payload["test_paper"]["n"],
    )
    return SIDECAR_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Scope-A test split assignment sidecar.")
    parser.add_argument(
        "--check", action="store_true",
        help="Don't write — just verify the sidecar matches the current "
             "evaluation_alerts.json. Exit 1 if drift detected.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.check:
        try:
            existing = load_test_split_assignment()
        except SplitAssignmentMissing:
            logger.error("Sidecar absent — run without --check to generate.")
            return 1
        with EVAL_ALERTS_PATH.open(encoding="utf-8") as f:
            eval_alerts = json.load(f)
        existing_demo = sorted(int(r) for r in existing["demo_pool"]["row_ids"])
        current_demo = sorted({int(a["sample_index"]) for a in eval_alerts})
        if existing_demo != current_demo:
            logger.error(
                "Drift: sidecar demo_pool=%s but evaluation_alerts unique sample_indices=%s",
                existing_demo, current_demo,
            )
            return 1
        logger.info("OK — sidecar matches evaluation_alerts.json.")
        return 0

    finalize()
    print(f"Saved: {SIDECAR_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  demo_pool : {len(demo_row_ids())} rows ({demo_row_ids()})")
    body = load_test_split_assignment()
    print(f"  test_paper: {body['test_paper']['n']} rows "
          f"({body['test_paper']['fraction_of_test']*100:.2f}% of test)")
    print(f"  invariants enforced by tests/test_test_split_role_assignment.py")
    return 0


__all__ = [
    "SIDECAR_PATH",
    "SIDECAR_FORMAT",
    "SplitAssignmentMissing",
    "load_test_split_assignment",
    "is_demo_pool",
    "is_test_paper",
    "demo_row_ids",
    "assert_no_paper_rows_in_eval_alerts",
    "finalize",
]


if __name__ == "__main__":
    raise SystemExit(main())
