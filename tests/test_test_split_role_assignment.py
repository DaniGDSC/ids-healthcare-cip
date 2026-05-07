"""Scope-A test-split role-assignment tests.

Locks the four invariants of v5.1's frozen-test/frozen-demo design:

  I1  Sidecar exists and parses with the expected ``format`` tag.
  I2  ``demo_pool ∩ test_paper = ∅`` and ``demo_pool ∪ test_paper``
      covers every row_id in test_phase1.
  I3  Every ``sample_index`` in ``results/reports/evaluation_alerts.json``
      is a member of demo_pool — no paper-metrics row is ever exposed
      to the operator UI.
  I4  Re-running ``finalize_test_split.finalize()`` is idempotent on
      identical inputs (apart from the ``generated_at`` timestamp).

These tests are the load-bearing guard for the v5.1 split design.
If any of them fails, the dashboard or M-metrics pipeline is on the
edge of leaking demo↔paper data; do not relax them lightly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from module1_preprocessing.phase1.finalize_test_split import (
    SIDECAR_FORMAT,
    SIDECAR_PATH,
    SplitAssignmentMissing,
    _demo_row_ids_set,
    assert_no_paper_rows_in_eval_alerts,
    demo_row_ids,
    finalize,
    is_demo_pool,
    is_test_paper,
    load_test_split_assignment,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PARQUET = PROJECT_ROOT / "data" / "processed" / "test_phase1.parquet"
EVAL_ALERTS = PROJECT_ROOT / "results" / "reports" / "evaluation_alerts.json"


# Each test invalidates the lru_cache so prior tests don't poison state.
@pytest.fixture(autouse=True)
def _clear_caches():
    load_test_split_assignment.cache_clear()
    _demo_row_ids_set.cache_clear()
    yield
    load_test_split_assignment.cache_clear()
    _demo_row_ids_set.cache_clear()


# ── I1: sidecar shape ─────────────────────────────────────────────────


def test_sidecar_exists_and_parses():
    body = load_test_split_assignment()
    assert body["format"] == SIDECAR_FORMAT
    assert body["split_strategy"] == "scope_a_split_flag_v1"
    assert "demo_pool" in body and "test_paper" in body


def test_sidecar_records_class_distribution_for_audit():
    body = load_test_split_assignment()
    for k in ("demo_pool", "test_paper"):
        assert "label_counts" in body[k]
        assert "attack_category_counts" in body[k]


# ── I2: demo / paper partition is total + disjoint ────────────────────


def test_demo_and_paper_are_disjoint():
    body = load_test_split_assignment()
    demo = set(int(r) for r in body["demo_pool"]["row_ids"])
    n_test = body["source"]["test_n_rows"]
    test_df = pd.read_parquet(TEST_PARQUET)
    all_ids = set(int(r) for r in test_df["row_id"].tolist())
    paper = all_ids - demo
    # Disjoint:
    assert demo & paper == set()
    # Total:
    assert demo | paper == all_ids
    # Counts in sidecar are consistent:
    assert body["demo_pool"]["n"] == len(demo)
    assert body["test_paper"]["n"] == len(paper)
    assert body["demo_pool"]["n"] + body["test_paper"]["n"] == n_test


def test_is_demo_pool_and_is_test_paper_are_complementary():
    test_df = pd.read_parquet(TEST_PARQUET)
    for rid in test_df["row_id"].sample(50, random_state=0):
        assert is_demo_pool(int(rid)) != is_test_paper(int(rid))


def test_demo_row_ids_are_sorted_and_in_range():
    test_df = pd.read_parquet(TEST_PARQUET)
    valid_range = (test_df["row_id"].min(), test_df["row_id"].max())
    rows = demo_row_ids()
    assert rows == sorted(rows)
    for r in rows:
        assert valid_range[0] <= r <= valid_range[1]


# ── I3: no paper-metrics row leaks to the operator UI ────────────────


def test_every_evaluation_alert_lives_in_demo_pool():
    """Critical guard: ``evaluation_alerts.json`` must reference only
    demo_pool rows. Otherwise the dashboard is exposing paper-metrics
    territory to the operator and the v5.1 split contract is broken."""
    assert_no_paper_rows_in_eval_alerts()


def test_unique_eval_sample_indices_match_sidecar_demo_pool():
    with EVAL_ALERTS.open(encoding="utf-8") as f:
        eval_alerts = json.load(f)
    eval_unique = sorted({int(a["sample_index"]) for a in eval_alerts})
    assert eval_unique == demo_row_ids()


# ── I4: idempotent regeneration (sans timestamp) ──────────────────────


def test_finalize_is_idempotent_modulo_timestamp():
    """Two runs back-to-back must produce the same content modulo
    the ``generated_at`` timestamp. Otherwise the assignment is
    non-deterministic and the v5.1 frozen-split claim is false."""
    finalize()
    body1 = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))

    load_test_split_assignment.cache_clear()
    _demo_row_ids_set.cache_clear()
    finalize()
    body2 = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))

    body1.pop("generated_at", None)
    body2.pop("generated_at", None)
    assert body1 == body2


# ── Failure modes the consumer-facing API guarantees ──────────────────


def test_missing_sidecar_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "module1_preprocessing.phase1.finalize_test_split.SIDECAR_PATH",
        tmp_path / "not_a_real_sidecar.json",
    )
    load_test_split_assignment.cache_clear()
    _demo_row_ids_set.cache_clear()
    with pytest.raises(SplitAssignmentMissing):
        load_test_split_assignment()
