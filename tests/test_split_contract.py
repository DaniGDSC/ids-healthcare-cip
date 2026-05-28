"""Phase 1 split-contract test — locks the 60:15:15:10 4-way split.

This test is the byte-level audit anchor for the split that powers every
downstream module (2-6). It re-runs the splitter against the raw CSV and
verifies the output matches the contract recorded in
``data/processed/split_metadata.yaml`` — same n, same attack_rate, same
Attack Category counts per split.

If this test fails, EITHER:
  - the splitter logic has drifted from the recorded contract, OR
  - the raw CSV has been replaced (the integrity baseline should catch
    this case first, but this test is a second defense), OR
  - split_metadata.yaml has been edited to claim something the code
    cannot produce.

In any of those cases, downstream model artifacts (Modules 2-5) are no
longer trustable until the cause is investigated and the parquets are
either regenerated or restored from the manifest in
``data/processed/split_artifact_manifest.txt``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RAW_CSV = _PROJECT_ROOT / "data/raw/WUSTL-EHMS/wustl-ehms-2020_with_attacks_categories.csv"
_METADATA = _PROJECT_ROOT / "data/processed/split_metadata.yaml"


@pytest.fixture(scope="module")
def contract() -> dict:
    """Load the contract this test validates against."""
    if not _METADATA.exists():
        pytest.skip(f"split_metadata.yaml not found at {_METADATA}")
    with open(_METADATA) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def splits() -> dict:
    """Run the 4-way splitter against the raw CSV (once per session)."""
    if not _RAW_CSV.exists():
        pytest.skip(f"raw CSV not found at {_RAW_CSV}")

    from module1_preprocessing.splitter import DataSplitter

    raw = pd.read_csv(_RAW_CSV, low_memory=False)
    raw = raw.reset_index().rename(columns={"index": "raw_csv_row"})

    splitter = DataSplitter(
        train_ratio=0.60,
        val_ratio=0.15,
        test_ratio=0.15,
        demo_ratio=0.10,
        random_state=42,
        label_column="Label",
        multi_label_column="Attack Category",
    )
    out = splitter.split(raw)
    return {
        "train": (out.y_train, out.y_multi_train),
        "val":   (out.y_val,   out.y_multi_val),
        "test":  (out.y_test,  out.y_multi_test),
        "demo":  (out.y_demo,  out.y_multi_demo),
    }


@pytest.mark.parametrize("name", ["train", "val", "test", "demo"])
def test_split_row_count(name: str, contract: dict, splits: dict) -> None:
    """Each split's row count must match split_metadata.yaml exactly."""
    expected = contract["splits"][name]["n"]
    y, _ = splits[name]
    assert len(y) == expected, (
        f"{name}: row count {len(y)} != contract {expected}"
    )


@pytest.mark.parametrize("name", ["train", "val", "test", "demo"])
def test_split_attack_rate(name: str, contract: dict, splits: dict) -> None:
    """Each split's attack rate must match the contract to 5 decimal places."""
    expected = float(contract["splits"][name]["attack_rate"])
    y, _ = splits[name]
    actual = float((y == 1).mean())
    assert abs(actual - expected) < 1e-5, (
        f"{name}: attack_rate {actual:.6f} != contract {expected:.6f}"
    )


@pytest.mark.parametrize("name", ["train", "val", "test", "demo"])
def test_split_category_counts(name: str, contract: dict, splits: dict) -> None:
    """Per-category counts must match exactly (stratified Attack Category)."""
    expected = contract["splits"][name]["attack_category_counts"]
    _, y_multi = splits[name]
    actual = pd.Series(y_multi).value_counts().to_dict()
    actual_str = {str(k): int(v) for k, v in actual.items()}
    assert actual_str == expected, (
        f"{name}: category counts diverged from contract\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual_str}"
    )


def test_splits_disjoint_by_row_count(contract: dict) -> None:
    """train + val + test + demo must equal n_total exactly."""
    n_total = contract["n_total"]
    summed = sum(contract["splits"][k]["n"] for k in ("train", "val", "test", "demo"))
    assert summed == n_total, (
        f"split sizes don't sum to n_total: {summed} != {n_total}"
    )


def test_strategy_string_recorded(contract: dict) -> None:
    """Defense Q&A asset — strategy string is part of the audit record."""
    assert "Strategy 1" in contract["strategy"]
    assert "4-way" in contract["strategy"].lower() or "4-way" in contract["strategy"]


def test_random_state_canonical(contract: dict) -> None:
    """random_state must be in the canonical vetted set for research integrity."""
    canonical = {0, 7, 42}
    assert contract["random_state"] in canonical, (
        f"random_state={contract['random_state']} not in canonical set {canonical}"
    )


def test_ratios_sum_to_one(contract: dict) -> None:
    """Defense invariant: train + val + test + demo = 1.0."""
    ratios = contract["ratios"]
    total = ratios["train"] + ratios["val"] + ratios["test"] + ratios["demo"]
    assert abs(total - 1.0) < 1e-6, f"ratios don't sum to 1.0: {total}"
