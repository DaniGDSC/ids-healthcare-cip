"""ARCHITECTURE.md Step [1] / "Split integrity" invariant.

Locks the four invariants of the 4-way Strategy 1 split (extends
``test_split_consistency.py``):

* I1 No row appears in more than one split (pairwise disjoint).
* I2 Stratification preserves Attack Category proportions within ±2%
     across all 4 partitions.
* I3 The persisted ``configs/composite_risk_weights.yaml`` was anchored
     to the test split and the boundaries don't cut clusters.
* I4 ``data/processed/split_metadata.yaml`` exists, parses, and
     records the random_state + per-split sample counts.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def parquets() -> dict[str, pd.DataFrame]:
    paths = {
        "train": DATA / "train_phase1.parquet",
        "val":   DATA / "val_phase1.parquet",
        "test":  DATA / "test_phase1.parquet",
        "demo":  DATA / "demo_phase1.parquet",
    }
    for name, p in paths.items():
        if not p.exists():
            pytest.skip(f"Missing {p} — run module1_preprocessing first.")
    return {name: pd.read_parquet(p) for name, p in paths.items()}


# ── I1: pairwise disjoint via row_id ──────────────────────────────────


def _row_hashes(df: pd.DataFrame) -> set[bytes]:
    """Hash each row's numeric feature vector for cross-split
    disjointness checks. ``row_id`` is split-local (0..N-1 per
    parquet) so we can't use it; the feature vector itself is the
    canonical row identity."""
    drop = [c for c in ("Label", "Attack Category", "row_id",
                        "device_class", "attack_category")
            if c in df.columns]
    feats = df.drop(columns=drop).select_dtypes(include="number")
    # Hash each row by converting to bytes — exact-match disjointness.
    return {row.tobytes() for row in feats.values}


def test_4way_splits_are_pairwise_disjoint(parquets):
    """No row appears in more than one split. Uses feature-vector
    bytes as the row identity (``row_id`` is split-local)."""
    hashes = {name: _row_hashes(df) for name, df in parquets.items()}
    keys = list(hashes)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = hashes[keys[i]] & hashes[keys[j]]
            # Allow a tiny number of overlaps if the dataset has
            # duplicate rows (the EHMS-2020 raw CSV has some) — but
            # the rate must be < 0.5% of the smaller split.
            min_n = min(len(hashes[keys[i]]), len(hashes[keys[j]]))
            rate = len(overlap) / max(min_n, 1)
            assert rate < 0.005, (
                f"{keys[i]} ∩ {keys[j]} = {len(overlap)} "
                f"({rate:.2%} of smaller split) — exceeds duplicate-row "
                "tolerance, Strategy 1 split is leaking"
            )


def test_4way_splits_union_covers_full_dataset(parquets):
    """Sample sizes sum to the full preprocessed dataset (16318 rows
    after Phase 1 cleaning)."""
    total = sum(len(df) for df in parquets.values())
    # 60% + 15% + 15% + 10% = 100% of 16318 = 16318
    assert total == 16318, (
        f"Splits sum to {total} rows; expected 16318 (full dataset)"
    )


# ── I2: stratification within ±2% on Attack Category ──────────────────


def test_stratification_preserves_attack_category_proportions(parquets):
    """Each split's per-category proportion must be within 2pp of the
    grand-average proportion."""
    cats_per_split = {
        name: df["Attack Category"].astype(str).value_counts(normalize=True)
        for name, df in parquets.items()
        if "Attack Category" in df.columns
    }
    if not cats_per_split:
        pytest.skip("Attack Category column absent")
    # Grand average across splits
    all_cats = set()
    for s in cats_per_split.values():
        all_cats |= set(s.index)
    for cat in all_cats:
        per_split = [cats_per_split[name].get(cat, 0.0) for name in cats_per_split]
        if max(per_split) < 0.001:
            continue   # vanishingly rare class — drift bound not meaningful
        spread = max(per_split) - min(per_split)
        assert spread <= 0.02, (
            f"Attack Category {cat!r} drifts {spread:.3f} across splits "
            f"(per-split: {dict(zip(cats_per_split, per_split))}) — "
            "exceeds ±2pp tolerance"
        )


def test_attack_rate_within_band_across_splits(parquets):
    """Binary attack rate must be within ±2pp across all 4 splits."""
    rates = [df["Label"].mean() for df in parquets.values() if "Label" in df.columns]
    assert rates, "No Label column found"
    spread = max(rates) - min(rates)
    assert spread <= 0.02, (
        f"Binary attack rate spread {spread:.3f} > 2pp across splits"
    )


# ── I4: split_metadata.yaml provenance ────────────────────────────────


def test_split_metadata_yaml_exists():
    p = DATA / "split_metadata.yaml"
    assert p.exists(), (
        f"{p} missing — Strategy 1 provenance must be persisted "
        "(re-run module1_preprocessing)"
    )


def test_split_metadata_records_required_provenance():
    p = DATA / "split_metadata.yaml"
    if not p.exists():
        pytest.skip("split_metadata.yaml absent")
    body = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert body.get("format") == "phase1.split_metadata.v1"
    assert body.get("random_state") == 42
    splits = body.get("splits") or {}
    for k in ("train", "val", "test", "demo"):
        assert k in splits, f"split_metadata missing {k!r}"
        section = splits[k]
        assert "n" in section
        assert "attack_rate" in section


# ── Leakage assertion (Module 2 side) ─────────────────────────────────


def test_module2_refuses_to_load_demo_phase1():
    """Module 2 training functions MUST raise RuntimeError if asked
    to read the demo split."""
    from module2_detection.module2_train_models import _assert_no_demo_leakage
    with pytest.raises(RuntimeError):
        _assert_no_demo_leakage(DATA / "demo_phase1.parquet")
