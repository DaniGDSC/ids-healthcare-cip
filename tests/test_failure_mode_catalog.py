"""Schema + invariant tests for the RQ2 failure mode catalog."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "results" / "rq2_failure_mode_catalog.json"

REQUIRED_CATEGORIES = {
    "TECHNICAL_LAYER_1", "MITRE_NOT_REFERENCED",
    "DO_NOT_IGNORED", "ROLE_VIEW_MISMATCH", "OTHER",
}
VALID_STATUSES = {
    "observed_not_fixed", "no_observations_collected", "source_unavailable",
}


@pytest.fixture(scope="module")
def catalog() -> dict:
    if not CATALOG.exists():
        pytest.skip("Run analysis/compile_failure_modes.py first")
    return json.loads(CATALOG.read_text())


def test_schema_complete(catalog: dict) -> None:
    for key in ("_meta", "_disclosure", "summary", "catalog"):
        assert key in catalog, f"Missing top-level key: {key}"


def test_all_five_categories_present(catalog: dict) -> None:
    missing = REQUIRED_CATEGORIES - set(catalog["catalog"].keys())
    assert not missing, f"Missing categories: {missing}"


def test_disclosure_framing_is_observation(catalog: dict) -> None:
    """Defense-critical: framing MUST NOT claim improvement."""
    d = catalog["_disclosure"]
    assert d["framing"] == "observation_not_improvement", (
        "Catalog framing must be 'observation_not_improvement' "
        "(RQ2.d rescoped to future work)."
    )
    assert d["iteration_performed"] is False, (
        "iteration_performed must be False — no iteration was done."
    )


def test_every_entry_has_status(catalog: dict) -> None:
    for cid, entry in catalog["catalog"].items():
        assert "_status" in entry, f"Missing _status for {cid}"
        assert entry["_status"] in VALID_STATUSES, (
            f"Invalid _status for {cid}: {entry['_status']}"
        )


def test_other_bucket_diagnostic_under_threshold(catalog: dict) -> None:
    """Sanity: if OTHER >40% of observations, the taxonomy is broken.

    Skipped when total < 10 — the percentage is uninformative at small N
    and would fire spuriously on near-empty catalogs.
    """
    summary = catalog["summary"]
    total = summary.get("total_observations", 0)
    if total < 10:
        pytest.skip(f"Total observations ({total}) too few for taxonomy "
                    "diagnosis.")
    other = summary["other_bucket_diagnostic"]
    assert other["size_pct"] < 0.40, (
        f"OTHER bucket is {other['size_pct']:.1%} of {total} observations — "
        "taxonomy likely needs new fixed category."
    )


def test_every_entry_has_recommended_iteration(catalog: dict) -> None:
    """Future-work column must be populated for every category."""
    for cid, entry in catalog["catalog"].items():
        ri = entry.get("recommended_iteration", "")
        assert ri and ri.strip(), (
            f"Category {cid} missing recommended_iteration. "
            "Update configs/rq2_failure_categories.yaml."
        )


def test_taxonomy_provenance_disclosed(catalog: dict) -> None:
    """The catalog must honestly state whether the taxonomy predates data."""
    d = catalog["_disclosure"]
    assert "taxonomy_predates_data" in d, (
        "_disclosure must include taxonomy_predates_data (defense-critical)."
    )
    assert d.get("taxonomy_source"), \
        "_disclosure.taxonomy_source must be populated."
