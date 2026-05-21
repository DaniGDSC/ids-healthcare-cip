"""RQ3 Track 4 — verify §4.2 truth-table claims against the canonical CSV.

CROSS-REF: the 8 critical rows from RQ3_expected_outputs.md §4.2 are
verified against results/rq1_tier_surfacing_truth_table.csv (produced
by module6_evaluation/make_rq1_truth_table.py — RQ1 Phase 7). Wildcards
expand to 16 concrete claims.

Real CSV state (Phase 0 confirmed):
  - 16 rows total (one per (tier, patchable, maintenance_active) triple;
    spec §1's "32 rows" is the spec's count, not the generator's).
  - Boolean encoding: ``True`` / ``False`` strings for both ``patchable``
    and ``maintenance_active``. ``_find_row`` maps spec's
    ``active``/``inactive`` semantics to these.
  - Extra columns beyond the spec's 5 (device_class, anomaly_score,
    adjusted_score, threshold, risk_multiplier) are ignored by the
    lookup; only the 5 spec columns participate in claim verification.

Side effect: writes results/rq3_truth_table_reference.json for the
RQ3 master aggregator (when compute_rq3_metrics.py lands).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "results" / "rq1_tier_surfacing_truth_table.csv"
JSON_OUT = REPO_ROOT / "results" / "rq3_truth_table_reference.json"

# Real CSV column names (Phase 0 confirmed — matches spec exactly).
COL_TIER = "risk_tier"
COL_PATCHABLE = "patchable"
COL_MAINTENANCE = "maintenance_active"
COL_SURFACE = "should_surface"
COL_REASON = "reason"

# The 8 rows from RQ3_expected_outputs.md §4.2, with wildcards.
# expected_surface is one of "TRUE", "FALSE", "DEPENDS" (non-binary).
# expected_reason_prefix is a substring fragment of the canonical reason
# string, or None when §4.2 explicitly accepts multiple reasons.
CRITICAL_CLAIMS: list[tuple[str, str, str, str, str | None]] = [
    ("CRITICAL", "False", "active",   "TRUE",  "safety_floor"),
    ("CRITICAL", "False", "inactive", "TRUE",  "safety_floor"),
    ("CRITICAL", "True",  "active",   "FALSE", "suppressed_maintenance"),
    ("CRITICAL", "True",  "inactive", "TRUE",  None),  # above_threshold OR normal
    ("HIGH",     "*",     "active",   "FALSE", "suppressed_maintenance"),
    ("HIGH",     "*",     "inactive", "DEPENDS", "normal"),
    ("MEDIUM",   "*",     "*",        "DEPENDS", "normal"),
    ("LOW",      "*",     "*",        "DEPENDS", "below_threshold"),
]


def _expand_wildcards(claims: list[tuple]) -> list[dict]:
    """Enumerate ``*`` wildcards into concrete (patchable, maintenance) pairs."""
    patchable_values = ["True", "False"]
    maintenance_values = ["active", "inactive"]
    out: list[dict] = []
    for tier, p, m, surface, reason in claims:
        p_vals = patchable_values if p == "*" else [p]
        m_vals = maintenance_values if m == "*" else [m]
        for pv, mv in product(p_vals, m_vals):
            out.append({
                "tier": tier,
                "patchable": pv,
                "maintenance": mv,
                "expected_surface": surface,
                "expected_reason_prefix": reason,
                "source_claim": f"{tier}|{p}|{m}",
            })
    return out


def _normalize_bool(value) -> str:
    """Map any common boolean variant to canonical ``True`` / ``False``."""
    s = "" if value is None else str(value).strip().lower()
    if s in {"true", "1", "yes", "active", "on"}:
        return "True"
    if s in {"false", "0", "no", "inactive", "off"}:
        return "False"
    return str(value).strip() if value is not None else ""


def _load_csv() -> list[dict]:
    if not CSV_PATH.exists():
        pytest.skip(
            f"{CSV_PATH.relative_to(REPO_ROOT)} missing. Run "
            "`python -m module6_evaluation.make_rq1_truth_table` first."
        )
    with CSV_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_row(rows: list[dict], tier: str, patchable: str,
              maintenance: str) -> dict | None:
    """Locate the row matching the concrete (tier, patchable, maintenance) triple.

    ``maintenance`` is the spec's semantic value (``active`` /
    ``inactive``); the CSV stores it as a boolean (``True`` / ``False``).
    Map via ``_normalize_bool`` so both encodings line up.
    """
    target_p = _normalize_bool(patchable)
    target_m = _normalize_bool(maintenance)
    for row in rows:
        if str(row.get(COL_TIER, "")).strip().upper() != tier.upper():
            continue
        if _normalize_bool(row.get(COL_PATCHABLE)) != target_p:
            continue
        if _normalize_bool(row.get(COL_MAINTENANCE)) != target_m:
            continue
        return row
    return None


def _evaluate_claim(claim: dict, rows: list[dict]) -> dict:
    """Verify one expanded claim against the CSV."""
    row = _find_row(rows, claim["tier"], claim["patchable"], claim["maintenance"])
    out: dict = {"claim": claim, "matched_row": row, "status": None, "details": None}

    if row is None:
        out["status"] = "row_missing"
        out["details"] = (
            f"No row matching tier={claim['tier']} "
            f"patchable={claim['patchable']} maintenance={claim['maintenance']}"
        )
        return out

    actual_surface = _normalize_bool(row.get(COL_SURFACE, ""))
    actual_reason = str(row.get(COL_REASON, "")).strip().lower()
    expected = claim["expected_surface"]

    if expected == "DEPENDS":
        out["status"] = "depends_ok"
        out["details"] = (
            f"Row present; outcome ({actual_surface}) is non-binary per §4.2."
        )
        return out

    # Map "TRUE"/"FALSE" claim shorthand to "True"/"False" CSV form.
    expected_canonical = "True" if expected == "TRUE" else "False"
    if actual_surface != expected_canonical:
        out["status"] = "fail"
        out["details"] = (
            f"Expected should_surface={expected_canonical}, "
            f"got {actual_surface}. Reason: {actual_reason!r}."
        )
        return out

    prefix = claim["expected_reason_prefix"]
    if prefix and prefix.lower() not in actual_reason:
        out["status"] = "fail"
        out["details"] = (
            f"Outcome matches ({actual_surface}) but reason mismatch: "
            f"expected prefix {prefix!r}, got {actual_reason!r}."
        )
        return out

    out["status"] = "pass"
    return out


def _write_json(results: list[dict]) -> dict:
    n_pass = sum(1 for r in results if r["status"] == "pass")
    n_depends = sum(1 for r in results if r["status"] == "depends_ok")
    n_fail = sum(1 for r in results if r["status"] in {"fail", "row_missing"})
    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "tests/test_rq3_truth_table_completeness.py",
            "source_csv": str(CSV_PATH.relative_to(REPO_ROOT)),
            "rq3_section_reference": "RQ3_expected_outputs.md §4.2 "
                                     "(via RQ3_TRUTH_TABLE_SPEC.md §4.1)",
            "_cross_reference_note": (
                "Track 4 verifies the 8 critical rows from RQ3 §4.2 against "
                "the canonical truth table produced by RQ1 Phase 7. The full "
                "table lives in results/rq1_tier_surfacing_truth_table.csv "
                "(16 rows: one per (tier, patchable, maintenance_active) "
                "triple)."
            ),
            "_encoding_note": (
                "Real CSV uses 'True'/'False' for both patchable and "
                "maintenance_active. _find_row maps spec's "
                "'active'/'inactive' semantics to these via _normalize_bool."
            ),
        },
        "headline": {
            "verification_pass": n_fail == 0,
            "n_claims_total": len(results),
            "n_pass": n_pass,
            "n_depends_ok": n_depends,
            "n_fail": n_fail,
        },
        "results": results,
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(out, indent=2, default=str))
    return out


# ─── pytest entry points ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def csv_rows() -> list[dict]:
    return _load_csv()


@pytest.fixture(scope="module")
def expanded_claims() -> list[dict]:
    return _expand_wildcards(CRITICAL_CLAIMS)


@pytest.fixture(scope="module")
def verification_results(csv_rows, expanded_claims) -> list[dict]:
    results = [_evaluate_claim(c, csv_rows) for c in expanded_claims]
    _write_json(results)
    return results


def test_all_critical_rows_present(verification_results):
    """Every concrete (tier, patchable, maintenance) combo must be in the CSV."""
    missing = [r for r in verification_results if r["status"] == "row_missing"]
    assert not missing, (
        f"{len(missing)} critical row(s) missing from "
        f"{CSV_PATH.relative_to(REPO_ROOT)}:\n"
        + "\n".join(
            f"  - {r['claim']['source_claim']} -> "
            f"({r['claim']['tier']}, {r['claim']['patchable']}, "
            f"{r['claim']['maintenance']}): {r['details']}"
            for r in missing[:10]
        )
    )


def test_critical_safety_floor_rows(verification_results):
    """DEFENSE-CRITICAL: CRITICAL+unpatchable always surfaces (Invariant 2)."""
    safety_floor = [
        r for r in verification_results
        if r["claim"]["tier"] == "CRITICAL"
        and r["claim"]["patchable"] == "False"
    ]
    assert len(safety_floor) == 2, (
        f"Expected 2 safety_floor claims; got {len(safety_floor)}"
    )
    for r in safety_floor:
        assert r["status"] == "pass", (
            f"Safety floor violation: {r['claim']['source_claim']} -> "
            f"{r['status']}. {r['details']}"
        )


def test_maintenance_suppression_holds_for_patchable(verification_results):
    """Maintenance window suppresses patchable HIGH/CRITICAL alerts."""
    target = [
        r for r in verification_results
        if r["claim"]["maintenance"] == "active"
        and r["claim"]["expected_surface"] == "FALSE"
    ]
    failures = [r for r in target if r["status"] != "pass"]
    assert not failures, (
        f"{len(failures)} maintenance-suppression claim(s) failed: "
        + "; ".join(f"{r['claim']['source_claim']} -> {r['status']}"
                    for r in failures[:5])
    )


def test_no_outcome_mismatches(verification_results):
    """No claim with status=='fail' (outcome or reason mismatch)."""
    mismatches = [r for r in verification_results if r["status"] == "fail"]
    assert not mismatches, (
        f"{len(mismatches)} truth-table outcome mismatch(es):\n"
        + "\n".join(f"  - {r['claim']['source_claim']}: {r['details']}"
                    for r in mismatches[:10])
    )


def test_depends_rows_present(verification_results):
    """'Depends on threshold' rows must exist (outcome is non-binary)."""
    depends = [r for r in verification_results
               if r["claim"]["expected_surface"] == "DEPENDS"]
    missing = [r for r in depends if r["status"] == "row_missing"]
    assert not missing, (
        f"{len(missing)} 'depends on threshold' row(s) absent from CSV: "
        + "; ".join(r["claim"]["source_claim"] for r in missing[:5])
    )
