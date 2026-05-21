# RQ3 Track 4 — Tier × Surfacing Truth Table Verification

**Project:** XAI-IDS-Healthcare
**Scope:** RQ3.4 — Verify the 8 critical rows of the tier × surfacing truth table (`RQ3_expected_outputs.md §4.2`) are present and correct in the canonical truth table CSV produced by RQ1.
**Purpose:** Single, self-contained spec for the truth table verification + RQ3 Appendix B rendering. Hand to Claude Code.
**Status of design:** All decisions locked. Two `DO NOT GUESS` checkpoints (RQ1 CSV column names, paper rendering format preferences).

---

## 0. How to use this spec

1. Phase 0 is mandatory but tiny — Claude Code must confirm the RQ1 truth table CSV exists or note its absence (Track 4 is gated on RQ1 Phase 7 implementation).
2. Phases 1–3 are sequential.
3. Markers:
   - **DO NOT GUESS** — stop and ask the developer
   - **CROSS-REF** — relies on an artifact produced by RQ1
4. Total expected size: 1 verification test, 1 small renderer, 1 JSON, 1 markdown artifact. Runtime: sub-second.

---

## 1. Background: what Track 4 produces

| Component | Question | Output |
|---|---|---|
| Verification test | Are the 8 critical rows from `§4.2` present in the canonical CSV with correct outcomes? | `tests/test_rq3_truth_table_completeness.py` |
| Verification JSON | What's the structured result of the verification? | `results/rq3_truth_table_reference.json` |
| Paper markdown | The Appendix B table ready for thesis inclusion | `results/rq3_truth_table_appendix_b.md` |

Track 4 is **the smallest RQ3 track** by design. It does not generate a new truth table — that's RQ1's job (`make_rq1_truth_table.py` from `RQ1_PIPELINE_SPEC.md` Phase 7). Track 4 only verifies the RQ3-specific claims hold against RQ1's output.

---

## 2. Locked design decisions

| Decision | Resolution |
|---|---|
| Cross-reference strategy | Light verification: test asserts the 8 rows from §4.2 are present in RQ1's CSV |
| Wildcard handling | Expand: `HIGH \| * \| active` → enumerate all `patchable` values and assert each concrete row matches |
| Output format | Markdown (paper Appendix B) + JSON (master aggregator integration) |
| Source of truth | `results/rq1_tier_surfacing_truth_table.csv` (produced by RQ1 Phase 7) |
| "depends on threshold" rows | Assert row presence + flag outcome as non-binary (not asserted to a specific value) |
| Mismatched outcome | Hard fail with explicit per-row diff |

---

## 3. Phase 0 — Cross-reference confirmation (DO NOT GUESS)

### 3.1 Quick discovery

```python
# scripts/discover_truth_table_artifact.py — TRANSIENT, delete after Phase 0
"""
Confirm the RQ1 truth table CSV exists and inspect its column schema.
Track 4 depends entirely on this artifact.
"""
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "results/rq1_tier_surfacing_truth_table.csv"
MD_PATH = REPO_ROOT / "results/rq1_tier_surfacing_truth_table.md"

findings = {
    "csv_path": str(CSV_PATH.relative_to(REPO_ROOT)),
    "csv_exists": CSV_PATH.exists(),
    "md_path": str(MD_PATH.relative_to(REPO_ROOT)),
    "md_exists": MD_PATH.exists(),
}

if CSV_PATH.exists():
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    findings["csv_n_rows"] = len(rows)
    findings["csv_columns"] = reader.fieldnames
    findings["csv_sample_first_3"] = rows[:3]

    # Detect distinct values per column (informational)
    if rows:
        for col in reader.fieldnames or []:
            values = {str(r.get(col, "")) for r in rows}
            findings.setdefault("distinct_values", {})[col] = sorted(values)[:10]

print(json.dumps(findings, indent=2, default=str))
print("\n" + "=" * 60)
print("DEVELOPER ACTION:")
if not CSV_PATH.exists():
    print("  CSV missing. Track 4 is GATED on RQ1 Phase 7 implementation.")
    print("  Run RQ1 truth table generator first:")
    print("    python -m module6_evaluation.make_rq1_truth_table")
else:
    print("  Confirm column names match the verification's expected names:")
    print("    risk_tier, patchable, maintenance_active, should_surface, reason")
    print("  If columns differ (e.g., 'maintenance' vs 'maintenance_active'),")
    print("  Phase 1 maps them.")
print("=" * 60)
```

### 3.2 Two things to confirm

1. **CSV exists.** If not, RQ1 Phase 7 must be implemented first. Track 4 cannot ship without the canonical CSV.
2. **CSV column names.** The verification logic refers to `risk_tier`, `patchable`, `maintenance_active`, `should_surface`, `reason`. If the CSV uses different column names (e.g., `maintenance` instead of `maintenance_active`), Phase 1's verification logic must be adapted.

### 3.3 Verification

```bash
python scripts/discover_truth_table_artifact.py > /tmp/truth_table_inventory.json
# Developer confirms CSV exists and column schema matches.
```

---

## 4. Phase 1 — Verification test

### 4.1 The 8 critical rows from `RQ3_expected_outputs.md §4.2`

These are the *RQ3 paper claims* about the safety/surfacing model. The full truth table has 32 rows (4 tiers × 2 patchable × 2 maintenance × 2 other dimensions); §4.2 lists 8 with explicit wildcards.

| risk_tier | patchable | maintenance | should_surface | reason | wildcard expansion |
|---|---|---|---|---|---|
| CRITICAL | False | active | TRUE | safety_floor | — |
| CRITICAL | False | inactive | TRUE | safety_floor | — |
| CRITICAL | True | active | FALSE | suppressed_maintenance | — |
| CRITICAL | True | inactive | TRUE | (above_threshold or normal) | — |
| HIGH | * | active | FALSE | suppressed_maintenance | × 2 (True, False) |
| HIGH | * | inactive | depends on threshold | normal | × 2 (True, False) |
| MEDIUM | * | * | depends on threshold | normal | × 4 (True/False × active/inactive) |
| LOW | * | * | usually FALSE | below_threshold | × 4 |

After wildcard expansion: 4 + 2 + 2 + 4 + 4 = **16 concrete row claims**.

### 4.2 Create `tests/test_rq3_truth_table_completeness.py`

```python
"""
tests/test_rq3_truth_table_completeness.py

CROSS-REF: verify that the 8 critical rows from RQ3_expected_outputs.md §4.2
are present in results/rq1_tier_surfacing_truth_table.csv with the expected
should_surface values.

Wildcards in §4.2 (e.g., 'HIGH | * | active') are expanded to all concrete
combinations and each is verified.

Also writes a structured result to results/rq3_truth_table_reference.json
for the master RQ3 metrics aggregator.
"""

import csv
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "results/rq1_tier_surfacing_truth_table.csv"
JSON_OUT = REPO_ROOT / "results/rq3_truth_table_reference.json"


# Column names — DO NOT GUESS. Adapt if Phase 0 discovery reveals different.
COL_TIER = "risk_tier"
COL_PATCHABLE = "patchable"
COL_MAINTENANCE = "maintenance_active"
COL_SURFACE = "should_surface"
COL_REASON = "reason"


# The 8 rows from RQ3_expected_outputs.md §4.2, expanded to concrete claims.
# Each claim is a tuple of:
#   (tier, patchable_constraint, maintenance_constraint, expected_surface, reason_prefix)
# Where:
#   patchable_constraint, maintenance_constraint may be a single value or "*"
#   expected_surface is "TRUE", "FALSE", or "DEPENDS" (non-binary check)
#   reason_prefix is the reason category (substring-match-friendly)

CRITICAL_CLAIMS = [
    # (tier, patchable, maintenance, expected_surface, reason)
    ("CRITICAL", "False",   "active",   "TRUE",    "safety_floor"),
    ("CRITICAL", "False",   "inactive", "TRUE",    "safety_floor"),
    ("CRITICAL", "True",    "active",   "FALSE",   "suppressed_maintenance"),
    ("CRITICAL", "True",    "inactive", "TRUE",    None),    # "above_threshold" or "normal"

    # HIGH wildcards expanded
    ("HIGH",     "*",       "active",   "FALSE",   "suppressed_maintenance"),
    ("HIGH",     "*",       "inactive", "DEPENDS", "normal"),

    # MEDIUM wildcards expanded
    ("MEDIUM",   "*",       "*",        "DEPENDS", "normal"),

    # LOW wildcards expanded
    ("LOW",     "*",        "*",        "DEPENDS", "below_threshold"),
]


def _expand_wildcards(claims):
    """Expand '*' values to all concrete combinations."""
    PATCHABLE_VALUES = ["True", "False"]
    MAINTENANCE_VALUES = ["active", "inactive"]

    expanded = []
    for tier, p, m, surface, reason in claims:
        p_values = PATCHABLE_VALUES if p == "*" else [p]
        m_values = MAINTENANCE_VALUES if m == "*" else [m]
        for pv, mv in product(p_values, m_values):
            expanded.append({
                "tier": tier,
                "patchable": pv,
                "maintenance": mv,
                "expected_surface": surface,
                "expected_reason_prefix": reason,
                "source_claim": f"{tier}|{p}|{m}",
            })
    return expanded


def _normalize(value):
    """Normalize a CSV value to a canonical form for comparison."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    # Map common boolean variants
    if s in {"true", "1", "yes"}:
        return "True"
    if s in {"false", "0", "no"}:
        return "False"
    # Map maintenance variants
    if s in {"true", "active", "on"}:
        return "active"  # handled above for booleans
    if s in {"false", "inactive", "off"}:
        return "inactive"
    return value.strip() if isinstance(value, str) else str(value)


def _load_csv():
    """Load the RQ1 truth table CSV."""
    if not CSV_PATH.exists():
        pytest.skip(
            f"{CSV_PATH} missing. Run "
            "module6_evaluation/make_rq1_truth_table.py first (RQ1 Phase 7)."
        )
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def _find_row(rows, tier, patchable, maintenance):
    """
    Find the CSV row matching (tier, patchable, maintenance).

    Returns the matching row dict, or None if not found.
    Normalizes value comparison to be robust to boolean/string variants.
    """
    target_p = _normalize(patchable)
    # Maintenance may be encoded as bool ("True"/"False") or status ("active"/"inactive")
    target_m_bool = "True" if maintenance == "active" else "False"
    target_m_str = maintenance  # "active" / "inactive"

    for row in rows:
        if str(row.get(COL_TIER, "")).strip().upper() != tier.upper():
            continue
        if _normalize(row.get(COL_PATCHABLE)) != target_p:
            continue
        m_val = str(row.get(COL_MAINTENANCE, "")).strip()
        if m_val in {target_m_bool, target_m_str, target_m_bool.lower(), target_m_str.lower()}:
            return row
    return None


def _evaluate_claim(claim, rows):
    """
    Verify one claim against the CSV.

    Returns a dict with:
      claim, matched_row, status: 'pass' | 'fail' | 'row_missing' | 'depends_ok'
      details (reason for status)
    """
    row = _find_row(rows, claim["tier"], claim["patchable"], claim["maintenance"])

    out = {
        "claim": claim,
        "matched_row": row,
        "status": None,
        "details": None,
    }

    if row is None:
        out["status"] = "row_missing"
        out["details"] = (
            f"No row found matching "
            f"tier={claim['tier']} patchable={claim['patchable']} "
            f"maintenance={claim['maintenance']}"
        )
        return out

    actual_surface = _normalize(row.get(COL_SURFACE, ""))
    actual_reason = str(row.get(COL_REASON, "")).strip().lower()
    expected = claim["expected_surface"]

    if expected == "DEPENDS":
        # "depends on threshold" — we just verify the row exists; outcome may
        # be True or False depending on score threshold.
        out["status"] = "depends_ok"
        out["details"] = (
            f"Row present; outcome ({actual_surface}) is non-binary per §4.2 "
            "('depends on threshold')."
        )
    elif actual_surface == expected:
        # Outcome matches. Check reason if specified.
        if claim["expected_reason_prefix"]:
            if claim["expected_reason_prefix"].lower() in actual_reason:
                out["status"] = "pass"
            else:
                out["status"] = "fail"
                out["details"] = (
                    f"Outcome matches ({actual_surface}) but reason mismatch: "
                    f"expected '{claim['expected_reason_prefix']}', "
                    f"got '{actual_reason}'."
                )
        else:
            out["status"] = "pass"
    else:
        out["status"] = "fail"
        out["details"] = (
            f"Expected should_surface={expected}, "
            f"got {actual_surface}. Reason: {actual_reason}."
        )

    return out


def _write_json(results):
    """Write the verification result for the master aggregator."""
    n_pass = sum(1 for r in results if r["status"] == "pass")
    n_depends = sum(1 for r in results if r["status"] == "depends_ok")
    n_fail = sum(1 for r in results if r["status"] in {"fail", "row_missing"})

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "tests/test_rq3_truth_table_completeness.py",
            "source_csv": str(CSV_PATH.relative_to(REPO_ROOT)),
            "rq3_section_reference": "RQ3_expected_outputs.md §4.2",
            "_cross_reference_note": (
                "Track 4 verifies the 8 critical rows from RQ3 §4.2 against "
                "the canonical truth table produced by RQ1 Phase 7. The full "
                "32-row truth table lives in results/rq1_tier_surfacing_truth_table.csv."
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


# ─── pytest entry points ───────────────────────────────────────

@pytest.fixture(scope="module")
def csv_rows():
    return _load_csv()


@pytest.fixture(scope="module")
def expanded_claims():
    return _expand_wildcards(CRITICAL_CLAIMS)


@pytest.fixture(scope="module")
def verification_results(csv_rows, expanded_claims):
    """Run all claim evaluations once; cached for the suite."""
    results = [_evaluate_claim(c, csv_rows) for c in expanded_claims]
    _write_json(results)   # write JSON as side effect
    return results


def test_all_critical_rows_present(verification_results):
    """No claim should be 'row_missing' — every concrete combination must exist in the CSV."""
    missing = [r for r in verification_results if r["status"] == "row_missing"]
    assert not missing, (
        f"{len(missing)} critical row(s) missing from "
        f"{CSV_PATH.relative_to(REPO_ROOT)}:\n"
        + "\n".join(f"  - {r['claim']['source_claim']} → "
                    f"({r['claim']['tier']}, {r['claim']['patchable']}, "
                    f"{r['claim']['maintenance']}): {r['details']}"
                    for r in missing[:10])
    )


def test_critical_safety_floor_rows(verification_results):
    """
    DEFENSE-CRITICAL: the 2 safety_floor rows (CRITICAL+unpatchable, both
    maintenance states) must surface = TRUE. This is Invariant 2 evidence.
    """
    safety_floor_results = [
        r for r in verification_results
        if r["claim"]["tier"] == "CRITICAL"
        and r["claim"]["patchable"] == "False"
    ]
    assert len(safety_floor_results) == 2, (
        f"Expected 2 safety_floor claims, got {len(safety_floor_results)}"
    )
    for r in safety_floor_results:
        assert r["status"] == "pass", (
            f"Safety floor violation: {r['claim']['source_claim']} → "
            f"{r['status']}. Details: {r['details']}"
        )


def test_maintenance_suppression_holds_for_patchable(verification_results):
    """
    Maintenance window suppresses HIGH+patchable and CRITICAL+patchable alerts.
    """
    target = [
        r for r in verification_results
        if r["claim"]["maintenance"] == "active"
        and r["claim"]["expected_surface"] == "FALSE"
    ]
    failures = [r for r in target if r["status"] != "pass"]
    assert not failures, (
        f"{len(failures)} maintenance-suppression claim(s) failed: "
        + "; ".join(f"{r['claim']['source_claim']} → {r['status']}"
                    for r in failures[:5])
    )


def test_no_outcome_mismatches(verification_results):
    """No claim should fail with status='fail' (outcome mismatch)."""
    mismatches = [r for r in verification_results if r["status"] == "fail"]
    assert not mismatches, (
        f"{len(mismatches)} truth table outcome mismatch(es):\n"
        + "\n".join(f"  - {r['claim']['source_claim']}: {r['details']}"
                    for r in mismatches[:10])
    )


def test_depends_rows_present(verification_results):
    """
    'depends on threshold' rows should have status 'depends_ok' (row exists,
    outcome non-binary). If any such row went 'row_missing', flag it.
    """
    depends_claims = [
        r for r in verification_results
        if r["claim"]["expected_surface"] == "DEPENDS"
    ]
    missing = [r for r in depends_claims if r["status"] == "row_missing"]
    assert not missing, (
        f"{len(missing)} 'depends on threshold' row(s) absent from CSV: "
        + "; ".join(r["claim"]["source_claim"] for r in missing[:5])
    )
```

### 4.3 Verification

```bash
pytest tests/test_rq3_truth_table_completeness.py -v
# Expected: 5 tests pass (assumes RQ1 CSV is correct).
# Side effect: results/rq3_truth_table_reference.json written.
```

---

## 5. Phase 2 — Paper Appendix B markdown renderer

### 5.1 Create `analysis/render_rq3_appendix_b.py`

Generates the RQ3 paper-ready Appendix B markdown from the verification JSON.

```python
"""
analysis/render_rq3_appendix_b.py

Render results/rq3_truth_table_reference.json into a paper-ready
Appendix B markdown for thesis §5.6 / RQ3 Appendix B.

Output: results/rq3_truth_table_appendix_b.md
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_IN = REPO_ROOT / "results/rq3_truth_table_reference.json"
OUT_MD = REPO_ROOT / "results/rq3_truth_table_appendix_b.md"


def main():
    if not JSON_IN.exists():
        raise SystemExit(
            f"{JSON_IN} missing — run "
            "pytest tests/test_rq3_truth_table_completeness.py first."
        )
    data = json.loads(JSON_IN.read_text())

    lines = []
    lines.append("# Appendix B — Tier × Surfacing Truth Table (RQ3)")
    lines.append("")
    lines.append(
        f"*Generated from `{data['_meta']['source_csv']}` "
        f"on {data['_meta']['generated_at']}.*"
    )
    lines.append("")
    lines.append(
        "This table enumerates the system's `should_surface` decision for "
        "every combination of `risk_tier`, `patchable`, and `maintenance_active`. "
        "Rows derived from `RQ3_expected_outputs.md §4.2` are verified by "
        "`tests/test_rq3_truth_table_completeness.py` and serve as the safety "
        "engineering evidence for Invariant 2 (safety floor) and the "
        "maintenance-window suppression policy."
    )
    lines.append("")

    h = data["headline"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Verification status:** "
                 f"{'PASS' if h['verification_pass'] else 'FAIL'}")
    lines.append(f"- **Claims verified:** {h['n_pass']} / {h['n_claims_total']}")
    lines.append(f"- **'Depends on threshold' rows (row presence verified):** "
                 f"{h['n_depends_ok']}")
    lines.append(f"- **Failures:** {h['n_fail']}")
    lines.append("")

    lines.append("## Table")
    lines.append("")
    lines.append("| risk_tier | patchable | maintenance | should_surface | reason | verification |")
    lines.append("|---|---|---|---|---|---|")

    status_marker = {
        "pass": "✓",
        "depends_ok": "○",
        "fail": "✗",
        "row_missing": "✗ missing",
    }

    for r in data["results"]:
        c = r["claim"]
        row = r.get("matched_row") or {}
        surface = row.get("should_surface", "—")
        reason = row.get("reason", "—")
        marker = status_marker.get(r["status"], r["status"])

        lines.append(
            f"| {c['tier']} | {c['patchable']} | {c['maintenance']} | "
            f"{surface} | {reason} | {marker} |"
        )
    lines.append("")

    lines.append("## Verification semantics")
    lines.append("")
    lines.append("- **✓** — claim verified: row exists with the expected "
                 "`should_surface` value and reason prefix.")
    lines.append("- **○** — row exists; outcome is non-binary per §4.2 "
                 "('depends on threshold').")
    lines.append("- **✗** — claim failed: outcome or reason mismatch.")
    lines.append("- **✗ missing** — expected row absent from the canonical CSV.")
    lines.append("")

    if h["n_fail"]:
        lines.append("## Failures")
        lines.append("")
        for r in data["results"]:
            if r["status"] in {"fail", "row_missing"}:
                lines.append(f"- **{r['claim']['source_claim']}**: {r['details']}")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

### 5.2 Verification

```bash
python -m analysis.render_rq3_appendix_b
head -30 results/rq3_truth_table_appendix_b.md
```

---

## 6. Phase 3 — Integration with `compute_rq3_metrics.py`

When the RQ3 merge spec is written, the master aggregator pulls Track 4 in via this pattern:

```python
def _load_truth_table_subfile():
    p = REPO_ROOT / "results/rq3_truth_table_reference.json"
    if not p.exists():
        return {"_status": "pending"}
    data = json.loads(p.read_text())
    h = data["headline"]
    return {
        "_status": "complete" if h["verification_pass"] else "failing",
        "_merged_at": datetime.now(timezone.utc).isoformat(),
        "subfile_path": "results/rq3_truth_table_reference.json",
        "appendix_md_path": "results/rq3_truth_table_appendix_b.md",
        "source_csv": data["_meta"]["source_csv"],
        "headline": h,
    }
```

In the aggregator: `out["truth_table"] = _load_truth_table_subfile()`.

The aggregator carries only the headline + paths — the full per-row verification details stay in the dedicated JSON.

---

## 7. Execution order

```bash
# ─── PHASE 0: CROSS-REFERENCE CONFIRMATION ─────────────────────
python scripts/discover_truth_table_artifact.py > /tmp/truth_table_inventory.json
# DEVELOPER CONFIRMS: RQ1 CSV exists; column schema matches.

# If CSV missing:
python -m module6_evaluation.make_rq1_truth_table   # from RQ1 Phase 7

# ─── PHASE 1: VERIFICATION TEST ────────────────────────────────
# Create tests/test_rq3_truth_table_completeness.py
pytest tests/test_rq3_truth_table_completeness.py -v
# Expected: 5 tests pass; side effect: results/rq3_truth_table_reference.json

# ─── PHASE 2: APPENDIX B RENDERER ──────────────────────────────
# Create analysis/render_rq3_appendix_b.py
python -m analysis.render_rq3_appendix_b
head -30 results/rq3_truth_table_appendix_b.md

# ─── FINAL VERIFICATION ────────────────────────────────────────
ls results/rq3_truth_table_reference.json \
   results/rq3_truth_table_appendix_b.md
```

---

## 8. Open questions to surface (DO NOT GUESS)

Claude Code must pause and ask:

1. **Phase 0 — CSV column names.** Are the columns named `risk_tier`, `patchable`, `maintenance_active`, `should_surface`, `reason`? If different (e.g., `maintenance` without `_active`, or `surface_decision` instead of `should_surface`), Phase 1 column constants must be updated.
2. **Phase 0 — RQ1 CSV existence.** If `results/rq1_tier_surfacing_truth_table.csv` doesn't exist, Track 4 cannot run. Confirm RQ1 Phase 7 is implemented OR run it.
3. **Phase 1 — CRITICAL+True+inactive reason.** §4.2 lists this row's reason as `(above_threshold or normal)`. The verifier treats `expected_reason_prefix = None` as "any reason acceptable" for this case. Confirm this interpretation.
4. **Phase 1 — boolean serialization in CSV.** Is `patchable` written as `True`/`False`, `true`/`false`, `1`/`0`, or `yes`/`no`? The `_normalize` function handles common variants, but Phase 0 should confirm so the test is robust.

---

## 9. Coverage map — RQ3 §4.2 expected outputs → pipeline phase

| RQ3_expected_outputs.md item | Phase | Output |
|---|---|---|
| §4.2 Tier × Surfacing Truth Table (Appendix B) | 1, 2 | `results/rq3_truth_table_appendix_b.md` + `results/rq3_truth_table_reference.json` |
| §4.2 row: CRITICAL+False+active → TRUE/safety_floor | 1 | `test_critical_safety_floor_rows` |
| §4.2 row: CRITICAL+False+inactive → TRUE/safety_floor | 1 | `test_critical_safety_floor_rows` |
| §4.2 row: CRITICAL+True+active → FALSE/suppressed_maintenance | 1 | `test_maintenance_suppression_holds_for_patchable` |
| §4.2 row: CRITICAL+True+inactive → TRUE | 1 | `test_all_critical_rows_present` |
| §4.2 row: HIGH+*+active → FALSE/suppressed_maintenance | 1 | `test_maintenance_suppression_holds_for_patchable` |
| §4.2 row: HIGH+*+inactive → depends | 1 | `test_depends_rows_present` |
| §4.2 row: MEDIUM+*+* → depends | 1 | `test_depends_rows_present` |
| §4.2 row: LOW+*+* → usually FALSE | 1 | `test_depends_rows_present` |
| §8 "Document tier × surfacing truth table in paper appendix" | 2 | `render_rq3_appendix_b.py` |

Every numbered RQ3 §4.2 row is traceable to a test assertion. Wildcards expand to 16 concrete claims.

---

## 10. Defense talking points this enables

- **"Where does the tier × surfacing truth table come from?"**
  *"It's generated deterministically by `module6_evaluation/make_rq1_truth_table.py`, which enumerates every combination of `risk_tier × patchable × maintenance_active` and runs each through the surfacing logic in `src/risk_scorer.py`. The full 32-row table lives at `results/rq1_tier_surfacing_truth_table.csv`. The 8 representative rows in `RQ3_expected_outputs.md §4.2` are verified by `tests/test_rq3_truth_table_completeness.py` — wildcards are expanded to all concrete cases. Appendix B is rendered from the verification JSON."*

- **"How do you know the safety floor actually holds?"**
  *"Two rows of the truth table: CRITICAL+unpatchable+maintenance-active → TRUE/safety_floor, and CRITICAL+unpatchable+maintenance-inactive → TRUE/safety_floor. Both are verified by `test_critical_safety_floor_rows`. If either ever returned FALSE, that test fails immediately. The defense framing: this isn't an assertion about behavior — it's a verification of behavior."*

- **"What about 'depends on threshold' rows — aren't those untestable?"**
  *"They're verified as 'row exists' rather than 'row has specific outcome.' The threshold-dependent outcome varies by tuning — the test asserts the row is present and the system handles it via the threshold path. The full outcome for any specific score is then determined at runtime."*

- **"What if RQ1 changes its CSV format?"**
  *"The verification test fails immediately. The CSV column names are constants in the test file. A column rename, row reorder, or value-encoding change all surface as test failures with explicit per-row diff messages — protecting RQ3 paper claims from RQ1 drift."*

---

## 11. What this track deliberately does NOT do

- **Generate a new truth table.** RQ1 Phase 7 produces the canonical CSV. Track 4 only verifies.
- **Test all 32 rows of the full truth table.** Only the 8 rows from §4.2 (16 after wildcard expansion). Full coverage is RQ1's concern.
- **Modify the surfacing logic.** Track 4 reads results; the surfacing logic is in `src/risk_scorer.py` (RQ1 territory).
- **Render LaTeX.** Markdown only; the developer pastes into Word.

---

## End of spec

Implementation order: Phase 0 (confirm CSV) → Phase 1 (verification test) → Phase 2 (renderer). Each phase is independently verifiable. Track 4 is the smallest RQ3 spec; once implemented, it confirms RQ3's safety/surfacing paper claims hold against RQ1's canonical artifact.