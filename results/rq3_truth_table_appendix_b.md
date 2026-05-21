# Appendix B - Tier x Surfacing Truth Table (RQ3)

*Generated from `results/rq1_tier_surfacing_truth_table.csv` on 2026-05-20T07:20:35.605502+00:00.*

This table enumerates the system's `should_surface` decision for every combination of `risk_tier`, `patchable`, and `maintenance_active`. Rows derived from `RQ3_expected_outputs.md §4.2` are verified by `tests/test_rq3_truth_table_completeness.py` and serve as the safety-engineering evidence for Invariant 2 (safety floor) and the maintenance-window suppression policy.

## Summary

- **Verification status:** PASS
- **Claims verified:** 6 / 16
- **'Depends on threshold' rows (presence verified):** 10
- **Failures:** 0

## Table

| risk_tier | patchable | maintenance | should_surface | reason | verification |
|---|---|---|---|---|---|
| CRITICAL | False | active | True | safety_floor | PASS |
| CRITICAL | False | inactive | True | safety_floor | PASS |
| CRITICAL | True | active | False | suppressed_maintenance | PASS |
| CRITICAL | True | inactive | True | above_threshold | PASS |
| HIGH | True | active | False | suppressed_maintenance | PASS |
| HIGH | False | active | False | suppressed_maintenance | PASS |
| HIGH | True | inactive | True | above_threshold | depends |
| HIGH | False | inactive | True | above_threshold | depends |
| MEDIUM | True | active | False | suppressed_maintenance | depends |
| MEDIUM | True | inactive | False | below_threshold | depends |
| MEDIUM | False | active | False | suppressed_maintenance | depends |
| MEDIUM | False | inactive | True | above_threshold | depends |
| LOW | True | active | False | suppressed_maintenance | depends |
| LOW | True | inactive | False | below_threshold | depends |
| LOW | False | active | False | suppressed_maintenance | depends |
| LOW | False | inactive | False | below_threshold | depends |

## Verification semantics

- **PASS** - row exists with the expected `should_surface` value and reason prefix.
- **depends** - row exists; outcome is non-binary per §4.2 ('depends on threshold').
- **FAIL** - outcome or reason mismatch.
- **MISSING** - expected row absent from the canonical CSV.

## Cross-references

- Canonical CSV: `results/rq1_tier_surfacing_truth_table.csv` (produced by `module6_evaluation/make_rq1_truth_table.py`).
- Spec reference: RQ3_expected_outputs.md §4.2 (via RQ3_TRUTH_TABLE_SPEC.md §4.1).
- Invariant 2 ('Safety floor') in `configs/invariants_manifest.yaml` is enforced by the CRITICAL+False rows in this table.

