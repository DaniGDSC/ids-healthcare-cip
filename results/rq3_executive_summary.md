# RQ3 — Executive Summary

*Generated on 2026-05-21T10:29:15.671567+00:00 by `module6_evaluation/compute_rq3_metrics.py`.*

**Research Question:** RQ3 — Does the system support distributed security responsibility across hospital roles while maintaining clinical safety?

## Defense Summary (Read First)

- **No Auto Execution**: PASS — 4-layer defense verified (grep + imports + negative test + runtime mock; 46 production files scanned)
- **Audit Tamper Evident**: PASS — chain intact across 5 entries; schema completeness verified
- **Safety Floor Invariant**: PASS — CRITICAL+unpatchable rows verified surface=TRUE (truth table audit)
- **Architectural Invariants**: PASS — 9/9 enforced, 0 pending, 0 failing
- **Distributed Responsibility Empirical**: PASS — escalation rate A=0% vs B=80% (overall, n=50/50; p=3.22e-16, Cramer's V=0.8165). Path C: LLM-persona simulation.

## Sub-RQ Status

| Sub-RQ | Status |
|---|---|
| RQ3.1 invariants | `complete` |
| RQ3.2 audit integrity | `complete` |
| RQ3.3 no auto execution | `complete` |
| RQ3.4 truth table | `complete` |
| RQ3.5 user study | `complete` |
| **Overall** | **complete** |

## Targets

| Target | Value | Pass | Defense-critical |
|---|---|---|---|
| `all_invariants_pass` | True | PASS | yes |
| `audit_chain_intact` | True | PASS | yes |
| `audit_schema_complete` | True | PASS | yes |
| `no_auto_exec_audit_pass` | True | PASS | yes |
| `truth_table_completeness` | True | PASS | no |
| `safety_floor_holds` | True | PASS | yes |
| `escalation_chi2_overall` | 3.215262727387118e-16 | PASS | no |

## Cross-References

- **Full invariant catalog:** `results/rq3_invariant_evidence.md`
- **Truth table (Appendix B):** `results/rq3_truth_table_appendix_b.md`
- **Audit chain status:** `results/rq3_audit_chain_verification.json` + `results/rq3_audit_schema_audit.json`
- **No-auto-execution audit:** `results/rq3_no_auto_execution.json`
- **User study (Path C):** `analysis/outputs/rq3_user_study.json`
- **Detailed JSON:** `results/rq3_metrics.json`

