# RQ3 Pipeline — Overview & Roadmap

**Project:** XAI-IDS-Healthcare
**Research Question:** *"Does the system support distributed security responsibility across hospital roles while maintaining clinical safety?"*
**Purpose of this document:** Single-page roadmap for the entire RQ3 evaluation pipeline. No code. Use alongside the detailed track specs (to be written following this overview).
**Status:** Pipeline structure locked. Track-specific specs pending.

---

## 0. How this document fits

You have three kinds of documents in this thesis project:

| Layer | Document | Audience |
|---|---|---|
| **Strategy** | `thesis_outline_latest.docx`, `RQ3_expected_outputs.md` | Defense committee, you, future-you |
| **Roadmap** | *this file* | Claude Code (orientation), you (planning) |
| **Implementation specs** | `RQ1_PIPELINE_SPEC.md`, RQ2 spec family, future RQ3 track specs | Claude Code (executable instructions) |

Claude Code should read this overview *before* picking up any individual RQ3 track spec — it provides the dependency picture, the cross-RQ overlap map, and answers questions like "what's already done by an RQ2 spec?"

---

## 1. RQ3 in one paragraph

RQ3 evaluates whether the system supports **distributed security responsibility** across hospital roles (IT Generalist, Biomed Engineer, Nurse Manager) while preserving **clinical safety constraints**. Unlike RQ2 (where the contribution is mostly the *explanation mechanism* of MVE), **RQ3's contribution is mostly architectural**: the design supports distributed responsibility by construction, and the safety properties are code-verifiable today. The remaining empirical work is the user study — which is shared with RQ2.c and gated on data collection.

---

## 2. What's different about RQ3

Three things distinguish RQ3 from the prior RQs:

**1. Architecture-heavy, evaluation-light.** RQ3's strongest claims (no auto-execution, audit integrity, safety floor, cross-role invariance) are deterministic tests on existing code. They don't depend on data, modeling, or external services — they pass or fail in seconds. This makes RQ3 the **defense-strongest** RQ.

**2. Cross-RQ overlap.** Several invariants serve both RQ2 and RQ3 (see §6 for the full cross-walk). The RQ3 specs should *reference* RQ2 outputs rather than duplicate them.

**3. Different statistical surface.** RQ3 adds a Chi-square test for **escalation rate** (binary/categorical), in addition to the Mann-Whitney tests RQ2 already specs for continuous metrics.

---

## 3. Pipeline tracks

### Track 1 — Invariant Evidence ⏳ pending spec

The architectural backbone. Aggregates the deterministic test results for all 9 invariants into a single JSON, with cross-references to the code/config that enforces each.

**Deliverables:**
- `analysis/compile_invariant_evidence.py` — pytest result aggregator
- `results/rq3_invariant_evidence.json` — structured pass/fail per invariant
- Documentation manifest tying each invariant to its enforcing artifact

**Why first:** highest defense value, no dependencies, runs in seconds.

**Estimated spec size:** ~600 lines.

### Track 2 — Audit Log Integrity ⏳ pending spec

Hash chain verification + schema completeness audit. The deliverable a HIPAA-aware reviewer reads to verify the "tamper-evident" claim.

**Deliverables:**
- `analysis/verify_audit_log_integrity.py` — chain verifier
- `analysis/audit_log_schema_completeness.py` — schema auditor
- `results/rq3_audit_integrity.json` — pass/fail + chain length + integrity report
- Test: `tests/test_step16_audit_integrity.py` (extended)

**Why second:** novel work, defense-critical for the "tamper-evident" claim, depends only on whether audit logs exist (a single empty log file is also valid).

**Estimated spec size:** ~500 lines.

### Track 3 — No-Auto-Execution Triple Layer ⏳ pending spec

Three-layer defense from the expected outputs `§9 reviewer-anticipated questions`:
- Layer A: code grep for execution patterns
- Layer B: import statement grep
- Layer C: negative test in CI

**Deliverables:**
- `analysis/audit_no_auto_execution.py` — runs the three layers, emits structured result
- `tests/negative_tests.py::test_no_automated_blocking` — verify exists; extend if missing
- `results/rq3_no_auto_execution.json` — three-layer report

**Why third:** small, important, fast. The grep test in §3.1 of expected outputs is the literal specification.

**Estimated spec size:** ~300 lines.

### Track 4 — Tier × Surfacing Truth Table ⏳ pending spec

Overlaps RQ1 (the RQ1_PIPELINE_SPEC `make_rq1_truth_table.py` was already specced). For RQ3, this is *thesis Appendix B*. Same artifact, different paper section.

**Deliverables:**
- `module6_evaluation/make_rq1_truth_table.py` — **already specced in RQ1_PIPELINE_SPEC.md**
- `results/rq1_tier_surfacing_truth_table.{csv,md}` — already produced
- *RQ3 specifically requires:* a paper-rendering helper that includes the table verbatim in §5.6 (Safety Validation) of the thesis

**Why fourth:** mostly already done. RQ3 spec for this is mainly the *reference* and the per-row column meaning.

**Estimated spec size:** ~250 lines (mostly cross-references, minor rendering).

### Track 5 — User Study (RQ3 lens) ⏳ pending spec, gated on data

The shared user study with RQ2.c, viewed through RQ3's lens. RQ2 cared about *role-differentiated outputs* (does MVE produce different explanations per role?). RQ3 cares about *role-distributed responsibility* (does the workflow work when roles are distributed?). Same data, different framings, mostly different metrics:

- **Same as RQ2.c:** decision time, decision accuracy, confidence — per role
- **RQ3-specific:** **appropriate escalation rate** (new metric; Chi-square)
- **RQ3 reuses:** the existing `compute_rq2c_per_role.py` analysis with extension for escalation

**Deliverables:**
- `analysis/compute_rq3_escalation.py` — Chi-square test for escalation rate
- `analysis/compute_rq3_per_role.py` — wraps the RQ2c analysis with RQ3 framing
- `analysis/outputs/rq3_user_study.json` — final analysis

**Why last:** data-gated. Also depends on Track 4 (cross-role consistency from RQ2 compliance spec).

**Estimated spec size:** ~600 lines.

### Merge + Figures + CI

**Phase 6 — `compute_rq3_metrics.py`** aggregates everything into `results/rq3_metrics.json` (mirror of RQ1/RQ2 pattern).

**Phase 7 — `make_rq3_figures.py`** produces:
- Invariant pass/fail matrix
- Audit chain growth + tamper detection sample
- Per-role × per-metric (similar to RQ2 figure but with escalation rate added)

**Phase 8 — CI verification** via `tests/acceptance_tests.py::test_rq3_targets_met`.

---

## 4. Dependency graph

```
                                                                          
  PREREQUISITES                                                           
  ─────────────                                                           
  RQ1 complete (risk_scores.npz schema v1.1, truth table generated)       
  RQ2 specs available (we reference, don't duplicate)                      
  Test suite passing baseline                                              
                                                                          
                                                                          
  TRACK 1 — INVARIANT EVIDENCE                                            
  ────────────────────────────                                            
                                                                          
    pytest suite outputs                                                  
        │                                                                 
        ▼                                                                 
    analysis/compile_invariant_evidence.py                                
        │                                                                 
        ▼                                                                 
    results/rq3_invariant_evidence.json                                   
                                                                          
                                                                          
  TRACK 2 — AUDIT LOG INTEGRITY                                           
  ─────────────────────────────                                           
                                                                          
    logs/llm_audit.jsonl (Mode A)                                         
    logs/decision_audit.jsonl (per-alert)                                 
        │                                                                 
        ├──→ verify_audit_log_integrity.py                                
        │       │                                                         
        │       ▼                                                         
        │   chain verification report                                     
        │                                                                 
        └──→ audit_log_schema_completeness.py                             
                │                                                         
                ▼                                                         
            schema audit                                                  
                │                                                         
                ▼                                                         
            results/rq3_audit_integrity.json                              
                                                                          
                                                                          
  TRACK 3 — NO-AUTO-EXECUTION                                             
  ───────────────────────────                                             
                                                                          
    pipeline/module5_response/ source                                     
        │                                                                 
        ├──→ Layer A: grep -rnE 'subprocess|os.system|...'                
        ├──→ Layer B: grep -rn '^import subprocess|^from subprocess'      
        └──→ Layer C: pytest tests/negative_tests.py                      
                │                                                         
                ▼                                                         
            results/rq3_no_auto_execution.json                            
                                                                          
                                                                          
  TRACK 4 — TIER × SURFACING TRUTH TABLE                                  
  ──────────────────────────────────────                                  
                                                                          
    src/risk_scorer.py logic                                              
        │                                                                 
        ▼                                                                 
    (already produced by RQ1 Phase 7)                                     
    results/rq1_tier_surfacing_truth_table.{csv,md}                       
        │                                                                 
        ▼                                                                 
    RQ3 paper §5.6 includes verbatim                                      
                                                                          
                                                                          
  TRACK 5 — USER STUDY (RQ3 LENS) — DATA-GATED                            
  ────────────────────────────────────────────                            
                                                                          
    survey/study_responses_*.json (collected via RQ2.c)                   
        │                                                                 
        ├──→ compute_rq2c_per_role.py (RQ2 spec)                          
        │       └─→ analysis/outputs/rq2c_per_role.json                   
        │                                                                 
        ├──→ compute_rq3_escalation.py (NEW)                              
        │       └─→ analysis/outputs/rq3_escalation.json                  
        │                                                                 
        └──→ compute_rq3_per_role.py (NEW; wraps RQ2c + escalation)       
                └─→ analysis/outputs/rq3_user_study.json                  
                                                                          
                                                                          
  MERGE + FIGURES + CI                                                    
  ────────────────────                                                    
                                                                          
    All track outputs ──→ compute_rq3_metrics.py [Phase 6]                
                          └──→ results/rq3_metrics.json                   
                                                                          
    rq3_metrics.json ──→ make_rq3_figures.py [Phase 7]                    
                          └──→ results/figures/rq3_*.pdf                  
                                                                          
    All artifacts ──→ tests/acceptance_tests.py [Phase 8]                 
```

---

## 5. Critical path

The path that gates "RQ3 is defensible":

```
Track 1 (Invariant evidence) — strongest defense leverage
Track 2 (Audit integrity)    — HIPAA-relevant
Track 3 (No-auto-execution)  — single most-asked reviewer question
Phase 6 (Aggregator)          — canonical source of truth
```

Track 4 is mostly already done (RQ1 Phase 7). Track 5 is data-gated.

If only the critical path completes, **RQ3 is still defensible** — because RQ3's contribution is primarily architectural, the empirical (Track 5) results are nice-to-have, not load-bearing. This is the opposite of RQ2 where empirical evidence was central.

---

## 6. Cross-RQ overlap map

Per `RQ3_expected_outputs.md §10.3`:

| Property | RQ1 | RQ2 | RQ3 |
|---|---|---|---|
| BENIGN_MEDIAN imputation | ✓ | | |
| Invariant 1 (DAE only elevates) | ✓ | | |
| **Invariant 2 (safety floor)** | ✓ | | ✓ |
| **Invariant 3 (no auto-execution)** | | | ✓ |
| **Invariant 4 (audit complete)** | | | ✓ |
| Invariant 5 (Layer 1 SHAP) | | ✓ | |
| **Invariant 6 (role auth)** | | ✓ | ✓ |
| **Invariant 7 (DO_NOT)** | | ✓ | ✓ |
| Invariant 8 (Layer 2 tier) | | ✓ | |
| **Invariant 9 (shared anchor)** | | ✓ | ✓ |

**Shared invariants (RQ2 ∩ RQ3):** 2, 6, 7, 9. These are tested *once* (in RQ2 spec files) and *referenced* by RQ3's Track 1.

**RQ3-only invariants:** 3, 4. These get new tests in RQ3 Tracks 2 and 3.

**RQ1-only invariants:** 1, 5, 8. Already covered.

The RQ3 Track 1 spec will inventory these explicitly so the evidence file points to the right test for each invariant.

---

## 7. File inventory (full RQ3)

### Pending specs (Tracks 1–5)
- `analysis/compile_invariant_evidence.py` (T1)
- `analysis/verify_audit_log_integrity.py` (T2)
- `analysis/audit_log_schema_completeness.py` (T2)
- `analysis/audit_no_auto_execution.py` (T3)
- `analysis/compute_rq3_escalation.py` (T5)
- `analysis/compute_rq3_per_role.py` (T5)

### Pending specs (Merge layer)
- `module6_evaluation/compute_rq3_metrics.py` (Phase 6)
- `module6_evaluation/make_rq3_figures.py` (Phase 7)

### Already specced or produced elsewhere
- `tests/test_step13_cross_role_consistency.py` — RQ2_COMPLIANCE_SPEC Phase 3
- `tests/test_step16_audit_integrity.py` — referenced; may exist already
- `tests/negative_tests.py::test_no_automated_blocking` — referenced; verify exists
- `module6_evaluation/make_rq1_truth_table.py` — RQ1_PIPELINE_SPEC Phase 7

### May already exist (verify before writing)
- `tests/test_step16_audit_integrity.py`
- `tests/negative_tests.py`
- `tests/test_step15_role_consistency.py`
- `tests/test_safe_failure.py` (8 tests existing)
- `tests/test_step10_surfacing_logic.py`

### Produced artifacts (full RQ3)

```
results/
├── rq3_metrics.json                          ← Phase 6 (canonical)
├── rq3_invariant_evidence.json               ← Track 1
├── rq3_audit_integrity.json                  ← Track 2
├── rq3_no_auto_execution.json                ← Track 3
├── (rq1_tier_surfacing_truth_table.* exists) ← Track 4 (from RQ1)
└── figures/
    ├── rq3_invariant_matrix.pdf              ← Phase 7
    ├── rq3_audit_chain_health.pdf            ← Phase 7
    └── rq3_per_role_with_escalation.pdf      ← Phase 7

analysis/outputs/
├── rq3_escalation.json                       ← Track 5
└── rq3_user_study.json                       ← Track 5
```

---

## 8. Execution order (full RQ3)

```bash
# ─── BLOCK A: PREREQUISITES ────────────────────────────────────
# 1. RQ1 pipeline complete (truth table at results/rq1_tier_surfacing_truth_table.csv)
# 2. RQ2 spec implementations underway (Tracks 1-3 give us shared invariant tests)
# 3. Existing tests passing baseline

# ─── BLOCK B: TRACK 1 — INVARIANT EVIDENCE ─────────────────────
# Phase T1-0: verify all referenced test files exist
# Phase T1-1: write analysis/compile_invariant_evidence.py
python -m analysis.compile_invariant_evidence
pytest tests/test_invariant_evidence.py -v

# ─── BLOCK C: TRACK 2 — AUDIT LOG INTEGRITY ────────────────────
# Phase T2-0: confirm audit log paths (logs/llm_audit.jsonl, etc.)
# Phase T2-1: verify_audit_log_integrity.py
# Phase T2-2: audit_log_schema_completeness.py
python -m analysis.verify_audit_log_integrity
python -m analysis.audit_log_schema_completeness
pytest tests/test_step16_audit_integrity.py -v

# ─── BLOCK D: TRACK 3 — NO-AUTO-EXECUTION ──────────────────────
python -m analysis.audit_no_auto_execution
pytest tests/negative_tests.py::test_no_automated_blocking -v

# ─── BLOCK E: TRACK 4 — TIER × SURFACING (mostly RQ1) ──────────
# Verify results/rq1_tier_surfacing_truth_table.csv exists
ls results/rq1_tier_surfacing_truth_table.csv

# ─── BLOCK F: TRACK 5 — USER STUDY (data-gated) ────────────────
# After RQ2.c data collection + analysis completes:
python -m analysis.compute_rq3_escalation
python -m analysis.compute_rq3_per_role

# ─── BLOCK G: MERGE + FIGURES + CI ─────────────────────────────
python -m module6_evaluation.compute_rq3_metrics
python -m module6_evaluation.make_rq3_figures
pytest tests/acceptance_tests.py::test_rq3_targets_met -v

# ─── FINAL VERIFICATION ────────────────────────────────────────
pytest tests/ -v
ls results/rq3_*.json results/figures/rq3_*.pdf
```

Blocks B, C, D can run in parallel (no dependencies between them). E is essentially a "verify existing artifact" step. F is gated on data. G is the merge.

---

## 9. Coverage map: RQ3_expected_outputs.md → pipeline track

| RQ3_expected_outputs.md item | Track | Phase |
|---|---|---|
| **§1.1** Role-based explanation routing (Inv 6) | T1 | references RQ2_COMPLIANCE Phase 3 |
| **§1.1** Tier recommendation routing (`tier_routing.yaml`) | T1 | invariant manifest entry |
| **§1.1** Action authorization (`role_action_authorization.yaml`) | T1 | invariant manifest entry |
| **§1.1** No auto-execution (Inv 3) | T3 | dedicated track |
| **§1.1** Audit trail per role (Inv 4) | T2 | dedicated track |
| **§1.1** Cross-role severity invariance (Inv 6) | T1 | references RQ2_COMPLIANCE Phase 3 |
| **§1.1** Shared anchor (Inv 9) | T1 | references RQ2_COMPLIANCE Phase 3 |
| **§2** A/B study aggregate group comparison | T5 | shared with RQ2.c |
| **§2** Per-role breakdown | T5 | extends RQ2.c per-role |
| **§2** Appropriate escalation rate (Chi-square) | T5 | RQ3-specific |
| **§3.1** No-auto-execution grep + import + test | T3 | 3-layer report |
| **§3.2** Audit trail completeness | T2 | schema audit |
| **§3.2** `verify_audit_log_integrity()` = True | T2 | chain verifier |
| **§4.1** Clinical safety constraints (Inv 1-9) | T1 | invariant manifest |
| **§4.2** Tier × Surfacing truth table | T4 | already produced by RQ1 |
| **§7** Test coverage list | T1 | each test referenced by invariant |
| **§8** Pre-defense checklist | T1+T2+T5 | aggregated in Phase 6 |
| **§10.3** Cross-RQ safety guarantees | overview §6 | this document |

Every numbered RQ3 item is traceable to a track. Cross-RQ overlap is acknowledged at the overview level rather than duplicated in track specs.

---

## 10. Decisions already locked (do not revisit)

Inherited from RQ2 spec work, where relevant:

| Decision | Resolution | Source |
|---|---|---|
| Operator roles | IT_GENERALIST, BIOMED_ENGINEER, NURSE_MANAGER | RQ2_USER_STUDY_SPEC |
| Statistical test for continuous metrics | Mann-Whitney U two-sided | RQ2_USER_STUDY_SPEC |
| Multiple-comparisons correction | None applied; disclosure in methodology_notes | RQ2_USER_STUDY_SPEC |
| Effect size | Cliff's delta | RQ2_USER_STUDY_SPEC |
| Sample size threshold | N=10 per cell warning flag | RQ2_USER_STUDY_SPEC |
| Cross-role consistency test | Implemented in RQ2_COMPLIANCE Phase 3 | Test file `test_step13_cross_role_consistency.py` |
| Audit log location | Per Phase 0 discovery in RQ2_COMPLIANCE | Confirmed-pending |
| PHI flow control | RQ2_COMPLIANCE Phase 1 covers this | Cross-reference |

RQ3-specific locked decisions (none yet — all to be discussed in track specs):

| Decision | Status |
|---|---|
| Audit log hash algorithm | Per `RQ3_expected_outputs.md §3.2`: SHA256 |
| Chi-square for escalation rate | Per `RQ3_expected_outputs.md §2.2` (new for RQ3 vs RQ2.c's Mann-Whitney) |
| Escalation definition | TO BE DEFINED in T5 spec — what constitutes "appropriate escalation"? |
| Out-of-scope scope statement | Steps 17 (outcome tracking) and 18 (continuous improvement) are future work per `§6.3` |

---

## 11. Open questions (per track)

These are the things Claude Code will need to ask before implementing each track. Surfacing them up front so they aren't surprises.

### Track 1 (Invariant evidence)
1. Do all 9 invariant tests currently exist? Which ones need creation?
2. How are pytest results captured into JSON? (`pytest --json-report`? Custom collector?)
3. What's the invariant manifest's source of truth — a YAML file or a Python module?

### Track 2 (Audit log integrity)
1. Audit log location and rotation policy (single file? daily rotation?).
2. Hash chain seed entry — what's `previous_hash` for the first entry?
3. Schema completeness validation: hardcoded required fields or read from `config/audit_log_schema.yaml`?
4. What does the verifier do on a chain break — fail immediately or report all breaks?

### Track 3 (No-auto-execution)
1. Confirm grep targets `pipeline/module5_response/` per `RQ3_expected_outputs.md §3.1`.
2. False-positive handling — what about `subprocess` in docstrings or simulated-attack tests?
3. Does `tests/negative_tests.py::test_no_automated_blocking` already exist? If not, creating it is part of the spec.

### Track 4 (Truth table)
1. Confirm `results/rq1_tier_surfacing_truth_table.csv` is the canonical artifact (from RQ1_PIPELINE_SPEC Phase 7).
2. For RQ3 paper rendering, does the table need any RQ3-specific column (e.g., "responsibility role")?

### Track 5 (User study)
1. **Escalation definition** — biggest open question. What action codes count as "escalation"? Likely needs a YAML manifest similar to `correct_action`.
2. Per-role expected escalation rate — is this scenario-dependent or globally pre-registered?
3. Chi-square contingency table shape — 2x2 (escalated/not × A/B) or 3x2 (role × A/B)?
4. Same Path C decision as RQ2.c affects framing.

---

## 12. Recommended next move

Three reasonable paths from here:

**Path 1 — Implement RQ1+RQ2, then spec RQ3 tracks one by one:**
Hand the existing 9 specs (RQ1 + 8 RQ2) to Claude Code. As Claude Code finishes each, spec the next RQ3 track. This serializes well but stretches calendar time.

**Path 2 — Spec all RQ3 tracks now, in parallel with RQ1+RQ2 implementation:**
Write the 5 RQ3 track specs while Claude Code implements RQ1+RQ2. By the time RQ1+RQ2 is done, the RQ3 specs are ready. This maximizes parallelism.

**Path 3 — Spec the critical path now (Tracks 1, 2, 3), defer 4 and 5:**
The critical path is the strongest-defense subset. Specs 4 (mostly already done) and 5 (data-gated) can wait.

Recommendation: **Path 3** — spec Tracks 1, 2, 3 next, in that order. Track 4 is a 250-line cross-reference document that takes minimal time to write later. Track 5 doesn't need a final spec until Track 4 of RQ2 (also data-gated) is resolved.

---

## End of overview

Track-specific specs to follow:

- ⏳ `RQ3_INVARIANT_EVIDENCE_SPEC.md` — Track 1 (recommended next)
- ⏳ `RQ3_AUDIT_INTEGRITY_SPEC.md` — Track 2
- ⏳ `RQ3_NO_AUTO_EXECUTION_SPEC.md` — Track 3
- ⏳ `RQ3_TRUTH_TABLE_SPEC.md` — Track 4 (mostly cross-reference)
- ⏳ `RQ3_USER_STUDY_SPEC.md` — Track 5 (data-gated)
- ⏳ `RQ3_MERGE_AND_FIGURES_SPEC.md` — Phases 6-8