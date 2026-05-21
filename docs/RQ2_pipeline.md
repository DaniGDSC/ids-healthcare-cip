# RQ2 Pipeline — Overview & Roadmap

**Project:** XAI-IDS-Healthcare
**Research Question:** *"Can MVE provide role-tailored security explanations enabling non-specialist hospital stakeholders to make informed threat triage decisions?"*
**Purpose of this document:** Single-page roadmap for the entire RQ2 evaluation pipeline. No code. Use alongside the detailed track specs (only `RQ2_FAITHFULNESS_SPEC.md` exists so far; others to be written).
**Status:** Pipeline structure locked. Detailed specs: Track 1 complete, Tracks 2–5 + merge pending.

---

## 0. How this document fits

You have three kinds of documents in this thesis project:

| Layer | Document | Audience |
|---|---|---|
| **Strategy** | `thesis_outline_latest.docx`, `RQ2_expected_outputs.md` | Defense committee, you, future-you |
| **Roadmap** | *this file* | Claude Code (orientation), you (planning) |
| **Implementation specs** | `RQ1_PIPELINE_SPEC.md`, `RQ2_FAITHFULNESS_SPEC.md`, future track specs | Claude Code (executable instructions) |

Claude Code should read this overview *before* picking up any individual track spec — it provides the dependency picture and answers questions like "what artifacts must already exist?" and "what's downstream of this script?"

---

## 1. RQ2 in one paragraph

RQ2 evaluates the **Minimum Viable Explanation** (MVE) layer — the role-adapted, three-layer natural-language output that turns a detection alert into something a clinician, biomed engineer, or IT generalist can act on. RQ2 has five sub-questions (a–e), of which one is rescoped to future work (d). The remaining four sub-questions cluster into **five tracks** of computational work, plus a merge layer.

---

## 2. Pipeline tracks

### Track 1 — Faithfulness (RQ2.b) ✅ specced

Measures whether MVE explanations are *faithful* to the underlying detection — i.e., whether SHAP is stable under input perturbation, and whether MVE Layer 1 actually mentions the features SHAP says drove the alert.

**Spec:** `RQ2_FAITHFULNESS_SPEC.md` (1,370 lines, ready)
**Artifacts produced:**
- `results/rq2_shap_stability.json`
- `results/rq2_mve_shap_alignment.json`
- `common/feature_metadata.py`, `common/perturbation.py`
- Caching at `results/cache/shap_stability_cache.npz`

### Track 2 — MITRE Grounding (RQ2.e) ⏳ pending spec

Validates that every attack category maps to a MITRE ATT&CK technique, and that MVE Layer 1 outputs actually reference those techniques. Two distinct deliverables:

- **Config audit:** does `config/attack_to_mitre_mapping.yaml` cover every attack category? Are confidence levels and framework versions set?
- **Grounding rate:** for each MVE output, does Layer 1 mention the mapped MITRE technique (by ID or human name)?

**Spec status:** not written. Estimated ~400 lines.
**Artifacts produced:**
- `results/rq2_mitre_audit.json` — config completeness
- `results/rq2_mitre_grounding.json` — per-alert reference rate

### Track 3 — Compliance & Cross-Role (RQ2.a) ⏳ pending spec

The HIPAA/audit-trail compliance track. Four components, each defense-critical:

- **Compliance mapping table** — literature requirement → MVE design choice → evidence file. Static markdown for paper appendix.
- **Word budget audit** — verifies MVE outputs stay within the 150-word total budget.
- **PHI flow control test** — scans Mode A audit logs to confirm no PHI ever crosses to the LLM. Highest-stakes test in RQ2.
- **Cross-role consistency test** — same alert generated for three roles; asserts Invariant 6 (shared anchor) and Invariant 9 (action-authorization scoping).

**Spec status:** not written. Estimated ~600 lines (4 sub-scripts/tests).
**Artifacts produced:**
- `results/rq2_compliance_mapping.md`
- `results/rq2_word_budget_audit.json`
- `tests/test_phi_not_in_llm_prompt.py`
- `tests/test_step13_cross_role_consistency.py`

### Track 4 — User Study (RQ2.c) ⏳ pending spec, gated on data

The empirical heart of RQ2. Three roles × three outcome measures (decision time, accuracy, confidence) × Group A (with MVE) vs Group B (without). Mann-Whitney U for each cell.

**Dependencies:** user study data must be collected first. `survey/study_loader.py` and `survey/study_analysis.py` are referenced in ARCHITECTURE.md and may already exist — Claude Code should verify before writing new code.

**Spec status:** not written. Estimated ~500 lines (4 scripts).
**Artifacts produced:**
- `survey/m5_result.yaml` — Mann-Whitney aggregates
- `analysis/outputs/rq2c_per_role.json` — per-role × per-metric structured results
- `analysis/outputs/rq2c_themes.json` — qualitative themes (optional)

### Track 5 — Failure Mode Catalog (RQ2.d, rescoped) ⏳ pending spec

Per the thesis outline, RQ2.d has been moved to future work in Section 7.2.3. What remains is a single-round failure-mode *catalog* — observations, not improvement claims. This pulls from outputs of Tracks 1, 3, and 4.

**Spec status:** not written. Estimated ~200 lines.
**Artifacts produced:**
- `results/rq2_failure_mode_catalog.json`

### Merge + Figures + CI

**Phase 18** — `compute_rq2_metrics.py` aggregator (mirrors RQ1 pattern) reads all track outputs and produces canonical `results/rq2_metrics.json`.

**Phase 19** — `make_rq2_figures.py` produces SHAP stability histogram, Mode A vs B alignment chart, per-role user study chart.

**Phase 20** — CI verification: extends `tests/acceptance_tests.py` with RQ2 target assertions.

---

## 3. Dependency graph

```
                                                                                
  PREREQUISITES                                                                 
  ─────────────                                                                 
  RQ1 complete ──→ risk_scores.npz schema v1.1                                  
  Module 5 batch run ──→ mve_outputs.jsonl                                      
  Study data collection (Track 4 only)                                          
                                                                                
                                                                                
  TRACK 1 (FAITHFULNESS)                                                        
  ──────────────────────                                                        
                                                                                
    feature_metadata.py [Phase 0] ──┐                                           
                                    ├──→ perturbation.py [Phase 1] ─┐          
    shap_explainer.py [Phase 2] ────┘                                │          
                                                                     │          
                                          compute_shap_stability.py ◄┘          
                                          [Phase 3]                             
                                              │                                 
                                              ▼                                 
                                          shap_stability_cache.npz              
                                              │                                 
                                              ▼                                 
                              compute_mve_shap_alignment.py [Phase 4]           
                              (also needs mve_outputs.jsonl)                    
                                                                                
                                                                                
  TRACK 2 (MITRE GROUNDING)                                                     
  ─────────────────────────                                                     
                                                                                
    config/attack_to_mitre_mapping.yaml                                         
        │                                                                       
        ├──→ audit_mitre_config.py [Phase 7]                                    
        │                                                                       
        └──→ compute_mitre_grounding.py [Phase 8]                               
             (also needs mve_outputs.jsonl)                                     
                                                                                
                                                                                
  TRACK 3 (COMPLIANCE)                                                          
  ────────────────────                                                          
                                                                                
    make_rq2_compliance_table.py [Phase 9]                                      
        (independent; static doc)                                               
                                                                                
    mve_outputs.jsonl                                                           
        │                                                                       
        └──→ audit_word_budgets.py [Phase 10]                                   
                                                                                
    Mode A audit logs                                                           
        │                                                                       
        └──→ test_phi_not_in_llm_prompt.py [Phase 11]                           
                                                                                
    MVE pipeline runnable                                                       
        │                                                                       
        └──→ test_step13_cross_role_consistency.py [Phase 12]                   
                                                                                
                                                                                
  TRACK 4 (USER STUDY)                                                          
  ────────────────────                                                          
                                                                                
    survey/study_responses_*.json (RAW)                                         
        │                                                                       
        ├──→ study_loader.py [Phase 13]                                         
        │       │                                                               
        │       ▼                                                               
        └──→ study_analysis.py [Phase 14] ──→ m5_result.yaml                    
                  │                                                             
                  ▼                                                             
              compute_rq2c_per_role.py [Phase 15] ──→ rq2c_per_role.json        
                  │                                                             
                  ▼                                                             
              analyze_qualitative_themes.py [Phase 16] ──→ rq2c_themes.json     
                                                                                
                                                                                
  TRACK 5 (FAILURE MODE CATALOG)                                                
  ──────────────────────────────                                                
                                                                                
    Outputs from Tracks 1, 3, 4 ──→ compile_failure_modes.py [Phase 17]         
                                    └──→ rq2_failure_mode_catalog.json          
                                                                                
                                                                                
  MERGE + FIGURES + CI                                                          
  ────────────────────                                                          
                                                                                
    All track outputs ──→ compute_rq2_metrics.py [Phase 18]                     
                          └──→ rq2_metrics.json                                 
                                                                                
    rq2_metrics.json + caches ──→ make_rq2_figures.py [Phase 19]                
                                   └──→ results/figures/rq2_*.pdf               
                                                                                
    All artifacts ──→ tests/acceptance_tests.py [Phase 20]                      
```

---

## 4. Critical path

The path that gates "RQ2 is defensible":

```
Phase 0 → 1 → 2 → 3 → 4   (Track 1: faithfulness)
Phase 11                  (PHI test — HIPAA-critical)
Phase 14 → 15             (User study — empirical heart)
Phase 18                  (Aggregator — single source of truth)
```

Everything else is supporting. If only the critical path completes, RQ2 is defensible. If any critical-path item is missing, RQ2 has a hole a reviewer will find.

---

## 5. File inventory (full RQ2)

### Already specced (Track 1)
- `common/feature_metadata.py` (Phase 0 — human-in-the-loop generated)
- `common/perturbation.py` (Phase 1)
- `module4_xai/shap_explainer.py` (Phase 2 — may already exist; verify)
- `analysis/compute_shap_stability.py` (Phase 3)
- `analysis/compute_mve_shap_alignment.py` (Phase 4)
- `tests/test_perturbation.py` (Phase 1)
- `tests/test_step11_shap_stability.py` (Phase 6)
- `tests/test_step12_mve_faithfulness.py` (Phase 6)

### Pending specs (Tracks 2–5)
- `analysis/audit_mitre_config.py` (Phase 7)
- `analysis/compute_mitre_grounding.py` (Phase 8)
- `analysis/make_rq2_compliance_table.py` (Phase 9)
- `analysis/audit_word_budgets.py` (Phase 10)
- `tests/test_phi_not_in_llm_prompt.py` (Phase 11)
- `tests/test_step13_cross_role_consistency.py` (Phase 12)
- `analysis/compute_rq2c_per_role.py` (Phase 15)
- `analysis/analyze_qualitative_themes.py` (Phase 16, optional)
- `analysis/compile_failure_modes.py` (Phase 17)

### Pending specs (Merge layer)
- `module6_evaluation/compute_rq2_metrics.py` (Phase 18 — verify if exists)
- `module6_evaluation/make_rq2_figures.py` (Phase 19)

### May already exist (verify before writing)
- `survey/study_loader.py` (Phase 13)
- `survey/study_analysis.py` (Phase 14)
- `tests/test_coverage_mve.py` (referenced as existing)

### Produced artifacts (full RQ2)
```
results/
├── rq2_metrics.json                          ← Phase 18 (canonical)
├── rq2_shap_stability.json                   ← Phase 3
├── rq2_mve_shap_alignment.json               ← Phase 4
├── rq2_mitre_audit.json                      ← Phase 7
├── rq2_mitre_grounding.json                  ← Phase 8
├── rq2_compliance_mapping.md                 ← Phase 9
├── rq2_word_budget_audit.json                ← Phase 10
├── rq2_failure_mode_catalog.json             ← Phase 17
├── cache/
│   └── shap_stability_cache.npz              ← Phase 3
└── figures/
    ├── rq2_shap_stability_histogram.pdf      ← Phase 19
    ├── rq2_mode_comparison.pdf               ← Phase 19
    └── rq2_per_role_results.pdf              ← Phase 19

survey/
└── m5_result.yaml                            ← Phase 14

analysis/outputs/
├── rq2c_per_role.json                        ← Phase 15
└── rq2c_themes.json                          ← Phase 16 (optional)
```

---

## 6. Execution order (full RQ2)

```bash
# ─── BLOCK A: PREREQUISITES ────────────────────────────────────
# Verify these are done before starting RQ2:
#   1. RQ1 pipeline complete (risk_scores.npz schema v1.1)
#   2. Module 5 batch run produced mve_outputs.jsonl
#   3. (For Track 4) user study data collected

# ─── BLOCK B: TRACK 1 — FAITHFULNESS (see RQ2_FAITHFULNESS_SPEC.md) ─
# Phase 0:  python scripts/bootstrap_feature_metadata.py [HUMAN-IN-LOOP]
# Phase 1:  pytest tests/test_perturbation.py
# Phase 2:  verify or create module4_xai/shap_explainer.py
# Phase 3:  python -m analysis.compute_shap_stability        (~30-60 min first run)
# Phase 4:  python -m analysis.compute_mve_shap_alignment    (seconds)
# Phase 6:  pytest tests/test_step11* tests/test_step12*

# ─── BLOCK C: TRACK 2 — MITRE GROUNDING (spec pending) ─────────
# Phase 7:  python -m analysis.audit_mitre_config
# Phase 8:  python -m analysis.compute_mitre_grounding

# ─── BLOCK D: TRACK 3 — COMPLIANCE (spec pending) ──────────────
# Phase 9:  python -m analysis.make_rq2_compliance_table
# Phase 10: python -m analysis.audit_word_budgets
# Phase 11: pytest tests/test_phi_not_in_llm_prompt.py
# Phase 12: pytest tests/test_step13_cross_role_consistency.py

# ─── BLOCK E: TRACK 4 — USER STUDY (spec pending, data-gated) ──
# Phase 13: verify survey/study_loader.py
# Phase 14: python -m survey.study_analysis
# Phase 15: python -m analysis.compute_rq2c_per_role
# Phase 16: python -m analysis.analyze_qualitative_themes  (optional)

# ─── BLOCK F: TRACK 5 — FAILURE MODES (spec pending) ───────────
# Phase 17: python -m analysis.compile_failure_modes

# ─── BLOCK G: MERGE + FIGURES + CI (spec pending) ──────────────
# Phase 18: python -m module6_evaluation.compute_rq2_metrics
# Phase 19: python -m module6_evaluation.make_rq2_figures
# Phase 20: pytest tests/acceptance_tests.py::test_rq2_targets_met

# ─── FINAL VERIFICATION ────────────────────────────────────────
pytest tests/
ls results/rq2_*.json results/figures/rq2_*.pdf
```

Blocks B, C, D, E, F can run in parallel within their tracks once prerequisites are met. The merge in Block G can only run after all upstream phases complete (or partial — the aggregator handles `_status: pending` placeholders, RQ1 pattern).

---

## 7. Coverage map: RQ2_expected_outputs.md → pipeline phase

| RQ2_expected_outputs.md item | Track | Phase | Spec status |
|---|---|---|---|
| **§1.1** Mapping table | 3 | 9 | pending |
| **§1.2** Compliance checklist evidence | 3 | 9–12 | pending |
| **§2.1** SHAP stability score / pass rate / histogram | 1 | 3 + 19 | ✅ specced (figure pending) |
| **§2.2** MVE-SHAP alignment all-3 / ≥2 | 1 | 4 | ✅ specced |
| **§2.2** Mode A vs Mode B | 1 | 4 | ✅ specced |
| **§2.2** Layer 1 references MITRE technique | 2 | 8 | pending |
| **§2.2** NOVEL_ANOMALY SHAP gap | 1 | 3 | ✅ specced |
| **§3.1** M5 Mann-Whitney per role × time/accuracy/confidence | 4 | 14 + 15 | pending |
| **§3.2** survey/study_responses_*.json | 4 | 13 (raw) | pending |
| **§3.2** survey/m5_result.yaml | 4 | 14 | pending |
| **§3.2** analysis/outputs/rq2c_per_role.json | 4 | 15 | pending |
| **§3.3** Qualitative themes | 4 | 16 | pending (optional) |
| **§3.4** Bedside nurse role question | — | (future work) | acknowledged |
| **§4** Failure mode catalog | 5 | 17 | pending |
| **§4.2** Two-round evaluation | — | (future work) | acknowledged |
| **§5.1** Per-class MITRE coverage | 2 | 7 + 8 | pending |
| **§5.2** config/attack_to_mitre_mapping.yaml audit | 2 | 7 | pending |
| **§5.3** % Layer 1 references MITRE > 90% | 2 | 8 | pending |
| **§8** test_step11_shap_stability.py | 1 | 6 | ✅ specced |
| **§8** test_step12_mve_faithfulness.py | 1 | 6 | ✅ specced |
| **§8** test_phi_not_in_llm_prompt.py | 3 | 11 | pending |
| **§8** test_step13_cross_role_consistency.py | 3 | 12 | pending |
| **§8** test_coverage_mve.py | — | existing | verify |

Every numbered RQ2 item is traceable to a phase. 11 items already specced, 14 pending, 2 acknowledged as future work, 1 to verify.

---

## 8. Decisions already locked (do not revisit)

These were resolved during prior conversations and constrain the spec work to come:

| Decision | Resolution | Source |
|---|---|---|
| SHAP perturbation rule | `x + uniform(-0.01, 0.01) * sigma_x` | Round 1 |
| SHAP perturbation scope | Hardcoded list in `common/feature_metadata.py` | Round 2 |
| SHAP seeding | `hash((alert_id, k)) & 0xFFFFFFFF` | Round 2 |
| Stability sample | Surfaced alerts only (`fusion_class != BENIGN`) | Round 2 |
| Top-k headline metric | `|A ∩ B| / 3` | Round 1 |
| Top-k appendix metric | Jaccard | Round 1 |
| Top-3 SHAP selection | Signed for predicted class | Round 3 |
| Alignment match rule | Human-readable substring, case-insensitive | Round 3 |
| Alignment search scope | `layer1_why` field only | Round 3 |
| Output integration | Extend `compute_rq2_metrics.py` (RQ1 pattern) | Round 3 |
| Faithfulness sub-files | Separate JSONs merged at aggregation time | Round 3 follow-up |
| Caching | Model-mtime-invalidated cache in `results/cache/` | Heads-up #1 |
| RQ2.d scope | Rescoped to single-round catalog (observation, not iteration) | Thesis outline |
| Bedside nurse role | Acknowledged future work; current scope is 3 roles | Council feedback |

---

## 9. Open questions (per track)

These are the things Claude Code will need to ask before implementing each track. Surfacing them up front so they aren't surprises.

### Track 1 (already in `RQ2_FAITHFULNESS_SPEC.md` §11)
1. `module4_xai/shap_explainer.py` exists? Schema?
2. XGBoost model path/extension?
3. Test parquet feature columns vs metadata columns?
4. `mve_outputs.jsonl` path and schema?
5. Mode A/B tagging field name?
6. `compute_rq2_metrics.py` exists already?

### Track 2 (to be asked when spec is written)
1. `config/attack_to_mitre_mapping.yaml` exists and current?
2. Does the YAML use T-IDs (T1565), human names ("Data Manipulation"), or both?
3. What "references MITRE" means — exact T-ID substring vs human name substring vs either?
4. MITRE framework version pinning convention?

### Track 3 (to be asked when spec is written)
1. Mode A audit log location and schema?
2. PHI patterns to scan for — explicit regex list, or use a library (presidio, scrubadub)?
3. Word budget enforcement — is it tested at MVE generation time or evaluation time? Both?
4. Role list for cross-role test — confirmed 3 (IT generalist, biomed engineer, nurse manager)?

### Track 4 (to be asked when spec is written)
1. `survey/study_loader.py` and `study_analysis.py` exist and current?
2. Per-participant JSON schema?
3. Group A vs Group B counterbalancing rule (MD5 seeding referenced)?
4. Ground-truth labels for accuracy measurement — where?
5. Sample size achieved vs planned?
6. Free-text response field name(s)?

### Track 5 (to be asked when spec is written)
1. What failure-mode categories are we cataloging — fixed taxonomy or emergent from observations?
2. Severity scoring per failure mode?

---

## 10. Recommended next move

Three reasonable paths from here:

**Path 1 — implement what's specced, then return:**
Hand Track 1 (`RQ2_FAITHFULNESS_SPEC.md`) to Claude Code. While it runs Phases 0–4 (mostly automatic after Phase 0 review), come back and spec the next track. Track 1's outputs are independent of the other tracks, so this parallelizes well.

**Path 2 — spec the merge layer next:**
Write the `compute_rq2_metrics.py` + `make_rq2_figures.py` spec now. It's the smallest pending spec and lets you ship a partial `rq2_metrics.json` (with `_status: pending` placeholders) as soon as Track 1 completes — useful for the paper's table scaffolding.

**Path 3 — spec the highest-stakes pending track:**
Write the Track 3 (Compliance + PHI test) spec next. The PHI test is HIPAA-adjacent and the first thing any defense reviewer will ask about. Getting this in code earlier reduces defense risk.

Recommendation: **Path 1 + Path 2 in parallel.** Claude Code implements Track 1 while you and I spec the merge layer. Then move to Track 3 (highest defense stakes), then Track 2 (MITRE — moderate stakes, low complexity), then Track 4 (user study — data-gated).

---

## End of overview

Track-specific specs to follow:
- ✅ `RQ2_FAITHFULNESS_SPEC.md` — Track 1
- ⏳ `RQ2_MITRE_GROUNDING_SPEC.md` — Track 2
- ⏳ `RQ2_COMPLIANCE_SPEC.md` — Track 3
- ⏳ `RQ2_USER_STUDY_SPEC.md` — Track 4
- ⏳ `RQ2_FAILURE_CATALOG_SPEC.md` — Track 5
- ⏳ `RQ2_MERGE_AND_FIGURES_SPEC.md` — Phases 18–20