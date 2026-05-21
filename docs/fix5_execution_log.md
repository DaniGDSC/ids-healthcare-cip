# Fix 5 — Online/Offline Framing Reconciliation: Execution Log

**Date:** 2026-05-21
**Branch:** docs/offline-framing-fix
**Chosen framing:** F-D (per-record specification, batch-only execution)
**Operator:** Claude Code under prompts/fix5_v2.md

## Branch Hygiene

- Started from `fix/shap-category-vocab` at commit `9e62358` ("feat(audit): implement no-auto-execution audit and verification scripts").
- Pre-existing modifications stashed on source branch as `stash@{0}`:
  - `results/rq3_no_auto_execution.json`
  - `tests/acceptance_tests.py`
- Stash message: `"pre-fix5-recovery (Session 7 quarantine — DO NOT pop on docs/offline-framing-fix)"`.
- TODO: when returning to `fix/shap-category-vocab`, handle `stash@{0}` (pop, drop, or merge as appropriate).

## Phase 0 — Discovery

See `Codebase_Investigation.html` Sessions 5 (markdown inventory + four HARD STOP conditions), 6 (code verification of streaming-evidence claims, all upgraded from [GREPPED] to [VERIFIED] or [REFUTED]), and 7 (v2 drift check — no line-number drift; `CHOSEN_FRAMING = F-D` validated).

## Phase 1 — Edit Plan

See decision matrix at `docs/fix5_decision_matrix.md`.

Two edits planned: **F-D.2** (in-place line 169 diagram-label tighten), **F-D.1** (insert subsection after line 1204).

User amendments to the matrix-derived F-D.1 text, applied before Phase 2:

1. Removed `, module3_risk_scores.py::main` from paragraph 2 — citation kept to `module4_online_explainer.py::main` only.
2. Changed `Phase-3 future work` → `future work` in the closing paragraph.

## Phase 2 — Execution

### F-D.2 — Diagram label change

- **BEFORE** (verbatim from BEFORE-VIEW via `awk 'NR==169' ARCHITECTURE.md`):

  ```text
  ┌──────────────────────── ONLINE INFERENCE (per alert) ────────────────────────┐
  ```

  Width: 80 chars · Left dashes: 24 · Right dashes: 24 · Center: ` ONLINE INFERENCE (per alert) ` · State: **[BEFORE-VERIFIED]**.

- **AFTER** (verbatim from AFTER-VIEW via `awk 'NR==169' ARCHITECTURE.md`):

  ```text
  ┌────────────────────── PER-RECORD INFERENCE (per alert) ──────────────────────┐
  ```

  Width: 80 chars (preserved) · Dashes rebalanced 22+22 to absorb the +4-char center growth · State: **[AFTER-VERIFIED]**.

- **git diff** (this hunk only):

  ```diff
  @@ -166,7 +166,7 @@
   └─────────────────────────────────────┘
                       │
                       ▼
  -┌──────────────────────── ONLINE INFERENCE (per alert) ────────────────────────┐
  +┌────────────────────── PER-RECORD INFERENCE (per alert) ──────────────────────┐
   │                                                                              │
   │  ╔══════════════════════════════════════════════════════════════════╗        │
   │  ║                  Network Flow Record (raw 25 features)            ║        │
  ```

### F-D.1 — Operational model subsection

- **Anchor** (verbatim from BEFORE-VIEW; this is the last line of `## Operational Model`, used as the unique str_replace anchor):

  ```text
  The Streamlit app is a presentation and study layer on top of the offline-computed demo artifacts, never the primary computation engine. Test split records do not appear in the dashboard at any point.
  ```

  Pre-insert: `wc -l ARCHITECTURE.md` → 1204. State: **[BEFORE-VERIFIED]**.

- **AFTER** — new subsection inserted immediately after the anchor (lines 1206–1228 post-edit):

  ```markdown
  ## Operational model: per-record specification, batch execution

  The pipeline distinguishes two orthogonal properties:

  1. **Per-record (alert-independent) semantics — a correctness property.**
     Steps [5]–[16] in the canonical workflow operate independently per
     alert: no temporal state is shared across alerts, the score of alert
     N does not depend on alerts 1..N-1, and the per-alert objects
     (`SHAPContext`, `MVEOutput`) are designed for load-once + call-many.
     This is what `module4_online_explainer.py`'s docstring means by
     "online-capable."

  2. **Batch execution — an operational property.** The implemented
     entrypoint runs the per-record sequence in batch over the test split
     (`module4_online_explainer.py::main`).
     The per-call latency profile
     (`results/reports/online_latency_profile.json`, n=677) is
     opportunistic timing collected inside the batch run.

  All thesis claims about detection, explanation, and clinical
  adaptation derive from this batch execution. The per-record semantics
  make the pipeline implementable as a streaming runtime — that
  implementation is future work, not a claim made in this thesis.
  ```

  Post-insert: `wc -l ARCHITECTURE.md` → 1228 (+24); `grep -n "^## Operational model: per-record specification" ARCHITECTURE.md` → `1206`. State: **[AFTER-VERIFIED]**.

- **git diff** (this hunk only):

  ```diff
  @@ -1202,3 +1202,27 @@
   6. **Analysis**: post-collection RQ3 analysis (`analyze_rq3.py`) reads `survey/study_responses_*.json`.

   The Streamlit app is a presentation and study layer on top of the offline-computed demo artifacts, never the primary computation engine. Test split records do not appear in the dashboard at any point.
  +
  +## Operational model: per-record specification, batch execution
  +
  +The pipeline distinguishes two orthogonal properties:
  +
  +1. **Per-record (alert-independent) semantics — a correctness property.**
  +   Steps [5]–[16] in the canonical workflow operate independently per
  +   alert: no temporal state is shared across alerts, the score of alert
  +   N does not depend on alerts 1..N-1, and the per-alert objects
  +   (`SHAPContext`, `MVEOutput`) are designed for load-once + call-many.
  +   This is what `module4_online_explainer.py`'s docstring means by
  +   "online-capable."
  +
  +2. **Batch execution — an operational property.** The implemented
  +   entrypoint runs the per-record sequence in batch over the test split
  +   (`module4_online_explainer.py::main`).
  +   The per-call latency profile
  +   (`results/reports/online_latency_profile.json`, n=677) is
  +   opportunistic timing collected inside the batch run.
  +
  +All thesis claims about detection, explanation, and clinical
  +adaptation derive from this batch execution. The per-record semantics
  +make the pipeline implementable as a streaming runtime — that
  +implementation is future work, not a claim made in this thesis.
  ```

Combined `git diff --stat ARCHITECTURE.md`: `1 file changed, 25 insertions(+), 1 deletion(-)`.

## Phase 3 — Verification

### V-1 — `grep -in "online inference\|real-time" ARCHITECTURE.md`

```text
324:│  │     device class, NOT real-time patient acuity. The same infusion      │  │
899:    rationale: "Clinical workflow but not real-time; PHI exposure concern"
1009:# L2: clinical_tier is device-class proxy, not real-time patient acuity
```

Zero matches for `online inference`. Three matches for `real-time`, all biomedical "real-time patient acuity" / "Clinical workflow but not real-time" — bucket-D clinical jargon, distinct from streaming-runtime claims; intentionally left alone per Phase 1.

### V-2 — `test -f docs/fix5_execution_log.md && grep -n "CHOSEN_FRAMING\|F-A\|F-B\|F-C\|F-D" docs/fix5_execution_log.md`

```text
FILE_PRESENT
4:**Chosen framing:** F-D (per-record specification, batch-only execution)
...
```

File present; `F-D` referenced in header and throughout.

### V-3 — `timeout 120 python -m pytest tests/test_safe_failure.py tests/negative_tests.py -q --no-header`

```text
======================== 47 passed, 6 warnings in 5.29s ========================
```

47 passed, 0 failures. Warnings are unrelated `PytestReturnNotNoneWarning` notices that predate this fix.

### V-4 — `git status --short`

```text
 M ARCHITECTURE.md
?? Codebase_Investigation.html
?? SPEC_DEVIATIONS.md
?? analysis/render_rq3_appendix_b.py
?? docs/fix5_decision_matrix.md
?? docs/fix5_execution_log.md
?? prompts/
?? results/rq3_truth_table_appendix_b.md
?? results/rq3_truth_table_reference.json
?? tests/test_rq3_truth_table_completeness.py
```

One modified file (`ARCHITECTURE.md`). Untracked entries are Fix-5 recovery artifacts (`Codebase_Investigation.html`, `docs/fix5_decision_matrix.md`, `docs/fix5_execution_log.md`, `prompts/`) and unrelated pre-existing untracked items that carried over from `fix/shap-category-vocab`. No `.py`, `.yaml`, `.toml`, or `.json` modifications. No `results/_pre_*/` changes.

### V-5d — `grep -n "Operational model: per-record specification, batch execution" ARCHITECTURE.md`

```text
1206:## Operational model: per-record specification, batch execution
```

Subsection present at line 1206.

## Done Criteria Self-Check

- [x] `grep -in "online inference\|real-time" ARCHITECTURE.md` returns only qualified matches (V-1: zero `online inference` matches; three biomedical `real-time` matches, all bucket-D clinical jargon).
- [x] "Operational model: per-record specification" section present (V-5d: line 1206).
- [x] Per-record framing preserved (not over-corrected) — only the canonical-workflow diagram label at line 169 was changed; per-alert objects (`SHAPContext`, `MVEOutput`), per-alert docstrings, and existing per-record terminology elsewhere in `ARCHITECTURE.md` and `docs/architecture.md` are intact.
- [x] Tests pass (V-3: 47 passed, 0 failed).
- [x] Only documentation files changed (V-4: one `M` line on `ARCHITECTURE.md`; one new file `docs/fix5_execution_log.md`; no code/config/test modifications).

## Open Items

- **Thesis Section 6 edit: deferred** (docx absent per Session 6 Q-V6).
  Action: hand off F-D.1 text as draft for human thesis author.
- **`module3_risk_scores.py::main` citation: removed pending verification.**
  Status: `def main()` is defined at `module3_risk_scoring/module3_risk_scores.py:1438` (verified via Session 4 `grep -n "^def |^class "` output). The removal was a stylistic choice by the user to keep the F-D.1 citation focused on the explicitly online-capable entrypoint. If a follow-up edit wants to broaden the citation to both batch entrypoints, the addition is factually supported.
- **Latency profile provenance** (no date / source-data hash / code-version metadata in `online_latency_profile.json`): noted, no action this fix. If a future framing change (e.g., move to F-B) requires citing the profile authoritatively, regenerating it with provenance keys would be a precondition.
