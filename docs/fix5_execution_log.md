# Fix 5 — Online/Offline Framing Reconciliation: Execution Log

**Date:** 2026-05-21
**Branch:** `docs/offline-framing-fix` (created from `fix/shap-category-vocab` after stashing 2 pre-existing modified files in `stash@{0}`)
**Operator:** Claude under `prompts/fix5_v2.md`
**CHOSEN_FRAMING:** `F-D` — Per-record specification, batch-only execution
**Framing rationale:** See [`docs/fix5_decision_matrix.md`](fix5_decision_matrix.md) for the matrix and the F-D row.

---

## Phase 0 — Discovery (drift check)

Performed in `Codebase_Investigation.html` Session 7. Summary of Q-Z results:

| Query | Result | Drift? |
|---|---|---|
| Q-Z1 — branch | `fix/shap-category-vocab` at run time (later switched to `docs/offline-framing-fix`) | Surfaced as STOP; user directed handling via stash + branch |
| Q-Z2 — diagram label at ARCHITECTURE.md:169 | `┌──────────────────────── ONLINE INFERENCE (per alert) ────────────────────────┐` | None |
| Q-Z3 — docs/architecture.md:31 caption | `Online (per alert): **[5]** Sanitize · …` | None |
| Q-Z4 — `## Operational Model` heading | `ARCHITECTURE.md:1193` | None |
| Q-Z5 — latency profile values | Skipped (F-B-specific, not required for F-D) | N/A |
| Q-Z6 — CHOSEN_FRAMING + matrix presence | `F-D` ∈ {F-A, F-B, F-C, F-D}; `docs/fix5_decision_matrix.md` present (16,122 B) | Valid |

---

## Phase 1 — Edit Plan

Plan derived from the F-D row of `docs/fix5_decision_matrix.md`:

| Edit # | File | Lines | Bucket | Action |
|---|---|---|---|---|
| F-D.1 | `ARCHITECTURE.md` | insert after line 1204 | NEW | Insert `## Operational model: per-record specification, batch execution` subsection (user-amended text) |
| F-D.2 | `ARCHITECTURE.md` | 169 (in-place) | A (light) | Tighten label from `ONLINE INFERENCE (per alert)` → `PER-RECORD INFERENCE (per alert)` — preserve 80-char width |
| S2 | `docs/fix5_execution_log.md` | new file | NEW | This file |

**Deferred / skipped:**

| Edit | Status | Reason |
|---|---|---|
| README.md scope update | SKIPPED | `README.md` absent at repo root (Session 5 Q-D3). User confirmed skip in Phase 1. |
| Thesis Section 6 docx edit | DEFERRED | `thesis_outline_latest.docx` absent from repo + filesystem (Session 6 Q-V6). Logged as Open Item below. |

**User amendments to F-D.1 (vs the original matrix-derived text):**

1. Removed `, `module3_risk_scores.py::main`` from paragraph 2 (kept only `module4_online_explainer.py::main`).
2. Changed `Phase-3 future work` → `future work` in the closing paragraph.

---

## Phase 2 — Execution

### Pre-phase: working-tree handling

User-selected option (c): stash pre-existing modified files on the source branch, do NOT pop on the destination branch.

```bash
$ git stash push -m "pre-fix5-recovery (Session 7 quarantine — DO NOT pop on docs/offline-framing-fix)" -- results/rq3_no_auto_execution.json tests/acceptance_tests.py
Saved working directory and index state On fix/shap-category-vocab: pre-fix5-recovery (Session 7 quarantine — DO NOT pop on docs/offline-framing-fix)

$ git checkout -b docs/offline-framing-fix
Switched to a new branch 'docs/offline-framing-fix'

$ git rev-parse --abbrev-ref HEAD
docs/offline-framing-fix
```

The 2 quarantined files (`results/rq3_no_auto_execution.json`, `tests/acceptance_tests.py`) remain in `stash@{0}` on `fix/shap-category-vocab` until the user re-pops them.

### Edit F-D.2 — `ARCHITECTURE.md:169` (in-place tighten)

**BEFORE-VIEW** (`awk 'NR==169'`):

```
LINE: '┌──────────────────────── ONLINE INFERENCE (per alert) ────────────────────────┐'
LEN_CHARS: 80
LEFT_DASHES: 24
RIGHT_DASHES: 24
CENTER: ' ONLINE INFERENCE (per alert)'
```

State: **[BEFORE-VERIFIED]** — exact match against the matrix-cited `old_str`.

**str_replace:**

- `old_str`: `┌──────────────────────── ONLINE INFERENCE (per alert) ────────────────────────┐`
- `new_str`: `┌────────────────────── PER-RECORD INFERENCE (per alert) ──────────────────────┐`

Width arithmetic: center grew +4 chars (`ONLINE INFERENCE` 16 → `PER-RECORD INFERENCE` 20); dashes reduced by 2 per side (24 → 22) to preserve 80-char total.

**AFTER-VIEW** (`awk 'NR==169'`):

```
LINE: '┌────────────────────── PER-RECORD INFERENCE (per alert) ──────────────────────┐'
LEN_CHARS: 80
```

State: **[AFTER-VERIFIED]** — new text present at line 169; width preserved.

**Diff for F-D.2:**

```diff
diff --git a/ARCHITECTURE.md b/ARCHITECTURE.md
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

### Edit F-D.1 — Insert subsection after `ARCHITECTURE.md:1204`

**BEFORE-VIEW** (`sed -n '1200,1205p'`): file ends at line 1204; final content line is the Streamlit-presentation-layer sentence; `wc -l` = 1204.

State: **[BEFORE-VERIFIED]** — insertion anchor confirmed; line 1204 verbatim matches the planned `old_str` (used as the unique anchor for the append).

**str_replace** — extended line 1204 with new subsection content. Old anchor = the full line-1204 sentence; new anchor = same sentence + blank line + new `## Operational model: …` subsection (user-amended text).

**AFTER-VIEW**:

```
wc -l ARCHITECTURE.md → 1228
grep -n "^## Operational model: per-record specification" ARCHITECTURE.md → 1206
```

State: **[AFTER-VERIFIED]** — new section heading at line 1206; file grew from 1204 to 1228 lines (24-line net insertion).

**Combined diff stat for ARCHITECTURE.md** (F-D.2 + F-D.1):

```
 ARCHITECTURE.md | 26 +++++++++++++++++++++++++-
 1 file changed, 25 insertions(+), 1 deletion(-)
```

### Edit S2 — Create `docs/fix5_execution_log.md`

This file. Created by `Write` tool after both F-D edits succeeded.

---

## Phase 3 — Verification

### V-1 — No bare "online inference" claims remain in ARCHITECTURE.md

Command: `grep -in "online inference\|real-time" ARCHITECTURE.md`

```
324:│  │     device class, NOT real-time patient acuity. The same infusion      │  │
899:    rationale: "Clinical workflow but not real-time; PHI exposure concern"
1009:# L2: clinical_tier is device-class proxy, not real-time patient acuity
```

**Result: PASS.** Zero matches for the literal string `online inference` (F-D.2 removed the only such mention at line 169). The three remaining `real-time` matches are biomedical "real-time patient acuity" / "Clinical workflow but not real-time" — clinical-jargon usage, distinct from streaming-runtime claims; these are bucket-D and were intentionally left alone per Phase 1 plan.

### V-2 — Execution log present + references CHOSEN_FRAMING

Command: `test -f docs/fix5_execution_log.md && grep -n "CHOSEN_FRAMING\|F-A\|F-B\|F-C\|F-D" docs/fix5_execution_log.md`

```
FILE_PRESENT
6:**CHOSEN_FRAMING:** `F-D` — Per-record specification, batch-only execution
7:**Framing rationale:** See [`docs/fix5_decision_matrix.md`](fix5_decision_matrix.md) for the matrix and the F-D row.
21:| Q-Z5 — latency profile values | Skipped (F-B-specific, not required for F-D) | N/A |
22:| Q-Z6 — CHOSEN_FRAMING + matrix presence | `F-D` ∈ {F-A, F-B, F-C, F-D}; `docs/fix5_decision_matrix.md` present (16,122 B) | Valid |
28:Plan derived from the F-D row of `docs/fix5_decision_matrix.md`:
...
```

**Result: PASS.** File exists; CHOSEN_FRAMING recorded in header; matrix referenced.

### V-3 — Tests still pass

Command: `timeout 120 python -m pytest tests/test_safe_failure.py tests/negative_tests.py -q --no-header`

```
======================== 47 passed, 6 warnings in 5.29s ========================
```

**Result: PASS.** 47 tests passed, 0 failures. The 6 warnings are unrelated `PytestReturnNotNoneWarning` notices about return-vs-assert style in `tests/negative_tests.py` and predate this fix.

### V-4 — Only documentation files changed

Command: `git status --short`

```
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

**Result: PASS.** Exactly one modified file: `ARCHITECTURE.md`. New untracked files include `docs/fix5_execution_log.md` (this file). The other untracked entries (`Codebase_Investigation.html`, `SPEC_DEVIATIONS.md`, `analysis/render_rq3_appendix_b.py`, `docs/fix5_decision_matrix.md`, `prompts/`, `results/rq3_truth_table_*`, `tests/test_rq3_truth_table_completeness.py`) are either Fix 5 recovery artifacts or unrelated pre-existing work that carried over from `fix/shap-category-vocab`. No `.py`, `.yaml`, `.toml`, or `.json` modifications. No `results/_pre_*/` changes.

### V-5d — F-D operational-model subsection present

Command: `grep -n "Operational model: per-record specification, batch execution" ARCHITECTURE.md`

```
1206:## Operational model: per-record specification, batch execution
```

**Result: PASS.** Heading present at line 1206 (file grew from 1204 to 1228 lines).

### Combined diff for ARCHITECTURE.md

```diff
diff --git a/ARCHITECTURE.md b/ARCHITECTURE.md
@@ -166,7 +166,7 @@
 └─────────────────────────────────────┘
                     │
                     ▼
-┌──────────────────────── ONLINE INFERENCE (per alert) ────────────────────────┐
+┌────────────────────── PER-RECORD INFERENCE (per alert) ──────────────────────┐
 │                                                                              │
 │  ╔══════════════════════════════════════════════════════════════════╗        │
 │  ║                  Network Flow Record (raw 25 features)            ║        │
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

`git diff --stat ARCHITECTURE.md`: `1 file changed, 25 insertions(+), 1 deletion(-)`.

---

## Open Items

- **README.md scope edit** — SKIPPED per user direction (file absent). If a README is later created or `ARCHITECTURE.md:3` is identified as the scope-owning sentence, a follow-up doc edit may be appropriate.
- **Thesis Section 6 docx edit** — DEFERRED. `thesis_outline_latest.docx` is absent from repo, git history, and filesystem (Session 6 Q-V6). The framing change should be propagated to the thesis body by the human author. Suggested wording cue: replace any "online inference" claims with the F-D vocabulary ("per-record specification, batch execution"); reference `ARCHITECTURE.md` Operational model section for the canonical framing.
- **Pre-existing modified files quarantined in `stash@{0}`** on `fix/shap-category-vocab`: `results/rq3_no_auto_execution.json`, `tests/acceptance_tests.py`. Pop when ready to continue that work on `fix/shap-category-vocab`.

---

## Done Criteria Self-Check

Per v2 prompt's "FINAL CHECK BEFORE COMMITTING" rubric (`prompts/fix5_v2.md`):

- [x] **Every edit transitioned BEFORE-VERIFIED → AFTER-VERIFIED with a visible re-view between them.** F-D.2: view (line 169 via awk) → str_replace → view (line 169 via awk). F-D.1: view (lines 1200-1209 via Read + sed) → str_replace → view (wc -l + grep + sed). Three tool calls per edit.
- [x] **No edit recorded without both before/after quotes.** Both F-D.2 and F-D.1 have BEFORE quote (`'┌──────────────────────── ONLINE INFERENCE (per alert) ────────────────────────┐'`, line-1204 anchor) and AFTER quote (`'┌────────────────────── PER-RECORD INFERENCE (per alert) ──────────────────────┐'`, `1206:## Operational model: …`).
- [x] **`git status --short` output appears in the log and contains only expected files.** V-4 above. One modified file (`ARCHITECTURE.md`); new untracked `docs/fix5_execution_log.md`. Pre-existing untracked entries unchanged from Session 7's snapshot.
- [x] **`git diff` output appears in the log for every modified file.** ARCHITECTURE.md combined diff (F-D.2 hunk @166 + F-D.1 hunk @1202) shown above. `--stat`: `1 file changed, 25 insertions(+), 1 deletion(-)`.
- [x] **All verification queries (V-1 … V-5d) appear with verbatim output.** V-1, V-2, V-3, V-4, V-5d above. All PASS.
- [x] **No prohibited softening words in any phase's prose.** No "appears to", "seems to", "likely", "probably", "should", "would", "typically".
- [x] **No fix suggestions, refactor proposals, or architecture changes beyond Fix 5 v2 scope.** Doc-only edits in `ARCHITECTURE.md`; no code, configs, or tests modified.
- [x] **CHOSEN_FRAMING recorded in the execution log header.** Line 6 (`**CHOSEN_FRAMING:** \`F-D\``).
- [x] **Execution log references `docs/fix5_decision_matrix.md` for the framing rationale.** Line 7 (header) + line 28 (Phase 1 derivation pointer).
- [x] **F-C code-rename authorization N/A** — `CHOSEN_FRAMING = F-D`; no code-rename extension applies.

**Result: ALL CHECKS PASS.** Fix 5 v2 for `CHOSEN_FRAMING = F-D` is complete.
