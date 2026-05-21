# TASK PROMPT — Fix 5 v2: Online/Offline Framing Reconciliation (Framing-Aware)

## REQUIRED INPUT — CHOSEN_FRAMING

Set this before running. Must be one of: `F-A`, `F-B`, `F-C`, `F-D`.

```
CHOSEN_FRAMING: F-D
```

The framing definitions, evidence base, and trade-offs are in
[`docs/fix5_decision_matrix.md`](../docs/fix5_decision_matrix.md). If the
matrix is missing or `CHOSEN_FRAMING` is unset/invalid, **STOP** and ask
the user.

Summary (consult the matrix for full evidence):

- **F-A** — Hybrid architecture: offline training + batch artifact generation; online per-alert SHAP/MVE inference. Both are first-class.
- **F-B** — Online prototype, batch production: batch is production; per-alert online code is a prototype with documented SLA gap; streaming = future work.
- **F-C** — Simulation only: production is batch; "Online Simulation" is a demo replay; all "Online" labels renamed to "Per-record (simulated)".
- **F-D** — Per-record specification, batch-only execution: per-record specifies correctness (alert independence); batch is the operational mode; all reported numerical claims derive from batch.

---

## INHERITED CONTEXT

You operate under all rules of `prompts/discovery.md` for Phases 0 and 1
(read-only). For Phases 2–3 the prohibition on file modification is LIFTED
— but only for the specific files identified in Phase 1, only at the
specific line ranges that pass Phase 1 classification, and only under
the Write-Mode Safety Protocol defined below.

If you have not loaded `prompts/discovery.md`, STOP and ask the user to load
it before proceeding.

---

## PRIOR CONTEXT

- `Codebase_Investigation.html` **Session 5** (Phase 0 of Fix 5 v1) inventoried "online" / "real-time" / "streaming" mentions across markdown and surfaced four HARD STOP conditions.
- `Codebase_Investigation.html` **Session 6** (Phase 0b of Fix 5 recovery) opened the actual code and upgraded every Session 5 [GREPPED] claim to [VERIFIED] or [REFUTED].
- [`docs/fix5_decision_matrix.md`](../docs/fix5_decision_matrix.md) (Phase 0c of Fix 5 recovery) enumerates the framings. This v2 prompt operationalizes the user's selected framing into an edit plan.
- The original Fix 5 v1 "DECISION ALREADY MADE" section is **revoked** — the user now selects the framing via `CHOSEN_FRAMING`.
- Fix 5 v2 is **DOCUMENTATION ONLY**. No code, no tests, no configs change unless `CHOSEN_FRAMING = F-C` and the user explicitly authorizes the code-rename extension surfaced in the decision matrix.

---

## REPOSITORY CONTEXT

- Repo root: `/home/un1/project/ids-healthcare-cip`
- Branch policy: this work belongs on its own branch `docs/offline-framing-fix`.
  - If current branch is already that, proceed.
  - If on `fix/shap-category-vocab` or any other branch, surface and ask whether to (a) create the branch now or (b) continue on the current branch.
- Date when the recovery decision matrix was authored: 2026-05-21.

---

## WRITE-MODE SAFETY PROTOCOL

Every file modification passes through these states, in order:

- **[BEFORE-VERIFIED]** — A `view` of the target line range was performed in
  THIS turn, immediately before the edit. The current text matches Phase 1's
  expected `old_str` byte-for-byte. Only this state authorizes `str_replace`.

- **[BEFORE-DIVERGED]** — The `view` shows text DIFFERENT from Phase 1's
  expected `old_str`. **STOP.** Do not retry. Do not "fix up" the `old_str`.
  Surface the divergence with both quoted texts and ask the user.

- **[AFTER-VERIFIED]** — `str_replace` returned success AND a fresh `view`
  of the same line range shows the new text present. The edit is complete.

- **[AFTER-FAILED]** — `str_replace` errored (no match, multiple matches,
  permission denied) OR the post-edit `view` does NOT show the new text.
  **STOP.** Surface the failure mode and tool output. Do not retry without
  explicit instruction.

**Critical:** no edit transitions from BEFORE-VERIFIED to AFTER-VERIFIED
without a visible re-`view` tool call between them. The pattern is:
view → str_replace → view. Three tool calls per edit. No exceptions.

---

## PHASE 0 — DISCOVERY (REDUCED FROM v1, STOP FOR USER CONFIRMATION)

The bulk of Phase 0 is already done (Sessions 5 + 6). This phase only
re-checks that nothing material drifted between the matrix being written
(2026-05-21) and this v2 prompt being invoked.

Queries:

```bash
# Q-Z1: Branch confirmation
git rev-parse --abbrev-ref HEAD

# Q-Z2: Diagram label still at ARCHITECTURE.md:169
grep -n "ONLINE INFERENCE\|OFFLINE\|per alert\|per-record" ARCHITECTURE.md

# Q-Z3: docs/architecture.md:31 workflow caption still present
sed -n '28,33p' docs/architecture.md

# Q-Z4: Operational Model section still at ARCHITECTURE.md:~1193
grep -n "^## Operational Model" ARCHITECTURE.md

# Q-Z5: Latency profile still at expected values (F-B only)
grep -E "p50|p95|p99" results/reports/online_latency_profile.json | head -10

# Q-Z6: CHOSEN_FRAMING validation
# The agent checks that CHOSEN_FRAMING is set and ∈ {F-A, F-B, F-C, F-D}
```

**Phase 0 output:** a short HTML block appendable to
`Codebase_Investigation.html` as Session 7, confirming each Q-Z. No edits
proposed. No file modifications.

**Phase 0 STOP condition:** After emitting the discovery confirmation,
**STOP**. Wait for the user to reply "proceed to Phase 1" or to amend the
discovery.

---

## PHASE 1 — EDIT PLAN (BRANCHES ON CHOSEN_FRAMING)

This phase emits an edit plan table. No edits execute. Stop at end.

### Common to all framings

| Edit # | File | Lines | Bucket | Current text (≤25 words) | Proposed action |
|---|---|---|---|---|---|
| **S2** | `docs/fix5_execution_log.md` | (new file) | NEW | — | Create execution log to record this fix; populated during Phase 2 |

### Plus: a deferred-or-skipped table

| Edit | Status | Reason |
|---|---|---|
| README.md scope update | **SKIPPED** | `README.md` absent at repo root (Session 5 Q-D3). Surface to user for redirect or new-file decision. |
| Thesis Section 6 docx edit | **DEFERRED** | `thesis_outline_latest.docx` absent from repo + filesystem (Session 6 Q-V6). Logged as Open Item in execution log; needs human author. |

### Framing-specific edit lists

#### If `CHOSEN_FRAMING = F-A` (Hybrid architecture)

| Edit # | File | Lines | Bucket | Current text (≤25 words) | Proposed action |
|---|---|---|---|---|---|
| **F-A.1** (optional) | `ARCHITECTURE.md` | insert after `## Operational Model` (line ~1204) | NEW | — | Insert short subsection clarifying that `ONLINE` in the workflow diagram means per-record semantics + low-latency design, not live network ingest. Reference the latency profile as feasibility evidence (n=677, p50=66 ms). Do not commit to an SLA. |

No code changes. No diagram-label rename. Existing labels at
`ARCHITECTURE.md:141, 169` and `docs/architecture.md:31` remain unchanged.

**Proposed F-A.1 text (≤180 words):**

```markdown
## Online and Offline: scope clarification

The workflow diagram (Canonical System Workflow above) splits processing
into OFFLINE (one-time training + threshold calibration) and ONLINE
INFERENCE (per alert). "Online" here denotes the per-record execution
semantics of Steps [5]–[16]: each alert flows through the same sequence,
no temporal state is shared, and the per-alert SHAP/MVE objects
(`SHAPContext`, `MVEOutput`) are designed for load-once + call-many use.

Operationally, the implemented entrypoint runs this per-record sequence
in batch over the test split — for reproducibility and for the paper
metrics pipeline (`module3_risk_scores.py::main`). The per-alert latency
profile (`results/reports/online_latency_profile.json`) is feasibility
evidence collected during this batch run, not a production SLA.

A production-grade streaming runtime (live NetFlow ingestion, per-record
output dispatch, runtime latency-budget enforcement) is named as
Phase-3 future work — see `## Operational Model` Steps 17–18.
```

---

#### If `CHOSEN_FRAMING = F-B` (Online prototype, batch production)

| Edit # | File | Lines | Bucket | Current text (≤25 words) | Proposed action |
|---|---|---|---|---|---|
| **F-B.1** | `ARCHITECTURE.md` | insert after `## Operational Model` (line ~1204) | NEW | — | Insert "Per-Alert Inference Prototype" subsection naming the `<150 ms` SLA target, the measured `p95 = 216.841 ms`, the gap, and the prototype framing. |
| **F-B.2** | `ARCHITECTURE.md` | insert after F-B.1 | NEW | — | Insert "Streaming as future work" subsection naming production-grade requirements (live ingest, per-record dispatch, runtime latency-budget enforcement). |

No code changes (the prototype already exists and is profiled).

**Proposed F-B.1 text (≤200 words):**

```markdown
## Per-Alert Inference Prototype

A per-alert SHAP + DAE decomposition + NLG pipeline is implemented in
`module4_explanations/module4_online_explainer.py`. The module is
described in its own docstring as "online-capable, validated in batch
mode on the test set."

**SLA target (documentation):** `< 150 ms` per alert (per module
docstring and the `STABILITY_*` constants' tuning comments).

**Measured (n=677 alerts on the test split):** `total_ms` p50 = 66.3 ms,
p95 = 216.8 ms, p99 = 230.5 ms (source:
`results/reports/online_latency_profile.json`).

**Gap:** the measured p95 exceeds the documented SLA by ~67 ms (~45%).
The module's own `main()` log emits "FAIL" under these values
(`module4_online_explainer.py:968–971`).

This pipeline is a research prototype. All thesis claims about
explanation quality, stability, and clinical relevance derive from this
prototype's outputs; thesis claims about latency are limited to the
measured profile above and are not deployment guarantees.
```

**Proposed F-B.2 text (≤120 words):**

```markdown
## Streaming as future work

The current implementation does not include a production-grade streaming
runtime. A streaming variant would require:

- Live NetFlow (or equivalent) ingestion replacing the parquet read in
  Module 1's preprocessing path.
- Per-record output dispatch outside the Streamlit replay page (the
  current "Online Simulation" iterates pre-loaded `alert_responses.json`
  with `time.sleep()`).
- Runtime latency-budget enforcement: the prototype's `< 150 ms` claim
  is observational, not guarded — closing the measured gap (p95 = 216.8
  ms vs target 150 ms) and adding per-alert assertion is a precondition
  for any production SLA claim.

These are named as Phase-3 work and are out of scope for this thesis.
```

---

#### If `CHOSEN_FRAMING = F-C` (Simulation only)

| Edit # | File | Lines | Bucket | Current text (≤25 words) | Proposed action |
|---|---|---|---|---|---|
| **F-C.1** | `ARCHITECTURE.md` | 169 | A | `┌──────────────────────── ONLINE INFERENCE (per alert) ────────────────────────┐` | Replace `ONLINE INFERENCE (per alert)` with `PER-RECORD INFERENCE (batch-processed)`. Preserve box-drawing characters and column count. |
| **F-C.2** | `docs/architecture.md` | 31 | A | `Online (per alert): **[5]** Sanitize · **[6a]** Track A · …` | Replace leading `Online (per alert):` with `Per-record (batch-processed):`. Preserve all step bullets. |
| **F-C.3** | `ARCHITECTURE.md` | insert in §1 (after the Overview, before the Module Overview) | NEW | — | Insert "Simulation framing" paragraph clarifying that "Online Simulation" in the Streamlit dashboard is a demo replay, not a streaming runtime. |
| **F-C.4** (optional, larger scope) | `docs/section312_offline_online_extraction.md`, `docs/chapter3_pipeline_extraction_report.md` | various | A/C mix | "online inference path" references that describe production behavior | Rephrase to "per-record reasoning path (executed in batch)". Defer if scope creep is a concern; surface as Open Items. |

**Code-rename extension (out of doc-only scope — requires explicit user authorization):**

| Edit | File | Lines | Action |
|---|---|---|---|
| F-C.X1 | `module6_evaluation/module6_app.py` | 3909, 3914, 2490 | Rename Streamlit page string `"Online Simulation"` &rarr; `"Per-Record Replay"` (or user-chosen alternative). |
| F-C.X2 | `module6_evaluation/module6_app.py` | 2194 | Rename `"IoMT IDS — Real-Time Dashboard"` &rarr; `"IoMT IDS — Per-Record Dashboard"`. |

**STOP for user before executing F-C.X1 / F-C.X2** — these are code edits
and require out-of-doc-only authorization per the "DO NOT DO" rules below.

**Proposed F-C.3 text (≤160 words):**

```markdown
## Simulation framing

The system is a batch-evaluation pipeline. The Streamlit dashboard
includes an **Online Simulation** page (`module6_app.py::simulation_mode`)
which iterates pre-computed alert responses with an artificial delay,
simulating per-alert arrival for demonstration purposes — it does not
ingest live network data.

All reported numerical claims (RQ1 detection metrics, RQ2 explanation
metrics, RQ3 user-study outcomes) derive from the batch path, not the
simulator. The simulator exists for usability demonstrations and is not
a production runtime claim.

References to "online inference" elsewhere in the documentation describe
per-record execution semantics (Steps [5]–[16] processed independently
per alert), not a streaming runtime. A production streaming deployment
is named as future work in the supplementary Phase-3 documents.
```

---

#### If `CHOSEN_FRAMING = F-D` (Per-record specification, batch-only execution)

| Edit # | File | Lines | Bucket | Current text (≤25 words) | Proposed action |
|---|---|---|---|---|---|
| **F-D.1** | `ARCHITECTURE.md` | insert after `## Operational Model` (line ~1204) | NEW | — | Insert "Operational model: per-record specification, batch execution" subsection distinguishing the per-record correctness property from the batch operational mode. |
| **F-D.2** (optional) | `ARCHITECTURE.md` | 169 | A (light) | `┌──────────────────────── ONLINE INFERENCE (per alert) ────────────────────────┐` | Tighten to `┌──────────────────────── PER-RECORD INFERENCE (per alert) ────────────────────────┐` (drop "ONLINE", keep "per alert"). Less disruptive than F-C.1. |

No code changes (this framing accepts existing code naming as honest
design vocabulary).

**Proposed F-D.1 text (≤200 words):**

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
   (`module4_online_explainer.py::main`, `module3_risk_scores.py::main`).
   The per-call latency profile
   (`results/reports/online_latency_profile.json`, n=677) is
   opportunistic timing collected inside the batch run.

All thesis claims about detection, explanation, and clinical
adaptation derive from this batch execution. The per-record semantics
make the pipeline implementable as a streaming runtime — that
implementation is Phase-3 future work, not a claim made in this thesis.
```

---

**Phase 1 STOP condition:** After emitting the edit-plan table specific
to `CHOSEN_FRAMING` plus the common S2 row and the deferred/skipped
table, **STOP**. Wait for the user to reply "proceed to Phase 2" or to
amend specific edits.

---

## PHASE 2 — EXECUTE EDITS (ONE AT A TIME, USE THE WRITE-MODE PROTOCOL)

For each row in the Phase 1 table (excluding deferred/skipped), execute
the canonical sequence: **view → str_replace → view → git diff**.

### Worked example (F-C.1, the diagram label edit)

```
Step (a) — BEFORE-VIEW:
  view ARCHITECTURE.md:165-175

Step (b) — VERIFY:
  Confirm line 169 contains EXACTLY:
    "┌──────────────────────── ONLINE INFERENCE (per alert) ──...─┐"
  If matches → state [BEFORE-VERIFIED] and proceed.
  If differs → state [BEFORE-DIVERGED], quote both, STOP.

Step (c) — EDIT:
  str_replace with the verbatim before-text and the framing's new-text.

Step (d) — AFTER-VIEW:
  view ARCHITECTURE.md:165-175

Step (e) — VERIFY:
  Confirm new text present.
  If yes → [AFTER-VERIFIED].
  If no  → [AFTER-FAILED], quote view, STOP.

Step (f) — DIFF:
  bash: git diff --stat ARCHITECTURE.md
  Confirm only ARCHITECTURE.md changed.
```

**Critical rules for Phase 2:**

- Process edits in this order: **structural inserts (F-X.1, etc.) LAST**;
  in-place edits FIRST. Inserts shift line numbers; doing them first
  invalidates the line ranges of subsequent edits.
- Process one file completely before moving to the next file.
- Within a file, process edits in DESCENDING line-number order — preserves
  line numbers for later edits in the same file.
- After all edits in a file complete, `git diff <file>` and quote the
  diff verbatim in the execution log.
- After ALL files complete, `git status --short`. Expected output depends
  on `CHOSEN_FRAMING`:
  - **F-A**: `M ARCHITECTURE.md` + `?? docs/fix5_execution_log.md`
  - **F-B**: `M ARCHITECTURE.md` + `?? docs/fix5_execution_log.md`
  - **F-C**: `M ARCHITECTURE.md` + `M docs/architecture.md` + `?? docs/fix5_execution_log.md` (+ optionally `M module6_evaluation/module6_app.py` if X1/X2 were authorized)
  - **F-D**: `M ARCHITECTURE.md` + `?? docs/fix5_execution_log.md`
  
  No other paths. If `results/_pre_*/` shows up, **STOP** — archived
  snapshots must not be edited.

**Phase 2 STOP conditions** (in addition to BEFORE-DIVERGED and AFTER-FAILED):

- Any unexpected file appears in `git status --short`.
- `git status` shows code files (`.py`, `.yaml`, `.toml`) modified, except
  for F-C.X1/F-C.X2 when explicitly authorized.
- `git status` shows changes inside `results/_pre_*/`.
- The Streamlit dashboard files appear modified without explicit Phase 1
  authorization (F-C.X1/F-C.X2).

---

## PHASE 3 — VERIFY

Run verification queries adjusted to `CHOSEN_FRAMING`:

```bash
# Common to all framings

# V-1: No bare unqualified "online inference" claims remain in ARCHITECTURE.md
# (Definition of "qualified": within 5 lines of "per alert", "batch", "future
# work", "prototype", or "would require")
grep -in "online inference\|real-time" ARCHITECTURE.md

# V-2: Execution log exists and references CHOSEN_FRAMING
test -f docs/fix5_execution_log.md && grep -n "CHOSEN_FRAMING\|F-A\|F-B\|F-C\|F-D" docs/fix5_execution_log.md

# V-3: Tests still pass (Fix 5 shouldn't touch code in most framings; confirm)
pytest tests/test_safe_failure.py tests/negative_tests.py -q

# V-4: Only documentation files changed (or page-rename code if F-C.X authorized)
git status --short

# Framing-specific verifications:

# F-A: V-5a — Operational Model section gained a scope-clarification subsection
grep -n "Online and Offline: scope clarification" ARCHITECTURE.md

# F-B: V-5b — Prototype + future-work subsections present
grep -n "Per-Alert Inference Prototype" ARCHITECTURE.md
grep -n "Streaming as future work" ARCHITECTURE.md
grep -E "p95.*216|216.*p95" ARCHITECTURE.md   # numbers from latency profile cited

# F-C: V-5c — Diagram labels renamed; simulation framing inserted
grep -n "PER-RECORD INFERENCE (batch-processed)" ARCHITECTURE.md
grep -n "Per-record (batch-processed):" docs/architecture.md
grep -n "Simulation framing" ARCHITECTURE.md

# F-D: V-5d — Per-record/batch operational subsection present
grep -n "Operational model: per-record specification, batch execution" ARCHITECTURE.md
```

For each verification, quote the command output verbatim in
`docs/fix5_execution_log.md`.

---

## OUTPUT ARTIFACTS

| Artifact | Path | Phase | Format |
|---|---|---|---|
| Phase 0 confirmation | `Codebase_Investigation.html` (append Session 7) | 0 | HTML matching existing investigation schema |
| Phase 1 edit plan | conversation-level output (not a file) | 1 | Markdown table per the structure above |
| Phase 2 edits | in-place edits to the files named in the framing's edit list | 2 | str_replace under Write-Mode Safety Protocol |
| Execution log | `docs/fix5_execution_log.md` (new file) | 2–3 | Markdown with Phases 0/1/2/3 sections |

---

## STOP CONDITIONS (HARD)

Stop and ask the user if:

- **`CHOSEN_FRAMING` is unset or not in {`F-A`, `F-B`, `F-C`, `F-D`}**.
- **The decision matrix is missing** (`docs/fix5_decision_matrix.md` absent).
- Current branch is not `docs/offline-framing-fix` and user has not confirmed continuing on the current branch.
- Phase 0's Q-Z2/Q-Z3/Q-Z4 grep returns line numbers materially different from those cited in this prompt (e.g., the diagram label moved away from `ARCHITECTURE.md:169`) — surface the divergence before Phase 1.
- Any edit hits BEFORE-DIVERGED or AFTER-FAILED state.
- `git status` shows changes outside the framing-specific expected file set.
- `CHOSEN_FRAMING = F-C` and Phase 2 is about to run F-C.X1/F-C.X2 without the user having explicitly authorized the code-rename extension.
- A planned read returns a file larger than 500 lines without an obvious truncation point — propose a smaller range first.
- README.md absent AND user has not directed where to redirect Edit 5 (skip, redirect to ARCHITECTURE.md:3, create new file).

---

## DO NOT DO

- Do not switch branches without explicit user direction.
- Do not edit anything under `results/_pre_*/`.
- Do not edit any `.py`, `.yaml`, `.toml`, `.cfg`, `.json`, or `.ini` file unless `CHOSEN_FRAMING = F-C` AND the user has explicitly authorized F-C.X1 / F-C.X2.
- Do not edit Streamlit dashboard UI strings unless authorized as F-C.X1 / F-C.X2.
- Do not chain edits across files in one tool-call sequence. One file at a time. Within a file: descending line order.
- Do not "fix up" a divergent `old_str` to make it match. Divergence is a STOP, not a parameter to tune.
- Do not retry a failed `str_replace` without user authorization.
- Do not propose architectural changes (e.g., pseudocode for a streaming variant). The fix produces framing documentation; that is the scope.
- Do not introduce new latency claims or SLA numbers beyond what the matrix's evidence already supports. The measured `p95 = 216.841 ms` is the only number authorized for F-B.
- Do not skip the post-edit `view` step. Three tool calls per edit. No exceptions.
- Do not re-decide the framing silently. If Phase 0 surfaces something that contradicts `CHOSEN_FRAMING`, **STOP** and surface to the user.
- Do not create new files outside the specified set (`docs/fix5_execution_log.md` is the only new file Phase 2 creates).

---

## FINAL CHECK BEFORE COMMITTING

Run the discovery-prompt self-grade rubric plus these write-mode additions:

- [ ] Every edit transitioned BEFORE-VERIFIED → AFTER-VERIFIED with a visible re-view between them.
- [ ] No edit is recorded in the log without both before/after quotes.
- [ ] `git status --short` output appears in the log and contains only expected files for the chosen framing.
- [ ] `git diff` output appears in the log for every modified file.
- [ ] All verification queries (V-1 … V-5x) appear in the log with verbatim output.
- [ ] No prohibited softening words in any phase's prose.
- [ ] No fix suggestions, refactor proposals, or architecture changes beyond the documented Fix 5 v2 scope for the selected framing.
- [ ] `CHOSEN_FRAMING` is recorded in the execution log header.
- [ ] The execution log references `docs/fix5_decision_matrix.md` for the framing rationale.
- [ ] If `CHOSEN_FRAMING = F-C` and the code-rename extension was authorized, the authorization is quoted verbatim in the execution log.

If any check fails, **do not commit the changes**. Surface the failure
and ask the user.

---

## END

Run Phase 0 only on first invocation. **STOP** after Phase 0 and wait for
`proceed to Phase 1`. Each subsequent phase has its own STOP condition;
do not chain phases without user direction.
