# Fix 5 Decision Matrix — Online/Offline Framing

**Date:** 2026-05-21
**Source:** `Codebase_Investigation.html` Sessions 5 (markdown inventory) and 6 (code verification).
**Purpose:** Enumerate framings the user can choose from. No recommendation. The user selects one framing; the v2 master prompt (`prompts/fix5_v2.md`) consumes the selection and branches its edit plan accordingly.

---

## Phase 0 / 0b Evidence Summary

The following facts are grounded in Session 6's `[VERIFIED]` code reads and are the load-bearing inputs to every framing below. Citations are file:line.

1. **An "online-capable" per-alert explanation module exists in code.** `module4_explanations/module4_online_explainer.py` is 977 lines (Session 5's doc-cited "887" was wrong); its docstring states "*Online-capable per-alert explanation pipeline with latency profiling*" and "*Design: online-capable, validated in batch mode on the test set*" (`module4_online_explainer.py:2-7`).

2. **No production driver invokes the full online pipeline.** Import-graph grep across `*.py` returns one production caller (`module4_explanations/module4_explanations.py:175`), and it imports only the helper `_feature_to_narrative` — not the `AlertExplainer` class or `main()`. The other importers are `tests/test_step11_shap_stability.py:22`, `tests/test_safe_failure.py:283,304`. `main()`'s data path reads `data/processed/test_phase1.parquet` + `results/models/xgboost_test_predictions.npz` (`module4_online_explainer.py:839,845`) — no live ingest anywhere.

3. **The `<150 ms` SLA is documentation, not enforcement.** It appears at `module4_online_explainer.py:5` (docstring), `:143` (comment), and `:968-971` (post-hoc log string `"PASS"` or `"FAIL"`). No `raise`, no `assert`. Measured `p95 = 216.841 ms` (`results/reports/online_latency_profile.json:56`) means the file's own log would emit `"FAIL"` under current data.

4. **The Streamlit "Online Simulation" is a replay page, not a streaming runtime.** `module6_evaluation/module6_app.py:838-846` defines `stream_simulator` as a generator that iterates a pre-loaded list with `time.sleep(delay)`. Source data is the pre-computed `alert_responses.json` (`module6_app.py:2493`). The page is sidebar-registered at `:3909` ("Online Simulation") and routed at `:3914`.

5. **`src/harness.py` "streaming" is test infrastructure.** Module docstring `:1-5` states "*Testing infrastructure only — not part of the production system.*" The streaming generator iterates `tests/fixtures/sample_alerts.yaml` (a static YAML). Sole caller: `run_tests.py:219`.

6. **`online_latency_profile.json` has no provenance metadata.** Top-level keys: `n_alerts_total` (677), `n_full_explanations` (604), `n_minimal_explanations` (73), `startup_ms` (8278.9), plus nested per-stage stats. No date, no source-data hash, no code version stamp, no sample population description. File `mtime`: Apr 9, 2638 bytes.

7. **`thesis_outline_latest.docx` is absent.** Never committed (`git log --all -- "thesis_outline_latest.docx"` &rarr; 0); never deleted (`--diff-filter=D --name-only -- "*.docx"` &rarr; 0); not on filesystem (`find / -name "thesis_outline*.docx"` &rarr; 0). Any framing's "thesis Section 6" edit is unverifiable in this repo context.

---

## Framings

Four framings are evaluated below. The required three (F-A, F-B, F-C) plus one additional (F-D) supported by Phase 0b evidence.

### Framing column key

- **One-line statement** — the sentence the user would write at the top of the scope-owning doc (ARCHITECTURE.md §1 or a new README).
- **Supporting evidence** — Phase 0/0b findings that uphold this framing.
- **Contradicting evidence** — Phase 0/0b findings that make this framing harder to defend.
- **ARCHITECTURE.md edits required** — approximate count and nature, derived from Session 5 inventory + Session 6 verification.
- **Code changes required** — explicit list of any non-doc edit (e.g., page-string rename). Per task prompt, doc-only is the scope; surfaced separately so the user can see what each framing implies if extended.
- **Defense risk** — questions a thesis examiner might raise.
- **Time estimate** — rough doc-edit hours (and code-change days if extended).

---

### F-A — Hybrid architecture

| Column | Content |
|---|---|
| One-line statement | "The system separates *offline training and batch artifact generation* from *per-alert SHAP/MVE inference*; both are first-class architectural pillars." |
| Supporting evidence | The per-alert architecture is genuine: `AlertExplainer` exists with one-time-load + per-call-explain semantics (`module4_online_explainer.py:264-313`); `SHAPContext` and `MVEOutput` are designed as in-memory per-alert objects; ARCHITECTURE.md:3 already states "offline-first … separates batch data preparation … from the online user interface". |
| Contradicting evidence | The "online" pipeline has no production driver invoking it; the entrypoint runs batch over test parquet (`module4_online_explainer.py:839`); the `<150 ms` SLA fails under measured latency (`p95 = 216.841 ms`). |
| ARCHITECTURE.md edits required | ~0–1. The existing `OFFLINE (one-time training)` / `ONLINE INFERENCE (per alert)` diagram labels (lines 141, 169) are consistent with this framing as-is. Optional: add a "Batch vs Streaming Equivalence" subsection clarifying that "online" means per-record semantics, not live network ingest. |
| Code changes required | None. |
| Defense risk | Examiner: "What does 'online' mean here if no live data source exists?" — Answer must explain that "online" denotes per-alert semantics + low-latency design, validated by batch replay of the test set. The `<150 ms` SLA gap (p95 = 217 ms) is an honest measurement that must be acknowledged. |
| Time estimate | Doc-only: 1–2 hours. |

---

### F-B — Online prototype, batch production

| Column | Content |
|---|---|
| One-line statement | "The production research evaluation is the offline batch path; a per-alert online-inference prototype is implemented and latency-profiled, with a documented SLA gap; production-grade streaming is named as future work." |
| Supporting evidence | The latency profile is real (`online_latency_profile.json` n=677, p95=216.841 ms); the SLA claim (`<150 ms`) is documented at `module4_online_explainer.py:5`; the measurement-vs-claim gap is concrete and defensible as "prototype findings"; no live data path exists, supporting the "future work" naming for streaming. |
| Contradicting evidence | The latency profile has no provenance metadata (date, code version, source hash) — without these, the prototype's reproducibility for thesis claims is weaker; "prototype" framing implicitly commits the thesis to acknowledging the SLA gap, which is non-trivial. |
| ARCHITECTURE.md edits required | 1–2. (i) Add a new "Per-Alert Inference Prototype" subsection that names the SLA target, the measured value, and the gap explicitly. (ii) Add a "Streaming as future work" subsection naming production-grade requirements (live NetFlow ingestion, per-record dispatch outside Streamlit replay, runtime latency-budget enforcement). |
| Code changes required | None (the prototype already exists and is profiled; no new code needed to support this framing). |
| Defense risk | Examiner: "Why is the SLA gap acceptable for a thesis claim?" — Answer must frame the prototype as research, not deployment, and the latency profile as evidence of feasibility-with-known-gaps rather than production-readiness. Examiner may also ask why the latency profile has no provenance metadata. |
| Time estimate | Doc-only: 2–4 hours (more text than F-A because the SLA gap needs explicit framing). |

---

### F-C — Simulation only

| Column | Content |
|---|---|
| One-line statement | "The system is a batch evaluation pipeline; the 'Online Simulation' page is a demo replay, not a streaming runtime; all 'online' labels are renamed to 'per-record (simulated)' to avoid ambiguity." |
| Supporting evidence | The Streamlit Online Simulation page provably replays pre-loaded JSON with `time.sleep()` (`module6_app.py:838-846, 2493`); `src/harness.py` is "Testing infrastructure only" per its own docstring (`:1-5`); the `<150 ms` SLA fails under measured latency, weakening any "online runtime" claim. |
| Contradicting evidence | The per-alert architecture is genuine (not just a simulation): `AlertExplainer` has real per-call semantics; `SHAPContext`/`MVEOutput` are per-record by design; renaming "online" to "simulated" understates the architectural intent and may confuse readers about why per-alert objects exist. |
| ARCHITECTURE.md edits required | 3–5. Rename the diagram label at line 169 (`ONLINE INFERENCE (per alert)` &rarr; `PER-RECORD INFERENCE (batch-processed)` or similar). Rename `Online (per alert):` at `docs/architecture.md:31`. Add a "Simulation framing" paragraph to ARCHITECTURE.md §1 clarifying that "Online Simulation" is a replay. Optionally rewrite references in chapter3 / section312 docs. |
| Code changes required | Optional but consistent: rename the Streamlit page string `"Online Simulation"` → `"Per-Record Replay"` in `module6_app.py:3909, 3914, 2490` and re-label the "Real-Time Dashboard" title at `:2194`. **Out of doc-only scope per task prompt's `DO NOT DO` rule** — surface as deferred work if this framing is chosen. |
| Defense risk | Examiner: "If the architecture isn't online, why are there per-alert objects like `SHAPContext`?" — Answer must explain per-record semantics as a correctness property (no temporal coupling across alerts) independent of any streaming runtime, but the framing makes this distinction harder to articulate. |
| Time estimate | Doc-only: 3–5 hours (multiple files touched). Code rename: +0.5 day if extended. |

---

### F-D — Per-record specification, batch-only execution

| Column | Content |
|---|---|
| One-line statement | "The pipeline specifies per-record (alert-independent) semantics for correctness; the implemented execution path is batch over the frozen test split; all reported numerical claims derive from this batch execution." |
| Supporting evidence | The "online-capable, validated in batch mode" docstring (`module4_online_explainer.py:7`) is *itself* this framing in the file's own words; no production driver runs per-alert; the test/sim harness uses generator-streaming patterns for memory, not for live ingest (`src/harness.py:48-54`); `AlertExplainer` is structured as load-once + call-many but is only ever called batch via `main()`. |
| Contradicting evidence | The Streamlit Online Simulation page name + the `"Real-Time Dashboard"` title (`module6_app.py:2194`) + the SLA log string (`:968-971`) all suggest a deployment-ready framing that F-D explicitly rejects; readers who saw the latency profile may expect a deployment claim. |
| ARCHITECTURE.md edits required | 1–2. Add an "Operational model: per-record specification, batch execution" subsection. Optionally tighten the diagram label at line 169 from `ONLINE INFERENCE (per alert)` to `PER-RECORD INFERENCE` (drop "ONLINE" without renaming to "simulated"). |
| Code changes required | None (this framing accepts the existing code naming as honest design vocabulary). |
| Defense risk | Examiner: "What is the difference between per-record-and-batch vs online-and-batch?" — Answer must articulate that per-record refers to alert independence (a correctness property), while batch refers to execution mode (an operational property); the two are orthogonal. Examiner may follow up: "Then what is the latency profile measuring?" — Answer: opportunistic per-call timing inside the batch run, useful as feasibility evidence but not a deployment SLA. |
| Time estimate | Doc-only: 1–2 hours. |

---

## Cross-framing comparison table

| Aspect | F-A | F-B | F-C | F-D |
|---|---|---|---|---|
| Acknowledges online code is real | Yes | Yes | No (calls it simulation) | Yes (as design intent) |
| Acknowledges SLA gap | Optional | **Required** | Optional | Optional |
| Requires renaming Streamlit pages | No | No | **Yes (if extended to code)** | No |
| Requires new ARCHITECTURE.md section | Optional | **Yes** (prototype + future work) | Yes (simulation framing) | Yes (per-record/batch operational model) |
| Doc-edit count | ~0–1 | 1–2 | 3–5 | 1–2 |
| Compatible with thesis emphasizing deployment-readiness | Weak (SLA gap unaddressed) | **Strong** | Weak (claims no deployment) | Moderate (claims feasibility, not deployment) |
| Compatible with thesis emphasizing safety / explainability architecture | **Strong** | Moderate | Moderate | **Strong** |
| Compatible with thesis emphasizing research-prototype framing | Moderate | **Strong** | Moderate | **Strong** |
| Effort (doc-only) | 1–2 h | 2–4 h | 3–5 h | 1–2 h |

---

## How to Choose

The criteria below are descriptive, not normative. They translate framing choices into the kind of thesis defense the user will face.

- **If the thesis emphasizes deployment-readiness** (e.g., the contribution is "an IDS that could be deployed in a hospital"): F-B is the most direct fit because it addresses the latency SLA gap head-on. F-A is plausible only if the user is willing to defend the gap in another way.

- **If the thesis emphasizes safety/explainability architecture** (e.g., the contribution is "a per-alert reasoning chain with stability/MVE guarantees, evaluated offline"): F-A or F-D both fit. F-A keeps the existing diagram labels intact; F-D tightens the operational vocabulary at the cost of one more subsection.

- **If the thesis emphasizes research-prototype framing** (e.g., the contribution is "an evaluation methodology + supporting prototype"): F-B or F-D both fit. F-B requires admitting the SLA gap; F-D sidesteps it by reframing latency as feasibility evidence rather than deployment SLA.

- **If the thesis avoids any deployment claim entirely** (e.g., the contribution is "an offline analysis of clinical-IDS trade-offs"): F-C fits, at the cost of renaming several diagram labels and potentially Streamlit page strings.

- **If the user is uncertain which thesis emphasis applies**: F-A is the lowest-edit-cost framing and is consistent with the current code state; choosing F-A defers the framing decision without committing to anything that the evidence refutes.

The choice also depends on whether the user is willing to introduce a "future work" subsection naming production-grade streaming requirements. F-B requires this; F-A, F-C, F-D do not require it but may include it optionally.

---

## What Happens After Selection

1. The user sets `CHOSEN_FRAMING` to one of `F-A`, `F-B`, `F-C`, `F-D` at the top of `prompts/fix5_v2.md` (produced in Phase 0d of this recovery).
2. The user invokes `prompts/fix5_v2.md`. Its Phase 1 branches on `CHOSEN_FRAMING` and emits the framing-specific edit plan derived from the "ARCHITECTURE.md edits required" + "Code changes required" rows above.
3. The user reviews Phase 1's plan and replies "proceed to Phase 2."
4. Phase 2 executes the edits under the Write-Mode Safety Protocol (BEFORE-VIEW → str_replace → AFTER-VIEW).
5. Phase 3 verifies via the standard query suite (V-1 … V-5), adjusted to the chosen framing.

The framing chosen here does *not* commit the user to any specific edit until Phase 1 of the v2 prompt produces a concrete plan the user can amend.

---

## Constraints inherited from the recovery prompt

- Doc-only scope. Code changes (page renames, harness restructuring) are surfaced for each framing but require explicit out-of-scope authorization to execute.
- No edits to `results/_pre_*/` archived snapshots.
- The `thesis_outline_latest.docx` edit named in the original Fix 5 master prompt cannot be performed in this repo — the file is absent from history and filesystem (Q-V6). Any framing's "thesis Section 6" edit must be deferred to the human author or surfaced as an Open Item in the execution log.

---

*This matrix records framings only. The v2 master prompt (`prompts/fix5_v2.md`, produced next in Phase 0d) is where the selected framing is operationalized into an edit plan.*
