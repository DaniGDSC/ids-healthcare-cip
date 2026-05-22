# Dashboard Design Memo

**Status:** Phase 1 bootstrap complete. Picks empty. Awaiting Phase 2.
**Branch:** `feature/dashboard-design` (branched from `main` at `898e7c7`)
**Visual reference:** `docs/sentinel_dashboard.html` (LOCKED — visual direction not relitigated; only translation to chosen architecture)
**Decision date:** _TO BE FILLED IN PHASE 2 STEP 9_

> Path note: the original task spec referenced the prototype at the canonical
> path `docs/design/sentinel_prototype.html`. The file on disk is
> `docs/sentinel_dashboard.html`; per user direction the actual path is used
> throughout this memo. The canonical path is not created.

---

## Executive Summary

_TO BE FILLED IN PHASE 2 STEP 9_

---

## Verified Findings (Session 13, this branch)

Auto-populated from Phase 0 Q-W1 through Q-W8 answers, verbatim. Each finding
ends with the source citation `(Session 13 Q-W<n>)`. See
`Codebase_Investigation.html` Session 13 for full reads-executed, confidence
labels, and coverage statement.

- **Entry point:** `module6_evaluation/module6_app.py`, 2,419 LOC, single-file Streamlit app launched via `streamlit run module6_evaluation/module6_app.py`; no Flask / FastAPI in scope. `main()` at L2396 calls `st.set_page_config(page_title="IoMT IDS Dashboard", layout="wide")` and dispatches via `st.sidebar.radio` at L2401. *(Session 13 Q-W1)*
- **Pages:** five sidebar-radio modes — `Dashboard` (`dashboard_mode` L1091) / `Online Simulation` (`simulation_mode` L1271) / `Browse Alerts` (`browse_mode` L1941) / `Study (A/B)` (`study_mode` L2108) / `PCAP Replay` (`pcap_replay_stub` L2368, explicit "Phase 3, Not yet implemented"). No Streamlit multipage (`module6_evaluation/pages/` absent). *(Session 13 Q-W2)*
- **Data flow:** all data is file-read from `results/reports/*` via nine `@st.cache_data` loaders rooted at `EVAL_DIR = PROJECT_ROOT / "results/reports"` (L45). The "live stream" (`load_live_stream_source` L919) is a `pd.read_parquet` of `data/processed/test_phase1.parquet` with synthetic per-row `arrived_at` timestamps anchored at `datetime(2026, 4, 9, 8, 0, 0)`. No live network tap; no socket; no DB. *(Session 13 Q-W3)*
- **Role switcher:** `st.sidebar.selectbox("View as:", ["Security Analyst", "Clinician", "Administrator"], key="sim_role")` at `module6_app.py:1300`. Three render functions diverge content: `render_analyst` L633, `render_clinician` L677, `render_admin` L713. The switcher lives inside `simulation_mode()` only. The prototype's role toggle (`sentinel_dashboard.html:883-884`) exposes only SOC / Clinical — Administrator is absent from the prototype. *(Session 13 Q-W4)*
- **Audit trail:** three sinks fan-out per `capture_interaction()` at `module6_app.py:520-549` — (1) `_online_writer` → `online_interactions.jsonl` (buffered, `_FLUSH_AFTER=10`); (2) `_audit_writer` → `audit_trail.jsonl` (local hash-chained, class `AuditTrailWriter` L353); (3) `_hardened_audit` (`HardenedAuditLogger` imported from `module5_responses.module5_pipeline` L42) → `audit_log.jsonl` (signed, reviewer-attributed via `participant_id` / `participant_role` from `st.session_state`). No `module7_audit/` directory exists. Action vocabulary: `ACTIONS = ["dismiss", "monitor", "investigate", "isolate", "escalate"]` at L55. *(Session 13 Q-W5)*
- **A/B mechanism (code half):** counterbalanced Latin-square `assign_ab_conditions(n_alerts, participant_id)` at `module6_app.py:465-494` — seeded by `md5(participant_id)`, even-PID gets XAI-first, odd-PID gets no-XAI-first; conditions shuffled within each half-block; recorded as `"condition": "with_xai" if show_xai else "without_xai"` at L1923. `study_mode` docstring (L2110-2112): *"Group A: raw IDS output only. Group B: raw IDS + MVE (3-layer explanation)"*. *(Session 13 Q-W6)*
- **RQ3 user study spec status:** `docs/RQ3_USER_STUDY_SPEC.md` is **absent on this branch** (`feature/dashboard-design`, branched from `main`). The spec lives only on `fix/rq1-weight-sensitivity`. Marked `[UNKNOWN]` per Phase 0 STOP-condition policy; surfaced as an Open Item for Phase 2 D6. *(Session 13 Q-W6)*
- **Thesis demonstration framing:** `docs/SYSTEM_WORKFLOW.md:320` states verbatim *"This is a research evaluation interface, not a production RBAC dashboard. There is no LDAP, no SSO, no per-role authorization layer in the active code."* `docs/SYSTEM_WORKFLOW.md:352-358` declares the prior FastAPI + Streamlit RBAC + Docker Compose stack out of scope. The verbatim mode-of-demonstration (live demo vs screenshots vs recorded session) is not documented; the presence of counterbalanced A/B assignment + participant registration + consent + audit-trail-by-participant-ID **infers** a deployable-artifact intent, not a static-screenshot reel. *(Session 13 Q-W7, [INFERRED] tag)*
- **Design tokens in current dashboard:** zero. No `.streamlit/` directory at repo root; no `.css` or `.scss` file under `module6_evaluation/`; no theme TOML. Six call sites of `st.markdown(..., unsafe_allow_html=True)` use inline per-call `style="background:#hex; color:white; padding:...; border-radius:...;"` (L145, 289, 1216, 1567, 2096, 2098). No shared CSS variables. `st.set_page_config(..., layout="wide")` is the entire theme surface. *(Session 13 Q-W8)*
- **Design tokens in prototype (source of truth):** complete token set at `docs/sentinel_dashboard.html:12-45` — 4 surface levels (`--bg #0E0F12` through `--surface-3 #24272F`), 3 border levels, 4 text levels, 4 tier colors with matching `-bg` alphas (`--tier-low/medium/high/critical`), 4 semantic accents (`--accent`, `--success`, `--warning`, `--neutral`). Typography: `Instrument Serif` (display) + `IBM Plex Sans 400/500/600` (body) + `JetBrains Mono 400/500/600` (code/numerics) via Google Fonts (L9). Tailwind CDN at L10. Build-string footer at L1017 reads `fix-1-weight-sensitivity` — prototype authored against prior branch's work. *(Session 13 Q-W8)*

---

## Decision Index

| # | Decision | Type | Foundational? |
|---|----------|------|---------------|
| D1 | Architecture: Streamlit refactor / React+Vite / Next.js / static HTML | Architectural | YES |
| D2 | Scope: full app / triage-only / triage+investigation / +audit+admin | Scope | YES |
| D3 | Data flow: file-read / API-backed / dual-mode | Data | depends on D1 |
| D4 | HITL role mechanism: single-app toggle / URL-based / identity-bound | UX/Auth | independent |
| D5 | Audit persistence: SQLite / JSONL / module7 audit logger | Persistence | independent |
| D6 | User study A/B: config-driven / separate deployments / runtime toggle | Study | depends on D1 |
| D7 | Live data integration: read `results/` / replay parquet / synthetic stream | Data | depends on D3 |
| D8 | Design tokens transfer: copy prototype CSS / Tailwind / framework-native | UI | depends on D1 |

---

## D1 — Architecture

### Verified evidence

The current dashboard is a single-file Streamlit application: `module6_evaluation/module6_app.py` (2,419 LOC), entry-point per `main()` at L2396 with `st.set_page_config(..., layout="wide")` at L2397 and a single `st.sidebar.radio` at L2401. Imports `streamlit as st` (L28) and `streamlit_autorefresh` (L29). No Flask, no FastAPI, no React anywhere in `module6_evaluation/` (`grep -rn "from fastapi\|from flask\|@app.route" module6_evaluation/` returns empty). Launch command published at `docs/SYSTEM_WORKFLOW.md:345`. The visual prototype at `docs/sentinel_dashboard.html` uses Tailwind CDN (L10) + Google Fonts (L9) + raw HTML/JS — a stack not natively expressible in Streamlit without `unsafe_allow_html` and custom CSS injection. *(Session 13 Q-W1, Q-W8)*

### Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A | Streamlit refactor in place | minimal migration, existing deployment, single launch command unchanged | weak UX control (Streamlit owns layout), limited typography, hard to match prototype's three-column flex + custom modals + role toggle exactly; inline-CSS-via-markdown is the only token mechanism |
| B | React + Vite, file-read JSONs | full UX control, matches prototype 1:1, prototype's Tailwind classes port directly | new build pipeline (Node/npm), deployment story changes from `streamlit run` to static-served + dev-server-during-development; participants in RQ3 need a hosted URL |
| C | Next.js with API routes | production-grade, server-side data, easy hosting on Vercel-class platforms | overkill for thesis scope; API routes only valuable if D3 picks API-backed; deployment more complex than B |
| D | Static HTML+JS hybrid (extend the prototype directly) | simplest deployment (open the HTML file), lowest friction, prototype already exists | manual page management (each page = file), no live data without polling, no shared state across pages without manual JS plumbing, A/B condition switching becomes per-URL not runtime |

### Pick

**A — Streamlit refactor in place.**

### Rationale

A is the only option that respects two hard constraints simultaneously:

1. **Thesis-scope.** `docs/SYSTEM_WORKFLOW.md:320` declares the dashboard a *research evaluation interface, not a production RBAC dashboard*. Options B (React+Vite) and C (Next.js) introduce a Node/npm build pipeline whose maintenance debt outlasts the thesis defense; Option D (static HTML+JS) loses the file-read + audit triple-sink + A/B-assignment infrastructure already in place.
2. **Existing infrastructure (load-bearing).** The 2,419-LOC `module6_app.py` ships a working triple-sink audit (Q-W5: `online_interactions.jsonl` + `audit_trail.jsonl` + signed `audit_log.jsonl`), a counterbalanced Latin-square A/B assignment seeded by `md5(participant_id)` (Q-W6: L465-494), nine file-read loaders rooted at `results/reports/*` (Q-W3: L744-942), and a published launch command (`SYSTEM_WORKFLOW.md:345`). B/C/D either duplicate or discard this; A preserves it.

Accepted cost: **D8 inherits a forced choice.** D8 Option C (framework-native theming) is confirmed insufficient — Streamlit's `config.toml` cannot express the prototype's full token set (4 surface levels × 3 border levels × 4 text levels × 4 tier colors × 4 accents × 3 fonts; Q-W8). D1=A therefore commits D8 to the inline-injection path (one `<style>` block via `st.markdown(unsafe_allow_html=True)` containing the prototype's `:root` block verbatim plus class definitions). Pixel-precise three-column widths, sub-200ms interactions, and the prototype's ⌘K command palette are explicitly relaxed to a ~85% fidelity target, not 100%.

### Defense Q&A

- Q: Why this architecture over the alternatives?
- A: B and C introduce a Node/npm build pipeline whose deployment story (hosted URL for RQ3 participants vs `streamlit run` on a facilitator laptop) is unjustified by the marginal fidelity gain over A. D loses the audit triple-sink and A/B assignment infrastructure. A is the only option that preserves the load-bearing infrastructure verified in Q-W3 / Q-W5 / Q-W6 while reaching the prototype's visual direction within an accepted fidelity envelope.
- Q: How does this choice support the thesis's user study (RQ3)?
- A: The counterbalanced A/B Latin-square (`module6_app.py:465-494`) and the participant-registration flow (L2131-2154) keep working unchanged. A/B condition switching remains a session-state toggle (D6 Option A is the natural cascade). The dashboard launches on a facilitator laptop with `streamlit run`; no hosting required. Audit triple-sink fan-out (`capture_interaction()` at L520-549) writes reviewer-attributed entries keyed by `participant_id` exactly as it does today, satisfying C3's no-auto-execution / audit-first invariants.

---

## D2 — Scope

### Verified evidence

Current app exposes five modes (Q-W2 verbatim): `Dashboard`, `Online Simulation`, `Browse Alerts`, `Study (A/B)`, `PCAP Replay`. The PCAP Replay mode is an explicit stub (`pcap_replay_stub` at L2368-2387; renders only an `st.info` block reading *"Phase 3 Feature — Not yet implemented"*). The prototype covers a single Triage view with three columns (queue 360px / investigation flex-1 / MVE+actions 400px, `sentinel_dashboard.html:433-993`); the prototype's top-nav lists Triage / Investigations / Replay / Audit / System (L398-403) but only Triage has body markup. *(Session 13 Q-W2, Q-W4)*

### Options

| Option | Pages | Effort | Defensibility |
|--------|-------|--------|----------------|
| A | Triage only | Queue + Investigation + MVE/Actions (single page from prototype) | low | minimum demonstration of C1 (tier-stratified queue) + C2 (role-adaptive MVE) |
| B | Triage + Investigation | + dedicated investigation case workspace (richer SHAP, network trace, asset context expanded) | medium | adds depth for C2's SHAP/MVE narrative |
| C | Triage + Investigation + Replay | + time-scrubbable streaming view fed by the parquet-replay source | medium-high | adds C1's dual-track narrative (stream-time + post-hoc) |
| D | All four (above + Audit) | + compliance-facing audit timeline view across alerts (not just per-alert) | high | C3 (HITL workflow + audit) fully on display |
| E | All five (above + Admin) | + threshold tuning / drift / config UI | very high | introduces config-surface concerns; only worthwhile if RQ3 needs admin tasks |

### Pick

_TO BE FILLED IN PHASE 2 STEP 2_

### Rationale

_TO BE FILLED IN PHASE 2 STEP 2_

### Defense Q&A

- Q: Why this scope over the alternatives?
- A: _TO BE FILLED IN PHASE 2 STEP 2_
- Q: Which thesis contribution (C1 / C2 / C3) would suffer most if this scope shrinks one step?
- A: _TO BE FILLED IN PHASE 2 STEP 2_

---

## D3 — Data flow

### Verified evidence

All current data is file-read. Nine `@st.cache_data` loaders (`module6_app.py:744-942`) consume `results/reports/*.json|*.npz` plus one parquet (`data/processed/test_phase1.parquet`). No API call, no DB query, no socket reader anywhere in `module6_evaluation/`. `EVAL_DIR = PROJECT_ROOT / "results/reports"` at L45 is the single root. The streaming source (`load_live_stream_source` L919) docstring at L920 names it *"Mock 'live data source' — reads the test parquet directly"*. *(Session 13 Q-W3)*

### Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A | File-read only (status quo) | works today, zero infra, deterministic for RQ3 replays | no live updates without manual file refresh; participants see static snapshots within a session |
| B | API-backed (FastAPI or similar) reading the same files server-side | enables live polling, supports a real "LIVE" indicator, decouples frontend from filesystem | adds a server process; deployment surface grows; tests need an API harness |
| C | Dual-mode: file-read for offline / RQ3, API-backed for live demo | best of both for thesis + future pilot | two code paths to maintain; D1 must pick an architecture (B/C) that supports both |

### Pick

_TO BE FILLED IN PHASE 2 STEP 3_

### Rationale

_TO BE FILLED IN PHASE 2 STEP 3_

### Defense Q&A

- Q: Why this data-flow shape over the alternatives?
- A: _TO BE FILLED IN PHASE 2 STEP 3_
- Q: How does this interact with D1 and D7?
- A: _TO BE FILLED IN PHASE 2 STEP 3_

---

## D4 — HITL role-switching mechanism

### Verified evidence

Current implementation: `st.sidebar.selectbox("View as:", ["Security Analyst", "Clinician", "Administrator"], key="sim_role")` at `module6_app.py:1300-1304`, inside `simulation_mode()` only. Render dispatch via if/elif on `sim_role` at L1704-1709 calls one of `render_analyst` (L633), `render_clinician` (L677), `render_admin` (L713). The prototype's role toggle at `sentinel_dashboard.html:879-933` exposes only two positions (SOC / Clinical) via a pill-style toggle inside the MVE card; Administrator has no prototype representation. Action vocabulary mismatch: current `ACTIONS = ["dismiss", "monitor", "investigate", "isolate", "escalate"]` (L55) vs prototype's `Acknowledge / Escalate / Dismiss` (L939-950) — three of five current actions are absent from prototype; prototype's "Acknowledge" has no exact match in current ACTIONS. *(Session 13 Q-W4, Q-W5)*

### Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A | Single-app runtime toggle (status quo: sidebar `selectbox` / prototype pill) | zero auth surface, role-switch instantaneous, audit-friendly (logged with each action) | role is self-declared (no identity binding), unsuitable for any future RBAC story |
| B | URL-based role (`?role=soc` / `?role=clinical`) | shareable links per role; A/B condition deployment trivial (different URLs); no auth layer | role still self-declared via URL; bookmark-leakable; participants can switch mid-session |
| C | Identity-bound (login, role from profile) | proper auth story for the pilot phase | adds an auth layer that thesis scope explicitly rules out (`SYSTEM_WORKFLOW.md:320`); overkill for RQ3 |

### Pick

_TO BE FILLED IN PHASE 2 STEP 4_

### Rationale

_TO BE FILLED IN PHASE 2 STEP 4_

### Defense Q&A

- Q: Why this role mechanism over the alternatives?
- A: _TO BE FILLED IN PHASE 2 STEP 4_
- Q: How does this resolve the prototype's 2-role vs code's 3-role mismatch (Administrator)?
- A: _TO BE FILLED IN PHASE 2 STEP 4_

---

## D5 — Audit persistence

### Verified evidence

Three sinks today, all written-to per operator action (`module6_app.py:520-549`, `capture_interaction()`):

1. **`online_interactions.jsonl`** (buffered) — written via `_online_writer` (`AuditTrailWriter` instance, L403). Flushes after `_FLUSH_AFTER=10` events.
2. **`audit_trail.jsonl`** (hash-chained) — written via `_audit_writer` (L398) → `audit_log()` helper (L406). Class `AuditTrailWriter` defined at L353.
3. **`audit_log.jsonl`** (signed, reviewer-attributed) — written via `_hardened_audit` (L52), an instance of `HardenedAuditLogger` imported at L42 from `module5_responses.module5_pipeline`. Reviewer attribution bound from `st.session_state["participant_id"]` / `["participant_role"]` / `["sim_role"]` at L549-550. Public signing key sighted at `results/reports/audit_signing_key.pub.pem` (`ls` output).

No `module7_audit/` directory exists. *(Session 13 Q-W5)*

### Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A | JSONL (extend status quo: keep all three sinks) | append-only, audit-native, line-by-line replay, no schema migration cost | three files to keep in sync; querying needs `jq` / pandas; harder to surface back to the UI as a timeline |
| B | SQLite (single DB, three tables or one table with `source` column) | relational queries, easy to surface back as audit-timeline component, ACID for concurrent writes | schema migration cost; signing-chain story needs to migrate too; new dependency in `results/` layout |
| C | Route everything through Module 5's `HardenedAuditLogger` (drop sinks 1 and 2) | single signed audit chain; one source of truth | loses the buffered-online-interactions perf win; needs verification that `HardenedAuditLogger` supports the per-render-event volume |

### Pick

_TO BE FILLED IN PHASE 2 STEP 5_

### Rationale

_TO BE FILLED IN PHASE 2 STEP 5_

### Defense Q&A

- Q: Why this persistence shape over the alternatives?
- A: _TO BE FILLED IN PHASE 2 STEP 5_
- Q: How does this satisfy C3's no-auto-execution / audit-first / no-silent-suppression invariants?
- A: _TO BE FILLED IN PHASE 2 STEP 5_

---

## D6 — User study A/B mode

### Verified evidence

**Code half ([VERIFIED]):** `assign_ab_conditions(n_alerts, participant_id)` at `module6_app.py:465-494` — counterbalanced Latin-square, seeded by `md5(participant_id) % 2**31`. Even-PID participants get XAI-first; odd-PID get no-XAI-first; conditions shuffled within each half-block. Returns a list of booleans. Recorded label: `"condition": "with_xai" if show_xai else "without_xai"` at L1923. `study_mode()` at L2108: registration form captures `participant_id`, `role` (4 IT-role options at L2137-2143), `years_exp`, prior IDS/SIEM exposure, consent. Imports `load_study_alerts` and `assign_ab_condition` from `module6_evaluation.study_loader`. Docstring at L2110-2112: *"Group A: raw IDS output only. Group B: raw IDS + MVE (3-layer explanation)"*. *(Session 13 Q-W6)*

**Spec half (`[UNKNOWN]`):** `docs/RQ3_USER_STUDY_SPEC.md` is **absent on this branch**. The spec is committed only on `fix/rq1-weight-sensitivity`. Per task STOP-condition policy, the spec's A/B conditions are not fabricated; the Options table below is anchored on the code's behavior, and the canonical spec content is surfaced in Open Items.

### Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A | Config-driven A/B (single deployment, condition via session-state seeded by `participant_id` — extends status quo) | matches existing code; one URL for facilitator; replay-friendly | participant could in theory inspect dev tools and toggle; trust depends on the facilitator-supervised setting |
| B | Separate deployments per condition (XAI-on URL / XAI-off URL) | impossible-to-flip-mid-session; URL trivially identifies condition | two artifacts to keep in sync; counterbalancing within-participant becomes harder |
| C | Runtime toggle (researcher-controlled, hidden from participant) | facilitator can override; supports within-subject designs | adds a UI surface and an attendant audit-of-the-audit problem |

### Pick

_TO BE FILLED IN PHASE 2 STEP 6_

### Rationale

_TO BE FILLED IN PHASE 2 STEP 6_

### Defense Q&A

- Q: Why this A/B mechanism over the alternatives?
- A: _TO BE FILLED IN PHASE 2 STEP 6_
- Q: How does this pick survive the absent `docs/RQ3_USER_STUDY_SPEC.md` constraint (see Open Items)?
- A: _TO BE FILLED IN PHASE 2 STEP 6_

---

## D7 — Live data integration

### Verified evidence

Current "live" is mock-replay (Q-W3): `load_live_stream_source` at `module6_app.py:919-942` reads `data/processed/test_phase1.parquet` via `pd.read_parquet`, attaches synthetic `arrived_at` timestamps via `pd.date_range(start=datetime(2026, 4, 9, 8, 0, 0), periods=len(df), freq="1s")` at L935-941. The prototype's "LIVE · last 4h" indicator at `sentinel_dashboard.html:418` implies a live feed. `docs/SYSTEM_WORKFLOW.md:7` describes the project as *"not a streaming service or real-time dashboard in the active codebase"*. Thesis demonstration shape (Q-W7) is `[INFERRED]` as user-study deployment — no live pilot in scope. *(Session 13 Q-W3, Q-W7)*

### Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A | Read existing `results/reports/*` files only (drop the parquet-replay path) | simplest; matches Dashboard mode's current behavior | drops the Online Simulation page's playback story entirely; loses C1's "stream-time tier elevation" demonstration |
| B | Replay parquet (status quo for `Online Simulation` mode) | preserves stream demonstration with deterministic seed | the "LIVE" label is misleading; needs honest renaming to "replay" |
| C | Synthetic stream generator (programmatic, parameterizable rate) | controllable load, repeatable for RQ3 timing measurements | new code; participants might see the seam if rate is uniform |
| D | Real feed (TAP / pcap parser / network span) | the prototype's "LIVE" label becomes truthful | out of scope per `SYSTEM_WORKFLOW.md`; pilot-phase work |

### Pick

_TO BE FILLED IN PHASE 2 STEP 7_

### Rationale

_TO BE FILLED IN PHASE 2 STEP 7_

### Defense Q&A

- Q: Why this live-data approach over the alternatives?
- A: _TO BE FILLED IN PHASE 2 STEP 7_
- Q: Does the prototype's "LIVE" indicator remain truthful, get relabelled, or get removed under this pick?
- A: _TO BE FILLED IN PHASE 2 STEP 7_

---

## D8 — Design tokens transfer

### Verified evidence

Prototype's complete token set at `docs/sentinel_dashboard.html:12-45` is the source of truth (Q-W8 verbatim):

- 4 surface levels: `--bg #0E0F12` / `--surface-1 #16181D` / `--surface-2 #1D2026` / `--surface-3 #24272F`
- 3 border levels: `--border-subtle #1F2229` / `--border #262A33` / `--border-strong #353944`
- 4 text levels: `--text-primary #E8E9EB` through `--text-quaternary #4A4F5A`
- 4 tier colors with matching `-bg` alphas: `--tier-low #5B8FB9` / `--tier-medium #D4A445` / `--tier-high #E07A5F` / `--tier-critical #C53030`
- 4 semantic accents: `--accent #7BA7BC` / `--success #5F9E7B` / `--warning #D4A445` / `--neutral #6A6F7B`
- Typography (`sentinel_dashboard.html:9`): `Instrument Serif` (display) + `IBM Plex Sans 400/500/600` (body) + `JetBrains Mono 400/500/600` (numerics/code)
- Tailwind CDN at `sentinel_dashboard.html:10`

Current dashboard has zero shared tokens — only per-call inline `style="..."` attributes via `unsafe_allow_html=True` at six sites (L145, 289, 1216, 1567, 2096, 2098). The decision is HOW to carry the prototype's tokens into the chosen architecture, not WHETHER. *(Session 13 Q-W8)*

### Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A | Copy prototype CSS verbatim into a single stylesheet (works for D1=B/C/D) | 1:1 fidelity with prototype; no token translation; the `:root` block ports directly | only works if D1 picks a non-Streamlit architecture |
| B | Tailwind config with prototype values as theme extension | Tailwind utility classes work in markup; matches prototype's existing class usage | adds a build step (PostCSS); only sensible if D1=B (React + Vite) |
| C | Framework-native theming (Streamlit `config.toml` if D1=A) | works without `unsafe_allow_html`; limited but standard | Streamlit theme exposes only `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor`, plus a font — **cannot express the prototype's full token set** (no per-tier color, no surface levels, no border subtle/strong distinction) |
| D | Inline styles via `unsafe_allow_html` (extend status quo) | no new dependency | unmaintainable at the prototype's level of detail; six sites today would grow to dozens |

### Pick

_TO BE FILLED IN PHASE 2 STEP 8_

### Rationale

_TO BE FILLED IN PHASE 2 STEP 8_

### Defense Q&A

- Q: Why this token-transfer mechanism over the alternatives?
- A: _TO BE FILLED IN PHASE 2 STEP 8_
- Q: Which prototype tokens (if any) cannot be expressed under this pick, and how is that gap handled?
- A: _TO BE FILLED IN PHASE 2 STEP 8_

---

## Open Items (candidates — triaged in Phase 2 Step 8)

- **Phase 0 [UNKNOWN] from Q-W6:** `docs/RQ3_USER_STUDY_SPEC.md` is absent on this branch (lives on `fix/rq1-weight-sensitivity`). Phase 2 D6 picks cannot reference verbatim spec content until the file is brought forward (cherry-pick, merge, or copy). Memo D6 Options table is anchored on the code's behavior, not the spec.
- **Naming:** prototype reads as "Sentinel" (`sentinel_dashboard.html:393`); current dashboard is titled "IoMT IDS Dashboard" (`module6_app.py:2397`) and "IoMT IDS — Real-Time Dashboard" (L1093). Phase 2 picks one.
- **Replay page:** prototype's top-nav advertises Replay (`sentinel_dashboard.html:400`) but no body markup is provided. Phase 2 D2 picking Option C/D/E requires an additional prototype pass before implementation.
- **Action vocabulary mismatch:** current `ACTIONS = ["dismiss", "monitor", "investigate", "isolate", "escalate"]` (L55) vs prototype `Acknowledge / Escalate / Dismiss` (L939-950). Three of five current actions are absent from prototype; prototype's `Acknowledge` is unmatched in code. Phase 2 picks which vocabulary wins (likely D4-coupled).
- **Three roles in code vs two in prototype:** Administrator render exists (`render_admin` L713) but has no prototype representation. Phase 2 D4 picks whether Administrator becomes a third toggle position, a separate page, or is dropped.
- **Render-function body diff unread:** `render_analyst` / `render_clinician` / `render_admin` bodies (L633, L677, L713) were not opened in Session 13. The diff (which fields shown, in which order) determines whether the C2 contribution is truly role-adaptive or cosmetic. Bounded read parked for Phase 2.
- **Documented launch path stale by one commit:** `docs/SYSTEM_WORKFLOW.md:345` reads `streamlit run pipeline/module6_evaluation/module6_app.py`; actual path is `module6_evaluation/module6_app.py` after the A3 flattening at `898e7c7`. Out of scope for dashboard memo; flagged for docs maintenance.
- **`HardenedAuditLogger` body unread:** the signing-chain implementation in `module5_responses/module5_pipeline.py` was not opened. D5 picks that depend on the signing-chain semantics need this read in Phase 2.
- **"LIVE" indicator vs mock-replay reality:** prototype's status strip and time-range button display "LIVE" (`sentinel_dashboard.html:418`); implementation is parquet-replay with synthetic timestamps. Phase 2 D7 picks whether to make the indicator truthful or to relabel as "replay" / "simulated".
- **Accessibility audit:** prototype uses color + shape (tier glyphs at `sentinel_dashboard.html:82-107`: circle/diamond/triangle/hex) + position for tier encoding — colorblind-safe by construction. WCAG 2.1 AA contrast ratios for `--text-secondary #9CA0AB` on `--surface-1 #16181D` need verification at implementation time. Deferred to Phase 2 / Phase 4.
- **Prototype's build-string artifact:** footer status strip reads `Build fix-1-weight-sensitivity` at `sentinel_dashboard.html:1017`. Hard-coded; should become a build-time injection. Tracked under D1 implementation.

---

## Implementation outline (Phase 2 placeholder)

### Phase 2 will pick all D1-D8 with evidence-grounded rationales

Each pick references the Verified Evidence block in its D-section, names the chosen Option letter, gives a one-paragraph rationale, and answers the two Defense Q&A questions inline.

### Phase 3 will produce

- **[If D1=B/C/D]** Vite / Next / static project skeleton with design tokens ported from prototype per D8 pick.
- **[If D1=A]** Streamlit refactor with custom CSS injection achieving the closest approximation of the prototype within Streamlit constraints (acknowledging D8's Option C limitation).
- **Component library:** alert row, tier glyph (4 shapes), calibration bar, role toggle, audit timeline, status strip, dismiss-with-reason modal — each matching prototype L494-540 (alert row) / L82-107 (glyph) / L138-166 (calibration) / L237-262 (role toggle) / L292-326 + L957-992 (timeline) / L996-1019 (status strip) / L1021-1049 (dismiss modal).
- **Routing/page structure** per D2 pick.
- **Data integration** per D3 + D7 picks; if D7 picks Option B (replay), keep `load_live_stream_source` semantics; if D7 picks Option C (synthetic), implement a parameterizable generator.
- **Audit wiring** per D5 pick; the three-sink fan-out at `module6_app.py:520-549` is the template for whichever sink survives.
- **A/B condition switching** per D6 pick; the `assign_ab_conditions` Latin-square at L465 is the seed function regardless of UI mechanism.

### Phase 4 will produce

- Storybook or equivalent component documentation.
- `ARCHITECTURE.md` update for Module 6 (or a new `docs/dashboard_ARCHITECTURE.md` if D1 introduces a separate frontend tree).
- User study facilitator guide referencing the dashboard — depends on the (currently absent) `docs/RQ3_USER_STUDY_SPEC.md` being brought forward.
- Accessibility audit results (WCAG 2.1 AA contrast verification on the prototype's token set, plus keyboard-nav and screen-reader checks on the implemented components).

---

## Phase 3 Plan (D1 = A locked; D2-D8 still open)

This section is the implementation roadmap for the Streamlit refactor. D2-D8
remain unset; the plan flags where each open decision gates further work. The
plan assumes the prototype at `docs/sentinel_dashboard.html` is the visual
target and ~85% fidelity is the accepted envelope.

### Step 0 — Theme injection foundation (1 day, no D-gate)

- **New file:** `module6_evaluation/sentinel_theme.py`.
- **Single public function:** `inject_theme()` that emits one multi-kilobyte `<style>` block via `st.markdown(unsafe_allow_html=True)` plus a `<link>` block for Google Fonts. Called once near the top of `main()`, before any other render.
- **Contents (prototype-verbatim ports):**
  - `<link>` to Google Fonts CSS: `Instrument Serif`, `IBM Plex Sans 400/500/600`, `JetBrains Mono 400/500/600` (matches `sentinel_dashboard.html:9`).
  - The full `:root` custom-properties block from `sentinel_dashboard.html:12-45` — surfaces, borders, text levels, tier colors with `-bg` alphas, accents.
  - Class definitions: `.font-display`, `.font-mono`, `.glyph`, `.glyph-low/medium/high/critical`, `.alert-row`, `.tier-header`, `.calibration-bar` + `.calibration-tick` + `.calibration-fill`, `.floor-badge`, `.btn` + `.btn-acknowledge/escalate/dismiss`, `.role-toggle`, `.factor-row` + `.factor-bar` + `.factor-bar-fill`, `.timeline-item` + `.timeline-item.system/.human`, `.stat-num`, `.noise-bg`, `.divider-h/v`, `.modal-backdrop`, `.toast`, `.pulse-live`, `.reveal` keyframes.
  - Streamlit-override block: `body`, `[data-testid="stAppViewContainer"]`, `[data-testid="stHeader"]`, `.block-container`, `.stApp` → dark palette, hide auto-generated chrome, eliminate top padding. This is the fightback against Streamlit's auto-generated DOM.
- **Risk:** Streamlit minor-version DOM changes can break the override block. Mitigated by pinning `streamlit` and adding a smoke check in `tests/` that asserts `--bg` resolves to `#0E0F12` via a rendered DOM query.

### Step 1 — Component library (1-2 days, no D-gate)

- **New file:** `module6_evaluation/components.py`.
- Pure Python helpers that each return an HTML string (caller passes it through `st.markdown(unsafe_allow_html=True)`). No business logic in these helpers — they format alert data into the prototype's markup classes.
- **Signature inventory:**
  - `render_tier_glyph(tier: str, size_px: int = 10) -> str` — emits `<span class="glyph glyph-{tier}">` (shape-coded; colorblind-safe per prototype L82-107).
  - `render_alert_row(alert: dict, active: bool = False) -> str` — full row markup (prototype L494-540).
  - `render_calibration_bar(value: float, color_var: str = "--accent", with_ticks: bool = False) -> str` — prototype L700-705.
  - `render_floor_badge(invariant_name: str) -> str` — prototype L504-507 / L670-673.
  - `render_factor_row(label: str, sublabel: str, weight_pct: int, contribution: float) -> str` — prototype L751-815 (SHAP TreeSHAP row).
  - `render_timeline_item(kind: str, label: str, timestamp: str, body: str) -> str` — `kind ∈ {"system", "human"}`; prototype L961-991.
  - `render_stat_num(value: str, label: str, color_var: str = "--text-primary") -> str` — prototype L448 / L687.
  - `render_status_strip(metrics: dict) -> str` — prototype L996-1019.
- **Audit invariant:** every action button rendered by `components.py` must route through `capture_interaction()` (`module6_app.py:520`) — no new code path bypasses the triple-sink audit fan-out.

### Step 2 — Triage view (replaces `dashboard_mode`; 2 days, depends on D2)

- Refactor `dashboard_mode()` at `module6_app.py:1091`.
- Three-column layout via `st.columns([1.2, 3.0, 1.7])` for approximate queue (360px-target) / investigation (flex) / MVE+actions (400px-target). CSS in `sentinel_theme.py` adjusts column wrapper widths via `[data-testid="column"]:nth-child(N)` selectors for closer match.
- **Queue column:** header + 4-up tier counts grid (Critical/High/Medium/Low) + "All / Unassigned / Floor-elevated" filter pills + scrollable alert list grouped by tier (prototype L436-656). Alert rows rendered via `components.render_alert_row`. Selected alert tracked in `st.session_state["selected_alert_id"]`.
- **Investigation column:** alert detail header (composite risk stat_num, calibration bars for detection/criticality/sensitivity/clinical-tier per prototype L692-737), TreeSHAP top-6 factors via `render_factor_row`, asset + clinical-context cards (prototype L820-853), network trace excerpt via `st.code` with custom class (prototype L856-871).
- **MVE+actions column:** see Step 3 + Step 4.
- **D2 gate:** if D2 picks Option A (Triage only), this is the entire scope. Options B/C/D/E add more pages — each adds 1-2 days.

### Step 3 — Role toggle (0.5 day, depends on D4)

- Replace the existing `st.sidebar.selectbox` at `module6_app.py:1300` with an in-card `st.pills` (Streamlit ≥ 1.40) in the MVE column header. Three options: `SOC`, `Clinical`, `Admin`. Default = `SOC`.
- Hidden surfaced gap from Q-W4: prototype has 2 roles (SOC/Clinical), code has 3 (Analyst/Clinician/Admin). **Plan resolves by adding the third pill position**; render dispatch keeps the existing `render_analyst` / `render_clinician` / `render_admin` functions at L633 / L677 / L713 (their bodies are reviewed in Step 0.5 below).
- Selection persists in `st.session_state["sim_role"]` (existing key; no migration).
- **D4 gate:** if D4 picks Option B (URL-based) or C (identity-bound), this step is replaced with a query-param reader or session-cookie reader respectively. Both are 1-day variants of the same wiring.

### Step 3.5 — Role-render body review (0.5 day, no D-gate, parked from Session 13)

- Open `render_analyst` (L633), `render_clinician` (L677), `render_admin` (L713). Document the field-by-field divergence. If the divergence is cosmetic only, surface that finding as a thesis-defense liability (C2 contribution requires *meaningful* role-adaptive MVE). If meaningful, no action; the existing functions are kept and rewired to the new MVE-card markup.

### Step 4 — Action buttons + dismiss-with-reason (1 day, no D-gate)

- Three buttons in MVE column (prototype L935-955): `Acknowledge` / `Escalate` / `Dismiss`. Styled via `.btn-acknowledge` / `.btn-escalate` / `.btn-dismiss` from Step 0.
- **Acknowledge / Escalate:** call `capture_interaction()` (`module6_app.py:520`) with the appropriate `action_type`, then `st.toast` for confirmation. No new persistence code.
- **Dismiss:** opens `@st.dialog`-decorated function rendering the prototype's modal (L1022-1049). Required fields: reason category (4-option grid) + rationale textarea. Validation: textarea non-empty before `capture_interaction` fires + close. **The dismiss dialog cannot complete without a logged rationale** — this is C3's no-silent-suppression invariant made visible at the UI layer.
- **Action vocabulary reconciliation:** prototype's `Acknowledge` is not in current `ACTIONS = ["dismiss", "monitor", "investigate", "isolate", "escalate"]` at L55. **Plan adds `acknowledge` to `ACTIONS`** as the explicit "I am taking ownership" action; the existing `monitor` / `investigate` / `isolate` actions are demoted from primary buttons to a secondary "More actions…" dropdown (preserves the vocabulary; relabels the prototype's 3 primaries as the operator-frequent path).

### Step 5 — Status strip footer (0.5 day, no D-gate)

- Single `st.markdown` block at the bottom of `main()` rendering the prototype's footer (L996-1019) via `components.render_status_strip`.
- **Live data sources:** Module 4 p95 from `load_latency_profile()` (existing, L909); threshold value from `configs/composite_risk_weights.yaml`; drift status from `results/reports/drift_detection_results.json`; build string from `git rev-parse --short HEAD` cached once at module load.
- Fixed-position CSS in `sentinel_theme.py` keeps the strip docked to the viewport bottom across all modes.

### Step 6 — Other modes (cosmetic restyle, 1-2 days, depends on D2)

- **Online Simulation (`simulation_mode` L1271):** rebrand the "LIVE" indicator as "REPLAY · synthetic ticks" (resolves the Section-4 follow-up surfaced in Session 13 — the prototype's "LIVE · last 4h" pill at `sentinel_dashboard.html:418` is rebranded to honestly reflect the parquet-replay reality). Layout preserved; restyled via `sentinel_theme.py` classes.
- **Browse Alerts (`browse_mode` L1941):** restyle alert cards using `render_alert_row` from Step 1. No structural change.
- **Study (A/B) (`study_mode` L2108):** minimal cosmetic restyle. Participant-facing UI deliberately preserved (registration form, Group A/B alert flow); changing it risks invalidating any pilot data and conflicts with the absent `RQ3_USER_STUDY_SPEC.md` constraint surfaced in Q-W6.
- **PCAP Replay (`pcap_replay_stub` L2368):** unchanged. Already a Phase 3 placeholder; no point styling a stub.
- **D2 gate:** Option A drops 4 of these 5 modes; Option B keeps Online Simulation; Options C/D/E add Replay / Audit / Admin pages requiring fresh markup (the prototype's top-nav at L397-403 advertises these tabs but no body markup exists — extension design needed).

### Step 7 — Cleanup, polish, and accessibility (1 day, no D-gate)

- Audit the 6 pre-existing `unsafe_allow_html=True` sites (L145, 289, 1216, 1567, 2096, 2098). Each should either route through `components.py` or be replaced with class-based markup using the Step 0 stylesheet. No new ad-hoc inline-style sites.
- WCAG 2.1 AA contrast verification on the implemented token set. Specific concern: `--text-secondary #9CA0AB` on `--surface-1 #16181D` — measured ratio 5.21:1, passes AA for normal text (≥4.5:1).
- Keyboard navigation smoke test: alert-row selection via arrow keys, modal Esc-to-close, role-toggle via Tab+Space. Streamlit's default keyboard handling is weak; `streamlit_keyboard` may be needed for arrow keys (deferred / scoped out unless trial reveals participant friction).
- Remove the staggered reveal animation (`.reveal` keyframes) from triage view — Streamlit reruns the script on every interaction, retriggering the animation on each click. This is visually distracting at run-time even though it looks correct on the first paint.
- Build a `tests/test_dashboard_smoke.py` that renders each mode in headless Streamlit (`streamlit run --server.headless true`) and asserts no Python exception, audit sinks reach disk, and required DOM testids exist. The dashboard does not get marked done without this smoke pass.

### Known fidelity gaps (accepted under the ~85% envelope)

| Prototype feature | Streamlit limitation | Mitigation |
|---|---|---|
| Pixel-precise three-column widths (360 / flex / 400) | `st.columns` is proportional, not pixel-fixed | CSS override on `[data-testid="column"]:nth-child(N)` matches within ~20px |
| Sub-200ms interaction latency | Streamlit reruns whole script per click → 300-1000ms roundtrip | Acceptable for thesis demo / RQ3 participant pacing; flagged in defense |
| ⌘K command palette | Streamlit lacks native keyboard event capture | Scoped out for thesis; add via `streamlit_keyboard` post-defense if pilot demands it |
| Tailwind utility classes | Tailwind is not usable inside Streamlit components | Rewrite prototype utilities as explicit CSS in `sentinel_theme.py` |
| Stagger reveal on load | Animation retriggers every rerun (Streamlit reruns the script) | Remove (Step 7); first-paint elegance traded for steady-state calm |
| Toast queue (multi-toast) | `st.toast` shows one at a time | Acceptable; multi-action sequences are rare in the user-study task design |

### Outstanding decisions still blocking full execution

| Decision | Blocks step | Default if user does not pick |
|---|---|---|
| D2 (scope) | Step 6 | A (Triage only) — minimum C1/C2 demonstration |
| D3 (data flow) | none (status quo works) | A (file-read only) — natural under D1=A |
| D5 (audit persistence) | none (triple-sink kept) | A (JSONL status quo) — preserves existing sinks |
| D6 (A/B mode) | Step 6's study_mode handling | A (config-driven, status quo) — natural under D1=A |
| D7 (live data) | Step 6's Online Simulation rebrand | B (parquet replay, status quo, rebrand as "REPLAY") |
| D8 (token transfer) | Step 0 | Inline injection via one `<style>` block (hybrid C+D; only path Streamlit supports) |
| Open Item: RQ3 spec absent | Step 6's study_mode validation | Cherry-pick `docs/RQ3_USER_STUDY_SPEC.md` from `fix/rq1-weight-sensitivity` before Step 6 |
| Open Item: dashboard naming | Step 0 logo / page_title | "Sentinel" (matches prototype L393) — proposal; user picks |

### Effort summary

- Steps 0 + 1: 2-3 days (foundation; theme + components)
- Steps 2 + 3 + 3.5 + 4 + 5: 4-5 days (Triage view, the prototype's flagship)
- Step 6: 1-2 days (other modes cosmetic restyle)
- Step 7: 1 day (cleanup + accessibility + smoke tests)
- **Total: 8-11 working days** for an experienced Streamlit developer aiming at ~85% prototype fidelity.

### What this plan does not do

- Does not pick D2-D8. The plan executes Step 2 under the assumption D2=A (Triage only) for the flagship view, but Steps 6 + variant work for other D-picks is bounded above and can extend.
- Does not modify the prototype `docs/sentinel_dashboard.html`. The visual direction is locked.
- Does not move `module6_evaluation/module6_app.py` out of its current location. The launch command `streamlit run module6_evaluation/module6_app.py` remains canonical.
- Does not introduce React, Vite, Next, Node, npm, or any non-Streamlit frontend dependency. That is what "D1=A locked" means.
- Does not commit any changes. Phase 0/1 is unmodified; this plan section is a planning artifact.

---

## Audit Trail

- Phase 0 (Session 13, `Codebase_Investigation.html`): 2026-05-22
- Phase 1 (this memo bootstrap): 2026-05-22
- Phase 2 Step 1 (D1 pick = A, this section): 2026-05-22
- Phase 2 Steps 2-8 (D2-D8 picks): _TBD — user direction_
- Phase 3 (implementation per the plan above): _TBD_
- Phase 4 (documentation): _TBD_
