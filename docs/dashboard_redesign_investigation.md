# Dashboard Redesign — Investigation Report

> Pre-redesign audit of `module6_evaluation/module6_app.py` answering 18 scoping questions across 5 priority tiers.
>
> Generated: 2026-05-06 · Branch: `fix/shap-category-vocab` · Source file: 2,485 lines

---

## Table of contents

| Tier | Questions | Topic |
|---|---|---|
| CRITICAL — current state | Q1, Q2, Q3 | What the dashboard is, why rebuild, what to preserve |
| CRITICAL — audience | Q4, Q5, Q6, Q7 | Who uses it, primary use case, demo, devices |
| HIGH — technical | Q8, Q9, Q10, Q11 | Streamlit, data pipeline, performance, browsers |
| HIGH — IA | Q12, Q13, Q14 | Page purposes, navigation, alert card density |
| MEDIUM — scope | Q19, Q20, Q21 | Must-haves, nice-to-haves, explicit cuts |

---

## Q1 — What the existing dashboard looks like

**File:** [module6_evaluation/module6_app.py](../module6_evaluation/module6_app.py) — **2,485 lines**. Single Streamlit script, **41 top-level functions, 1 class**, ~114 `st.session_state.*` references.

### Page topology — `main()` at [module6_app.py:2462](../module6_evaluation/module6_app.py#L2462)

Sidebar `st.radio` selects one of 4 modes:

| # | Mode | Entry | Purpose |
|---|---|---|---|
| 1 | **Dashboard** | [`dashboard_mode()` :1082](../module6_evaluation/module6_app.py#L1082) | 5-tier metric strip → tier+category bar charts → risk-gauge + 15-row alert feed → SHAP waterfall + clinician NLG → response panel + DO-NOT → global SHAP. Auto-refresh toggle (30 s). |
| 2 | **Online Simulation** | [`simulation_mode()` :1268](../module6_evaluation/module6_app.py#L1268) | Streaming playhead (0.5×/1×/2×/4× via `st.fragment`), role switcher (Analyst/Clinician/Admin), per-alert FDA audit-record export, latency profile + adaptive-threshold + drift panels. ~570 lines. |
| 3 | **Browse Alerts** | [`browse_mode()` :1927](../module6_evaluation/module6_app.py#L1927) | Slider-driven single-alert review with XAI on/off toggle and "Recommended Action" affordance. |
| 4 | **Study (A/B)** | [`study_mode()` :2107](../module6_evaluation/module6_app.py#L2107) | Registration → 20-alert A/B (raw vs. raw+MVE) → checkpoint every 5 alerts → 2 proxy questions → save `study_responses_<PID>.json`. |

### Reusable display helpers (all in same file)

- [`render_device_criticality` :138](../module6_evaluation/module6_app.py#L138), [`render_prioritized_actions` :184](../module6_evaluation/module6_app.py#L184), [`render_do_not_constraint` :212](../module6_evaluation/module6_app.py#L212), [`render_mve_layers` :239](../module6_evaluation/module6_app.py#L239) — Layer 1/2/3 expanders with `_DO_NOT_FALLBACKS` mapped per device class.
- Three role views: [`render_analyst` :608](../module6_evaluation/module6_app.py#L608), [`render_clinician` :652](../module6_evaluation/module6_app.py#L652), [`render_admin` :688](../module6_evaluation/module6_app.py#L688).
- [`AuditTrailWriter` :367](../module6_evaluation/module6_app.py#L367) — buffered append-only JSONL with SHA-256 hash chain. Plus a `HardenedAuditLogger` from Module 5 wired in at line 58.
- [`likert_form` :430](../module6_evaluation/module6_app.py#L430) — 5-point trust/usefulness/comprehensibility/actionability + free-text + reclassify.

### Backend performance (artifacts already on disk)

[results/reports/online_latency_profile.json](../results/reports/online_latency_profile.json): n=677 alerts, total p50=66 ms · **p95=217 ms** · p99=230 ms. `predict_ms` dominates (mean 96 ms). NLG/risk/DAE-decompose are sub-millisecond.

---

## Q2 — Why a rebuild is being considered (pain points)

### A. Code-quality issues

1. **Single ~2,450-line god-file.** Everything — page routing, data loading, audit log, session-state setup, 4 mode functions, 3 role renderers, helpers, FDA-record builder — lives in one file.
2. **Layered patches with cryptic markers.** Grep finds **"Gap 1/2/3", "FIX B/C", "Issue 1–11", "M6-A1/A4", "UX-S-01/B-01/X-01/X-02"** scattered as comments. Retrofitted fixes, not designed-in features.
3. **`presentation_v4.py` is built but NOT wired in.** [presentation_v4.py](../module6_evaluation/presentation_v4.py) defines the 9-class `BADGE_FOR_ALERT_TYPE`, 4-level `CONFIDENCE_INDICATOR`, and `MODE_INDICATOR` (Mode A/B). Only `validate_nine_alert_types.py` and tests import it — **`module6_app.py` never does**. Dashboard still renders the legacy 4-tier `TIER_COLORS` palette.
4. **Session-state sprawl** — ~114 references; each mode adds its own keys (`_sim_acc`, `_tier_history`, `_processed_alerts`, `_fda_payload_cache`, `_fda_filename_cache`, `_live_preview_cache`, `_render_ms_history`, `_latency_df_cache`…). Easy to leak across modes.

### B. UX / feature gaps (documented)

5. **Heuristic eval H7 = PARTIAL** ([docs/heuristic_evaluation.md](heuristic_evaluation.md) §3.1): "no keyboard shortcuts/bulk-action UX in Streamlit dashboard" — recorded as **GAP-HE-4**.
6. **9-alert-type taxonomy not visible.** Layer 3 v4 emits 9 `AlertType` values (incl. unique `DISAGREEMENT_ANOMALY` purple/adversarial badge); operators currently see only the 4-tier colour code.
7. **Confidence + Mode A/B indicators not surfaced.** Same root cause — `presentation_v4.py` not imported.
8. **DAE per-dimension errors not rendered.** `anomalous_dims_markdown()` exists; no caller in the dashboard.
9. **Cognitive overload in Dashboard mode.** UX-S-01 / UX-X-02 comments show panels were retroactively wrapped in collapsed expanders to reduce triage distraction — designed-in problem, patched late.
10. **NIST RMF "Manage" = PARTIAL** — drift dashboard is simulation-only, not live (GAP-HE-3).

### C. Performance (mostly mitigated, but tells you the original was slow)

12. **Issue 1–11 fixes** are O(n)→O(1) optimizations on the simulation playhead (Counter rebuilds, DataFrame reconstructions, deque slices, regex compiles, file open-per-event). The fact that 11 such patches were needed reveals the original render loop didn't budget for fragment-rate updates at 4× speed (0.5 s tick).
13. Backend p95 = 217 ms is fine; visible latency budget is in Streamlit re-renders, which are not measured systemically (only an opt-in `/tmp/sim_render_timings.jsonl`).

### D. Architecture coupling

14. **No screenshots, no live URL captured.** No `docs/figures/dashboard*.png` exists. Cannot review actual visual design without running it.
15. **No Streamlit-specific tests.** Only `tests/test_layer5_v4_presentation.py` (19 pure-function tests) covers the new helpers. No harness exercises the 4-page flow end-to-end.

---

## Q3 — What works well and must be preserved

### Hard requirements (cited as INVARIANTs / passing heuristics)

1. **Append-only signed audit chain.** [`AuditTrailWriter` :367](../module6_evaluation/module6_app.py#L367) + [`HardenedAuditLogger` import :48](../module6_evaluation/module6_app.py#L48). INVARIANT 4. Backed by `tests/test_audit_append_only.py`. Format = JSONL with `prev_hash` + SHA-256 `integrity_hash`. **Never break this.**
2. **3-layer MVE rendering with DO-NOT prominence.** `render_mve_layers` + `render_do_not_constraint` + `_DO_NOT_FALLBACKS` per device class. INVARIANT 7.
3. **Three role views (Analyst/Clinician/Admin)** — H4/H6 PASS in heuristic eval.
4. **A/B study harness.** `study_mode` + `study_loader.py` (MD5 deterministic shuffle, counterbalanced A/B). Output schema is locked — downstream `study_analysis.py` and `analyze_rq3.py` read it.
5. **Checkpoint + resume.** `study_responses_*.json` and `study_checkpoint_*.json` survive a closed browser. Preserve the PID-conflict dialog (Resume / Overwrite / Cancel).
6. **Severity tier palette** (`TIER_COLORS` + `_CRIT_COLOR_HEX`) is consistent across pages — H4 PASS evidence.
7. **FDA-style audit-record export per alert** (`build_fda_record_for_alert`) — referenced as production-readiness evidence.

### Soft preserves (good UX patterns)

8. **Mode-radio sidebar** is simple and works — keep the 4-mode model.
9. **Per-tier expander auto-expansion**: `expanded=(level in ("CRITICAL", "HIGH"))` at line 1684.
10. **Streaming controls UX**: Pause/Resume/Step/Reset/Jump-to-#. Speed selector. Status pill.
11. **Inline truth labels in Browse mode sidebar** (researcher self-test).

### Data pipelines / fixtures (don't touch)

12. **Loaders** (`load_alerts`, `load_all_responses`, `load_admin_dashboard`, `load_clinician_summaries`, `load_response_policy`, `load_audit_trail`, `load_latency_profile`, `load_live_stream_source`) all read from `results/reports/` JSON/parquet artifacts. **Schema is the contract.**
13. **`@st.cache_data` decorations** on loaders + `_cached_png_bytes` for SHAP charts — already correctly memoized.

---

## Q4 — Who actually uses this dashboard

**Answer: Option A (thesis demo only) + Option C-LLM (LLM personas as research subjects). Option B (real operators) is explicitly out of scope.**

### Evidence — direct quote from architecture lock document

> [docs/system_architecture_final.md:562](system_architecture_final.md#L562)
> **"Real-user component: none (deliberately — all evaluation methods are standards-grounded or simulation-based; suitable for thesis defence without IRB-bound user-study evidence)."**

### Concrete user inventory

| Actor | Count | Role | Evidence |
|---|---|---|---|
| **Examiners / advisor** | small | Watch a defense demo | [docs/heuristic_evaluation.md:116](heuristic_evaluation.md#L116) "passes the threshold for thesis defence" |
| **Solo developer** | 1 | Test fixes, run study mode for self-checks | git status, single-author repo |
| **LLM personas (Method 1)** | **100** | "Subjects" of A/B study. **Never see the Streamlit UI** — consume `group_a_display`/`group_b_display` strings as text | [analysis/run_llm_persona_simulation.py](../analysis/run_llm_persona_simulation.py) — 50 IT × 30 Biomed × 20 Nurse = 2000 LLM calls |
| "P01–P25" study response files | 25 | All marked **`participant_role: "Legacy Survey Participant"`** — leftover/synthetic, not recruited humans | `cat results/reports/study_responses_P19.json` |
| **Real hospital IT staff** | **0** | Phase-3 only, gated on field deployment (C5 untestable) | [research_spec.yaml:613-617](../research_spec.yaml#L613) `phase: "Phase 3 — requires field deployment"` |

### What this drives

- **Visual polish vs. functional:** polish matters for the ~5 minutes during defense; the rest of the time nobody is looking.
- **Onboarding / help system:** **near-zero**. Examiners don't onboard.
- **Performance requirements:** must handle a single user clicking through ~20 alerts smoothly with a projector. Multi-operator concurrency, 50-alerts/day throughput — out of scope until Phase 3.

---

## Q5 — Primary use case

**Answer: B + C, with B (research demonstration) dominant. NOT A (live monitoring).**

### Priorities

1. **B — Research demonstration.** Architecture doc lists the dashboard at [Layer 5 Presentation](system_architecture_final.md#L65-L72). Each "page" maps to a research claim (C1 narrative, C2 risk-adaptive, C3 actionability, C4 A/B, C5 future).
2. **C — Evaluation tool.** Study mode is fully wired. The 100-persona LLM simulation **bypasses the UI entirely** — so the "evaluation" use of the dashboard is for the future human study (Phase 2), not the current LLM run.
3. **A — Live monitoring: NOT a real use case.** No live data source: simulation_mode reads from pre-computed `evaluation_alerts.json`. Drift detection and adaptive threshold are explicitly **simulation-only** (GAP-HE-3).

### What this drives

- **Default landing page:** Dashboard mode (visual story for examiners).
- **Information architecture:** organize pages by **what claim they evidence**:
  - Dashboard → "what operators would see" (C1, C3, C4)
  - Online Simulation → "real-time at 50–100ms p50" (C2 risk-adaptive)
  - Browse Alerts → "every alert type renders with full MVE" (C7, M1)
  - Study (A/B) → "the protocol producing 100-persona evidence" (C4)
- **Feature prioritization:** unwired `presentation_v4.py` (9 alert-type badges, confidence dots, Mode A/B indicator) is **the highest-leverage visual upgrade.**

---

## Q6 — Demo constraints

**No explicit demo time/format is written down anywhere in the repo.** Reasonable inference:

### What the repo tells us

- **No `docs/demo.md`, no `presentation/`, no recorded screencasts** in the project tree.
- Heuristic eval frames everything around "thesis defence" (British spelling — possibly UK/EU institution).
- Study mode is "30–40 minutes" per in-app instructions ([module6_app.py:2179](../module6_evaluation/module6_app.py#L2179)) — **future participant** experience, **not the defense demo**.

### Reasonable assumption (confirm with user)

A typical thesis defense demo is **5–10 minutes of live UI** within a 45–60 minute defense. That window has to:

1. Show one alert end-to-end (input → MVE → recommended action) — **2 min**
2. Show one CRITICAL+unpatchable case to demonstrate the safety floor — **1 min**
3. Show the role switcher (IT vs Biomed vs Nurse) on the same alert — **1 min**
4. Show A/B differentiation (one alert with MVE vs without) — **1 min**
5. Buffer / Q&A trigger — **1–2 min**

### What this drives

- **Optimize the "single alert deep-dive" path.** Browse mode is closest, but forces sidebar interaction. Presenter wants **one click per beat**.
- **Pre-load demo alerts.** A "demo playlist" of 4–5 hand-picked alerts (CRITICAL ventilator, HIGH EHR, MEDIUM IoT, LOW false-positive, DISAGREEMENT_ANOMALY).
- **Avoid features that need >3 clicks to reach the payoff.** Examiners won't wait.
- **Skip features that fail under projector light.** The `BADGE_FOR_ALERT_TYPE` palette uses two yellows (`#FACC15` SUSPICIOUS vs `#EAB308` CONFIRMED) — risky on projector.

---

## Q7 — Operator device context

**Answer: Desktop only. Zero responsive design exists in the codebase.**

### Evidence

```bash
grep -rE "responsive|tablet|mobile|breakpoint|viewport|@media|min-width" \
  --include='*.py' --include='*.md' --include='*.css'
# → 2 hits, both unrelated:
#   "Insulin pump ... mobile" (device class, not screen size)
```

- `st.set_page_config(page_title="IoMT IDS Dashboard", layout="wide")` at [module6_app.py:2463](../module6_evaluation/module6_app.py#L2463) — `layout="wide"` is the ONLY layout directive. Streamlit's `wide` mode optimizes for ≥1280 px width.
- All `st.columns(...)` use proportional splits.
- No CSS, no `@media` queries, no viewport-aware styles.
- Defense projectors are typically 1920×1080 or 1280×720. **Both are within desktop-wide envelope.**

### What this drives

- **Optimize for one screen size: 1920×1080 projector / laptop external display.** Don't waste effort on tablet or mobile.
- **Remove the 5-column metric strip on Dashboard mode** — competes for attention. Two columns reads better at projector distance.
- **Font sizing**: Streamlit defaults are too small for projector viewing at >2 m. Consider `html { font-size: 18px; }` CSS bump.
- **Ignore touch-friendly targets, gesture support, accelerometer, etc.**

---

## Q8 — Streamlit framework constraints

**Answer: Option A is the *current* state, but you are NOT structurally locked in. Migration cost is contained to one file.**

### Evidence

| Signal | Finding |
|---|---|
| `requirements.txt` includes streamlit? | **No.** Only numpy/pandas/sklearn/tensorflow/shap. |
| CLAUDE.md classification | "**Optional**: anthropic, imbalanced-learn (SMOTE), **streamlit**" |
| Files that import streamlit | **1 file only**: [module6_evaluation/module6_app.py](../module6_evaluation/module6_app.py). Everything else (Modules 0–5, all `src/`, `analysis/`, `tests/`) is pure Python. |
| Streamlit version | 1.55.0 installed (recent enough for `st.fragment(run_every=...)`, `st.cache_data`, `width="stretch"`) |
| `streamlit_autorefresh` | Optional, wrapped in `try/except ImportError` with no-op fallback ([module6_app.py:30-35](../module6_evaluation/module6_app.py#L30)) |
| `.streamlit/config.toml` | **Does not exist** — all defaults |
| Custom CSS / `st.html` / custom components | **None used.** Some `unsafe_allow_html=True` for severity colour spans. |
| Session-state / `st.rerun()` references | **118 occurrences** — heavy coupling to Streamlit's rerun model |

### Three structural realities

1. **Non-UI codebase is framework-agnostic.** Modules 3–5 return plain Python dicts/dataclasses. Any frontend can consume them.
2. **4-page UI is fully encapsulated** in one file. Migration replaces this file; nothing else changes.
3. **Rewrite cost is the session-state untangling**, not the framework switch. 118 `st.session_state.*` references means every page carries implicit state.

### Recommendation

**Stay with Streamlit for the thesis.** Audience doesn't care. Migration to React/FastAPI buys flexibility you won't use in 5–10 demo minutes. **Document UI debt** for Phase 3.

What you CAN do inside Streamlit, in order of leverage:
- `st.html()` (Streamlit 1.33+) for fully custom alert cards
- Custom CSS via `st.markdown(<style>...</style>, unsafe_allow_html=True)`
- `st.fragment(run_every=...)` — already used; expand to limit rerun blast radius
- Don't bother with custom `streamlit-component` packages (NPM build chain)

---

## Q9 — Data pipeline

**Answer: Option A (static files) — entirely. The "live" toggle is a labelled mock.**

### Evidence

All 11 dashboard loaders read from `results/reports/*.json` (`@st.cache_data`-wrapped):

| Loader | Reads | Size | Refresh model |
|---|---|---|---|
| `load_alerts` | `evaluation_alerts.json` | **102 KB, 20 alerts** | Cached on first call, regenerated by `python module6_evaluation/module6_evaluation.py` |
| `load_all_responses` | `alert_responses.json` (+ join `evaluation_alerts.json`) | (file present) | Same |
| `load_admin_dashboard` | `admin_dashboard.json` | — | Same |
| `load_clinician_summaries` | `clinician_summaries.json` | 182 KB | Same |
| `load_response_policy` | `response_policy.json` | — | Same |
| `load_audit_trail` | `audit_trail.json` | **3.7 MB** ⚠️ | Same — large file, fully loaded into memory |
| `load_risk_scores` | `risk_scores.npz` | — | numpy .load |
| `load_latency_profile` | `online_latency_profile.json` | — | Module 4 produces |
| `_cached_png_bytes` | `results/charts/*.png` | per-file, max 64 entries | LRU |
| `load_live_stream_source` | `data/processed/test_phase1.parquet` | parquet | "**Live parquet (mock TAP)**" — synthetic timestamps, deterministic per session |

### Total data footprint

`du -sh results/reports/` = **7.6 MB**. All cold-loaded on first script run; subsequent reruns hit the cache.

### What this drives

- **Polling vs. WebSocket: neither needed.** No real-time data.
- **Caching strategy: already correct.** `@st.cache_data` everywhere; `_cached_png_bytes` for SHAP charts is the load-bearing perf optimization.
- **Loading states: minimal needed.** A 102 KB JSON loads in <50ms. The 3.7 MB `audit_trail.json` could need a spinner.
- **Demo risk: file generation is offline.** Plan a **fixed demo dataset** ahead of defense.
- **Watch out for `audit_trail.json` (3.7 MB).** Consider lazy-loading only on FDA-export click.

---

## Q10 — Performance budget

**Documented budgets exist for the backend; no formal frontend render budget — but there's a soft 150 ms SLA marker in code.**

### What's documented

| Layer | Budget | Source | Measured | Status |
|---|---|---|---|---|
| **Per-alert end-to-end** | **150 ms** | [docs/threat_model.md:237](threat_model.md#L237) "T-D1 Model overload" | total p50 = **66 ms**, p95 = **217 ms** | **p95 EXCEEDS** target — surfaced in UI as "⚠️ exceeds 150 ms SLA" / "✅ within 150 ms SLA" warning ([module6_app.py:1664](../module6_evaluation/module6_app.py#L1664)) |
| **Layer 2 detector p95** | **500 ms** | [tests/test_layer2_v4_invariants.py:207](../tests/test_layer2_v4_invariants.py#L207) | n=50 calls, asserts < 500 ms | PASS |
| **Frontend page load** | not documented | — | not measured | — |
| **Frontend render per rerun** | not documented | — | instrumented opt-in (`/tmp/sim_render_timings.jsonl`) | unknown |
| **Filter / form / interaction** | not documented | — | not measured | — |

### Backend stage breakdown

| Stage | mean | p95 | p99 |
|---|---|---|---|
| `predict_ms` | 96 ms | **199 ms** | 210 ms ← **dominates** |
| `treeshap_ms` | 9 ms | 21 ms | 23 ms |
| `dae_decompose_ms` | 0.034 ms | 0.064 ms | 0.106 ms |
| `nlg_ms` | 0.013 ms | 0.028 ms | 0.031 ms |
| `risk_decompose_ms` | 0.046 ms | 0.098 ms | 0.13 ms |
| **`total_ms`** | **104 ms** | **217 ms** | 230 ms |
| Startup | — | — | **8.3 s** (one-time) |

### Recommended budgets (none documented for frontend today)

| Metric | Recommended target | Rationale |
|---|---|---|
| Initial cold page load | **< 3 s** | Streamlit 8.3 s startup is pre-demo — keep server warm |
| Switch between pages (mode radio) | **< 500 ms** | Already fast since data is cached |
| Alert card render (Browse mode slider) | **< 150 ms** | Matches per-alert SLA |
| Form submission (Likert / study response) | **< 200 ms** | JSON write + `st.rerun()` cycle |
| Online Simulation playhead tick (1× speed = 2 s) | **render < 500 ms** | Otherwise auto-refresh stutters |

### What this drives

- **Don't add more visual complexity to Dashboard mode.** `predict_ms` p99 of 210 ms already eats the per-alert SLA.
- **Keep the `_cached_png_bytes` pattern.**
- **Issue 1–11 fixes** are over-engineered for the audience (single user, 5 min demo) but harmless.

---

## Q11 — Browser targets

**Answer: No requirements documented. Inherit Streamlit's defaults.**

### Evidence

- Zero references to "Chrome", "Firefox", "Safari", "Edge", "IE", "browser", "user-agent", "webkit" in any doc, spec, or config.
- No `.streamlit/config.toml` exists — Streamlit serves with all defaults.
- Streamlit 1.55.0 officially supports recent Chrome, Firefox, Safari, Edge (no Internet Explorer).
- No PWA manifest, no service worker, no `meta viewport` overrides.

### What this drives

- **Test on Chrome/Edge for the demo.** That's what's typically on a presentation laptop.
- **Specifically test on the actual defense projector setup.** Yellow rendering on projectors washes out — verify `#FACC15` vs `#EAB308` distinction in `BADGE_FOR_ALERT_TYPE`.
- **Skip mobile / Safari iOS / IE compatibility work.**
- **No need for browser feature-detection code.**

---

## Q12 — What each page actually does (and where they overlap)

### Per-page actual purpose (code-true)

| Page | Entry | Real purpose | Data source |
|---|---|---|---|
| **Dashboard** | [`dashboard_mode` :1082](../module6_evaluation/module6_app.py#L1082) | **Aggregate summary** of pre-computed alerts — 5 tier-count metrics → distribution charts (collapsed) → risk-gauge for ONE selected alert (selectbox over first 20) → 15-row alert-feed table → SHAP + NLG side-by-side → Response panel + DO-NOT → global SHAP (collapsed). Auto-refresh toggle (30 s) — but data is static, refresh just re-runs. | `alert_responses.json`, `admin_dashboard.json`, `clinician_summaries.json`, charts |
| **Online Simulation** | [`simulation_mode` :1268](../module6_evaluation/module6_app.py#L1268) | **Sequential replay** with playback controls (0.5×–4×). Shows last 3 alerts as expanders, auto-expand on CRITICAL/HIGH. **Only role switcher** in app. Latency + threshold + drift panels (collapsed). FDA audit-record export per alert. | Same `alert_responses.json` + optional mock parquet stream |
| **Browse Alerts** | [`browse_mode` :1927](../module6_evaluation/module6_app.py#L1927) | **Random-access single-alert review.** Slider 0..n-1. XAI toggle. Sidebar exposes ground-truth + correct-action (researcher mode). | `evaluation_alerts.json` (canonical 20-alert set) |
| **Study (A/B)** | [`study_mode` :2107](../module6_evaluation/module6_app.py#L2107) | **Forced sequential study.** Registration → 20 MD5-shuffled alerts → counterbalanced A/B → forced decision form → checkpoint every 5 → save `study_responses_<PID>.json`. Hides truth labels. | `evaluation_alerts.json` + `study_loader.py` |

### Overlap matrix

| Pair | Overlap | What's truly different | Verdict |
|---|---|---|---|
| **Dashboard ↔ Online Simulation** | ~70% | Sim adds: playback controls, role switcher, latency/threshold/drift panels, FDA export, last-3 expander window. Dashboard adds: 15-row feed table + global SHAP. Both use `load_all_responses()`. | Could merge into **one page** with "Static / Replay" toggle. |
| **Online Simulation ↔ Browse Alerts** | ~50% | Sim sequential with auto-advance. Browse random-access via slider. Browse exposes truth labels and XAI on/off toggle. | Could merge with sequential vs. random toggle + truth/XAI toggles. |
| **Browse Alerts ↔ Dashboard** | ~30% | Dashboard's `st.selectbox` picker IS a slider in disguise. | Different framing; keep distinct. |
| **Study Mode** | Structurally isolated | Forces sequence, forces form, hides truth, no toggles. | **Keep separate.** Output schema is locked. |

### Page-purpose recommendation: Three pages, not four

1. **Triage Console** ← merge Dashboard + Online Simulation. Default = static aggregate. Toggle to "Replay". Role selector promoted to top of page.
2. **Alert Inspector** ← keep Browse Alerts. Random-access deep dive. XAI on/off remains. Truth labels collapsible (researcher-only).
3. **Study (A/B)** ← keep as-is. Locked output schema, isolated UX.

---

## Q13 — Navigation pattern

### Current state

| Sidebar element | Where defined | Lives in |
|---|---|---|
| App title `IoMT IDS` | [line 2466](../module6_evaluation/module6_app.py#L2466) | Always visible |
| **Mode radio** | [line 2467](../module6_evaluation/module6_app.py#L2467) | Always visible |
| Dashboard: Auto-refresh toggle | [line 1087](../module6_evaluation/module6_app.py#L1087) | Dashboard only |
| **Sim: "Stakeholder View" role selectbox** | [line 1299](../module6_evaluation/module6_app.py#L1299) | **Online Simulation only** |
| Sim: Debug toggles | line 1306-1315 | Online Simulation only |
| Sim: Data Source radio | line 1320 | Online Simulation only |
| Browse: XAI toggle, Alert # slider, Truth labels | line 1932-1940 | Browse only |

### Three structural problems

1. **Role selector buried in one page.** Role is the most important conceptual axis (Pillar P2) but Dashboard, Browse, and Study render only one default view per page — burying claim C1 and metric M5.
2. **Sidebar mixes navigation, persistent state, and per-page filters.**
3. **No breadcrumb or contextual title.** No "you are watching this stream **as a clinician**" indication.

### Recommended structure

```
┌─────────────────────────────────────────────────────────────────┐
│ TOP BAR (always visible)                                         │
│   IoMT IDS    [Role: ▾ IT Generalist / Biomed / Nurse]   [Help] │
├──────────┬──────────────────────────────────────────────────────┤
│ SIDEBAR  │ MAIN                                                  │
│ Pages:   │   Page-contextual filters at TOP of main area:        │
│ • Triage │     ┌──────────────────────────────────────────────┐  │
│ • Insp.  │     │ Severity: ▾ All ▾   Status: ▾ Active ▾       │  │
│ • Study  │     └──────────────────────────────────────────────┘  │
│          │   Content...                                          │
└──────────┴──────────────────────────────────────────────────────┘
```

- **Top bar**: app title + **role selector** (persistent across pages) + help.
- **Sidebar**: 3 pages only (Triage / Inspector / Study).
- **Page-contextual filters** at top of main area, not sidebar.

How to do this in Streamlit:
- Top bar = `st.columns([4, 2, 1])` at top of `main()`, before page dispatch
- Role selectbox writes to `st.session_state.role` — single source of truth
- Per-page filters live inside each `*_mode()` function as the first row

---

## Q14 — Alert card density (per page)

### Current state

| Page | Density today | Pattern | Problem |
|---|---|---|---|
| **Dashboard** | **Hybrid (B + A)** | 15-row compact `st.dataframe` feed + 1 selected alert at full fidelity | Selected-alert section consumes ~70% of vertical space → scrolling required. Visual overload on projector. |
| **Online Simulation** | **Hybrid C (expandable)** | `current_batch_local = responses[idx-2 : idx+1]` — last 3 alerts as `st.expander`, **auto-expanded for CRITICAL/HIGH** | Best density choice in codebase. Auto-expand-on-severity is clean. But three full role-rendered cards stacked is heavy. |
| **Browse Alerts** | **Pure A (one at a time)** | Slider → render `display_alert(alert, show_xai)` | Cannot compare alerts. XAI on/off is the only comparison primitive — forces mode switch instead of side-by-side. |
| **Study Mode** | **Pure A** | Forced sequence, one alert | Correct for use case. Don't change. |

### What the audience actually needs

- **At-a-glance evidence of scale (B compact list)** — answers "does this work on more than one alert?"
- **Deep narrative on one specific alert (A focused)** — answers "show me the MVE actually working on a CRITICAL ventilator alert."
- **Comparison between two alerts (C expandable side-by-side)** — evidences v4 9-class taxonomy by showing e.g. KNOWN_ATTACK vs DISAGREEMENT_ANOMALY badges side-by-side. **Not currently supported.**

### Recommendation: density per page

| Page | Density | Why |
|---|---|---|
| **Triage Console** | **B compact list (15-25 rows) + drill-down to A focused on click** | Most common demo flow: scan → click → narrate. Use `st.dataframe` row-selection (Streamlit 1.35+). |
| **Alert Inspector** | **C expandable (2-up side-by-side)** | New capability. Two alert columns, each with role-rendered MVE. This is where the v4 9-class story lives. |
| **Study Mode** | **A focused (unchanged)** | Locked by study protocol. |

### One specific change worth making early

**Replace Dashboard's "selectbox over first 20 alerts" + always-rendered detail panel** ([line 1144](../module6_evaluation/module6_app.py#L1144)) **with `st.dataframe` row selection** (Streamlit 1.35+: `selection_mode="single-row"` and `on_select="rerun"`). 30-line change that removes the worst piece of current IA.

---

## Q19 — Must-haves for thesis defense

### Tier 1 — Required by INVARIANTs / acceptance tests

| Must-have | Backing | Status today | Work needed |
|---|---|---|---|
| **3-layer MVE display** | INVARIANT 7, M1, M1b | ✅ wired ([render_mve_layers :239](../module6_evaluation/module6_app.py#L239)) | None |
| **DO NOT prominent box** for CRITICAL clinical | INVARIANT 7, M4 | ✅ wired ([render_do_not_constraint :212](../module6_evaluation/module6_app.py#L212)) | None |
| **Audit log writes on operator decision** | INVARIANT 4, REQ-MVE-15 | ✅ wired ([AuditTrailWriter :367](../module6_evaluation/module6_app.py#L367) + HardenedAuditLogger) | None |
| **Operator decision form** | INVARIANT 4 + Method 5 study contract | ✅ wired ([likert_form :430](../module6_evaluation/module6_app.py#L430)) | None |
| **Role-tailored views** (IT / Biomed / Nurse) | INVARIANT 6, DARPA P2, Method 1 | ⚠ exists but **role selector only in Online Simulation** | Promote role selector to top bar |
| **Severity tier colour coding** | Nielsen H4, REQ-MVE-04 | ✅ wired (`TIER_COLORS`, `_CRIT_COLOR_HEX`) | None |
| **Recommendation-only output** | INVARIANT 3 | ✅ architectural | None |
| **Maintenance-window safety floor visible** | INVARIANT 2 | ⚠ enforced in scoring, not surfaced visually | Add "Safety floor invoked" indicator on card — small change, high demo impact |

### Tier 2 — Required by v4 contract / claims, currently UNWIRED

These are the **largest gap** and the **highest-leverage demo additions**.

| Must-have | Backing | Status | Work needed |
|---|---|---|---|
| **9 alert types with badges** (purple `#9333EA` for DISAGREEMENT_ANOMALY exclusively) | Layer 5 v4 contract, [validate_nine_alert_types.py](../module6_evaluation/validate_nine_alert_types.py) | ❌ **`badge_for_alert_type` exists, never imported** | ~50 lines: import in `module6_app.py` and call inside alert card |
| **4-level Confidence indicator** (●●●● VERY_HIGH … ● LOW) | Layer 5 v4 contract, DARPA P4 | ❌ **`confidence_display` exists, never imported** | ~10 lines |
| **Mode A (LLM) vs Mode B (rule-based) indicator** | Layer 5 v4 contract, DARPA P4 | ❌ **`mode_display` exists, never imported** | ~10 lines |
| **MITRE ATT&CK technique per role** | C8 / Layer 4 v4 contract | ❌ `format_mitre_for_alert_type` exists in `module4_explanations/triage_v4_adapter.py`, **never called by dashboard** | ~20 lines |
| **DAE anomalous-dims rendering** | Layer 2 v4 contract, [presentation_v4.py :177](../module6_evaluation/presentation_v4.py#L177) | ❌ helper exists, never imported | ~15 lines, behind "Show DAE details" expander |

### Tier 3 — Required by demo narrative

| Must-have | Why | Status |
|---|---|---|
| **Default landing page that works without setup** | First 30 seconds set the tone | ✅ Dashboard mode loads from cached JSON |
| **One-click drill-down to a specific alert** | "Show me the ventilator case" without sidebar nav | ⚠ today: sidebar slider in Browse, dropdown in Dashboard. Replace with row-click. |
| **Visible audit log of operator's last decision** | Demonstrates INVARIANT 4 visually | ⚠ Audit writes happen, but never displayed back. Add "Last 5 decisions" panel. |
| **Stable 5-min demo "happy path"** with hand-picked alerts | Examiners need CRITICAL ventilator + DISAGREEMENT_ANOMALY + benign-watch | ❌ no curated demo playlist |

---

## Q20 — Nice-to-haves (cuttable)

Ranked by drop-cost (lower number = drop first).

| # | Nice-to-have | Why nice | Why droppable |
|---|---|---|---|
| 1 | **Filter / search bar** | Triage UX expectation | Only 20 alerts. Sortable table covers 90% of value. |
| 2 | **Auto-refresh toggle** ([line 1087](../module6_evaluation/module6_app.py#L1087)) | "Real-time" vibe | Static data — refresh re-runs same script. Misleading. **Drop now.** |
| 3 | **Online Simulation playback** | Showy temporal demo | Heavy implementation cost (11 Issue-fix patches). Same story can be told by clicking through 3 alerts manually. **High candidate for cut.** |
| 4 | **Latency / threshold / drift panels** ([line 1403](../module6_evaluation/module6_app.py#L1403)) | Evidence of perf engineering | Already collapsed by default. Move to "System Diagnostics" expander or drop. |
| 5 | **FDA-style audit-record per-alert export** | Hints at production-readiness | Examiners won't click download. Audit log existence is enough. |
| 6 | **Live parquet (mock TAP) toggle** | "We can ingest from a TAP" claim | It's a mock. **Drop or hide.** |
| 7 | **"Confirm/Reject/Note" buttons in Online Simulation** | Feedback-loop UI | Captured in `online_interactions.jsonl` but never analyzed. **Cut.** |
| 8 | **Global SHAP / beeswarm panels** | Background credibility | Behind `expanded=False`. Keep collapsed. |
| 9 | **Role-authority-violation indicator** | Would make INVARIANT 6 visible | `role_authority_violations()` exists. Could surface as red banner. Skip if time-bound. |
| 10 | **Help/Onboarding/Tooltips** | Standard UX polish | Audience is examiners + LLM personas. **Drop.** |

---

## Q21 — Explicit cuts (will not build)

### Already cut by CLAUDE.md "DO NOT BUILD"

| Cut | Source | Rationale |
|---|---|---|
| **Device discovery / network scanning** | [CLAUDE.md](../CLAUDE.md) | Out of research scope |
| **Automated enforcement / blocking** | INVARIANT 3 | Architectural invariant; no "Execute" button ever |
| **RF / proprietary wireless detection** | DO NOT BUILD | Out of dataset coverage |
| **Ransomware early-detection claims** | DO NOT BUILD | C5 untestable without field deployment |
| **Database / persistence** | DO NOT BUILD | "in-memory + YAML fixtures only" |
| **Authentication / authorization** | DO NOT BUILD | Single-user prototype |
| **CVSS scores in severity** | mve_generator must_not | Use clinical CRITICAL/HIGH/MEDIUM/LOW only |
| **Raw SHAP values in MVE text** | Negative test `test_no_model_internals_exposed` | INVARIANT 5 |

### Additional cuts based on Q4–Q14

| Cut | Rationale |
|---|---|
| **Multi-user authentication / login** | Audience = solo developer + examiners |
| **Real-time WebSocket / SSE updates** | All data is static pre-computed JSON |
| **Mobile / tablet responsive design** | Desktop / projector only |
| **Internationalization** | English only |
| **Dark mode** | One theme is enough for 5–10 min demo |
| **Live data ingestion from network TAP** | Phase 3. Hide the mock toggle. |
| **Drift-monitoring live dashboard** | GAP-HE-3 — Phase 3 explicitly |
| **Bulk operator actions / keyboard shortcuts** | GAP-HE-4 — Phase 3, not the audience |
| **Multi-tenant / multi-hospital deploy** | Single 200–500 bed prototype |
| **PCAP file upload + replay** | Already a stub. Delete the page. |
| **Email / SMS / Slack notifications** | Recommendation-only invariant — no outbound side-effects |
| **PDF / report-export "for management"** | JSON audit record is sufficient evidence |
| **Self-service A/B test creation** | Study mode is locked to one A/B protocol |
| **Comments / annotations / "@mention" workflow** | Single-user; out of HITL scope |
| **Custom alert authoring** | Alerts come from `evaluation_alerts.json`. UI is read-only. |

---

## Synthesis: a demo-defendable scope in 3 tiers

```
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1: SHIP (absolute floor — invariants + already-wired)     │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     │
│  • 3-layer MVE rendering            ✅ already done             │
│  • DO NOT box on CRITICAL clinical  ✅ already done             │
│  • Audit log writes                 ✅ already done             │
│  • Likert decision form             ✅ already done             │
│  • Severity tier colours            ✅ already done             │
│  • Operator role-tailored views     ⚠  promote role to top bar  │
│  • Default Dashboard renders        ✅ already done             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  TIER 2: SHOULD SHIP (closes v4 gap — the headline upgrade)     │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     │
│  • 9-class alert badges (esp. DISAGREEMENT_ANOMALY purple)      │
│  • 4-level Confidence indicator                                  │
│  • Mode A/B indicator                                            │
│  • MITRE ATT&CK technique per role                               │
│  • DAE anomalous-dims expander                                   │
│  • Curated 5-alert demo playlist                                 │
│  • Replace dropdown with row-click drill-down                    │
│                                                                   │
│  Total ~150 lines of glue code. No new logic — all helpers exist.│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  TIER 3: CUT (until Phase 2/3)                                  │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     │
│  • Online Simulation playback (cut or fold into Triage Console)  │
│  • Live parquet TAP toggle (hide)                                │
│  • Auto-refresh toggle (drop — misleading)                       │
│  • FDA-export buttons (hide behind "Researcher" toggle)          │
│  • Drift / latency / threshold panels (single collapsed expander)│
│  • Confirm/Reject/Note buttons in Sim                            │
│  • Filter/search/keyboard shortcuts/dark mode/i18n               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Open questions for user (from across the investigation)

1. **Confirm "1964 lines" was a stale snapshot** — current is 2,485.
2. **Demo time budget and format** — 5 min? 10 min? in-person? hybrid? Drives depth of "single alert deep-dive" path.
3. **Defense date** — if imminent (weeks), favor wiring `presentation_v4.py` in-place + cutting PCAP. If farther out (months), file split + role-aware redesign is on the table.
4. **Tier 2 commitment** — is wiring `presentation_v4.py` *the* defense-deciding feature, or "if I have a weekend"?
5. **Online Simulation: keep or cut?** Most ornate page (570 lines, 11 perf patches) and only one with role selector. Cutting playback + folding role into top bar saves ~400 lines.
6. **MITRE ATT&CK display** — per-alert badge or expander? `format_mitre_for_alert_type` returns a per-role string.
7. **Merged "Triage Console"** — does static-aggregate need separate URL for examiners to bookmark, or mode toggle inside one page?
8. **2-up Inspector** — examiner-facing or only researcher-facing? If only researcher, hide behind `?researcher=1` query param.
9. **Role selector default** — sticky across session, or default per-page (e.g. Inspector defaults to "IT Generalist" because that's `target_user`)?
10. **`audit_trail.json` (3.7 MB)** — keep eager-load, or lazy-load only on FDA-export click?
