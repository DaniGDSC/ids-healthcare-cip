# Security Review — Tier 2 (Training/Offline Pipelines + Dashboard)

**Date:** 2026-05-29
**Branch:** `main` @ `49a57db`
**Scope:** [module2_detection/](module2_detection/) (training + tuning), [module3_risk_scoring/](module3_risk_scoring/), [module4_explanations/](module4_explanations/) (offline), [module6_evaluation/](module6_evaluation/) (Streamlit dashboard + user study), all `np.load` / `pd.read_*` call-sites in these modules.

## Executive summary

| Severity | Count |
| --- | --- |
| HIGH | 2 |
| MEDIUM | 3 |
| LOW / INFO | 4 |

The training pipeline is **the cleanest layer in the codebase**. Every classifier pickle is written via `dumps_signed` ([module2_train_models.py:331](module2_detection/module2_train_models.py#L331), [tuning/_runner.py:94](module2_detection/tuning/_runner.py#L94)); only bare classifiers (not SMOTE wrappers) are persisted; `.npz` prediction files contain pure numeric arrays (no `allow_pickle` needed); DAE saves use JSON + Keras weights (no pickle).

Two new threat vectors surface at this tier:

1. **`risk_scores.npz` is an unsigned pickle deserialization sink.** Five distinct readers across module4/module6 load it with `allow_pickle=True` because module3 writes `risk_levels` / `formula_version` / `schema_version` as object-dtype string arrays. Any attacker who can write `results/reports/risk_scores.npz` gets code execution on every interactive dashboard session and every offline regen run.
2. **The Streamlit dashboard ([module6_app.py](module6_evaluation/module6_app.py)) is the codebase's only inbound network surface** — `streamlit run …` defaults to `0.0.0.0:8501` with no auth. Every visitor can write attributed records into the ECDSA-signed Module 5 audit log via `_capture_dashboard_action(participant_id="dashboard_user")`. The HTML escaping inside the dashboard is correct (`from html import escape` is used consistently on every alert-derived value across `components.py`), so XSS is not the issue — audit-chain forgery via the open port is.

The PHI plumbing is clean: every `unsafe_allow_html` rendering passes through `html.escape()` on the dynamic portion; biometric *values* never enter the dashboard payload (only column NAMES via `top_features` lists); the user-study free-text fields are stored to JSONL but never re-rendered as HTML.

---

## Findings

### F1 (HIGH) — `risk_scores.npz` loaded with `allow_pickle=True` at five reader sites; no signature

**Where:**

- Writer: [module3_risk_scoring/io.py:179-187](module3_risk_scoring/io.py#L179-L187) — `np.savez(..., risk_levels=levels, formula_version=np.array(formula_version, dtype=str), **version_kwarg_for(npz_path.name))`. The string-dtype 0-d arrays force consumers to load with `allow_pickle=True`.
- Readers (all use `allow_pickle=True`):
  - [module4_explanations/example_explanations.py:46](module4_explanations/example_explanations.py#L46)
  - [module4_explanations/module4_online_explainer.py:97](module4_explanations/module4_online_explainer.py#L97)
  - [module4_explanations/module4_explanations.py:231](module4_explanations/module4_explanations.py#L231)
  - [module6_evaluation/alerts.py:91](module6_evaluation/alerts.py#L91)
  - [module6_evaluation/module6_app.py:1124](module6_evaluation/module6_app.py#L1124) (cached via `@st.cache_data`, so a malicious load persists in the dashboard session)

**Trigger path:** `np.load(allow_pickle=True)` accepts `.npz` entries whose dtype is `object` and unpickles them. An attacker with write access to `results/reports/risk_scores.npz` crafts a malicious 0-d object array for the `risk_levels` field (or any field), and every reader gets RCE the moment the field is dereferenced. The Streamlit dashboard reaches the load path on every fresh session (line 1124 is wrapped in `@st.cache_data`, which serializes the result — the object's `__reduce__` runs at unpickle time inside Streamlit's worker).

**Impact:** Same threat class as the DAE artefact in Tier 0 F2 (unsigned integrity-critical pickle), but with **five independent attack surfaces** including the only network-facing process in the repo.

**Fix:**
- **Short-term:** stop writing the object-dtype string fields. `formula_version`, `schema_version`, and the `risk_levels` array can all be JSON sidecar fields next to the .npz, with the .npz reserved for purely-numeric arrays. Then every reader can drop `allow_pickle=True`.
- **Medium-term:** even after dropping `allow_pickle`, extend the `signed_pickle` envelope concept to `.npz`+JSON pairs (same signature scheme as the proposed DAE fix in Tier 0).
- **Defense in depth at the readers:** wrap every `np.load(..., allow_pickle=True)` in a helper that first runs `assert_compatible(path)` from `common.artifact_versioning` AND a sha256 check against a signed manifest.

---

### F2 (HIGH) — Streamlit dashboard binds default `0.0.0.0:8501` with no auth; any visitor writes signed audit records

**Where:** [module6_evaluation/module6_app.py](module6_evaluation/module6_app.py) (entire file), [module6_evaluation/audit_writer.py:59-96](module6_evaluation/audit_writer.py#L59-L96), [module6_evaluation/module6_app.py:2101-2130](module6_evaluation/module6_app.py#L2101-L2130) (`_capture_dashboard_action`).

**Trigger path:**
1. Operator runs `streamlit run module6_evaluation/module6_app.py` per the docstring / `module6_evaluation/__main__.py`. Streamlit defaults to `server.address = 0.0.0.0` and listens on `:8501`. The repo has no `.streamlit/config.toml` overriding either.
2. A visitor on the network reaches the dashboard. No auth, no session signing — Streamlit treats every browser as a session.
3. The visitor clicks `acknowledge` / `escalate` / `dismiss` on any alert. `_capture_dashboard_action` fans out to three sinks, including:
   - `get_hardened_audit().log({…}, reviewer_id=st.session_state.get("participant_id", "dashboard_user"), reviewer_role=st.session_state.get("sim_role", ""))` — this writes a **signed** record into the Module 5 ECDSA-signed audit chain ([audit_writer.py:34](module6_evaluation/audit_writer.py#L34) → `EVAL_DIR / "audit_log.jsonl"`).
   - The `participant_id` and `sim_role` come straight from `st.session_state`, which the visitor can populate by walking through the sidebar — there is no validation that they correspond to a real enrolled participant (P01..P10 is just a frozen list in [study_loader.py:59-62](module6_evaluation/study_loader.py#L59-L62), not an auth principal).
4. The visitor can also dismiss alerts with a free-text rationale ([module6_app.py:2153-2174](module6_evaluation/module6_app.py#L2153-L2174)). That rationale is written into the signed chain.
5. With the Module 5 ECDSA key as the trust anchor (auto-bootstrapped per Tier 0 F3), the chain now contains attacker-attributed records that are cryptographically valid. Forensic review cannot distinguish them from genuine reviewer actions.

This is amplified by Tier 0 F3 (signing key auto-bootstrap) and Tier 1 F1 (`legacy_ok=True` default in verifier): an attacker who can also reach `~/.iomt-ids/` doesn't even need the dashboard.

**Impact:**
- Audit chain forgery via the open port. The chain's purpose — tamper-evident reviewer attribution for a regulated context — is defeated.
- Information disclosure: the dashboard renders attack categories, risk scores, sample indices, response actions, MVE narratives, and (where present) device classification per sample. None of this is biometric VALUES (the allow-list filter in Tier 1 covers that), but the operational telemetry across patients is sensitive.

**Fix:**
- Add `.streamlit/config.toml` with `server.address = "127.0.0.1"`, `browser.serverAddress = "127.0.0.1"`, and `server.enableCORS = true`. Document the operator-only invocation pattern (`streamlit run … --server.address 127.0.0.1`).
- Front the dashboard with a reverse proxy that enforces a shared secret or OIDC. Streamlit's own auth story has matured (`st.user`) — wire it.
- In `_capture_dashboard_action`, refuse to call `get_hardened_audit().log(...)` unless `st.session_state["participant_id"]` resolves through a validated enrolment table (not just the `_FROZEN_PID_PARITY` lookup which only checks parity). Default to **unsigned** local JSONL for unauthenticated users.
- Surface `streamlit run … --server.address 0.0.0.0` as a CRITICAL audit event so an operator misconfiguration is visible.

---

### F3 (MEDIUM) — `assign_ab_conditions` (legacy) uses MD5; `assign_ab_condition` (current) uses SHA-256 for new participants only

**Where:** [module6_evaluation/module6_app.py:547](module6_evaluation/module6_app.py#L547), [module6_evaluation/study_loader.py:78-86](module6_evaluation/study_loader.py#L78-L86)

The repo carries two assignment functions:
- `module6_app.py:539-568` — the legacy version uses `hashlib.md5(participant_id.encode())` as the seed for `random.Random`. MD5 is fine *as a non-security seed*, but the surrounding code is the reviewer-attribution layer; using MD5 in code paths adjacent to signed audit trails invites the "why is MD5 anywhere near our crypto" question on the next compliance scan.
- `study_loader.py` — already migrated to SHA-256, with a "frozen parity map" for the 10 enrolled participants (P01..P10) to preserve already-collected analysis. This is the *correct* approach.

**Impact:** Low operational risk (MD5 → seed → counterbalancing, no security claim attached) but flagged because compliance tooling will hit on it.

**Fix:** Replace the live MD5 call in `module6_app.assign_ab_conditions` with the SHA-256-based `study_loader.assign_ab_condition`. The frozen-PID compatibility is already handled by the lookup table.

---

### F4 (MEDIUM) — `audit_log` silently swallows hardened-sign failures with `except Exception`

**Where:** [module6_evaluation/audit_writer.py:84-96](module6_evaluation/audit_writer.py#L84-L96)

```python
try:
    get_hardened_audit().log(payload, reviewer_id=..., ...)
except Exception as exc:  # noqa: BLE001
    logger.warning(
        "audit_log: hardened sign failed (%s) — plain JSONL still wrote.", exc,
    )
```

A failure to write to the signed chain falls through to a `logger.warning` and the plain JSONL keeps writing. Operationally this means an attacker who can DoS the signing key path (e.g., chmod 000 on `~/.iomt-ids/`) silently converts every "signed" audit record into "unsigned plain JSONL only," with no escalation. The reviewer rendering layer cannot distinguish.

Same pattern in `_capture_dashboard_action` ([module6_app.py:2129-2130](module6_evaluation/module6_app.py#L2129-L2130)) — `except Exception: pass`.

**Fix:** When `sign=True` was requested and the hardened logger errors, fail the user-facing operation. Surface "audit chain unavailable — refusing to record dismissal" in the dashboard rather than recording it unsigned. Bonus: emit a `SECURITY_INCIDENT` to whatever monitoring exists.

---

### F5 (MEDIUM) — Streamlit `@st.cache_data` over `np.load(allow_pickle=True)` persists deserialised state across user sessions

**Where:** [module6_evaluation/module6_app.py:1119-1124](module6_evaluation/module6_app.py#L1119-L1124)

`load_risk_scores` is wrapped in `@st.cache_data`. The decorator memoises the *result* across all dashboard sessions in the same process. Combined with F1, this means: a single malicious `risk_scores.npz` loaded once at server start serves its payload to every subsequent visitor's session. Streamlit's cache is keyed on the function arguments, not on file mtime or hash — even if an admin replaces the npz with a clean copy, the dashboard keeps serving the cached (malicious) result until restart.

**Fix:** Add `ttl=` and a hash-of-file argument to bust the cache when the npz changes:
```python
@st.cache_data(ttl=60)
def load_risk_scores(file_sha256: str):
    ...
```
Or: drop `@st.cache_data` for any path whose source file is a deserialization sink, and rely on the `signed_pickle`/`assert_compatible` checks done at load time.

---

### F6 (LOW) — `from_artefacts` casts JSON-supplied dims to `int` / `float` without bound checking

**Where:** [module2_detection/models/DAE.py:683-733](module2_detection/models/DAE.py#L683-L733)

`encoding_dims=list(hp.get("encoding_dims", [16, 8, 16]))` and `n_features = int(body.get("n_features", instance._feat_weights.shape[0]))` go directly into `instance._build_model(n_features)` which allocates a Keras model with that input width. A malicious `dae_detector.json` can specify `n_features = 2**31` and (combined with [PYSEC-2026-73](https://osv.dev/vulnerability/PYSEC-2026-73)) trigger memory exhaustion. The check `if expected != actual` at [engine.py:187-194](detection_engine/engine.py#L187-L194) catches the *runtime* mismatch, but only *after* the model has been built.

**Fix:** Bound `n_features`, `encoding_dims`, `batch_size`, `epochs` against sensible upper limits before constructing the Keras model. Reject with `ValueError` instead of allocating.

---

### F7 (LOW) — `module6_app` plain JSONL audit writer has the same umask issue as Tier 1 F3

**Where:** [module6_evaluation/audit_writer.py:55-56](module6_evaluation/audit_writer.py#L55-L56) — `open(self.path, "a", encoding="utf-8")` then `f.write(json.dumps(event) + "\n")`. No `os.chmod` on the file. World-readable by default.

Same fix: `os.chmod(self.path, 0o640)` post-open.

---

### F8 (INFO) — Dashboard HTML rendering is correctly escaped

This is a positive finding. Every `unsafe_allow_html=True` call site in [module6_app.py](module6_evaluation/module6_app.py) and in [module6_evaluation/components.py](module6_evaluation/components.py) interpolates dynamic content through `html.escape(...)`. The `components.py` helper functions are particularly disciplined: `render_alert_row`, `render_factor_row`, `render_status_strip`, `render_timeline_item`, and `render_consensus_badge` all escape every untrusted field. Inline cases (e.g., [module6_app.py:1805-1810](module6_evaluation/module6_app.py#L1805-L1810) `subtitle_html`) also use `escape()`.

The user-study free-text fields (feedback, dismiss rationale) are stored to JSONL but *not* re-rendered as HTML anywhere I could find. If a future page does render them, it must use `escape()` (or call `st.write`, which escapes by default).

---

### F9 (INFO) — Training pipeline is uniformly signed

Module 2 writes the only production pickles (`*_final_pipeline.pkl`) through `dumps_signed`. The DAE artefact is JSON + Keras weights — pickle-free (the integrity gap is Tier 0 F2, not a training-time finding). The `.npz` prediction files contain only numeric arrays and load without `allow_pickle`.

The remaining offline-tool exception is [tools/rq2_compute_faithfulness.py:105](tools/rq2_compute_faithfulness.py#L105), already captured in Tier 0 F6.

---

### F10 (INFO) — No HTTP / RPC server outside the Streamlit dashboard

Reaffirms Tier 1 F9. The only inbound listener in the repo is the dashboard's Streamlit instance. CLI runners (`run_xgboost.py`, `run_random_forest.py`, etc.) and `make` targets are operator-invoked.

---

## Surface that was checked and is clean

- **Training pickles signed.** `dumps_signed(classifier_only, pipeline_path)` is the only writer at [module2_train_models.py:331](module2_detection/module2_train_models.py#L331) and [tuning/_runner.py:94](module2_detection/tuning/_runner.py#L94).
- **No `subprocess` injection.** All callsites in `module2`/`module3`/`module4`/`module6` use list args; the only command invoked is `git rev-parse` (no user-controlled input).
- **No `eval` / `exec` outside test files.**
- **HTML escaping in dashboard renders.** Confirmed via grep + read of `components.py` and inline `module6_app.py` blocks.
- **Free-text user input not re-rendered as HTML.** Feedback / dismiss rationale paths terminate at JSONL writers.
- **Parquet reads are passive.** `pd.read_parquet` opens files that go through Phase 1 → Phase 1's output is already in `data/processed/`. No user-controlled paths.

---

## Recommended next steps

1. **F1 + F5 together** — refactor `risk_scores.npz` writer to drop the object-dtype fields, then remove `allow_pickle=True` at every reader; gate `@st.cache_data` on file hash. Smallest blast-radius change for the biggest reduction in attack surface.
2. **F2** — `.streamlit/config.toml` with `127.0.0.1` binding + auth-or-no-signed-audit policy. Before the dashboard is run anywhere with an external network.
3. **F3 + F4** — MD5 swap + propagate sign failures. Two small PRs.
4. **F6 + F7** — bound-check DAE JSON fields, chmod audit log. Low priority but trivial.
5. The architectural note in F10 / Tier 1 F9 — document the dashboard as the codebase's *only* inbound listener so future additions are conscious decisions.
