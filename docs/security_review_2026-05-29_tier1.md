# Security Review — Tier 1 (Runtime / Serving Path)

**Date:** 2026-05-29
**Branch:** `main` @ `49a57db`
**Scope:** [detection_engine/engine.py](detection_engine/engine.py), [module4_explanations/online_explainer.py](module4_explanations/online_explainer.py), [module4_explanations/module4_online_explainer.py](module4_explanations/module4_online_explainer.py), [module5_responses/audit/logger.py](module5_responses/audit/logger.py), [module5_responses/audit/verify.py](module5_responses/audit/verify.py), [module5_responses/audit/retention.py](module5_responses/audit/retention.py), [module5_responses/executor.py](module5_responses/executor.py), [module5_responses/policy.py](module5_responses/policy.py), [module5_responses/pipeline.py](module5_responses/pipeline.py), [module5_responses/adaptive.py](module5_responses/adaptive.py), LLM/MVE generator path ([src/mve_generator.py](src/mve_generator.py), [configs/llm_data_flow.yaml](configs/llm_data_flow.yaml)), runtime input boundary search.

## Executive summary

| Severity | Count |
| --- | --- |
| HIGH | 2 |
| MEDIUM | 4 |
| LOW / INFO | 4 |

**Architectural observation that shrinks the threat model substantially:** there is **no inference server** in this repository. No Flask, FastAPI, gunicorn, socketserver, or HTTP listener. The "online" in `online_explainer.py` refers to per-alert *latency budget*, not network exposure. Every runtime call site (`module3_risk_scores.py`, `module4_online_explainer.py`, `module5_responses/pipeline.py`) is a batch process iterating a pre-loaded parquet. The attack surface is therefore the **on-disk artefact set** and the **LLM API egress**, not an inbound request channel. This makes most of the input-validation findings you might expect (rate limiting, auth, header parsing) inapplicable.

The audit log layer is well-designed at the cryptographic core (ECDSA P-256 over canonical JSON, hash-chained `prev_hash → integrity_hash`, ECDSA signature over the record including its hash). The weaknesses are in the **migration default** (`legacy_ok=True` accepts unsigned chain restarts) and the **tail-recovery primitive** (last 4 KB of file determines chain continuation).

The LLM data-flow boundary is excellent: an allow-list YAML at [configs/llm_data_flow.yaml](configs/llm_data_flow.yaml) drives `_filter_for_llm`, biometric column values never reach the API, and an explicit forbidden-list raises `AssertionError` on PHI leak. The remaining concerns are non-enforcement of the YAML's own `log_full_prompt: true` clause, and `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` lifecycle.

---

## Findings

### F1 (HIGH) — `AuditLogger` accepts unsigned legacy chain restarts by default

**Where:** [module5_responses/audit/verify.py:99-119](module5_responses/audit/verify.py#L99-L119), [module5_responses/audit/logger.py:110-143](module5_responses/audit/logger.py#L110-L143)

**Trigger path:**
1. Attacker has write access to `results/reports/audit_log.jsonl` (filesystem-level, e.g. a compromised service user or a shared-volume escape).
2. They append a record with `prev_hash="0"*64`, no `signature` field, and a self-consistent `integrity_hash`. The record's body can claim anything.
3. `AuditLogger.__init__` next process start calls `_recover_prev_hash`, which reads only the **last 4 KB** of the file. It picks up the attacker's `integrity_hash` and silently chains legitimate future writes onto it.
4. `verify_audit_log(..., legacy_ok=True)` walks the file. At [verify.py:103-110](module5_responses/audit/verify.py#L103-L110) the unsigned + genesis-`prev_hash` + `line_no > 1` triple is explicitly accepted as a "legacy chain restart" — `result["legacy_chain_restarts"] += 1` and the expected `prev_hash` is reset to genesis. The attacker's record becomes `valid_legacy`.
5. The rotation pre-check ([retention.py:62](module5_responses/audit/retention.py#L62)) also passes `legacy_ok=True`, so an attacker-poisoned log can still be archived as "clean."

Why this is a problem despite "the application user owning the file": the application user is the one Module 5 runs as — but the audit log is supposed to be tamper-evident across that boundary, so a compromise of the application user is exactly the case the chain exists for. With `legacy_ok=True`, the chain provides only weak detection (cross-rotation breaks visible in the *manifest*, but per-restart break invisible).

**Impact:** Forgeable audit trail across process boundaries. Forensics cannot distinguish a benign restart of the pre-hardening logger from an attacker rewriting history.

**Fix:**
- Flip `legacy_ok` default to `False` in `AuditLogger.verify` and `verify_audit_log`. The migration is presumably done (signed-pickle layer is the same key).
- For real legacy migration, archive existing `audit_log.jsonl` once, start fresh, and never re-encounter legacy records on the active log.
- Treat any `legacy_chain_restarts > 0` as a CRITICAL audit event in `rotate_and_purge`, not an informational counter.

---

### F2 (HIGH) — `_recover_prev_hash` trusts tail without cross-checking against any external anchor

**Where:** [module5_responses/audit/logger.py:110-143](module5_responses/audit/logger.py#L110-L143)

`_recover_prev_hash` reads the last 4 KB of the audit log, parses the last JSON line, and returns its `integrity_hash` as the new chain head. There is no comparison against:
- The previous-rotation archive's `last_integrity_hash` (stored in the `.manifest.json` sidecar).
- Any external attestation (e.g., a periodic sealed checkpoint outside the file).
- The signature on the last record itself — `_recover_prev_hash` does not call `verify_audit_log`.

So an attacker who can truncate (or, with the wrapper, append+truncate) the file can control the next chain head. Combined with F1, the chain becomes append-only-forge-able rather than append-only-tamper-evident.

**Fix:**
- When `audit.path` exists at constructor time, run `verify_audit_log(audit.path, audit.public_key_path, legacy_ok=False)` and refuse to construct if the chain is broken (or run in a read-only "forensics" mode).
- On rotation, write the active log's last `integrity_hash` into the new active log's *first* record as `cross_rotation_anchor` so a verifier can walk archives → active without trusting filesystem ordering.

---

### F3 (MEDIUM) — Audit log file written with default umask

**Where:** [module5_responses/audit/logger.py:196-197](module5_responses/audit/logger.py#L196-L197)

`open(self.path, "a", encoding="utf-8")` uses the process umask. On a typical Linux box that's `022` → `0644`. The audit log is therefore world-readable by default — every record contains the alert's `sample_index`, `risk_score`, `attack_category`, `simulated_outcome`, and (when reviewers are attached) the `reviewer_id` / `reviewer_role` / `review_action` triple.

**Fix:** After the first write, `os.chmod(self.path, 0o640)` (or `0o600` if there is no group reader role). Match the pattern already used in `security.py:_write_signed_metadata` ([module0_analysis/security.py:377](module0_analysis/security.py#L377)).

---

### F4 (MEDIUM) — `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` read from env on every call, no validation

**Where:** [src/mve_generator.py:1043-1057](src/mve_generator.py#L1043-L1057), [src/mve_generator.py:1120-1132](src/mve_generator.py#L1120-L1132)

The MVE generator's two LLM paths each read their respective API key from `os.environ` directly and pass it to the SDK. There is:

- **No validation** that the key is present, non-empty, and looks like an API key (e.g. starts with `sk-` for OpenAI). A misconfigured env var with stray whitespace currently fails inside the SDK with a generic error → swallowed → fall through to rule-based, with no log line that says "config error vs. rate-limit."
- **No proxy / outbound network gating.** A compromised host can replace `OPENAI_API_KEY` with the attacker's own key and `OPENAI_MVE_MODEL=…` with their own model, and exfiltrate the contents of every MVE prompt (which contain `attack_category`, `device_class`, `device_criticality` — not PHI per the allow-list, but operationally sensitive).
- **No rate / cost cap.** A bug that fires `generate_mve` in a loop would burn through quota at $0.0002/call × thousands of alerts before anyone notices.
- **`@functools.lru_cache(maxsize=1)` on `_client(key)`** caches the OpenAI client keyed on the API key. The key value lives inside the cache for the process lifetime, which is fine, but means a key rotation requires a process restart even though the rotation may have been driven by a *security incident*. Document or invalidate.

**Fix:**
- Validate keys at startup, not at first use.
- Pin `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` in code or refuse to start when they are overridden by env, to prevent endpoint redirection.
- Wire a per-process call counter; raise once a soft cap is exceeded.
- Document key rotation procedure (clear `lru_cache`, restart).

---

### F5 (MEDIUM) — `validation.log_full_prompt: true` / `log_full_response: true` not enforced

**Where:** [configs/llm_data_flow.yaml:74-79](configs/llm_data_flow.yaml#L74-L79), grepping `log_full_prompt` returns zero Python references.

The YAML declares that full prompts and full responses are logged "for audit reproducibility." No code reads `validation.log_full_prompt` — it is documentation, not enforcement. Two failure modes:

- If a reviewer believes prompts are logged and reviews accordingly, they may approve a flow that is actually opaque.
- If a future refactor *does* start logging prompts, the prompt content is fine (the allow-list ran first), but the **response** is not. The LLM could hallucinate PHI-like content (a synthetic patient name, a plausible MRN) into `layer_2.severity_rationale` or `layer_3.clinical_constraint`, and `log_full_response: true` would commit it to disk.

**Fix:** Either (a) implement an enforced `LLMAuditLogger` that the prompt path actually writes through, OR (b) move `log_full_prompt`/`log_full_response` out of the YAML so it does not mislead reviewers. If enforced, the response must run through `sanitize_for_log` *before* hitting disk.

---

### F6 (MEDIUM) — `sanitize_for_log` regexes are narrow English-PHI; LLM-response content is unsanitised before being placed into alert envelope

**Where:** [src/__init__.py:6-31](src/__init__.py#L6-L31), [src/mve_generator.py:1080-1088](src/mve_generator.py#L1080-L1088)

`sanitize_for_log` redacts SSN, MRN, "patient/pt name", and US-format DOB. That is:
- English-only — Vietnamese, Spanish, etc. clinical strings are passed through.
- US-only ID formats — UK NHS numbers, EU healthcare IDs are not matched.
- DOB regex `\d{2}/\d{2}/\d{4}` does not match ISO-8601 dates (`2026-05-29`).

More importantly, the LLM **response** (`raw_text = response.choices[0].message.content`) is passed to `_parse_llm_json` and then placed into `MVEOutput` and written into `alert_responses.json` *without* `sanitize_for_log`. The model is instructed not to mention SHAP / CVSS / vendor protocols, but the prompt does not forbid PHI-shaped hallucinations, and an attacker who can manipulate the conversation upstream (e.g. via a poisoned `device_context` from a compromised parquet) can plausibly drive the model to emit fake-PHI strings that then live in the signed audit envelope.

**Fix:**
- Run `sanitize_for_log` over `raw_text` before parsing, OR add explicit anti-PHI instructions to `_LLM_SYSTEM_PROMPT` ("never include patient identifiers, names, ages, dates").
- Expand `_PHI_PATTERNS` to cover ISO-8601 dates and a generic NNN-NN-NNNN-like pattern for non-US IDs.

---

### F7 (LOW) — `DetectionEngine._sanitise` silently zeros NaN/Inf in features

**Where:** [detection_engine/engine.py:137-150](detection_engine/engine.py#L137-L150), [module4_explanations/online_explainer.py:108-114](module4_explanations/online_explainer.py#L108-L114)

Both the engine and the online explainer replace `NaN` / `Inf` in the feature vector with `0.0`. This is documented (OOD-05 guard) and prevents crashes on a malformed row. But:
- A `WARNING` log is emitted on a code path that processes thousands of rows per second; either every malformed row floods the log or — if a future refactor downgrades the level — bad inputs become invisible.
- Zero is meaningful in this feature space (post-scaling), so an attacker who can inject `NaN`/`Inf` into a single row can force a known-class prediction. Mild attack-detector evasion.

**Fix:** Count and emit one summary line per batch (`"sanitised N/total rows"`); raise on a per-batch ratio threshold (e.g. > 5% NaN/Inf is suspicious).

---

### F8 (LOW) — `AlertExplainer.explain` instantiates a fresh `DetectionEngine()` per call

**Where:** [module4_explanations/online_explainer.py:162-163](module4_explanations/online_explainer.py#L162-L163)

`from detection_engine import DetectionEngine; x_augmented = DetectionEngine().build_augmented(x_2d)`. The `model_registry` is `@lru_cache(maxsize=None)`, so the heavy work is cached — but a brand-new `DetectionEngine` object is allocated per `explain()` call, and its `_load()` runs once per object to populate `self._classifiers`. Not a security issue at face value; flagged because it makes per-alert latency variance harder to reason about, and there is a code path where an attacker who can starve memory can force repeated allocations.

Also: `build_augmented` triggers `_track_a_probas_all` → `predict_proba` → a *second* model forward pass when the caller of `explain()` already ran one at line 158. Not security, but it cancels part of the perf reasoning that the gate (`TAU_SKIP_DAE`) was designed for.

**Fix:** Reuse a module-level `DetectionEngine` singleton.

---

### F9 (INFO) — No HTTP/RPC server in the runtime path

This is a positive finding for the threat model. `grep` for `Flask | FastAPI | uvicorn | gunicorn | HTTPServer | socketserver` across the repo returns nothing. Every "online" entry point is a batch job iterating a pre-loaded numpy array. The previously-anticipated checklist for inference services (input validation, auth, rate limit, CORS) is **inapplicable here** — and worth documenting in [docs/Threat Model and Scope.html](docs/Threat%20Model%20and%20Scope.html) so a future architect adding a server understands what work the addition implies.

---

### F10 (INFO) — `ActionExecutor.execute` is simulation-only; no real side effects

**Where:** [module5_responses/executor.py:7-66](module5_responses/executor.py#L7-L66)

The executor only appends to `self.execution_log` and returns the record — it does not perform isolation, traffic restriction, or re-authentication despite the action names. This is intentional ("Path B · commit 6 — field renames disambiguate 'recommendation only' from 'actually executed'") and the field rename from `auto_executed` → `auto_executed_simulated` makes the property obvious to a reviewer. Captured here so future work that *does* wire executors to real network controls understands that the IAM/authz boundary does not exist yet.

---

## Surface that was checked and is clean

- **PHI allow-list to LLM.** `_filter_for_llm` enforces the YAML at [configs/llm_data_flow.yaml](configs/llm_data_flow.yaml). Biometric values (`Temp`, `SpO2`, …) and explicit identifiers (`patient_id`, `mrn`, …) cannot reach the API by construction; forbidden-list presence raises `AssertionError`.
- **`shap_context` not sent to LLM.** `pipeline.py` calls `generate_mve(shap_context=None, …)`. The `shap_context` enrichment in `mve_generator` at line 1345 runs in the *rule-based* template, after the LLM call. SHAP feature names do not cross the API boundary.
- **Signature payload binding.** `AuditLogger.log` signs `_canonical_json(record)` with the record's own `integrity_hash` included; `verify_audit_log` reconstructs the same payload by popping `signature`/`signing_key_id`/`signature_alg` before hashing. Verified consistent — no signature-malleability gap.
- **Numerical safety in fusion.** `np.clip(np.maximum(c_track_a, c_track_b), 0.0, 1.0)` bounds the fused score; DAE-skip path leaves `c_track_b = 0` so the gate is safe by construction.
- **`PolicyEngine.recommend` and `clinical_safety_check`.** Pure functions over dict config; no user-controlled paths, no `eval`/`exec`, no shell.
- **`subprocess` in pipeline.** `git rev-parse HEAD` only — list args, no `shell=True`, no user-controlled fields.
- **Rotation refuses tampered chains.** `retention.rotate_and_purge` checks `verify_report["first_break_at"]` and logs `SECURITY_INCIDENT` instead of rotating ([retention.py:69-82](module5_responses/audit/retention.py#L69-L82)). Good failure mode — but undermined by F1 (legacy_ok lets tamper hide).

---

## Recommended next steps

1. F1 + F2 together (`legacy_ok=False` default + cross-rotation anchor verification on construction). Smallest change, biggest assurance.
2. F3 (chmod 0640 on audit log) — one-line fix, immediate.
3. F4 + F5 + F6 together — LLM-egress hygiene PR: validate keys, pin base URLs, enforce or remove `log_full_*`, sanitise LLM response before envelope.
4. F7 + F8 — perf-meets-security cleanup; safe to defer.
5. Document F9 (no inference server) in the threat-model doc so a future addition of FastAPI is a deliberate scope expansion, not a quiet drift.
