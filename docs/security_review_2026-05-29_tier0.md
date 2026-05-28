# Security Review — Tier 0 (PHI & Integrity Core)

**Date:** 2026-05-29
**Branch:** `main` @ `49a57db`
**Scope:** `common/phi.py`, `common/signed_pickle.py`, `common/artifact_versioning.py`, `common/model_registry.py`, `module5_responses/signing.py`, `module5_responses/audit/signing.py`, `module0_analysis/security.py`, `module0_analysis/bootstrap_integrity.py`, `module1_preprocessing/phase1/hipaa.py`, all deserialization call-sites, DAE load path, `pip-audit` on `requirements.txt`.

## Executive summary

| Severity | Count |
| --- | --- |
| HIGH | 3 |
| MEDIUM | 2 |
| LOW / INFO | 3 |

The signing layer (`signed_pickle` + Module 5 ECDSA) is well-designed in isolation — it correctly fails closed on tamper, sha mismatch, missing sidecar, and key-id rotation. The systemic weakness is not in the verification path; it is in the **trust-anchor lifecycle** and in artefacts that sit *outside* `signed_pickle`'s envelope (DAE JSON/HDF5 pair, baseline integrity file). An attacker with write access to the host's signing-key directory or to the integrity-baseline file before an operator re-runs `bootstrap` can launder malicious state under a valid signature.

PHI handling looks correct at this layer: biometric column names are frozen in a single source of truth, integrity logs redact biometric keys defensively, and the Phase 1 HIPAA sanitizer drops only network identifiers (which is the intended design — biometrics stay in the training data on purpose).

---

## Findings

### F1 (HIGH) — `bootstrap()` resigns over an unverified baseline

**Where:** [module0_analysis/security.py:100-155](module0_analysis/security.py#L100-L155), [security.py:339-380](module0_analysis/security.py#L339-L380)

**Trigger path:**
1. Attacker writes a row into `dataset_integrity.json` mapping a malicious file's SHA-256 to a `{filename, size_bytes, bootstrapped_at}` record. The attacker cannot produce a valid signature, so `verify_and_read()` would refuse to read the metadata via `_read_metadata_verified()`.
2. Operator later runs `python -m module0_analysis.bootstrap_integrity` to baseline a new (legitimate) file.
3. `bootstrap()` calls `self._read_metadata()` ([security.py:125](module0_analysis/security.py#L125)) — the **unverified** read. The attacker's entry is loaded into `entries`.
4. `_write_signed_metadata()` calls `_read_metadata()` again ([security.py:356](module0_analysis/security.py#L356)) and then re-signs the entire body, including the attacker's row.
5. On the next `verify_and_read()`, the malicious file passes both the size pre-check and the hash lookup, and gets fed to the parser.

**Impact:** Attacker-controlled bytes are accepted into the dataset loader with a valid operator signature. Breaks the central A02 control in `Threat Model and Scope.html`.

**Fix:** Inside `bootstrap()` (and `_write_signed_metadata`), if `meta.get("entries")` is non-empty, call `_read_metadata_verified()` instead of `_read_metadata()` and refuse to proceed on signature failure. The docstring claim in [bootstrap_integrity.py:11-16](module0_analysis/bootstrap_integrity.py#L11-L16) that "deleting the JSON does not whitewash a tampered file" should be revisited — deletion still allows whitewash because the bootstrap path is empty-baseline-permissive. Either (a) require an operator-provided "first baseline" attestation under VCS, or (b) make `bootstrap` reject when the file is missing unless an explicit `--initial` flag is passed and emit a CRITICAL audit event.

---

### F2 (HIGH) — DAE artefact is loaded without signature verification

**Where:** [common/model_registry.py:103-121](common/model_registry.py#L103-L121), [module2_detection/models/DAE.py:638-740](module2_detection/models/DAE.py#L638-L740)

**Trigger path:**
- `get_dae()` (used in the runtime engine and online explainer) calls `DAEDetector.from_artefacts(json_path=results/models/dae_detector.json, weights_path=results/models/dae_model.weights.h5)`.
- Neither path is verified by `signed_pickle` or any equivalent. The JSON carries the detector's **threshold**, **clip bounds**, **feature weights**, and **proba scaling params**; the HDF5 carries the model weights.
- An attacker with write access to `results/models/` can:
  - Set `threshold` to a value that suppresses every anomaly score → silent false-negative inflation, and the SHA256 in the M5 audit log will not catch this because the DAE artefacts have no entry there.
  - Substitute `feature_weights` so reconstruction error is driven by irrelevant features.
  - Combined with [PYSEC-2026-73](https://osv.dev/vulnerability/PYSEC-2026-73), craft a `model.weights.h5` whose dataset shape exhausts memory → DoS on every inference host.
  - Combined with [CVE-2026-1462](https://nvd.nist.gov/vuln/detail/CVE-2026-1462) ("Keras untrusted deserialization"), achieve RCE through the `load_weights` call in [DAE.py:733](module2_detection/models/DAE.py#L733).

**Impact:** Confidentiality/integrity/availability of the entire Track-B (DAE) detection path. The signed_pickle docstring explicitly cites the threat — "tampered build host" → "code execution on every machine that runs inference" — and the DAE escapes that protection by design.

**Fix:**
- Wrap `save_artefacts` / `from_artefacts` in a sidecar `.sig` analogous to `signed_pickle`. Sign `sha256(canonical_json(body) || sha256(weights_bytes))` so both files are bound to one signature.
- Update `get_dae()` to verify before calling `from_artefacts`.
- Upgrade Keras (see F4).

---

### F3 (HIGH) — Signing key has no out-of-band trust anchor; verifier accepts whatever key is on disk

**Where:** [module5_responses/audit/signing.py:41-78](module5_responses/audit/signing.py#L41-L78), [module5_responses/audit/signing.py:81-120](module5_responses/audit/signing.py#L81-L120), [common/signed_pickle.py:119-144](common/signed_pickle.py#L119-L144), [module0_analysis/security.py:382-389](module0_analysis/security.py#L382-L389)

**Trigger path:**
1. Attacker with write access to `~/.iomt-ids/` (or whichever path the operator chose via `IOMT_AUDIT_SIGNING_KEY`) deletes `audit_signing_key.pem` and `results/reports/audit_signing_key.pub.pem`.
2. Next inference / write run calls `_load_signing_key()` → no key file present → `_bootstrap_local_key()` silently generates a fresh ECDSA P-256 keypair and writes it. The warning is a single `logger.warning(...)` line.
3. The freshly bootstrapped key's `signing_key_id` is the SHA-256 of the new public key (computed at line 119). The verifier in `_get_verifying_key()` and `_load_phase0_public_key()` reads the **same** PEM the bootstrapper just wrote. Trust collapses to "whoever can write to that directory."
4. Attacker now calls `dumps_signed(malicious_obj, results/models/xgboost_final_pipeline.pkl)` and the sidecar verifies cleanly on every subsequent load.

Existing artefacts signed by the **previous** key start raising `SignedPickleError("signing_key_id mismatch")`, which is detectable — but the system has no policy for what to do on detection (no alerting wiring outside of a log line).

**Impact:** RCE on every host that loads the resigned pickle. The signed_pickle threat model lists "compromise of the private key" as "game over"; this finding reframes that — silent re-bootstrap is functionally equivalent to private-key compromise but requires no key exfiltration.

**Fix (defense in depth):**
- Refuse to auto-bootstrap when there are signed artefacts present on disk: `_load_signing_key()` should fail loudly if `results/models/*.pkl.sig` or `results/reports/audit_log.jsonl` exists but no private key does.
- Pin an expected `signing_key_id` in a VCS-tracked config (e.g. `config/signing_key_id.txt`) and refuse to load any key whose ID does not match. Rotation becomes a deliberate, code-reviewed change.
- Move the private key location to a secret store (HSM/KMS or at minimum a chmod-700 directory owned by a non-application user). The current default location is in the application user's home directory.
- Surface re-bootstrap as a CRITICAL Module 5 audit event, not just a `logger.warning`.

---

### F4 (MEDIUM) — `requirements.txt` is open-ended; installed Keras has known CVEs

**Where:** [requirements.txt](requirements.txt)

`pip-audit -r requirements.txt` flags one direct issue:

| Package | Version | ID | Fixed in | Note |
| --- | --- | --- | --- | --- |
| keras | 3.12.2 | PYSEC-2026-73 | 3.13.1 | HDF5 weight-loading DoS via crafted shape |
| keras | 3.12.2 | CVE-2026-1462 | 3.13.2 | Untrusted deserialization |

`pip-audit` against the installed environment also flags vulnerable `gitpython`, `idna`, `pillow`, `prefect`, `pygments`, `requests`, `starlette`, `urllib3`, etc. — these come in transitively (none are pinned in `requirements.txt`).

Compounding factor: requirements.txt uses **`>=`** for every dependency. A pinned, hash-locked dependency set is the only thing that prevents a fresh `pip install -r` on a CI runner from picking up an attacker's prerelease of `imbalanced-learn` or `python-dotenv`.

**Fix:**
- Pin: `keras==3.13.2`, then add hashes via `pip-compile --generate-hashes`.
- Add `pip-audit -r requirements.txt --strict` as a CI gate in [.github/workflows/](.github/workflows/).
- For runtime hosts, document a separate `requirements-runtime.lock` distinct from training requirements.

---

### F5 (MEDIUM) — `model_registry._load_thresholds` reads JSON `optimal_threshold` without signature check

**Where:** [common/model_registry.py:133-153](common/model_registry.py#L133-L153)

`get_track_a_thresholds()` reads `results/models/xgboost_final_report.json` and extracts `optimal_threshold` — a value that directly gates which samples become alerts. If an attacker can tamper with the report JSON (same directory as the signed pickle, but no sidecar), they can shift the threshold to e.g. 0.99 and silently kill the alert stream. The JSON is the analogue of the DAE JSON in F2; same class of bug, lower blast radius (the XGBoost pickle itself is signed, so model behaviour is intact — only the decision threshold is unsigned).

**Fix:** Embed the threshold inside the signed pickle's metadata (or sign the report JSON alongside). At a minimum, validate `0 ≤ optimal_threshold ≤ 1` and emit a hardened audit event on each load with the value.

---

### F6 (LOW) — `tools/resign_models.py` and `tools/rq2_compute_faithfulness.py` invoke `joblib.load` directly

**Where:** [tools/resign_models.py:88](tools/resign_models.py#L88), [tools/rq2_compute_faithfulness.py:105](tools/rq2_compute_faithfulness.py#L105)

`resign_models.py` documents the bypass intentionally — it has to load stale pickles before re-signing. Acceptable as a one-shot operator tool, but the bypass should be gated behind an explicit flag (`--I-trust-local-bytes`) so a future operator does not run it on attacker-controlled bytes (e.g., during incident response when they fetch a "known-good" pickle from a peer).

`rq2_compute_faithfulness.py` does NOT document a bypass — it loads the XGBoost pickle directly. Replace with `loads_signed`.

**Fix:** Add the trust-flag to `resign_models.py`; switch `rq2_compute_faithfulness.py` to `loads_signed`.

---

### F7 (INFO) — `_resolve_inside_workspace` is correct, but absolute paths short-circuit `self._root / path`

**Where:** [module0_analysis/security.py:485-498](module0_analysis/security.py#L485-L498)

`self._root / path` returns `path` directly when `path` is absolute (pathlib semantics). The `.resolve()` and `.relative_to(self._root)` calls catch this — if the absolute path is outside the root, `relative_to` raises and `PermissionError` is raised. So the validator is sound. Calling out for completeness only.

---

### F8 (INFO) — `_load_phase0_public_key` re-exports the public PEM from the private key on every load

The trust anchor for Phase 0 integrity verification is the **private key file** (which `_load_signing_key` reads to derive and re-write the public key). An attacker who modifies `audit_signing_key.pub.pem` alone has no effect — it gets overwritten. Defensive. Captured under F3 (the systemic issue is the private key's anchor).

---

### F9 (INFO) — `dumps_signed` sidecar/pickle ordering is safe

[common/signed_pickle.py:212-213](common/signed_pickle.py#L212-L213) writes sidecar first, pickle second. A racing reader may briefly see "sidecar with no pickle" or "sidecar with stale pickle" — both produce safe failures (`FileNotFoundError` or sha256 mismatch). The opposite order would allow a brief "pickle with no sidecar" window, which `loads_signed` would (correctly) refuse, but the current order is preferable because it never leaves an inconsistency that looks like a legitimate unsigned pickle.

---

## Surface that was checked and is clean

- **PHI scrubbing in audit log.** `log_phase0_event` defensively redacts biometric column keys in the payload before logging ([security.py:611-613](module0_analysis/security.py#L611-L613)). Producer code is the responsible party, but the defensive check is correct.
- **HIPAA sanitizer.** `HIPAASanitizer.transform` drops network-identifier columns by name only; the dropped-column report contains only column names, not values ([module1_preprocessing/phase1/hipaa.py](module1_preprocessing/phase1/hipaa.py)). Biometric values flow through training intentionally (per the threat model).
- **`yaml.load` usage.** Every call uses `yaml.safe_load`. No unsafe yaml deserialization sinks found.
- **`subprocess` usage.** All call-sites pass list args, none use `shell=True`, and the only commands invoked are `git rev-parse` (constant arg) and `[sys.executable, module_script]` (constant arg). No injection surface.
- **`.env` hygiene.** `.env.local` (mode `0600`) is not tracked. `.env.example` is committed and contains only path constants. Both are covered by `.gitignore`.
- **`signed_pickle.loads_signed` itself.** Reads bytes, hashes, refuses on (a) missing sidecar, (b) corrupt sidecar JSON, (c) sha mismatch, (d) `signing_key_id` mismatch, (e) `InvalidSignature`, (f) any other exception — all paths fail closed before `joblib.load` is invoked.
- **`artifact_versioning`** — read/check/embed semantics are correct. No deserialization risk: it only reads `_schema_version` from `json.loads` and `np.load(allow_pickle=True)` of `.npz` (note: `allow_pickle=True` is needed because schema_version is a 0-d str array; this is the safe usage — `np.savez` writes only the str payload).

---

## Recommended next steps

1. Apply the F1 fix (verify-before-resign in bootstrap) and the F3 fix (no-auto-bootstrap-when-artefacts-present + VCS-pinned key id) together — they are the same trust-anchor problem at two layers.
2. Promote DAE artefacts to signed-pickle parity (F2), then pin Keras (F4).
3. Patch F5 (threshold) and F6 (rq2 tool) in the same PR — both are one-line swaps to `loads_signed`.
4. Add `pip-audit -r requirements.txt --strict` to all four phase workflows in `.github/workflows/`.
