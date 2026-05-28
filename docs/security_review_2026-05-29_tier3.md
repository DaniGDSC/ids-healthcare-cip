# Security Review — Tier 3 (Infrastructure & Repo Hygiene)

**Date:** 2026-05-29
**Branch:** `main` @ `49a57db`
**Scope:** [.github/workflows/](.github/workflows/) (phase0/phase1/phase2/prototype), [requirements.txt](requirements.txt) (pinning + transitive coverage), git history (secrets scan), file permissions of on-disk artefacts, [Makefile](Makefile), [run_all_modules.py](run_all_modules.py), [.vscode/](.vscode/), pre-commit / Dockerfile / pyproject status, `.gitignore` coverage.

## Executive summary

| Severity | Count |
| --- | --- |
| HIGH | 2 |
| MEDIUM | 4 |
| LOW / INFO | 4 |

The headline finding is that **real RSA private keys remain in the git history.** Three PKCS#8 PEM files (`config/certs/test/{ca,server,client}-key.pem`) were committed in `05d2ea9c` (2026-03-30) and deleted in `5a6f3bee` (2026-04-04). They are still reachable via `git show 05d2ea9c:config/certs/test/ca-key.pem`. Even though the directory is labelled "test", any deployment that used these certs is compromised; from a compliance perspective, the keys must be considered leaked.

CI declares a strong security posture (`bandit`, `pip-audit --strict`, `cyclonedx-bom`, biometric-leak regression) but **two of the CI gates reference files that do not exist in HEAD** — `pyproject.toml` (bandit config) and `Dockerfile` (build job). The workflow will fail at those steps, and the subsequent "test" / "build" jobs will not execute. The security scan is therefore *partially active* — bandit will error, `pip-audit` will run, the docker build will fail. Net effect: no green build has actually run all of this in CI.

Dependency pinning is **entirely open** in `requirements.txt` (0/21 lines pinned with `==`; 21 use `>=`). `pip-audit -r requirements.txt` audited 151 transitive packages — the production install is whatever PyPI is offering on CI day.

File permissions on the live signing key (`~/.iomt-ids/audit_signing_key.pem`) are correct (`0600`), but its parent directory is `drwxrwxr-x` (group-writable) — directly amplifying Tier 0 F3 (anyone in the operator's group can replace the key). On-disk model artefacts, the audit log, and the integrity baseline are all `0664` (group-writable).

---

## Findings

### F1 (HIGH) — Real RSA private keys in git history

**Where:** Commits `05d2ea9cc0453fcc90ae7830a1b81e88bf700679` (added 2026-03-30) and `5a6f3bee53366d2ed3c0fbee6a76af7c06e17739` (deleted 2026-04-04), at paths:
- `config/certs/test/ca-key.pem`
- `config/certs/test/server-key.pem`
- `config/certs/test/client-key.pem`

**Evidence:**
```
$ git show 05d2ea9c:config/certs/test/ca-key.pem | head -2
-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDbhLgp04m/8D5F…
```
The headers are PKCS#8 PEM. Material is intact and recoverable from the pack file. There is no record of a `git filter-repo` / `git filter-branch` history rewrite, so the blobs remain in every clone of the repo.

**Impact:**
- If these certs were ever used to terminate TLS on any internal service (the name "ca/server/client" suggests an mTLS prototype), an attacker who clones the repo has the private keys for the corresponding identities.
- Even if "test" means "scratch / never deployed," the *appearance* of leaked private keys is enough to fail a SOC 2 / HIPAA tabletop exercise and will be picked up by every automated secret scanner pointed at this repo (`gitleaks`, GitHub secret scanning, TruffleHog).

**Fix (in order of operational cost):**
1. **Treat the keys as compromised.** Rotate any service identity that used them. If they were never deployed, document that in writing — auditors will ask.
2. **Rewrite history** with `git filter-repo --invert-paths --path config/certs/test/` (or BFG Repo-Cleaner) and force-push. Coordinate with every collaborator to re-clone — `git pull --rebase` will NOT clean their objects.
3. **Add `*.pem` and `*-key.pem` and `config/certs/**` to `.gitignore`**. Current `.gitignore` does not mention `.pem`, `.key`, or `certs/`.
4. **Add a pre-commit hook** (gitleaks / detect-secrets) so this cannot recur. The repo has no `.pre-commit-config.yaml` today.

---

### F2 (HIGH) — Two CI security gates reference missing files; gates do not run as advertised

**Where:**
- [.github/workflows/phase0.yml:51](.github/workflows/phase0.yml#L51) — `bandit -r module0_analysis/ common/ -c pyproject.toml …`
- [.github/workflows/phase0.yml:54](.github/workflows/phase0.yml#L54) — same again with `-ll` (the actual gate).
- [.github/workflows/phase1.yml:99](.github/workflows/phase1.yml#L99) and [.github/workflows/phase2.yml:106](.github/workflows/phase2.yml#L106) — same `-c pyproject.toml` pattern.
- [.github/workflows/phase0.yml:155](.github/workflows/phase0.yml#L155) — `docker build -t analyst/phase0:${{ github.sha }} .`
- [.github/workflows/phase1.yml:466](.github/workflows/phase1.yml#L466) and [.github/workflows/phase2.yml](.github/workflows/phase2.yml) — same docker build.

**Observed:**
- `pyproject.toml` does not exist in HEAD or git history (verified via `git ls-files | grep -i pyproject`).
- `Dockerfile` (any case) does not exist in HEAD or git history (verified via `git ls-files | grep -i dockerfile`).
- Bandit invoked with a missing `-c` config aborts. `docker build .` with no `Dockerfile` aborts.

**Impact:**
- The advertised "Bandit / SBOM / pip-audit" security gate is functionally **bandit-broken**: the line-51 invocation has `|| true` (ignores failure), but the line-54 invocation (the actual fail-on-high gate) errors at config-resolve before any rule runs. CI history will show red on this step, which either means (a) the workflows have never gone green, or (b) the team has been ignoring red CI on these jobs.
- The "build" job never runs because `needs: test` and `test` is gated behind `security-scan`.
- Future contributors look at the workflow file and see "bandit + pip-audit + SBOM" — believing the gate exists when it doesn't.

**Fix:**
- Add a `pyproject.toml` with a minimal `[tool.bandit]` section (or remove the `-c pyproject.toml` flag and rely on Bandit defaults — but lose the ability to per-rule exclude).
- Add a `Dockerfile` (or remove the `build` and `Verify imports` jobs until the image exists).
- After fixing both, run the workflows on a feature branch end-to-end and confirm green.

---

### F3 (MEDIUM) — `requirements.txt` is fully unpinned; CI has no hash verification

**Where:** [requirements.txt](requirements.txt) — 21 dependency lines, 0 with `==`, 21 with `>=` (or no constraint). `pip-audit` against the *installed* environment surfaced CVEs across `gitpython`, `idna`, `pillow`, `prefect`, `pygments`, `requests`, `starlette`, `urllib3`, `keras`, `langchain-core`, `langsmith`, `lupa`, `mako`, `python-multipart`, `pytest` — already captured in Tier 0 F4.

CI installs with bare `pip install -r requirements.txt` ([phase0.yml:89](.github/workflows/phase0.yml#L89), [phase1.yml:61](.github/workflows/phase1.yml#L61), [phase2.yml:65](.github/workflows/phase2.yml#L65), [prototype.yml:46](.github/workflows/prototype.yml#L46)) — no `--require-hashes`, no constraints file, no resolver lock.

**Impact:**
- Every CI run picks up whatever PyPI advertises. A typosquat or compromised maintainer of any transitive dependency (e.g., `python-json-logger`, `tqdm`, `joblib`) gets execution inside CI, which has access to `${{ github.sha }}` and any artefact-upload permissions.
- Reproducibility claim in [docs/reproducibility_report.md](docs/) (referenced via `module0_analysis/reproducibility_report.py`) is false — same `git rev` + same `requirements.txt` + different `pip install` time = different installed package set.

**Fix:**
- `pip-compile --generate-hashes --output-file=requirements.lock requirements.in` to produce a hashlocked lockfile.
- In CI: `pip install --require-hashes -r requirements.lock`.
- Add `pip-audit -r requirements.lock --strict` as an *enforcing* gate (not behind a missing-Dockerfile chain).

---

### F4 (MEDIUM) — `~/.iomt-ids/` directory is group-writable; live signing key amplifies Tier 0 F3

**Where:** `~/.iomt-ids/` on the development host has mode `drwxrwxr-x` (`0775`). The key file itself is correctly `0600` ([audit/signing.py:70](module5_responses/audit/signing.py#L70) calls `os.chmod(private_path, 0o600)`).

**Trigger path:**
1. Attacker is in the operator's primary group (e.g., teammate, CI runner that shares group).
2. Attacker does `rm ~/.iomt-ids/audit_signing_key.pem; touch …` — group-writable directory permits this even though the file itself is `0600`.
3. On the operator's next process start, `_bootstrap_local_key` regenerates a fresh keypair (per Tier 0 F3). The chain is reset, signed pickles fail verification, audit log starts a new chain — but the attacker now has write access to perform F3's full attack.

**Fix:**
- `chmod 0700 ~/.iomt-ids/` and document this in the runbook.
- In `_bootstrap_local_key`, after `private_path.parent.mkdir(parents=True, exist_ok=True)`, call `os.chmod(private_path.parent, 0o700)`. Defense-in-depth.

---

### F5 (MEDIUM) — Production on-disk artefacts are group-writable (`0664`)

**Where:**
- `results/models/*.pkl` and `*.pkl.sig` — `0664`.
- `results/models/dae_detector.json`, `dae_model.weights.h5` — `0664`.
- `results/reports/audit_log.jsonl` — `0664` (13 MB live audit log, group-writable).
- `module0_analysis/dataset_integrity.json` — `0664` (despite [security.py:377](module0_analysis/security.py#L377) requesting `0o640` — the `try…except OSError: pass` suggests this is the wrong mode setting AND the call failed silently, OR the file was created before the chmod was added).
- `data/processed/*.parquet` — `0664` (Phase 1 output, training-data integrity boundary).

**Impact:** Any user in the application group can tamper with model artefacts (combine with Tier 0 F2 for the DAE / Tier 0 F3 for the signing key), the audit log (combine with Tier 1 F1 for chain forgery), or the dataset integrity baseline. The signing layer was designed to protect against attackers without the private key — group-write breaks that assumption from a different angle.

**Fix:**
- Tighten `os.chmod` in every writer site:
  - `signed_pickle.dumps_signed` → `0o640` on the `.pkl` and `.pkl.sig`.
  - `DAEDetector.save_artefacts` → `0o640` on JSON + weights.
  - `AuditLogger.__init__` / first write → `0o640` on `audit_log.jsonl` (already requested by Tier 1 F3).
  - `_write_signed_metadata` → verify the existing `0o640` chmod actually applies on Linux (the `except OSError: pass` may be swallowing the failure when the file already exists with different ownership).
- Document the operational expectation: production deployment puts these files on a volume mounted as the application user, **group-readable not group-writable**.

---

### F6 (MEDIUM) — Workflow installs build-time deps inline before `pip-audit`, including `bandit[toml]` itself

**Where:** [phase0.yml:48](.github/workflows/phase0.yml#L48), [phase1.yml:95](.github/workflows/phase1.yml#L95), [phase2.yml:103](.github/workflows/phase2.yml#L103).

```yaml
- name: Install security tools
  run: pip install bandit[toml] pip-audit cyclonedx-bom
```

Then later: `pip install -r requirements.txt && pip-audit --strict --desc`. Order matters:
1. The audit runs against the installed environment, which now includes `bandit`, `pip-audit`, `cyclonedx-bom` *and their transitive trees* (e.g., `stevedore`, `rich`, `cyclonedx-python-lib`).
2. CVEs in those tools are surfaced as if they were *application* dependencies, polluting the report.
3. A failing CVE in `bandit` itself blocks the `--strict` gate even though it has nothing to do with the production runtime.

This is exactly what Tier 0 saw when I ran `pip-audit` without `-r requirements.txt` — many of the CVEs surfaced (`langchain-core`, `langsmith`, `mako`, `pygments`, `lupa`) are dev / build tooling, not production.

**Fix:**
- Install security tools in a separate venv: `python -m venv /tmp/sec-tools && /tmp/sec-tools/bin/pip install bandit[toml] pip-audit cyclonedx-bom`.
- Run `pip install -r requirements.txt` in the *primary* env and `/tmp/sec-tools/bin/pip-audit -r requirements.txt --strict --desc` cross-environment.
- This also cuts noise — the report will show production CVEs only.

---

### F7 (LOW) — `.gitignore` does not block `.pem`, `.key`, or `certs/`

**Where:** [.gitignore](.gitignore).

The current `.gitignore` covers `.env`, `.env.local`, model files (`*.pkl`, `*.h5`, `*.keras`, `*.joblib`), and the standard Python ignores. It does **not** mention:
- `*.pem` (cf. F1 — exactly what should have been blocked from being committable)
- `*.key`
- `config/certs/` or `**/certs/`
- `secrets/`, `.secrets/`
- `*.crt`, `*.cer`, `*.p12`, `*.pfx`

**Fix:** Append:
```gitignore
# Crypto material
*.pem
*.key
*.crt
*.cer
*.p12
*.pfx
config/certs/
secrets/
.secrets/
```

---

### F8 (LOW) — Workflows allow `workflow_run` chaining without `pull_request_target`-style isolation

**Where:** [phase1.yml:8-11](.github/workflows/phase1.yml#L8-L11), [phase2.yml:14-17](.github/workflows/phase2.yml#L14-L17).

```yaml
workflow_run:
  workflows: ["Phase 0 — Lint · Security · Test · Build"]
  types: [completed]
  branches: [main]
```

The `workflow_run` trigger fires on completion of the upstream workflow (Phase 0). The workflows are gated to `branches: [main]`, which means they execute against `main` HEAD — but `workflow_run` runs with the *workflow file from `main`*, not from the PR. That's the recommended Microsoft / Snyk pattern and avoids the canonical `pull_request_target` privilege escalation. Good as-is.

What is missing: there is no `if: github.event.workflow_run.conclusion == 'success'` at the job level for Phase 2 except for `lint-phase2` ([phase2.yml:32-34](.github/workflows/phase2.yml#L32-L34)). Other Phase 2 jobs (`test-phase2`, `security-scan-phase2`, `integration-test`, `build`) would attempt to run even on a Phase 0 failure that the `workflow_run` propagated. They only `needs: lint-phase2`, so they'll skip transitively — works in practice, but the dependency chain is brittle.

**Fix:** Add the same `if:` guard to every top-level Phase 2 job, mirroring Phase 1 where the pattern is correctly applied.

---

### F9 (INFO) — `permissions: contents: read` is correctly scoped at workflow level

[phase0.yml:9-10](.github/workflows/phase0.yml#L9-L10), [phase1.yml:13-14](.github/workflows/phase1.yml#L13-L14), [phase2.yml:19-20](.github/workflows/phase2.yml#L19-L20), [prototype.yml:9-10](.github/workflows/prototype.yml#L9-L10).

All four workflows declare `permissions: contents: read` at the top. No job over-claims (e.g., `id-token: write` for OIDC, `packages: write` for registry). Artifact upload uses the default ephemeral artifact storage, not container-registry pushes. Good baseline.

---

### F10 (INFO) — `run_all_modules.py` and `Makefile` use list-arg subprocess; no shell injection

[run_all_modules.py:79-82](run_all_modules.py#L79-L82) — `subprocess.run([sys.executable, module["script"]], cwd=str(PROJECT_ROOT))`. The `module["script"]` is a hardcoded path from the `MODULES` list (constant). Safe.

[Makefile](Makefile) — targets only invoke `$(PYTHON) -m tools.…` with hardcoded module names. No shell expansion of untrusted input.

---

## Surface that was checked and is clean

- **No real API tokens in git history.** `git log -p -S 'sk-'` matched only `"sk-fake"` in test files. No live `sk-…`, `AKIA…`, `ghp_…`, `xox[bpsa]-…` tokens.
- **`.env.local` is not tracked**, only `.env.example` is. The example file contains path constants, no secrets.
- **CI uses pinned major versions of GitHub Actions** (`@v4`, `@v5`) — not SHA-pinned, but at least using the maintained majors rather than `@master`.
- **Workflows are `branches: [main]`-scoped** and use `concurrency: cancel-in-progress: true` to avoid CI thrash.
- **CI explicitly bootstraps the signed integrity baseline** via the production CLI ([phase1.yml:151-189](.github/workflows/phase1.yml#L151-L189)), exercising the same code path the runtime uses. The comment at L138 explicitly calls out the prior failure pattern ("Anything short of this re-introduces the legacy unsigned format that the hardened verifier refuses"). Good defense-in-depth.
- **CI has a regression guard against biometric value leakage** in Phase 0 artefacts ([phase0.yml:102-128](.github/workflows/phase0.yml#L102-L128)) — regex scans every `.json`/`.md`/`.csv` for biometric column names followed by min/max/median numeric fields. This is the kind of property-based test that catches PHI exposure better than schema validation.
- **`run_all_modules.py` and `Makefile`** — no shell injection, no `eval`, no `os.system`. Hardcoded module paths.

---

## Recommended next steps

1. **F1 (history rewrite + rotation)** — escalate immediately. Even if "test only," the appearance of leaked private keys in a HIPAA-context repo is a compliance fire drill. Use `git filter-repo`, force-push, coordinate re-clones, add to `.gitignore` (F7), wire `gitleaks` pre-commit.
2. **F2 (CI gate functionality)** — add the missing `pyproject.toml` + `Dockerfile`, then run all four workflows green on a feature branch before merging anything else. Today's CI gives false assurance.
3. **F3 (dependency pinning)** — `pip-compile --generate-hashes` once, then every dep bump is a deliberate diff. Pair with F6 (split sec-tools venv) so the audit signal is clean.
4. **F4 + F5 (filesystem perms)** — `chmod 0700 ~/.iomt-ids/`; fix the chmod calls in `signed_pickle`, `audit_writer`, `DAE.save_artefacts` to apply `0o640` on first write.
5. **F7 (`.gitignore` extension)** — append the crypto-material patterns in the same PR as F1.
6. Document F2/F3/F4/F5/F7 as **known infrastructure gaps** in [docs/Threat Model and Scope.html](docs/Threat%20Model%20and%20Scope.html) until the fixes land — they substantially weaken the assurances stated by the in-code signing layer.

---

## Tier 0–3 roll-up

| Tier | Sev | Count | Theme |
| --- | --- | --- | --- |
| 0 | H | 3 | Trust-anchor lifecycle (bootstrap-resign, DAE unsigned, signing-key auto-bootstrap) |
| 0 | M+L+I | 8 | Threshold JSON unsigned, Keras CVE, scaler bypass, path validator, sidecar order |
| 1 | H | 2 | Audit chain `legacy_ok` default + tail recovery without anchor |
| 1 | M+L+I | 8 | Audit log umask, LLM env-key, log_full_prompt non-enforcement, sanitize_for_log gaps, no inference server (positive) |
| 2 | H | 2 | `risk_scores.npz` allow_pickle deserialization + Streamlit `0.0.0.0:8501` no-auth |
| 2 | M+L+I | 8 | MD5 seed, sign-failure swallow, `@st.cache_data` pinning, DAE bound checks, training pipeline clean (positive) |
| 3 | H | 2 | Real RSA keys in git history + non-functional CI security gate |
| 3 | M+L+I | 8 | Open dep pinning, group-writable signing dir, group-writable artefacts, sec-tools env pollution, `.gitignore` gaps |

**Cross-tier pattern that keeps recurring:** the signing layer is well-designed in code, but **the trust anchor (private key on disk + permissions + history of related keys) is the weak link**. Tier 0 F3 (auto-bootstrap), Tier 2 F2 (open dashboard writes to signed chain), Tier 3 F1 (real keys in history), Tier 3 F4 (group-writable signing dir) all reduce to "whoever can touch `~/.iomt-ids/` or the audit-log directory effectively owns the audit chain." A single PR that (a) pins the expected `signing_key_id` in a VCS-tracked config, (b) chmods the parent directory, and (c) refuses to auto-bootstrap when artefacts already exist would close the largest gap in the system.

The Tier 2 finding on the Streamlit dashboard (`0.0.0.0:8501`, no auth, writes through the signed chain) is the operationally most exposed fault if the dashboard is ever run on a host with a reachable interface. That fix is a single `.streamlit/config.toml`.

If only three fixes can be prioritised across the entire review, they are:
1. **Tier 3 F1** — git history rewrite + key rotation.
2. **Tier 0 F3 + Tier 3 F4** — pin signing key id, lock down the directory, refuse auto-bootstrap.
3. **Tier 2 F2** — `.streamlit/config.toml` with localhost binding and an auth pre-check before any signed-chain write.
