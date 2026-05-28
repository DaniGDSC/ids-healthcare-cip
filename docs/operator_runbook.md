# Operator runbook — IoMT IDS

Audience: the engineer responsible for installing, rotating, and verifying
the cryptographic trust roots of this pipeline. Companion to
`docs/Threat Model and Scope.html`.

## 1. Trust anchors

| Component | Where | Owner | Mode |
| --- | --- | --- | --- |
| Audit signing private key | `~/.iomt-ids/audit_signing_key.pem` (or `$IOMT_AUDIT_SIGNING_KEY`) | the application user | `0600`, parent `0700` |
| Audit signing public key | `results/reports/audit_signing_key.pub.pem` | committed to VCS | `0644` |
| Expected key id pin | `config/signing_key_id.txt` | committed to VCS | `0644` |
| Dataset integrity baseline | `module0_analysis/dataset_integrity.json` | the application user | `0640` |
| Module 5 audit log | `results/reports/audit_log.jsonl` | the application user | `0640` |
| Signed model pickles | `results/models/*.pkl`, `*.pkl.sig` | the application user | `0640` |

The signing private key directory MUST be `0700` (or stricter). The CI
workflow verifies that signed artefacts plus an absent private key trip
the auto-bootstrap refusal — that exception is the canary that the trust
root has been disturbed.

## 2. First-time install

1. Clone the repo, install dependencies:
   ```
   python3.11 -m venv .venv
   . .venv/bin/activate
   pip install --require-hashes -r requirements.lock
   ```
   (Or `make lock` first if `requirements.lock` is stale.)
2. Bootstrap the audit signing key by running any signed-write code path
   ONCE (e.g. `python -m module5_responses` against a tiny smoke input).
   The first call generates the keypair, applies `0700` to the parent
   directory, `0600` to the key file, and writes the public key sidecar.
3. Copy the generated `key_id` (`grep ecdsa-p256 ~/.iomt-ids/...` or
   inspect the log) into `config/signing_key_id.txt`, commit the change
   alongside the public-key sidecar (`results/reports/audit_signing_key.pub.pem`).
4. Bootstrap the dataset integrity baseline:
   ```
   python -m module0_analysis.bootstrap_integrity --config module0_analysis/config.yaml
   ```
5. Run the full pipeline once to seed signed model pickles, DAE sidecars,
   risk score artefacts, and the initial audit log entries.

## 3. Routine operation

- Every signed-write code path verifies the key id matches the pin
  before it produces a signature. A pin mismatch fails the operation —
  CRITICAL audit event surfaces in `results/reports/audit_log.jsonl`.
- `verify_audit_log(..., legacy_ok=False)` is the default since the
  pre-hardening migration is complete. CI runs this against every
  archive after each rotation.

## 4. Key rotation (deliberate)

A planned rotation:

1. Stop every process that holds the cached private key (`@lru_cache`).
2. Move the existing key to a sealed offline location:
   ```
   mv ~/.iomt-ids/audit_signing_key.pem ~/.iomt-ids/audit_signing_key.retired-$(date +%Y%m%d).pem
   chmod 0400 ~/.iomt-ids/audit_signing_key.retired-*.pem
   ```
3. Trigger the bootstrap path (run any signed-write smoke). It will
   **refuse** because signed artefacts exist on disk. That refusal is the
   correct behaviour. The operator must explicitly run the rotation CLI:
   ```
   python -m module5_responses.audit.rotate_key --i-understand-this-orphans-old-signatures
   ```
   (The CLI emits a `SIGNING_KEY_ROTATED` event into the audit chain,
   archives the old log via `rotate_and_purge`, and seeds a new chain
   with `cross_rotation_anchor` pointing at the archive's last
   `integrity_hash`.)
4. After the new key is live, update `config/signing_key_id.txt` with
   the new fingerprint and commit. Re-sign the model pickles via
   `python -m tools.resign_models`.
5. Re-bootstrap the dataset integrity baseline:
   ```
   python -m module0_analysis.bootstrap_integrity --config module0_analysis/config.yaml
   ```

## 5. Emergency: key was lost

If `~/.iomt-ids/audit_signing_key.pem` is gone and there is no archived
copy:

1. Existing signed artefacts can no longer be verified by future loads.
   Treat them as forensically read-only.
2. Capture the pre-loss state for audit:
   ```
   tar czf /secure/iomt-loss-$(date +%Y%m%dT%H%M%S).tar.gz \
       results/reports/audit_log.jsonl \
       results/reports/audit_archive/ \
       module0_analysis/dataset_integrity.json \
       results/models/*.pkl.sig \
       config/signing_key_id.txt
   ```
3. Run the rotation CLI with the `--key-lost` flag. The CLI will:
   - Refuse if the signed-artefacts presence guard is on (it is, by
     default).
   - Require an explicit operator override (`--i-acknowledge-chain-break`).
   - Emit a CRITICAL `SIGNING_KEY_LOST` event into a fresh chain whose
     genesis record references the archived prior chain hash.
4. Update `config/signing_key_id.txt` and re-bootstrap as in §4.

## 6. Compromise indicators

Watch for these in CI / monitoring:

- `INTEGRITY_BOOTSTRAP_REFUSED` — someone tampered with
  `dataset_integrity.json` before the operator ran bootstrap.
- `INTEGRITY_VIOLATION` — a loaded dataset file's SHA-256 does not match
  any baseline entry.
- `INTEGRITY_METADATA_FORGED` — the baseline signature does not validate.
- `INTEGRITY_SIZE_MISMATCH` — a load was attempted against a file whose
  size matches no baseline (cheap DoS guard, possibly hostile).
- `SigningKeyTrustError` raised in any signed-write context — either
  the key id pin mismatches OR the bootstrap-refusal guard tripped.
- `RAW_DATASET_WRITABLE` — `PHASE0_PROD=1` environment had a writable
  raw dataset; raise to operator immediately.
- `legacy_chain_restarts > 0` in `rotate_and_purge` — historical only;
  after the post-Sprint-3 migration this counter MUST be 0.

## 7. Streamlit dashboard (Module 6)

The dashboard at `module6_evaluation/module6_app.py` writes
reviewer-attributed records into the audit chain. Tier 2 F2 hardening:

- `.streamlit/config.toml` pins `server.address = "127.0.0.1"` so the
  process binds loopback only.
- Operator-only invocation: `streamlit run module6_evaluation/module6_app.py`.
- Remote participants: do NOT expose port 8501 directly. Front with an
  authenticated reverse proxy (OIDC / shared secret) and propagate the
  operator identity into `st.session_state["participant_id"]` via the
  proxy's header pass-through. The dashboard refuses to call
  `get_hardened_audit().log` unless a validated participant id is
  present (Sprint 4 hardening).

## 8. CI gates

Defined in `.github/workflows/phase{0,1,2}.yml`:

- `bandit -r <module> -c pyproject.toml -ll` — fails on high/critical.
- `pip-audit -r requirements.in --strict` — fails on any unfixed CVE.
- `cyclonedx-py requirements -i requirements.in -o sbom.json` — uploads SBOM.
- `python -m module0_analysis.bootstrap_integrity` — exercises the
  production CLI on a 100-row CI subset so format drift trips CI, not
  prod.

`make security-scan` runs the same matrix locally.
