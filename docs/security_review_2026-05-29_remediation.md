# Security review remediation — 2026-05-29 → 2026-05-29

Completion record for the 6-sprint fix plan derived from
`docs/security_review_2026-05-29_tier{0..3}.md`.

## Sprint roll-up

| Sprint | Commit | Findings closed | Notes |
| --- | --- | --- | --- |
| 0 — CI gates | `1df97f9` | tier 3 F2 / F3 / F6 / F8 + tier 0 F4 | Dockerfile + pyproject.toml + requirements.in + sec-tools venv split + Phase 2 workflow_run guards + Keras pin floor. |
| 1 — Trust anchor | `f29e3f8` | tier 0 F1 / F3 + tier 3 F1 / F4 / F7 | VCS-pinned key id, no-auto-bootstrap-when-artefacts-present, chmod 0700 on parent dir, verify-before-resign in `bootstrap_integrity`, `rotate_key` CLI, `.gitignore` crypto patterns, pre-commit hook. Git history rewrite escalated to operator. |
| 2 — Signed artefacts | `cb1fd6c` | tier 0 F2 / F5 / F6 + tier 2 F1 / F5 / F6 | New `common.signed_sidecar` for json+weights and npz+meta pairs. DAE detector signs (json, weights). `risk_scores.npz` schema migration drops `allow_pickle=True` at all five readers. `signed_pickle.dumps_signed` accepts a `metadata` dict bound into the signature; `model_registry` reads `optimal_threshold` from there. Bound checks in `DAEDetector.from_artefacts`. `chmod 0640` on every writer. |
| 3 — Audit chain | `9a6962d` | tier 1 F1 / F2 / F3 + tier 2 F4 / F7 + tier 3 F5 | `legacy_ok=False` default flip, `AuditLogger.__init__` verifies on open, `cross_rotation_anchor` in rotation marker, `seal_legacy` CLI for the one-time migration, `audit_log` raises `HardenedAuditUnavailable` instead of silent degrade, dashboard signed-chain write requires enrolled participant id, chmod 0640 on first JSONL write. |
| 4 — Dashboard | `f2c2bfc` | tier 2 F2 / F3 | `.streamlit/config.toml` pins `127.0.0.1:8501`, CORS + XSRF on, no usage telemetry. MD5 → SHA-256 in `assign_ab_conditions` (frozen P01..P10 keep their lookup-table parity). |
| 5 — LLM egress | `93bcdfe` | tier 1 F4 / F5 / F6 | Base URL pin, env-override refusal, API key prefix validation, per-process call cap, response sanitisation, anti-PHI clauses in the system prompt, `log_full_*` flipped to false with explicit policy `metadata_only`. |
| 6 — Cleanup | _this commit_ | tier 1 F7 / F8 / F9 + tier 2 F10 | Batch-summary NaN/Inf log + `IDS_NAN_RATIO_GATE` threshold, `get_shared_engine` process-singleton, threat-model doc updated with the 3 positive findings. |

## What was deferred to operator action

These items require coordination outside the code repository:

1. **Git history rewrite** for `config/certs/test/*-key.pem`.
   `git filter-repo --invert-paths --path config/certs/` + force-push +
   coordinate re-clones. Plan worst case until provenance of the leaked
   PEMs is verified (user answered "I don't know — need to check"; the
   plan assumes compromised).
2. **Identity rotation** if the leaked PEMs were ever issued through a
   real CA / used on a deployed service.
3. **Audit-log seal** in environments that carry a pre-Sprint-3
   `audit_log.jsonl`. Procedure in `docs/operator_runbook.md` §8. The
   dev environment specifically has a tamper at line 18255 of the
   13 MB legacy log; the operator must either restore from backup or
   quarantine the post-break prefix before the next module-5 run.
4. **`requirements.lock` regeneration** in a Python 3.11+ environment.
   `make lock` will refuse on Python 3.10 because `keras>=3.13.2` does
   not publish wheels for 3.10. Until the lock is committed, CI falls
   back to `pip install -r requirements.txt`.

## Smoke coverage

Each sprint includes a `python -c "..."` smoke check that runs
end-to-end against the touched code paths:

- Sprint 1: pin-check passes against the current dev key id, signing
  module imports + symbols exposed.
- Sprint 2: signed-pickle round-trip with and without metadata,
  signed-sidecar pair tamper detection, dashboard wrapper imports.
- Sprint 3: chmod 0640 applied on first write, `verify_audit_log`
  passes a fresh signed chain under `legacy_ok=False`,
  `AuditLogger.__init__` successfully reopens a verified file.
- Sprint 4: `.streamlit/config.toml` parses with `127.0.0.1` binding,
  SHA-256 confirmed as the only hash in the file.
- Sprint 5: base URL override detection, API key prefix validation,
  call-cap enforcement, sanitiser pattern coverage for UK NHS + ISO
  date, anti-PHI system prompt clause present.
- Sprint 6: shared engine singleton identity, NaN/Inf batch summary,
  threat-model doc additions.

CI gates that exercise the same paths end-to-end (bandit, pip-audit,
cyclonedx, integration test with `bootstrap_integrity`) live in
`.github/workflows/phase{0,1,2}.yml` and are now functional after
Sprint 0.
