# RQ1 — Threat Model Scope Statement

**Artifact reference:** `results/rq1_metrics.json`, `results/reports/risk_scores.npz`
**Last updated:** 2026-05-25
**Status:** Locked for Phase 1 evaluation

---

## 1. In-scope detection surface — NetFlow-only

The detection pipeline evaluated in RQ1 operates **exclusively on
NetFlow-shaped network telemetry** (post-feature-extraction flow records).
Specifically:

- **Input modality**: bidirectional flow records with the 47 numeric +
  categorical features documented in `module1_preprocessing/config.py`.
- **Aggregation window**: per-flow (not per-packet, not per-session).
- **Vantage point**: clinical-network span port / TAP downstream of the
  edge firewall, upstream of the IoMT VLAN.
- **Sample population**: 2,448 frozen test-split records drawn from the
  IDS-HC-IoMT-2024 corpus; 307 attacks (12.5%) across 2 attack categories
  (Spoofing, Data Alteration).

## 2. Threat classes covered

The current Track A + Track B detector pair targets two specific attack
families because the labeled corpus only contains these two:

| Attack category | n samples | Coverage |
|-----------------|-----------|----------|
| Spoofing        | 169       | ✓ trained + evaluated |
| Data Alteration | 138       | ✓ trained + evaluated |

Per-class detection performance is reported in
`results/rq1_track_b_per_class.json`.

## 3. Explicitly out-of-scope

The following threat surfaces are **NOT** assessed by the RQ1 evaluation
and require separate evidence to defend any production claim:

1. **Payload-content attacks** — DPI / HTTP body inspection / TLS-decrypted
   content are not in the input feature space. SQLi, XSS, command
   injection against clinical web apps are not detectable.
2. **Endpoint-only attacks** — file-system tampering, USB drops, local
   privilege escalation produce no network signal until lateral
   movement and are not in scope until that point.
3. **Encrypted exfiltration with normal flow shape** — adversary
   tunneling data over a long-lived TLS connection to a permitted
   destination at trickle bandwidth would not perturb the flow-shape
   features the model relies on.
4. **Zero-day attack categories** — the Track A ensemble is supervised
   on the 2 corpus classes; novel attack families would route through
   Track B's anomaly path (DAE cascade) with uncertain recall — see
   limitations in §5.
5. **Insider misuse with valid credentials** — no behavioral baseline
   tied to user identity; the detector sees flow shape only.
6. **Physical-layer attacks** — RF interference, signal jamming,
   physical device tampering produce no NetFlow signal.

### 3.1 Trust boundaries documented by the 2026-05-29 security review

Captured here because each is a *positive* property of the current
codebase that a future architect adding a new entry point must
deliberately preserve (or surrender, with operator awareness):

- **No inbound HTTP / RPC server.** `grep -rE "Flask|FastAPI|uvicorn|
  gunicorn|HTTPServer|socketserver"` returns zero hits in production
  code (tier 1 F9, tier 2 F10). Every runtime entry point is a batch
  job iterating a frozen parquet; "online" in `online_explainer.py`
  refers to per-alert latency, not network exposure.
- **One inbound listener: the Streamlit dashboard at module 6.** Pinned
  to `127.0.0.1:8501` via `.streamlit/config.toml` (tier 2 F2). The
  dashboard's signed-chain writes are additionally gated on a
  participant id resolved through the enrolment table in
  `module6_evaluation/study_loader.py`.
- **`module5_responses.executor.ActionExecutor` is simulation-only**
  (tier 1 F10). It appends to an in-memory log and never performs
  isolation / traffic-restriction / re-auth side effects. Any future
  PR that wires the executor to a real network control MUST add an
  authz boundary; the current codebase has none.

Adding a real inference server, exposing the dashboard externally,
or wiring real side effects into the executor are each a scope
expansion that requires fresh threat-model work — they are not
covered by the current evidence.

## 4. Assumptions on adversary

- **Knowledge**: adversary may know the architecture but does not have
  white-box access to model weights or training data ordering.
- **Capability**: adversary can issue arbitrary NetFlow-shaped traffic
  from any IP within the clinical network or from external sources
  routed through the span port.
- **Adaptive attacks**: NOT assessed. The reported metrics assume
  non-adaptive attackers. Evasion via flow-shape perturbation is a
  known weakness; mitigation depends on the cascade re-baselining
  flagged by Module 4's drift detection (separate evaluation).

## 5. Detection-only, not response

This document scopes **detection**. The HITL response surface (Modules
5 / 6) — operator acknowledge / escalate / dismiss decisions, audit
chain integrity, role-adaptive explanation rendering — is evaluated
under RQ3 (user study) and RQ2 (response policy correctness).
RQ1 metrics make no claim about whether surfaced alerts trigger the
correct downstream action.

## 6. Safety floor — Invariant 2

Within RQ1's scope, the detection pipeline is wired to a Module-5
safety floor (Invariant 2): any alert on a life-critical device
(`d_crit >= 0.8`) that the composite scoring would otherwise route to
LOW gets surfaced anyway. This is **not** a detection improvement — it
is a policy override that trades specificity for FNR_critical.

In the evaluated split:
- `n_critical_device_attacks = 138` (all on `d_crit >= 0.8` devices)
- `n_critical_attacks_surfaced = 138`
- `FNR_critical = 0.000` (target was <0.05)

This metric is gated by the safety floor; raw detector recall on the
same subset is reported separately in the ablation table.

---

## Appendix — Linked artifacts

- `results/rq1_metrics.json` — headline metrics with this scope
- `results/rq1_ablation_track_a.json` — per-model breakdown (XGBoost / RF / DT)
- `results/rq1_ablation_track_b.json` — DAE raw vs cascade
- `results/rq1_track_b_per_class.json` — Spoofing / Data Alteration per-class
- `results/rq1_weight_sensitivity.json` — composite-risk weight grid
- `results/figures/roc_curves.png` — all four detectors
- `results/figures/confusion_matrix.png` — surfacing-decision matrix
