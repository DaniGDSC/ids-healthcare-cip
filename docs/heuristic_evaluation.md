# Heuristic Evaluation — Method 4

> Standards-grounded compliance check across 4 frameworks (26 heuristics).
> Companion to [`results/reports/heuristic_compliance.yaml`](../results/reports/heuristic_compliance.yaml).

Generated: 2026-05-06  ·  Branch: `fix/shap-category-vocab`  ·  Method 4

---

## 1. Frameworks and method

| Framework | Source | Heuristics |
|---|---|---|
| **Nielsen 10 Usability Heuristics** | Nielsen (1994; reaffirmed 2020) | 10 |
| **DARPA XAI 4 Principles** | Gunning & Aha (2019) | 4 |
| **NIST AI Risk Management Framework** | NIST AI 100-1 (2023) — Map / Measure / Manage / Govern | 4 |
| **Healthcare-specific standards** | HIPAA Security Rule (NIST 800-66r2 2024), FDA Premarket Cybersecurity (2023), IEC 62443-4-1 | 8 |
| **Total** | | **26** |

Each heuristic is graded `PASS`, `PARTIAL`, `FAIL`, or `NOT_APPLICABLE` against concrete artifact evidence (code path, test name, doc section). `PARTIAL` is used when the spirit of the heuristic is honoured but a measurable artifact is missing; `NOT_APPLICABLE` only when the heuristic targets a system layer outside the prototype's scope (e.g. physical security).

---

## 2. Aggregate compliance

![heuristic compliance heatmap](../results/figures/heuristic_compliance_heatmap.png)

| Verdict | Count | Pass-rate |
|---|---|---|
| PASS | 21 | |
| PARTIAL | 4 | |
| FAIL | 0 | |
| N/A | 1 | |
| **Total evaluated (excluding N/A)** | **25** | |
| **Strict pass-rate** | | **84%** |
| **With-partial-credit pass-rate** | | **92%** |
| **Threshold (acceptance criterion)** | | ≥80% |

**Both rates exceed the 80% threshold** — Method 4 acceptance criteria PASS.

---

## 3. Per-framework results

### 3.1 Nielsen 10 — 9 PASS / 1 PARTIAL (90%)

| ID | Heuristic | Verdict | Evidence anchor |
|---|---|---|---|
| H1 | Visibility of system status | PASS | Step [10] timestamped audit log + Step [16] OperatorDecision.timestamp + Mode-B fallback badge |
| H2 | Match real-world language | PASS | `_feature_to_narrative` clinician strings + clinical severity ladder (REQ-MVE-04) |
| H3 | User control and freedom | PASS | INVARIANT 3 (NO AUTO-EXECUTION) — operator can ignore any recommendation |
| H4 | Consistency and standards | PASS | OperatorRole + FusionClass + severity-tier enums used identically across modules |
| H5 | Error prevention | PASS | Safety floor (REQ-MVE-13) + ROLE_FORBIDDEN (REQ-MVE-09) + EA-06 (REQ-MVE-17) |
| H6 | Recognition over recall | PASS | 3-layer MVE renders all decision-relevant context inline; ATT&CK ID surfaced (REQ-MVE-18) |
| H7 | Flexibility and efficiency of use | **PARTIAL** | Per-role views accelerate triage; **gap**: no keyboard shortcuts/bulk-action UX in Streamlit dashboard |
| H8 | Aesthetic and minimalist design | PASS | Word-budget enforcement (Layer 1 ≤60w, total ≤150w); REQ-MVE-03 no raw SHAP |
| H9 | Help recover from errors | PASS | `suppression_reason` text; DataQuality.DEGRADED/FAILED surfaced; no vague Layer 3 verbs |
| H10 | Help and documentation | PASS | ARCHITECTURE.md + docs/architecture.md + threat_model.md + REQ trace matrix |

### 3.2 DARPA XAI 4 — 4 PASS / 0 PARTIAL (100%)

| ID | Principle | Verdict | Evidence anchor |
|---|---|---|---|
| P1 | Explanation | PASS | SHAPContext + MVE Layer 1 references (REQ-MVE-08); M5=1.0 |
| P2 | Meaningful (role-appropriate) | PASS | 3 OperatorRole views (REQ-MVE-09); Method 7 info-gain 8/8 dims; Method 1 severity-accuracy 55-58% in Group B |
| P3 | Explanation accuracy | PASS | INVARIANT 5 + SHAP stability score (REQ-MVE-20); M5=1.0 |
| P4 | Knowledge limits | PASS | DataQuality flag + SHAPContext.confidence_from_shap + honest-limitation docs |

DARPA XAI is the strongest framework for this prototype — all 4 principles fully satisfied with concrete code references and quantitative evidence.

### 3.3 NIST AI RMF — 3 PASS / 1 PARTIAL (75%)

| Function | Verdict | Evidence anchor |
|---|---|---|
| **Map** (context, classification, capabilities, third-party) | PASS | ARCHITECTURE.md + threat_model.md + research_spec.yaml; pinned dependencies + signed_pickle protection |
| **Measure** (metrics, uncertainty, explainability, safety) | PASS | 5 YAML deliverables (detection_baseline / track_a_performance / two_stage_fusion / risk_adaptive / novelty); Wilson CIs throughout |
| **Manage** (prioritise, residual, monitoring, response) | **PARTIAL** | Threat→mitigation matrix complete; **gap**: continuous monitoring is simulation-only (drift_detection.py is a simulation, not live operational dashboard) |
| **Govern** (policies, accountability, workforce, engagement) | PASS | DO NOT BUILD list + INVARIANTS 1-7 + HITL boundary + 50-participant M5 study + multi-role Method 1 |

### 3.4 Healthcare standards — 5 PASS / 2 PARTIAL / 1 N/A (86% excluding N/A)

| Standard | Verdict | Evidence anchor |
|---|---|---|
| HIPAA Administrative Safeguards | PASS | Audit controls (REQ-MVE-15) + access management via HITL (INVARIANT 3) |
| HIPAA Physical Safeguards | NOT_APPLICABLE | Physical security is hospital-facility responsibility (threat_model.md §3.3) |
| HIPAA Technical Safeguards | PASS | Sanitiser integrity + audit-log SHA-256 tamper detection |
| HIPAA PHI minimum-necessary | PASS | REQ-MVE-03 no raw SHAP; word-budget enforcement; PHI weight in S_data |
| FDA Premarket Cybersecurity | **PARTIAL** | STRIDE threat model + threat→mitigation matrix; **gap**: formal CycloneDX/SPDX SBOM not yet generated |
| FDA Post-market Surveillance | **PARTIAL** | drift_detection.py provides foundation; **gap**: no formal CVD policy, no SOAR integration |
| IEC 62443-4-1 Security-by-design | PASS | 7 invariants + two-stage fusion + safety floor + per-device thresholds + EA-06 = layered defence |
| IEC 62443-4-1 Secure-development lifecycle | PASS | Threat model + 177 automated tests + signed_pickle + negative tests |

---

## 4. Strengths

1. **Defence-in-depth architecture matches IEC 62443-4-1 expectations.** The prototype layers seven invariants on top of: two-stage fusion (catches Track-A-evading attacks), safety floor (CRITICAL+unpatchable always surfaces), per-device thresholds (clinical-tier-aware sensitivity), EA-06 mitigation (NaN-injection cannot mask anomalies), and a recommendation-only invariant that keeps the IDS from becoming an attack surface itself.

2. **DARPA XAI 4/4 PASS** — explanation, meaningfulness (validated by Method 7 information-gain showing 8/8 dimensions covered), accuracy (M5=1.0 SHAP narrative alignment), and explicit knowledge-limit signalling (DataQuality flag + SHAP stability + honest-limitation docs).

3. **Audit + traceability is unusually rigorous for a research prototype.** Append-only log with SHA-256 tamper detection (`tests/test_audit_append_only.py`), `OperatorDecision` schema validation, and a 22-requirement formal trace matrix that maps each spec line to its implementation file and verification test.

4. **Nielsen H4/H6/H8** — consistency, recognition-over-recall, minimalist design — MVE word budget + 3-layer structure + role-specific views deliver the canonical UX virtues without retro-fitting.

---

## 5. Gaps (all production-deployment scoped)

| Gap ID | Heuristic | Description | Closure phase |
|---|---|---|---|
| **GAP-HE-1** | FDA Premarket | Generate CycloneDX or SPDX SBOM artifact alongside model release | Production deployment |
| **GAP-HE-2** | FDA Post-market | Coordinated Vulnerability Disclosure policy + SOAR integration | Phase 3 |
| **GAP-HE-3** | NIST RMF Manage | Live drift-monitoring dashboard (drift_detection.py is a simulation) | Phase 3 |
| **GAP-HE-4** | Nielsen H7 | Streamlit accelerators for high-volume operators (50+ alerts/day) | Production UX phase |

None of the four gaps are fundamental research limitations — all are production-deployment hardening tasks. The prototype passes the threshold for thesis defence; the gaps are pre-deployment work.

---

## 6. Cross-method corroboration

| Other method | What it confirms about Method 4 |
|---|---|
| **Method 6** REQ trace matrix | 22/22 PASS at requirement level; Method 4 confirms *which standards frameworks* the requirements satisfy |
| **Method 7** information gain | Confirms DARPA P2 (meaningfulness): MVE adds 5 dimensions over raw IDS view |
| **Method 1** LLM persona simulation | 100% DO_NOT compliance + 85% cross-role severity consistency confirm Nielsen H4 (consistency) and DARPA P3 (accuracy) |
| **Method 5** case studies | 92.5% MVE rubric pass-rate confirms Nielsen H8 (minimalist) and DARPA P2 (meaningfulness) at the per-alert level |
| **Method 2** self-consistency | 100% within-persona temporal agreement at temperature=0 confirms Nielsen H4 (consistency) at the model-output level |

---

## 7. Acceptance criteria — 3/3 PASS

| Criterion | Status | Evidence |
|---|---|---|
| Per-framework compliance table | **PASS** | All 4 frameworks have evaluation block + summary block + per-heuristic evidence in `heuristic_compliance.yaml` and §3 above |
| Identified strengths and gaps | **PASS** | §4 (4 strengths) + §5 (4 gaps with closure phases) |
| ≥80% PASS coverage (excluding NOT_APPLICABLE) | **PASS** | 21/25 = 84% strict; 92% with partial credit; both exceed 80% threshold |

---

## 8. Reproducibility

```bash
# Verify the YAML deliverable
python3 -c "import yaml; d = yaml.safe_load(open('results/reports/heuristic_compliance.yaml')); print(d['aggregate_compliance'])"

# Regenerate the figure
python3 - << 'EOF'
# (See the embedded matplotlib script in this commit's audit trail)
EOF
```

Source artifacts: `results/reports/heuristic_compliance.yaml`, `results/figures/heuristic_compliance_heatmap.png`.
