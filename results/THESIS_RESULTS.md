# Thesis Results — Computed Outputs

**Generated:** 2026-05-16 10:20 UTC
**Code commit:** `52521ee16850dd6a0dae6a1e76c716f53dbf359b`
**Random seed:** 42 (applied throughout)

This document summarises the quantitative evidence produced by `analysis/compute_rq*.py` for the three research questions.
Every per-RQ JSON includes a `provenance` block (timestamp, git commit, input file SHA-256s, schema version).

## Executive Summary

| RQ | Status | Notes |
|---|---|---|
| RQ1 (Detection + Sensitivity) | OK | 5 subsections; baseline metrics, ablation extract, sensitivity analysis, truth table, correlation |
| RQ2 (MVE Faithfulness + Study) | OK | 4 subsections; SHAP stability, MVE alignment, MITRE coverage, LLM-persona study |
| RQ3 (Safety + HITL) | OK | 5 subsections; pytest summary, no-auto-execution, audit chain, role consistency, HITL study |

## RQ1 — Detection + Sensitivity Analysis

### RQ1.1 — Baseline detection metrics (test split)

- Source: `xgboost_test_predictions.npz` (n=2448 samples, 307 attacks / 2141 benign)
- Threshold (F2-tuned): 0.0185
- Sensitivity: **0.9739**
- Specificity: **0.9752**
- F2 score: **0.9462**
- F1 score: 0.9074
- AUC: **0.9952**
- PR-AUC: 0.9806
- FNR_CRITICAL (proxy = FN/(FN+TP)): **0.0261**
- Confusion matrix: TP=299 FN=8 FP=53 TN=2088
- Hard assertions: PASSED
- Figures: `results/figures/roc_curves.pdf`, `pr_curves.pdf`, `confusion_matrix.pdf`

### RQ1.2 — Track B per-class AUC

- EHMS: `Data Alteration`=0.996, `Spoofing`=0.549
- MEDSEC25: `Exfiltration`=0.852, `Initial access`=0.905, `Lateral movement`=0.833, `Reconnaissance`=0.927

### RQ1.3 — Composite-risk weight sensitivity

- 30 perturbations (±10% per weight, renormalised to sum=1):
  - mean agreement: **0.9823**
  - std / min / max: 0.0082 / 0.9669 / 0.9947
  - IQR p25–p75: [0.9755, 0.9905]
- Baselines (vs ARCHITECTURE.md weights 0.40/0.25/0.15/0.20):
  - `equal_weights`: agreement=0.7345, FNR_CRITICAL Δ=0.0106
  - `c_detect_only`: agreement=0.7659, FNR_CRITICAL Δ=0.0000
  - `multiplicative`: agreement=0.7929, FNR_CRITICAL Δ=0.0000
- N alerts evaluated: 2448
- Figure: `results/figures/sensitivity_histogram.pdf`

### RQ1.4 — Tier × Patchable × Maintenance truth table

- 16 (tier × patchable × maintenance) combinations derived from `src.risk_scorer.score_alert()`
- Discrepancies between code and documented expected behaviour: **2**
  - ['MEDIUM', True, False]: code=False vs doc=True
  - ['MEDIUM', False, True]: code=True vs doc=False
- Full table: `results/rq1_truth_table.md` (and `.yaml`)

### RQ1.5 — D_crit vs D_clinical_tier correlation

- Pearson r = **0.6120** (p = 0.0454)
- N devices: 11
- Interpretation: moderate correlation (0.4–0.7) indicates partial overlap

## RQ2 — MVE Faithfulness + User Study

### RQ2.1 — SHAP stability

- Method: TreeSHAP on signed Track A pipeline; 10 perturbations × U(0.99, 1.01) multiplicative noise; Jaccard top-3.
- N alerts: 18
- Mean stability: **0.8628**
- Median stability: 1.0000
- Fraction stable (≥0.90): **0.6667**
  - `BENIGN`: mean=0.8628, n=18, pct_stable=0.6667
- Figure: `results/figures/shap_stability_distribution.pdf`

### RQ2.2 — MVE-SHAP alignment (stratified)

- Top-3 XGBoost SHAP features matched against Layer-1 narrative (full feature name, narrative phrase, or token of length ≥4 from configured feature_categories).
- `BENIGN` (n=16): all-3=0.0000, 2+=0.0000, any=1.0000, MITRE ref'd=0.0000, xgb_low_conf SHAP src=0
- Aggregate (caveat: see by_fusion_class): all-3=0.0000 over n=16
- _Overall metric averages across fusion classes including those where SHAP is acknowledged as not faithful (NOVEL_ANOMALY); see by_fusion_class for stratified reporting._

### RQ2.3 — MITRE ATT&CK coverage

- Attack categories defined: 7
- Categories with ≥1 technique: 6
- Orphan categories: ['normal']
- MITRE framework version: `v14.1`
- Techniques by confidence: {'HIGH': 4, 'MEDIUM': 8, 'LOW': 1}
- Layer-1 MITRE technique-ID grounding: 0/20 alerts (0.0000)
  - _Layer-1 MVE text typically describes baseline deviation in clinical language; MITRE technique IDs may not appear in Layer-1 explicitly. Coverage of the mapping file itself is more informative._

### RQ2.4 — User study (LLM-persona) faithfulness analysis

- Stat test: Mann-Whitney U (two-sided) + Cliff's delta; Holm-Bonferroni correction
- N survey files: 100
- **IT_generalist** (n_responses=1000)
  - `accuracy`: A median=0.0000 vs B median=0.0000 → U=75500.0000, p=0.0000, p_holm=0.0000, Cliff δ=-0.3960
  - `confidence`: A median=4.0000 vs B median=2.0000 → U=182885.0000, p=0.0000, p_holm=0.0000, Cliff δ=1.0000
- **biomed_engineer** (n_responses=600)
  - `accuracy`: A median=0.0000 vs B median=0.0000 → U=44550.0000, p=0.5407, p_holm=0.5407, Cliff δ=-0.0100
  - `confidence`: A median=4.0000 vs B median=2.0000 → U=80262.0000, p=0.0000, p_holm=0.0000, Cliff δ=1.0000
- **nurse_manager** (n_responses=400)
  - `accuracy`: A median=0.0000 vs B median=0.0000 → U=19900.0000, p=0.8054, p_holm=0.8054, Cliff δ=-0.0050
  - `confidence`: A median=4.0000 vs B median=2.0000 → U=25748.0000, p=0.0000, p_holm=0.0000, Cliff δ=1.0000
- _LLM-persona simulation data; not human user study. Decision_time field absent in LLM responses. Bootstrap CI computed when N<30._

## RQ3 — Architectural Safety + HITL

### RQ3.1 — Test suite summary

- Pytest result: **PASSED** (passed=635, failed=0, skipped=1)
- Test files: 37
- Pytest exit code: 0
- Raw log: `results/rq3_pytest_raw.log`
  - `tests/test_coverage_mve.py` — 73 tests (passed)
  - `tests/test_day2_dashboard_polish.py` — 40 tests (passed)
  - `tests/test_safe_failure.py` — 40 tests (passed)
  - `tests/test_v4_render_helpers.py` — 40 tests (passed)
  - `tests/test_role_authority.py` — 39 tests (passed)

### RQ3.2 — No-auto-execution verification

- Grep check (subprocess/os.system/iptables/netcat/etc.): **PASSED** (matches=0)
- Import check (`import subprocess`): **PASSED** (matches=0)
- Verdict: **PASSED — No auto-execution verified**

### RQ3.3 — Audit log hash chain

- Total logs checked: 3
- Total entries scanned: 26530
- All hash chains intact: **True**
  - `results/reports/audit_log.jsonl`: n_entries=22898, status=PASSED (chain_restarts=3)
  - `results/reports/alert_responses.json`: n_entries=1632, status=N/A — no hash chain fields
  - `survey/study_responses_*.json (n=100)`: n_entries=2000, status=N/A — no hash chain in survey JSON

### RQ3.4 — Cross-role consistency

- Alerts checked: 20, anchors_present=20
- Invariant 9 (shared anchor): all_identical=True (n_violations=0)
- Invariant 6 (severity consistency): all_identical=True (n_violations=0)
- Invariant 6 (action authorization, nurse_manager most-restrictive): all_authorized=True (n_violations=0)
- Overall: **PASSED**
  - _Checked against the most-restrictive role (nurse_manager). Full per-role comparison requires per-role MVE renderings, which evaluation_alerts.json does not include (each alert has one mve_structured for the active study condition)._

### RQ3.5 — HITL user study (LLM-persona simulation)

- N survey files: 100
- **IT_generalist**: 1000 responses, action distribution: {'investigate': 483, 'null': 142, 'restrict': 112, 'isolate': 263}
  - Condition A: n=500, accuracy=0.0480, mean_confidence=3.7019, escalation_rate=0.0000
  - Condition B: n=500, accuracy=0.4440, mean_confidence=2.0000, escalation_rate=0.0000
- **biomed_engineer**: 600 responses, action distribution: {'escalate': 198, 'investigate': 369, 'null': 33}
  - Condition A: n=300, accuracy=0.0367, mean_confidence=3.9898, escalation_rate=0.0233
  - Condition B: n=300, accuracy=0.0467, mean_confidence=2.0000, escalation_rate=0.6367
- **nurse_manager**: 400 responses, action distribution: {'monitor': 321, 'null': 79}
  - Condition A: n=200, accuracy=0.0400, mean_confidence=3.6707, escalation_rate=0.0000
  - Condition B: n=200, accuracy=0.0450, mean_confidence=2.0000, escalation_rate=0.0000
- χ² escalation A vs B:
  - `biomed_engineer`: χ²=252.4975, p=0.0

## Pending Items

None.

## Caveats and known discrepancies

- **fusion_class is `BENIGN` for all 20 alerts** in `evaluation_alerts.json` (the current snapshot). RQ2.2 stratification only reports the BENIGN slice as a result. The fusion classifier did not write KNOWN_ATTACK / CONFIRMED_ANOMALY / NOVEL_ANOMALY labels in this evaluation export — investigate before citing per-class alignment numbers in the thesis.
- **Test split metrics** are computed against `xgboost_test_predictions.npz` (the model's frozen test predictions). The `compute_rq1_metrics.py` script in `module6_evaluation/` reads `evaluation_alerts.json` (demo-sourced), which is a different population — see prior `[DISCREPANCY]` flag in `docs/section313_data_flow_extraction.md`.
- **RQ1.4 truth-table** flags 2 discrepancies between code and a hypothesised documented expectation (MEDIUM × patchable × no_maint, MEDIUM × unpatchable × maint). These reflect the score level chosen for the MEDIUM probe (0.45) vs the F2-tuned base threshold (0.425) plus `risk_adaptive_thresholds.yaml` multipliers. Not a code bug — the discrepancy is between the assumed coarse policy and the actual fine-grained policy. Authoritative behaviour is in `rq1_truth_table.md`.
- **User study data is LLM-persona simulation**, not human participants. Statistical tests still apply (Mann-Whitney U over per-alert correctness/confidence) but the interpretation must reflect that LLM personas standing in for IT-generalist/biomed-engineer/nurse-manager are an approximation; the response variances differ from human responses.
- **DAE-driven novel-anomaly faithfulness** — RQ2.2 cannot evidence reduced SHAP faithfulness for novel-anomaly alerts because the snapshot contains no NOVEL_ANOMALY rows; see `shap_source_xgb_low_conf` counter (zero across all classes in the snapshot).
- **Pearson r for ordinal D_crit/D_clinical_tier** is a coarse estimator; Spearman ρ may be more appropriate for the device-tier ordinal scale. The reported r=0.612 with p=0.045 (n=11) suggests partial overlap rather than complete redundancy — consistent with ARCHITECTURE.md limitation L3.

## Reproducibility

- Random seed: 42 (numpy, sklearn, scipy bootstrap)
- Code commit: `52521ee16850dd6a0dae6a1e76c716f53dbf359b`
- Python: 3.10.20
- Provenance metadata (input file SHA-256 hashes) embedded in every per-RQ JSON.
- Re-run sequence:
  ```bash
  python -m analysis.compute_rq1
  python -m analysis.compute_rq2
  python -m analysis.compute_rq3
  python -m analysis.build_thesis_results
  ```

## Output organisation

```
results/rq1_metrics.json (1605 bytes)
results/rq1_track_b_per_class.json (1569 bytes)
results/rq1_sensitivity_analysis.json (2357 bytes)
results/rq1_truth_table.md (1014 bytes)
results/rq1_truth_table.yaml (2467 bytes)
results/rq1_dcrit_dclinical_correlation.json (4726 bytes)
results/rq2_shap_stability.json (6914 bytes)
results/rq2_mve_shap_alignment.json (1651 bytes)
results/rq2_mitre_coverage.json (3152 bytes)
results/rq2_user_study.json (5295 bytes)
results/rq3_test_summary.json (8086 bytes)
results/rq3_no_auto_execution.json (1576 bytes)
results/rq3_audit_integrity.json (1820 bytes)
results/rq3_cross_role_consistency.json (1692 bytes)
results/rq3_user_study.json (4035 bytes)
results/computation_log.txt (8596 bytes)

results/figures/
  roc_curves.pdf (15870 bytes)
  pr_curves.pdf (14057 bytes)
  confusion_matrix.pdf (15241 bytes)
  sensitivity_histogram.pdf (16889 bytes)
  shap_stability_distribution.pdf (17159 bytes)
```

_Generated by `analysis/build_thesis_results.py` at 2026-05-16T10:20:03Z._
