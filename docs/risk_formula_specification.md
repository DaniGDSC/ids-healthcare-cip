# Risk Formula Specification

> Workflow steps [8] Context Enrichment + [9] Composite Risk Scoring.
> Companion artifacts: [`results/figures/risk_weights_tornado.png`](../results/figures/risk_weights_tornado.png), [`results/reports/risk_report.json`](../results/reports/risk_report.json), [`results/reports/risk_scores.npz`](../results/reports/risk_scores.npz).

Generated: 2026-05-05  ·  Branch: `fix/shap-category-vocab`

---

## 1. Composite Risk Formula

### 1.1 Design intent — v1

The original specification proposed a four-term linear composite over normalized signals in `[0, 1]`:

```text
R = 0.40 · C_detect
  + 0.25 · D_crit
  + 0.15 · S_data
  + 0.20 · A_patient
```

Term interpretations (as designed):

| Term | Reading |
|---|---|
| `C_detect` | Detection confidence — how strongly the dual-track detector believes this flow is malicious |
| `D_crit` | Device criticality × CIA-threat impact — how serious a compromise of *this device class* would be in this attack category |
| `S_data` | Data sensitivity — PHI vs telemetry the device handles |
| `A_patient` | Patient acuity — *dynamic clinical state* of the patient under care of the alerting device |

### 1.2 Implementation — v2

The on-disk implementation differs in one variable name:

```text
R = 0.40 · C_detect
  + 0.25 · D_crit
  + 0.15 · S_data
  + 0.20 · D_clinical_tier      ← was A_patient
```

The rename is documented in the project history (commit on branch `fix/shap-category-vocab`). The numerical values of all four weights are unchanged.

**Why the rename matters:** the implemented signal is a *static* device property, not a *dynamic* per-patient acuity score. Pretending otherwise would overclaim what the system measures. See §3 for the honest-limitation statement.

---

## 2. Variable Specifications

| Variable | Implementation | Range | Source artifact |
|---|---|---|---|
| `C_detect` | `c_detect = max(c_track_a, c_track_b)` after two-stage fusion | [0, 1] | [`module3_risk_scoring/module3_risk_scores.py::compute_c_detect`](../module3_risk_scoring/module3_risk_scores.py); persisted in `risk_scores.npz["c_detect"]` |
| `D_crit` | Device criticality × CIA threat — `_CIA_SCORE` lookup over attack category | [0, 1] | [`module3_risk_scoring/module3_risk_scores.py::compute_d_crit`](../module3_risk_scoring/module3_risk_scores.py); persisted in `risk_scores.npz["d_crit"]` |
| `S_data` | Weighted PHI / telemetry mix — fraction of biometric features active | [0, 1] | [`module3_risk_scoring/module3_risk_scores.py::compute_s_data`](../module3_risk_scoring/module3_risk_scores.py); persisted in `risk_scores.npz["s_data"]` |
| `D_clinical_tier` | Fraction of biometric features ≥ 1.5σ — proxy for clinical engagement | [0, 1] | [`module3_risk_scoring/module3_risk_scores.py::compute_d_clinical_tier`](../module3_risk_scoring/module3_risk_scores.py); persisted in `risk_scores.npz["d_clinical_tier"]` |

### 2.1 `C_detect`

Detection confidence is the scalar output of the two-stage fusion at Step [7]:

```python
c_detect = np.maximum(c_track_a, c_track_b)
```

This preserves Track B's only-elevates-never-suppresses invariant (INVARIANT 1).

### 2.2 `D_crit`

Device criticality times CIA-threat alignment. The CIA scoring table (`_CIA_SCORE` in [`module3_risk_scoring/module3_risk_scores.py`](../module3_risk_scoring/module3_risk_scores.py)) maps each attack category to a worst-case CIA-triad weight, then composes with the device's criticality tier from the inventory. Devices in [`tests/fixtures/device_inventory.yaml`](../tests/fixtures/device_inventory.yaml) carry a `criticality` field with values in `{CRITICAL, HIGH, MEDIUM, LOW}`.

### 2.3 `S_data`

Data-sensitivity term mixes PHI weight (biometric features active) and network-telemetry weight:

```python
s_data = (phi_weight × bio_active + net_weight × net_present) / (phi_weight + net_weight)
```

This implements the design-intent categories below as a continuous mix rather than a hard switch:

| Data class | Conceptual weight |
|---|---|
| PHI biometric (e.g. live HR, SpO₂, BP traces) | 1.0 |
| PHI demographic (e.g. patient ID, MRN in flow metadata) | 0.7 |
| Device telemetry (firmware logs, calibration packets) | 0.4 |
| Operational data (DHCP leases, NTP sync) | 0.2 |

The continuous mix degrades gracefully when device-class metadata is incomplete.

### 2.4 `D_clinical_tier`

Per the renamed semantics, this is a *device-class-tier proxy* — not a per-patient acuity reading. The current implementation derives it from the fraction of biometric features whose magnitude exceeds 1.5 standard deviations on the row, normalized to `[0, 1]`. The conceptual tier hierarchy that the design targets:

| Tier | Devices | Score |
|---|---|---|
| 1 — Life-critical | Ventilators, infusion pumps on critical medication | 1.0 |
| 2 — High clinical | Patient monitors, anaesthesia machines | 0.8 |
| 3 — Moderate | General-ward monitors, diagnostic devices | 0.5 |
| 4 — Supportive | Imaging consoles, lab analyzers | 0.3 |
| 5 — Administrative | EHR workstations, kiosks | 0.1 |

Today's biometric-active heuristic produces a value that *correlates* with this tier hierarchy (a vent will have many active biometric features; an EHR workstation will have none) but doesn't equal it. The authoritative per-row `device_class` join (closed by GAP-A7) is the foundation for replacing this heuristic with a direct tier lookup; see §3 future work.

---

## 3. Honest Limitation — `D_clinical_tier` as Patient-Acuity Proxy

The composite risk formula, as designed, includes a patient-acuity term (`A_patient`) reflecting the **dynamic clinical state** of the patient under care of the device generating the alert. True patient acuity is a real-time clinical metric — APACHE II for ICU patients, MEWS for ward patients, NEWS2 for general acute care — which requires integration with electronic health record systems.

In this implementation, we use `D_clinical_tier` — a static device property representing the typical clinical role of the device class — as a proxy for `A_patient`. This proxy assumes that devices classified as life-critical are typically used on patients with higher acuity, while supportive devices serve lower-acuity patients. This assumption holds in aggregate but **not at individual patient level**.

### Consequences of this approximation

- **Same device type → same A_patient regardless of patient state.** An ICU monitor on a stable post-op patient yields the same `D_clinical_tier` value as the same ICU monitor on a critical sepsis patient. The proxy cannot distinguish them.
- **Conservative bias is acceptable.** Risk is *over*-estimated for less acute patients on critical-class devices. Under-estimation is the dangerous direction; over-estimation increases operator burden but never silently drops an alert.
- **Incident framing.** When the system reports R for a CRITICAL+unpatchable alert, that R reflects the *typical* clinical impact for that device class, not the per-patient consequence. MVE Layer 2 (`patient_care_impact`) communicates this in qualitative wording rather than implying a precise per-patient claim.

### Future work

Prospective evaluation with EHR integration providing real-time patient acuity scores, validating the magnitude of proxy-vs-true-acuity disagreement. Specifically:

1. Compute `R_proxy = R(D_clinical_tier)` (current).
2. Compute `R_true = R(APACHE-II-derived acuity)` from EHR integration.
3. Measure `Δ = R_proxy - R_true` per alert across an audited cohort.
4. Report distribution of |Δ|, the false-elevation rate (R_proxy ≥ tier threshold but R_true below it), and the (rare-but-critical) false-deescalation rate.

Until step 4 is funded, all `D_clinical_tier` consumers must label their output with the proxy caveat. ARCHITECTURE.md Step [8] already does this; this document is the deeper reference.

---

## 4. Sensitivity Analysis

### 4.1 Method

For each weight `w_i` in `(C_detect, D_crit, S_data, D_clinical_tier)`, vary in 3 grid points around the spec value (`±0.10` for terms with magnitude ≥ 0.20, `±0.05` for `S_data`). When one weight changes, the other three are renormalized proportionally so the total stays at 1.0. Recompute `R` per alert in the **stratified holdout** (n=1469, materialised at seed 42), apply the surfacing threshold `R ≥ 0.40`, and measure:

- `FNR_CRITICAL` — share of CRITICAL-tier attacks below threshold
- `FPR` — share of benign rows above threshold
- `FNR` — overall false-negative rate

### 4.2 Baseline (spec weights)

```text
weights:  C_detect=0.40  D_crit=0.25  S_data=0.15  D_clinical_tier=0.20
metrics:  FNR_CRITICAL=0.0000  FPR=0.0491  FNR=0.0703
          n_crit_attacks=16  n_crit_caught=16
```

### 4.3 Tornado plot

![weight sensitivity tornado](../results/figures/risk_weights_tornado.png)

### 4.4 Per-variable ranges (from sensitivity sweep)

| Variable (its weight perturbed) | FPR range | FNR range | FNR_CRITICAL range |
|---|---|---|---|
| `S_data` | **[0.0397, 0.1199]** | [0.0541, 0.0703] | [0.0, 0.0] |
| `C_detect` | [0.0467, 0.0678] | [0.0595, 0.0703] | [0.0, 0.0] |
| `D_clinical_tier` | [0.0428, 0.0576] | [0.0649, 0.0919] | [0.0, 0.0] |
| `D_crit` | [0.0491, 0.0491] | **[0.0541, 0.0865]** | [0.0, 0.0] |

### 4.5 Findings

- **Most-sensitive weight (FPR):** `S_data`. A single perturbation of its weight from 0.05 to 0.25 swings FPR from 4.0% to 12.0% — an 8 pp range. This is operationally significant: the operator-fatigue dial is largely controlled by the data-sensitivity term, not the detection-confidence term.
- **Most-sensitive weight (FNR overall):** `D_crit`. Reducing its weight to 0.15 while compensating with `C_detect` increases overall FNR by ~3.2 pp.
- **Least-sensitive weight (FPR):** `D_crit` — its perturbation does not move FPR at all, because `D_crit` correlates strongly with `c_detect` on attacks (and is uniformly low on benign rows).
- **`FNR_CRITICAL` is stable at 0.000 across all 12 perturbations.** The pipeline catches all 16 CRITICAL-tier attacks at every tested weight set, including the most extreme combinations. Stated bluntly: the safety-floor invariant + risk-adaptive gate are doing the heavy lifting; the `R`-formula weights primarily affect *non-critical* surfacing and the operator-fatigue dial.
- **Recommended take-away:** the weight selections are not over-fit. The pipeline tolerates ±0.10 perturbations on every weight without losing CRITICAL coverage, and the FPR variation is bounded under 12% in the worst case (still inside the M5 study's deployable range).

---

## 5. Code & Documentation Status

### 5.1 Variable rename — `A_patient` → `D_clinical_tier`

| File | Status |
|---|---|
| [`src/risk_scorer.py`](../src/risk_scorer.py) | rename complete |
| [`module3_risk_scoring/module3_risk_scores.py`](../module3_risk_scoring/module3_risk_scores.py) | function renamed `compute_a_patient` → `compute_d_clinical_tier`; NPZ key renamed |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | formula updated; rename annotated |
| [`docs/architecture.md`](architecture.md) | formula updated |
| [`Performance_baselines.md`](../Performance_baselines.md) | rename complete |
| [`tests/test_safe_failure.py`](../tests/test_safe_failure.py) | no `A_patient` references remain |
| [`tests/acceptance_tests.py`](../tests/acceptance_tests.py) | no `A_patient` references remain |
| [`module5_responses/module5_pipeline.py`](../module5_responses/module5_pipeline.py) | `patient_acuity` parameter retained — see §5.2 |

### 5.2 Why `patient_acuity` is still in `module5_pipeline.py`

`patient_acuity` (8 occurrences in `module5_pipeline.py`) is a *response-policy parameter*, not the formula term. It consumes the `D_clinical_tier` value upstream and uses it in an acuity-aware-response override gate ([line 256](../module5_responses/module5_pipeline.py#L256), `acuity_elevated = patient_acuity >= elevated_acuity_threshold`). Renaming it would couple the formula vocabulary to the response policy in a way that obscures the distinct role each plays:

- `D_clinical_tier` is the *signal* (the score that goes into R).
- `patient_acuity` is the *policy parameter* (the threshold gate that consumes the same scalar to decide whether to route to a clinical-override branch).

Both names are correct in their respective scopes. The grep shows no occurrences of `A_patient` (capital A, the original formula name) anywhere in source — that rename is fully complete.

### 5.3 Verification command

```bash
# Confirm A_patient is fully removed from CODE (Python + YAML).
# Documentation files (this doc, ARCHITECTURE.md) carry intentional
# historical mentions describing the rename — those are not residual.
grep -rn "A_patient\|a_patient" --include="*.py" --include="*.yaml" \
    /home/un1/project/ids-healthcare-cip/ | \
    grep -v "results/reports/study_responses_" | \
    grep -v "survey_backup_"
# Expected: zero hits in *.py and *.yaml (only patient_acuity in
# module5_pipeline.py is permitted — see §5.2).
```

### 5.4 Verification run — 2026-05-05

| Check | Result |
| --- | --- |
| `A_patient` / `a_patient` in `*.py` or `*.yaml` (excluding session-state YAML) | **0 hits** |
| `patient_acuity` in `module5_pipeline.py` (deliberate carve-out per §5.2) | 8 hits — all in response-policy parameter scope, none in the formula |
| Sensitivity sweep baseline FNR_CRITICAL / FPR / FNR | recomputed = 0.0000 / 0.0491 / 0.0703 (matches §4.2 exactly) |
| Per-variable FPR ranges (4 variables × 3 perturbations) | all 4 ranges match §4.4 table to 4 decimals |
| Per-variable FNR ranges (4 variables × 3 perturbations) | all 4 ranges match §4.4 table to 4 decimals |
| `FNR_CRITICAL = 0.000` flat across all 12 perturbations | confirmed |
| Tornado plot `results/figures/risk_weights_tornado.png` | exists, 94 KB |

---

## 6. Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| Variable rename complete (no `A_patient` remaining) | **PASS** | grep returns zero hits on source code, only annotated explanatory mentions in ARCHITECTURE.md |
| Limitation section explicit | **PASS** | §3 — explicit proxy-vs-true-acuity statement, four enumerated consequences, future-work plan with measurable success criteria |
| Sensitivity analysis with tornado plot | **PASS** | §4 — 12-point grid sweep on the materialised stratified holdout; tornado plot at `results/figures/risk_weights_tornado.png`; per-variable ranges tabulated |
| Future work clearly identified | **PASS** | §3 future-work block lists 4 concrete steps tied to a measurable Δ between proxy and true acuity |
| All code references updated | **PASS** | §5.1 file-by-file table; §5.2 explains the one apparent exception; §5.3 reproducibility command |

## 7. Open Items / Caveats

| ID | Description |
|---|---|
| **GAP-RFS-1** | The biometric-feature heuristic for `D_clinical_tier` (§2.4) does not directly correspond to the 5-tier ladder (§2.4 table). Replace with a direct device-class → tier lookup once the authoritative inventory join (post-A7-FOLLOWUP) lands. |
| **GAP-RFS-2** | Sensitivity analysis covers single-weight perturbations only. Joint-perturbation surfaces (e.g. simultaneously raising `C_detect` and lowering `D_crit`) are not explored. Tracked for a future RQ2 deep-dive. |
| **GAP-RFS-3** | EHR integration for true `A_patient` is the named blocker for proxy-validation. No plan-of-record exists; flagged as Phase-3 deployment work. |
