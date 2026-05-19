# RQ1 Pipeline — Implementation Spec for Claude Code

**Project:** XAI-IDS-Healthcare
**Research Question:** *"Can risk-adaptive scoring + dual-track detection provide effective threat detection in IoMT environments while preserving clinical safety constraints?"*
**Purpose of this document:** Single, self-contained spec for implementing the full RQ1 evaluation pipeline. Hand directly to Claude Code.
**Status of decisions:** All design questions through Phase 5 are locked. Phase 4 sub-stage 5B (weight sensitivity) and Phase 4 sub-stage 5D (MedSec sibling metrics) have remaining design questions flagged inline — implement the rest first.

---

## 0. How to use this spec

1. Implement in the order given (Phases 1 → 6).
2. After each phase, run the verification command shown. Do not proceed if it fails.
3. Each new file has a "Contract" section: inputs, outputs, invariants. Do not deviate from these without surfacing the question to the developer.
4. Where the spec says **DO NOT GUESS**, stop and ask. These are points where wrong guesses corrupt downstream metrics silently.
5. Comments in code samples are normative — keep them in the output.

---

## 1. Background: what RQ1 needs

The pipeline produces seven artifacts that together support every claim in the paper's Chapter 5.2 (RQ1 Results) and the defense talking points:

| Artifact | Path | Stage |
|---|---|---|
| Headline metrics JSON | `results/rq1_metrics.json` | 5A + 5E |
| Weight sensitivity JSON | `results/rq1_weight_sensitivity.json` | 5B |
| Cascade ablation JSON | `results/rq1_cascade_ablation.json` | 5C |
| MedSec sibling JSON | `results/rq1_metrics_medsec.json` | 5D |
| Risk scores NPZ | `results/reports/risk_scores.npz` | Stage 4 (extended) |
| Figures (4 PDFs) | `results/figures/*.pdf` | 6 |
| Tier × surfacing truth table | `results/rq1_tier_surfacing_truth_table.{csv,md}` | 7 |

The paper's tables and figures are thin presentation over these files. The CI suite asserts the artifacts meet RQ1 targets (FNR_critical < 5%, AUC ≥ 0.99, invariant checks pass).

---

## 2. Locked design decisions (do not revisit)

These were resolved through prior design conversation. They are constraints, not options.

| Decision | Resolution |
|---|---|
| FNR_critical definition | Union of three criteria: ground-truth severity = CRITICAL OR R_counterfactual ≥ 0.80 OR device_criticality = CRITICAL |
| Counterfactual R computation | `R_cf = 0.40·1.0 + 0.25·d_crit + 0.15·s_data + 0.20·d_clinical_tier` (hypothetical full detection) |
| Track A ablation scope | XGBoost, RF, DT only. NO max-fusion row. (Senior review: max of correlated models is an anti-pattern; documented as rejected.) |
| `risk_scores.npz` schema | Extended to include row_id, attack_category, device_class, device_criticality, patchable, true_severity, R_counterfactual (Path A) |
| Headline `y_pred` | `y_pred = (fusion_class != "BENIGN")` — matches what the system actually surfaces |
| Figures | Separate script, NOT folded into compute_rq1_metrics.py |
| MedSec coverage | Full reproducibility — sibling metrics script + cascade ablation script |
| Cross-cutting additions | Tier boundary histogram (in figures) + D_crit/D_clinical_tier correlation (in JSON) |

---

## 3. Phase 1 — Refactor (safe, no behavior change)

### 3.1 Create `common/context_mappings.py`

**Why:** `DEVICE_CONTEXT` and `map_true_severity` currently live in `module6_evaluation/module6_evaluation.py` (lines 46–103 and 327–339). Module 3 needs them too, and we want one source of truth.

**File contents — write exactly this:**

```python
# common/context_mappings.py
"""
Shared device context and severity-derivation mappings.

Used by Module 3 (batch risk scoring) and Module 6 (evaluation).
This is the single source of truth — do not redefine these elsewhere.
"""

from typing import Dict, Any

# Device class → (criticality, patchable, d_crit numeric weight)
# Matches the original DEVICE_CONTEXT in module6_evaluation.py.
DEVICE_CONTEXT: Dict[str, Dict[str, Any]] = {
    "infusion_pump":   {"device_criticality": "CRITICAL", "patchable": False, "d_crit": 0.80},
    "ventilator":      {"device_criticality": "CRITICAL", "patchable": False, "d_crit": 0.80},
    "patient_monitor": {"device_criticality": "HIGH",     "patchable": False, "d_crit": 0.72},
    "ehr_workstation": {"device_criticality": "HIGH",     "patchable": True,  "d_crit": 0.72},
    "pacs_server":     {"device_criticality": "HIGH",     "patchable": False, "d_crit": 0.72},
    "insulin_pump":    {"device_criticality": "HIGH",     "patchable": False, "d_crit": 0.72},
    "pharmacy_system": {"device_criticality": "HIGH",     "patchable": True,  "d_crit": 0.72},
    "server":          {"device_criticality": "MEDIUM",   "patchable": True,  "d_crit": 0.40},
    "other":           {"device_criticality": "MEDIUM",   "patchable": True,  "d_crit": 0.40},
}

# Fallback for any unmapped device_class — DO NOT silently change this.
UNKNOWN_DEVICE_FALLBACK = DEVICE_CONTEXT["other"]


def lookup_device_context(device_class: str) -> Dict[str, Any]:
    """Safe lookup with explicit fallback to 'other'."""
    return DEVICE_CONTEXT.get(device_class, UNKNOWN_DEVICE_FALLBACK)


def map_true_severity(attack_category: str, device_class: str) -> str:
    """
    Derive ground-truth severity from raw labels.

    Rule (matches module6_evaluation.py:327-339):
      - "normal"                                              → "LOW"
      - {Data Alteration, Spoofing} on life-critical devices  → "CRITICAL"
      - other attacks on life-critical devices                → "HIGH"
      - "Data Alteration" on other devices                    → "HIGH"
      - "Spoofing" on other devices                           → "MEDIUM"
      - everything else                                       → "MEDIUM"

    Life-critical devices are: ventilator, patient_monitor, infusion_pump.
    """
    if attack_category == "normal":
        return "LOW"
    life_critical = {"ventilator", "patient_monitor", "infusion_pump"}
    if device_class in life_critical:
        if attack_category in {"Data Alteration", "Spoofing"}:
            return "CRITICAL"
        return "HIGH"
    if attack_category == "Data Alteration":
        return "HIGH"
    if attack_category == "Spoofing":
        return "MEDIUM"
    return "MEDIUM"
```

### 3.2 Update `module6_evaluation/module6_evaluation.py`

Replace the inline `DEVICE_CONTEXT` dict (lines ~46–103) and `map_true_severity` function (lines ~327–339) with:

```python
from common.context_mappings import DEVICE_CONTEXT, map_true_severity, lookup_device_context
```

Everywhere these symbols are used in the module, the imports above resolve them. Do not duplicate the definitions.

### 3.3 Verification

```bash
pytest tests/
```

**Expected:** all existing tests pass. This refactor changes no behavior; if any test fails, the refactor is wrong, not the test.

---

## 4. Phase 2 — Extend Module 3

### 4.1 Modify `module3_risk_scoring/module3_risk_scores.py`

**Goal:** persist 7 additional arrays plus a schema version to `risk_scores.npz`.

#### 4.1.1 New arrays to add (each length = test split size, currently 2,448)

| Array name | Dtype | Source / formula |
|---|---|---|
| `row_id` | `int64` | Range over parquet rows: `np.arange(len(test_df))`. Persisted so downstream scripts can audit alignment. |
| `attack_category` | `<U20` | Direct passthrough from `test_phase1.parquet["attack_category"]` (or whatever column name is used). |
| `device_class` | `<U20` | Direct passthrough from `test_phase1.parquet["device_class"]`. |
| `device_criticality` | `<U10` | `lookup_device_context(device_class)["device_criticality"]` per row. |
| `patchable` | `bool` | `lookup_device_context(device_class)["patchable"]` per row. |
| `true_severity` | `<U10` | `map_true_severity(attack_category, device_class)` per row. |
| `R_counterfactual` | `float64` | `0.40*1.0 + 0.25*d_crit + 0.15*s_data + 0.20*d_clinical_tier` per row. |

#### 4.1.2 Schema version

Add a sidecar file `results/reports/risk_scores.meta.json` containing:

```json
{
  "schema_version": "1.1",
  "generated_at": "<ISO-8601 timestamp>",
  "dataset": "WUSTL-EHMS-2020",
  "split": "test",
  "n_rows": 2448,
  "arrays": [
    "row_id", "y_true", "attack_category", "device_class",
    "device_criticality", "patchable", "true_severity",
    "d_crit", "d_clinical_tier", "s_data",
    "c_track_a", "c_track_b", "c_detect",
    "R", "R_counterfactual", "risk_levels", "fusion_class", "data_quality"
  ]
}
```

(Sidecar JSON is cleaner than stuffing a string array into the npz; downstream code reads this file to validate.)

#### 4.1.3 Implementation notes

- Import from `common.context_mappings` — do **not** duplicate the dict.
- `R_counterfactual` invariant: must satisfy `R_counterfactual >= R` for every row (since c_detect=1.0 is the maximum). Add a runtime assertion right before saving.
- `row_id` is `np.arange(len(test_df))` — this is *parquet row order*. If Module 3 ever shuffles or filters rows, persist the *original* parquet index, not the post-shuffle position. **DO NOT GUESS** if Module 3 currently filters rows; check the code and surface the question.

### 4.2 Modify `tests/test_step9_composite_risk.py`

Add this test:

```python
def test_risk_scores_npz_schema_v1_1():
    """Asserts the extended npz schema (Phase 2 of RQ1 pipeline)."""
    import json
    import numpy as np
    from pathlib import Path

    npz_path = Path("results/reports/risk_scores.npz")
    meta_path = Path("results/reports/risk_scores.meta.json")
    assert npz_path.exists(), "Run Module 3 to generate npz first"
    assert meta_path.exists(), "Sidecar meta file missing"

    meta = json.loads(meta_path.read_text())
    assert meta["schema_version"] == "1.1", \
        f"Expected schema v1.1, got {meta['schema_version']}"

    data = np.load(npz_path, allow_pickle=False)
    required = {
        "row_id", "attack_category", "device_class",
        "device_criticality", "patchable", "true_severity",
        "R_counterfactual",
    }
    missing = required - set(data.files)
    assert not missing, f"Missing required arrays: {missing}"

    # All arrays same length
    n = len(data["y_true"])
    for name in required:
        assert len(data[name]) == n, f"{name} length {len(data[name])} != {n}"

    # Invariant: R_counterfactual >= R (counterfactual is upper bound)
    assert np.all(data["R_counterfactual"] >= data["R"] - 1e-9), \
        "R_counterfactual must be >= R for every row"

    # Invariant: row_id is the identity range (post-Phase-2 assumption)
    assert np.array_equal(data["row_id"], np.arange(n)), \
        "row_id must be identity range over test parquet rows"
```

### 4.3 Verification

```bash
pytest tests/test_step9_composite_risk.py::test_risk_scores_npz_schema_v1_1
# Expected: FAIL (npz not regenerated yet)

python -m module3_risk_scoring.module3_risk_scores
# Should complete in seconds; regenerates risk_scores.npz

pytest tests/test_step9_composite_risk.py::test_risk_scores_npz_schema_v1_1
# Expected: PASS
```

---

## 5. Phase 3 — Headline metrics aggregator

### 5.1 Create `module6_evaluation/compute_rq1_metrics.py`

**Contract:**
- **Input:** `results/reports/risk_scores.npz` only. Pure aggregator. No model loading. No retraining.
- **Output:** `results/rq1_metrics.json`. Idempotent — running twice produces identical output (modulo `_meta.generated_at`).
- **Runtime:** seconds.
- **Side effects:** writes one file. Does not modify the npz.

#### 5.1.1 Complete script

```python
"""
compute_rq1_metrics.py
Aggregates RQ1 headline metrics, ablations, surfacing summaries,
and correlation diagnostics into results/rq1_metrics.json.

Pure aggregator — reads one .npz, writes one .json. No model loading.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = REPO_ROOT / "results/reports/risk_scores.npz"
META_PATH = REPO_ROOT / "results/reports/risk_scores.meta.json"
OUT_PATH = REPO_ROOT / "results/rq1_metrics.json"

FNR_CRITICAL_TARGET = 0.05
SENSITIVITY_TARGET = 0.90
SPECIFICITY_TARGET = 0.95
AUC_A_TARGET = 0.99
SCHEMA_VERSION = "1.0"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _assert_npz_schema():
    """Fail loudly if the npz is the old (pre-Phase 2) schema."""
    meta = json.loads(META_PATH.read_text())
    if meta.get("schema_version") != "1.1":
        raise RuntimeError(
            f"risk_scores.npz schema is {meta.get('schema_version')}, "
            f"expected 1.1. Re-run Module 3 (Phase 2)."
        )


def build_meta(data) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "module6_evaluation/compute_rq1_metrics.py",
        "dataset": "WUSTL-EHMS-2020",
        "split": "test",
        "inputs": {
            "risk_scores_npz": str(NPZ_PATH.relative_to(REPO_ROOT)),
            "risk_scores_sha256": _sha256_file(NPZ_PATH),
            "n_samples": int(len(data["y_true"])),
            "n_benign": int((data["y_true"] == 0).sum()),
            "n_malicious": int((data["y_true"] == 1).sum()),
        },
        "config": {
            "y_pred_definition": "fusion_class != 'BENIGN'",
            "risk_weights": {
                "c_detect": 0.40,
                "d_crit": 0.25,
                "s_data": 0.15,
                "d_clinical_tier": 0.20,
            },
            "tier_boundaries": {
                "critical": 0.80,
                "high": 0.60,
                "medium": 0.40,
            },
            "fnr_critical_definition": (
                "union(true_severity=='CRITICAL', "
                "R_counterfactual>=0.80, "
                "device_criticality=='CRITICAL')"
            ),
            "targets": {
                "fnr_critical": FNR_CRITICAL_TARGET,
                "sensitivity": SENSITIVITY_TARGET,
                "specificity": SPECIFICITY_TARGET,
                "auc_track_a": AUC_A_TARGET,
            },
        },
    }


def compute_critical_union(data) -> dict:
    """A sample is critical if ANY of three criteria hold."""
    c1 = data["true_severity"] == "CRITICAL"
    c2 = data["R_counterfactual"] >= 0.80
    c3 = data["device_criticality"] == "CRITICAL"
    union = c1 | c2 | c3
    return {
        "mask": union,
        "by_gt_severity": int(c1.sum()),
        "by_counterfactual_tier": int(c2.sum()),
        "by_device_criticality": int(c3.sum()),
        "overlap_all_three": int((c1 & c2 & c3).sum()),
        "union_total": int(union.sum()),
    }


def compute_headline(data) -> dict:
    y_true = data["y_true"]
    y_pred = (data["fusion_class"] != "BENIGN").astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)

    auc_a = roc_auc_score(y_true, data["c_track_a"])
    auc_b = roc_auc_score(y_true, data["c_track_b"])
    auc_fused = roc_auc_score(y_true, data["c_detect"])

    # FNR_critical
    crit = compute_critical_union(data)
    union_mask = crit["mask"]
    fn_mask = (y_true == 1) & (y_pred == 0)
    n_crit_total = int(union_mask.sum())
    n_crit_missed = int((union_mask & fn_mask).sum())
    fnr_crit = n_crit_missed / max(n_crit_total, 1)

    return {
        "fnr_critical": fnr_crit,
        "fnr_critical_target": FNR_CRITICAL_TARGET,
        "fnr_critical_pass": fnr_crit < FNR_CRITICAL_TARGET,
        "fnr_critical_n_total": n_crit_total,
        "fnr_critical_n_missed": n_crit_missed,
        "fnr_critical_breakdown": {
            k: v for k, v in crit.items() if k != "mask"
        },
        "sensitivity": float(sens),
        "sensitivity_pass": sens > SENSITIVITY_TARGET,
        "specificity": float(spec),
        "specificity_pass": spec > SPECIFICITY_TARGET,
        "f2_score": float(f2),
        "auc_track_a": float(auc_a),
        "auc_track_a_pass": auc_a > AUC_A_TARGET,
        "auc_track_b": float(auc_b),
        "auc_fused": float(auc_fused),
        "confusion_matrix": {
            "tp": int(tp), "fn": int(fn),
            "fp": int(fp), "tn": int(tn),
        },
    }


def compute_track_a_ablation(data) -> dict:
    """
    Per-model headline metrics for XGBoost, RF, DT.
    NO max-fusion row (decision locked: omit, do not report).

    Requires per-model probabilities. If only c_track_a (the max) is in the npz,
    this function can only fill in xgboost from c_track_a (since XGBoost dominates
    per F2-optimal thresholds — verify before relying on this).

    DO NOT GUESS: If per-model probas (P_xgb, P_rf, P_dt) are not in the npz,
    surface this and add them in Phase 2.5 before completing this function.
    """
    # Expected fields if Phase 2.5 adds them:
    #   data["P_xgb"], data["P_rf"], data["P_dt"]
    # And per-model F2-optimal thresholds from results/models/_thresholds.json
    # If those are not available, return a placeholder block:
    return {
        "_status": "pending — requires per-model probabilities in npz",
        "_note": (
            "Phase 2.5 may need to extend Module 3 to persist P_xgb, P_rf, P_dt "
            "separately. Until then, only c_track_a (= max of the three) is available, "
            "which is insufficient for per-model ablation."
        ),
        "selected_for_production": "xgboost",
        "selection_rationale": (
            "XGBoost selected per comparative evaluation (AUC 0.9941 on EHMS test). "
            "Max-fusion of XGB/RF/DT is omitted from this table — see senior engineer "
            "review: max of correlated models inflates FPR without FNR benefit."
        ),
    }


def compute_track_b_per_class(data) -> dict:
    """
    Per-attack-category AUC for Track B (DAE confidence only).
    For each non-benign class: positive set = samples of that class,
    negative set = all benign samples.
    """
    y_true = data["y_true"]
    c_track_b = data["c_track_b"]
    attack_cat = data["attack_category"]
    benign_mask = (y_true == 0)

    result = {}
    for cat in np.unique(attack_cat):
        cat_str = str(cat)
        if cat_str in ("normal", ""):
            continue
        pos_mask = (attack_cat == cat)
        if pos_mask.sum() < 5:
            result[cat_str] = {
                "auc": None,
                "n_positive": int(pos_mask.sum()),
                "n_negative": int(benign_mask.sum()),
                "verdict": "insufficient_data",
            }
            continue

        eval_mask = pos_mask | benign_mask
        try:
            auc = float(roc_auc_score(y_true[eval_mask], c_track_b[eval_mask]))
        except ValueError:
            auc = None

        result[cat_str] = {
            "auc": auc,
            "n_positive": int(pos_mask.sum()),
            "n_negative": int(benign_mask.sum()),
            "verdict": _classify_auc(auc),
        }
    return result


def _classify_auc(auc):
    if auc is None:
        return "insufficient_data"
    if auc >= 0.90:
        return "good_to_excellent"
    if auc >= 0.75:
        return "acceptable"
    if auc >= 0.60:
        return "weak"
    return "fails — benign-mimicking"


def compute_fusion_class_summary(data) -> dict:
    fc = data["fusion_class"]
    y_true = data["y_true"]
    out = {}
    for cls in ["KNOWN_ATTACK", "CONFIRMED_ANOMALY", "NOVEL_ANOMALY", "BENIGN"]:
        mask = (fc == cls)
        n = int(mask.sum())
        if n == 0:
            out[cls.lower()] = {"count": 0, "precision_within": None,
                                "recall_of_attacks": None}
            continue
        # Among samples in this class, what fraction are true attacks?
        precision_within = float(y_true[mask].mean()) if cls != "BENIGN" \
            else float((y_true[mask] == 0).mean())
        # Of all attacks, what fraction landed in this class?
        if cls == "BENIGN":
            recall_of_attacks = None
        else:
            recall_of_attacks = float((mask & (y_true == 1)).sum() /
                                       max((y_true == 1).sum(), 1))
        out[cls.lower()] = {
            "count": n,
            "precision_within": precision_within,
            "recall_of_attacks": recall_of_attacks,
        }
    return out


def compute_tier_distribution(data) -> dict:
    tiers = data["risk_levels"]
    n = len(tiers)
    out = {}
    for t in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = int((tiers == t).sum())
        out[t.lower()] = {"count": count, "fraction": count / n}
    return out


def compute_surfacing_summary(data) -> dict:
    """Includes invariant checks (1 and 2) from ARCHITECTURE.md."""
    tiers = data["risk_levels"]
    crit_unpatch = (data["device_criticality"] == "CRITICAL") & \
                   (~data["patchable"].astype(bool))
    is_critical_tier = (tiers == "CRITICAL")

    # Invariant 1: c_detect = max(c_track_a, c_track_b)
    inv1_violations = int(np.sum(data["c_detect"] + 1e-9 < data["c_track_a"]))

    # Invariant 2 (proxy): every CRITICAL+unpatchable should be in CRITICAL tier
    # (full safety floor is enforced in src/risk_scorer.py at surfacing time,
    # but tier assignment is a necessary precondition.)
    inv2_violations = int(np.sum(crit_unpatch & ~is_critical_tier))

    return {
        "total_alerts": int(len(tiers)),
        "tier_counts": {
            t.lower(): int((tiers == t).sum())
            for t in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        },
        "critical_unpatchable_device_count": int(crit_unpatch.sum()),
        "_invariant_check": {
            "invariant_1_dae_only_elevates": {
                "violations": inv1_violations,
                "pass": inv1_violations == 0,
                "description": "c_detect = max(c_track_a, c_track_b); "
                               "DAE cannot suppress Track A.",
            },
            "invariant_2_safety_floor_tier_proxy": {
                "violations": inv2_violations,
                "pass": inv2_violations == 0,
                "description": (
                    "Every CRITICAL+unpatchable device should be tiered CRITICAL. "
                    "This is a proxy for the full safety floor in src/risk_scorer.py."
                ),
            },
        },
    }


def compute_correlation_diagnostics(data) -> dict:
    """L3 evidence: are D_crit and D_clinical_tier double-counting?"""
    d_crit = data["d_crit"].astype(float)
    d_ct = data["d_clinical_tier"].astype(float)
    pr, pp = pearsonr(d_crit, d_ct)
    sr, sp = spearmanr(d_crit, d_ct)
    abs_r = abs(pr)
    if abs_r >= 0.7:
        interp = "high — possible double-counting (L3 concern)"
    elif abs_r >= 0.4:
        interp = "moderate — partial overlap"
    else:
        interp = "low — features capture distinct signals"
    return {
        "d_crit_vs_d_clinical_tier": {
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_r": float(sr),
            "spearman_p": float(sp),
            "n": int(len(d_crit)),
            "interpretation": interp,
        }
    }


def documented_failure_modes(track_b_per_class) -> list:
    """Surface known failure modes as structured data, not free text."""
    failures = []
    # Track B Spoofing failure (3.1 in expected outputs)
    spoofing = track_b_per_class.get("Spoofing")
    if spoofing and spoofing.get("auc") is not None:
        failures.append({
            "id": "FM-TB-01",
            "title": "Track B fails on benign-mimicking attacks",
            "evidence": {
                "attack_class": "Spoofing",
                "auc": spoofing["auc"],
                "n_samples": spoofing["n_positive"],
            },
            "mitigation": (
                "Track A supervised classification detects these via signature. "
                "max() fusion ensures Track A signal is preserved (Invariant 1)."
            ),
            "paper_section_ref": "Section 11 (Limitations) + threat model",
        })
    return failures


def limitations_acknowledged() -> list:
    return [
        {
            "id": "L1",
            "title": "Linear weighted sum vs multiplicative semantics",
            "description": (
                "R uses linear additive combination of four signals. "
                "A multiplicative formulation would enforce that any one zero "
                "signal zeroes R. Discussed in Section 11."
            ),
        },
        {
            "id": "L2",
            "title": "D_clinical_tier is device-class proxy for patient acuity",
            "description": (
                "Same device on stable vs unstable patient gets same weight. "
                "Production deployment would integrate EHR acuity (NEWS2/MEWS)."
            ),
        },
        {
            "id": "L3",
            "title": "D_crit / D_clinical_tier potential double-counting",
            "description": (
                "Both signals derive from device class. Correlation diagnostics "
                "in this file quantify the overlap."
            ),
        },
        {
            "id": "L4",
            "title": "Tier boundaries calibrated to test split",
            "description": (
                "Thresholds 0.40 / 0.60 / 0.80 are policy choices. "
                "Tier boundary histogram figure shows the empirical distribution."
            ),
        },
    ]


def main():
    _assert_npz_schema()
    data = np.load(NPZ_PATH, allow_pickle=False)

    track_b_per_class = compute_track_b_per_class(data)

    out = {
        "_meta": build_meta(data),
        "headline": compute_headline(data),
        "track_a_ablation": compute_track_a_ablation(data),
        "track_b_per_class": track_b_per_class,
        "track_b_ablation": {
            "cascade": {
                "_status": "pending — filled by "
                           "analysis/compute_track_b_cascade_ablation.py",
                "_merged_at": None,
            }
        },
        "fusion_classes": compute_fusion_class_summary(data),
        "risk_tier_distribution": compute_tier_distribution(data),
        "surfacing_summary": compute_surfacing_summary(data),
        "correlation_diagnostics": compute_correlation_diagnostics(data),
        "weight_sensitivity": {
            "_status": "pending — filled by analysis/compute_weight_sensitivity.py",
            "_merged_at": None,
        },
        "documented_failure_modes": documented_failure_modes(track_b_per_class),
        "limitations_acknowledged": limitations_acknowledged(),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"FNR_critical: {out['headline']['fnr_critical']:.4f} "
          f"(target < {FNR_CRITICAL_TARGET})")
    print(f"AUC Track A: {out['headline']['auc_track_a']:.4f} "
          f"(target > {AUC_A_TARGET})")


if __name__ == "__main__":
    main()
```

### 5.2 Phase-2.5 question (DO NOT GUESS)

The `compute_track_a_ablation` function currently emits a `_status: pending` block. For full Track A ablation, the npz needs per-model probabilities `P_xgb`, `P_rf`, `P_dt`, not just `c_track_a` (their max).

**Stop and ask the developer:** "Are per-model probabilities already persisted in `risk_scores.npz`, or only the max as `c_track_a`? If only max, should we extend Module 3 (Phase 2.5) to persist all three?"

Do not implement Track A ablation until this is resolved.

### 5.3 Add CI assertion in `tests/acceptance_tests.py`

```python
def test_rq1_targets_met():
    """Asserts RQ1 headline metrics meet defense targets."""
    import json
    from pathlib import Path

    metrics_path = Path("results/rq1_metrics.json")
    assert metrics_path.exists(), \
        "Run module6_evaluation/compute_rq1_metrics.py first"

    m = json.loads(metrics_path.read_text())
    h = m["headline"]
    s = m["surfacing_summary"]

    assert h["fnr_critical_pass"], (
        f"FNR_critical = {h['fnr_critical']:.4f} >= "
        f"{h['fnr_critical_target']} (target)"
    )
    assert h["auc_track_a_pass"], \
        f"AUC Track A = {h['auc_track_a']:.4f} <= 0.99"
    assert h["sensitivity_pass"], \
        f"Sensitivity = {h['sensitivity']:.4f} <= 0.90"
    assert h["specificity_pass"], \
        f"Specificity = {h['specificity']:.4f} <= 0.95"

    inv = s["_invariant_check"]
    assert inv["invariant_1_dae_only_elevates"]["pass"], \
        "Invariant 1 violated: DAE suppressed Track A on some rows"
    assert inv["invariant_2_safety_floor_tier_proxy"]["pass"], \
        "Invariant 2 (proxy) violated: CRITICAL+unpatchable tiered below CRITICAL"
```

### 5.4 Verification

```bash
python -m module6_evaluation.compute_rq1_metrics
# Should print FNR_critical and AUC, write results/rq1_metrics.json

cat results/rq1_metrics.json | head -40
# Inspect output manually

pytest tests/acceptance_tests.py::test_rq1_targets_met
# Expected: PASS (or fail loudly if a target is missed)
```

---

## 5b. Phase 3.5 — Fusion threshold calibration (`a_high`)

**Resolves:** the `a_high` half of Stage 5B (§6.1). Joint sensitivity over `a_low` and `b` remains future work.

### 5b.1 Motivation

Before Phase 3.5, `a_high` was hard-coded to `P_XGB_HIGH_CONF = 0.85` in [src/data_models.py](src/data_models.py). On the EHMS test split this caused 32 Spoofing samples to be classified `BENIGN` despite XGBoost ranking them in the 0.4–0.71 confidence band — XGBoost couldn't fire `KNOWN_ATTACK` (needs ≥ 0.85) and the DAE couldn't corroborate `CONFIRMED_ANOMALY` (Spoofing is benign-mimicking, documented FM-TB-01). Result: sensitivity = 0.8958 < 0.90 target, while XGBoost alone at its F2-optimal threshold already achieves sens = 0.987.

### 5b.2 Procedure (deterministic, locked)

1. **Score the tuning split.** [module3_risk_scoring/score_split.py](module3_risk_scoring/score_split.py) runs the calibrated XGBoost + DAE on an arbitrary parquet. Verified bit-exact against the persisted test-set probas (`xgboost_test_proba_calibrated.npy`, `c_track_b` in `risk_scores.npz`).
2. **Sweep.** [analysis/calibrate_fusion_threshold.py](analysis/calibrate_fusion_threshold.py) scores [data/processed/val_phase1.parquet](data/processed/val_phase1.parquet) (2,448 rows; canonical Phase-1 held-out validation split; disjoint from train/test/demo per [split_metadata.yaml](data/processed/split_metadata.yaml)) and sweeps `a_high ∈ [0.30, 0.85]` at step 0.01 with `a_low = 0.40`, `b = 0.70` fixed.
3. **Selection rule.** Smallest `a_high` such that (a) `a_high > a_low + step` (preserves the four-class CONFIRM band), and (b) sensitivity > 0.90 AND specificity > 0.95 on val. Tiebreak: max F2. Fails loudly if no `a_high` satisfies all conditions — signals that single-knob calibration is insufficient and a joint `(a_low, b)` sweep is needed.
4. **Persistence.** Writes `results/models/_fusion_thresholds.json` with full sweep table + provenance (split sha256, git commit, selection rule, all 56 sweep rows).
5. **Runtime loading.** [`load_fusion_thresholds()`](module3_risk_scoring/module3_risk_scores.py) reads the JSON; `classify_fusion()` calls it for default values. Falls back to `(P_XGB_HIGH_CONF, 0.40, 0.70)` when the JSON is absent.
6. **Verification.** [analysis/verify_fusion_threshold_holdout.py](analysis/verify_fusion_threshold_holdout.py) reports val/test/stratified-holdout metrics side by side; CI-gates a >5pp val→test degradation in either sensitivity or specificity.

### 5b.3 Why `val_phase1.parquet` not `stratified_calibration.parquet`

`stratified_calibration.parquet` (and `stratified_holdout.parquet`) were produced by the deprecated [docs/_archive/build_stratified_eval_set.py](docs/_archive/build_stratified_eval_set.py) and exhibit a preprocessing-drift signal vs the current Phase-1 pipeline: Track A AUC = 0.949 on these splits vs 0.995 on current test. Using them for fusion-threshold selection would tune to a distribution the model no longer faces. `val_phase1.parquet` is in the live preprocessing pipeline, disjoint from test, and was previously used only for isotonic XGBoost calibration — a different decision than fusion-threshold selection. `stratified_holdout.parquet` is still scored by the verification script as an informational "tougher distribution" check (not a gate).

### 5b.4 Calibration result (current)

- Picked `a_high = 0.41` (smallest value above `a_low = 0.40 + step`).
- Tuning (val) metrics: sens = 0.9414, spec = 0.9566, F2 = 0.8975.
- Test metrics after applying picked threshold: sens = 0.9511, spec = 0.9542, F2 = 0.9023.
- val→test delta: Δsens = +0.97pp, Δspec = −0.24pp (well within ±5pp gate).
- Headline target gate (`tests/acceptance_tests.py::test_rq1_targets_met`): all four targets PASS.

### 5b.5 Tests

- [tests/test_fusion_threshold_loading.py](tests/test_fusion_threshold_loading.py) — JSON schema, loader contract, fallback behaviour, classify_fusion wiring, end-to-end gate on test split.
- [tests/test_two_stage_fusion.py](tests/test_two_stage_fusion.py) and [tests/test_safe_failure.py](tests/test_safe_failure.py) fusion tests updated to pin explicit thresholds — they validate the function contract independently of the calibrated runtime defaults.

---

## 6. Phase 4 — Supporting analyses (independent, parallel-safe)

### 6.1 Stage 5B — Weight sensitivity (SPEC PENDING, `a_high` half RESOLVED in §5b)

**Resolved:** the `a_high` half of this stage was completed in [Phase 3.5](#5b-phase-35--fusion-threshold-calibration-a_high) and persisted to `results/models/_fusion_thresholds.json`. Calibrated value: `a_high = 0.41` on val_phase1.

**Still pending:** joint sensitivity over `(a_low, b)` plus full risk-weight perturbation. Open questions for the developer before implementation:

- Perturbation protocol: one-at-a-time vs joint sampling vs Dirichlet?
- Number of perturbations per condition?
- Agreement metric: exact tier match, Cohen's κ, both?
- Multiplicative R: implement as a separate condition or skip?

**Until specified, the broader sensitivity analysis is deferred.** `compute_rq1_metrics.py` already emits a `pending` placeholder under `weight_sensitivity`; that's sufficient for the JSON to be valid.

When ready to implement, create `analysis/compute_weight_sensitivity.py` writing to `results/rq1_weight_sensitivity.json`. The merge script (Stage 5E) folds it in.

### 6.2 Stage 5C — Track B cascade ablation

**File:** `analysis/compute_track_b_cascade_ablation.py`

**Contract:**
- **Inputs:** trained DAE models (raw input and cascade input), test splits for EHMS and MedSec-25.
- **Output:** `results/rq1_cascade_ablation.json`.
- **Side effect:** none beyond writing the JSON.

**DO NOT GUESS:** before implementing, verify the following with the developer:
1. Are there TWO trained DAE models (one for 25-feature input, one for 28-feature cascade input), or only one?
2. If only one cascade DAE exists, is there a Module 2 task to also train a raw-input DAE for the ablation?
3. What is the path to MedSec test data?

**Skeleton (do not fill in until above is resolved):**

```python
"""
compute_track_b_cascade_ablation.py
Runs DAE inference in both configurations (raw-input vs cascade-input)
on both datasets (EHMS-2020 and MedSec-25). Computes AUC per config.

Produces results/rq1_cascade_ablation.json.
"""

# Expected output structure:
# {
#   "_meta": { ... },
#   "results": {
#     "ehms_2020": {
#       "dae_raw": {"auc": float, "n_samples": int},
#       "dae_cascade": {"auc": float, "n_samples": int},
#       "delta": float
#     },
#     "medsec_25": { same }
#   },
#   "verdict": "Cascade design rejected — regression on MedSec-25 ..."
# }
```

### 6.3 Stage 5D — MedSec sibling metrics

**File:** `module6_evaluation/compute_rq1_metrics_medsec.py`

**Contract:**
- Same schema as `compute_rq1_metrics.py`.
- Reads a sibling npz: `results/reports/risk_scores_medsec.npz`.
- Writes `results/rq1_metrics_medsec.json`.

**DO NOT GUESS:** before implementing, verify:
1. Does MedSec-25 data include `device_class` annotations? If not, the per-tier metrics cannot be computed for MedSec — scope this stage to **"Track B per-class AUC only"** in that case.
2. Are Module 1 (preprocessing) and Module 3 (scoring) already runnable on MedSec, or does the developer need to extend them?

If MedSec lacks device context, implement only `compute_track_b_per_class` and emit a partial JSON with `_meta.dataset = "MedSec-25"` and an explanatory `_status` field on the missing blocks.

### 6.4 Stage 5E — Merge script

**File:** `analysis/merge_rq1_metrics.py`

**Contract:**
- **Inputs:** `results/rq1_metrics.json` (Phase 3 output), `results/rq1_weight_sensitivity.json` (5B), `results/rq1_cascade_ablation.json` (5C).
- **Output:** updates `results/rq1_metrics.json` in place.
- **Idempotent:** safe to re-run.

```python
"""
merge_rq1_metrics.py
Folds outputs of compute_weight_sensitivity.py and compute_track_b_cascade_ablation.py
into rq1_metrics.json under their respective placeholder blocks.

Idempotent — re-running with the same inputs produces the same result.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RQ1 = REPO_ROOT / "results/rq1_metrics.json"
WS = REPO_ROOT / "results/rq1_weight_sensitivity.json"
CA = REPO_ROOT / "results/rq1_cascade_ablation.json"


def main():
    if not RQ1.exists():
        raise SystemExit(f"{RQ1} not found — run compute_rq1_metrics.py first")
    metrics = json.loads(RQ1.read_text())
    now = datetime.now(timezone.utc).isoformat()

    if WS.exists():
        ws = json.loads(WS.read_text())
        metrics["weight_sensitivity"] = ws
        metrics["weight_sensitivity"]["_merged_at"] = now
        print(f"Merged weight_sensitivity from {WS.name}")

    if CA.exists():
        ca = json.loads(CA.read_text())
        metrics.setdefault("track_b_ablation", {})["cascade"] = ca
        metrics["track_b_ablation"]["cascade"]["_merged_at"] = now
        print(f"Merged cascade ablation from {CA.name}")

    RQ1.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Updated {RQ1.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

---

## 7. Phase 5 — Figures

### 7.1 Create `module6_evaluation/make_rq1_figures.py`

**Contract:**
- **Inputs:** `results/reports/risk_scores.npz` and `results/rq1_metrics.json`.
- **Outputs:** four PDFs in `results/figures/`.

```python
"""
make_rq1_figures.py
Produces RQ1 figures: ROC, PR, confusion matrix, tier boundary histogram.

Reads results/reports/risk_scores.npz and results/rq1_metrics.json.
Writes results/figures/*.pdf.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    precision_recall_curve,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NPZ = REPO_ROOT / "results/reports/risk_scores.npz"
METRICS = REPO_ROOT / "results/rq1_metrics.json"
FIG_DIR = REPO_ROOT / "results/figures"


def make_roc(data, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, key in [("Track A (XGBoost)", "c_track_a"),
                      ("Track B (DAE)", "c_track_b"),
                      ("Fused (max)", "c_detect")]:
        fpr, tpr, _ = roc_curve(data["y_true"], data[key])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Track A, Track B, Fused")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def make_pr(data, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, key in [("Track A (XGBoost)", "c_track_a"),
                      ("Track B (DAE)", "c_track_b"),
                      ("Fused (max)", "c_detect")]:
        p, r, _ = precision_recall_curve(data["y_true"], data[key])
        ax.plot(r, p, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — Track A, Track B, Fused")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def make_confusion(metrics, out):
    cm_dict = metrics["headline"]["confusion_matrix"]
    cm = np.array([[cm_dict["tn"], cm_dict["fp"]],
                   [cm_dict["fn"], cm_dict["tp"]]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Benign", "Attack"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Test Split")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def make_tier_histogram(data, out):
    """Histogram of R values with tier boundary lines.
    Visual argument for calibration of 0.40 / 0.60 / 0.80 boundaries."""
    R = data["R"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(R, bins=50, alpha=0.7, edgecolor="black")
    for boundary, label, color in [(0.40, "MEDIUM", "gold"),
                                   (0.60, "HIGH", "orange"),
                                   (0.80, "CRITICAL", "red")]:
        ax.axvline(boundary, linestyle="--", color=color, linewidth=1.5,
                   label=f"{label} threshold ({boundary})")
    ax.set_xlabel("Composite risk R")
    ax.set_ylabel("Count")
    ax.set_title("Risk score distribution with tier boundaries (test split)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out, format="pdf")
    plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(NPZ, allow_pickle=False)
    metrics = json.loads(METRICS.read_text())

    make_roc(data, FIG_DIR / "roc_curves.pdf")
    make_pr(data, FIG_DIR / "pr_curves.pdf")
    make_confusion(metrics, FIG_DIR / "confusion_matrix.pdf")
    make_tier_histogram(data, FIG_DIR / "tier_boundary_histogram.pdf")
    print(f"Wrote 4 PDFs to {FIG_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

---

## 8. Phase 5 (cont.) — Truth table

### 8.1 Create `module6_evaluation/make_rq1_truth_table.py`

**Purpose:** enumerate every (tier × patchable × maintenance) cell and report the expected `should_surface` decision plus its reason. Shared output between RQ1 (Appendix B) and RQ3 (invariant evidence).

```python
"""
make_rq1_truth_table.py
Enumerates (risk_tier × patchable × maintenance_active) cells and reports
the expected should_surface decision. Calls src.risk_scorer.score_alert
with synthetic inputs.

Writes:
  - results/rq1_tier_surfacing_truth_table.csv
  - results/rq1_tier_surfacing_truth_table.md
"""

import csv
from itertools import product
from pathlib import Path

# DO NOT GUESS: the exact import path may differ. Confirm with developer:
# from src.risk_scorer import score_alert, AlertContext

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = REPO_ROOT / "results/rq1_tier_surfacing_truth_table.csv"
OUT_MD = REPO_ROOT / "results/rq1_tier_surfacing_truth_table.md"

TIERS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
PATCHABLE_OPTIONS = [True, False]
MAINTENANCE_OPTIONS = [True, False]


def synthesize_inputs(tier, patchable, maintenance_active):
    """Build a synthetic AlertContext matching the (tier, patchable, maint) cell.
    Pick an R value within the tier band; pick a device_class consistent with patchable."""
    # Tier → representative R value
    r_by_tier = {"CRITICAL": 0.85, "HIGH": 0.70, "MEDIUM": 0.50, "LOW": 0.20}
    # Patchable → representative device class
    device_class = "ehr_workstation" if patchable else "infusion_pump"
    return {
        "r": r_by_tier[tier],
        "risk_tier": tier,
        "device_class": device_class,
        "patchable": patchable,
        "maintenance_active": maintenance_active,
    }


def evaluate_cell(tier, patchable, maintenance):
    """Returns dict with should_surface decision and reason."""
    inputs = synthesize_inputs(tier, patchable, maintenance)
    # DO NOT GUESS the score_alert signature — confirm and adapt:
    # result = score_alert(...)
    # For now, encode the expected logic from ARCHITECTURE.md Step 10:
    if tier == "CRITICAL" and not patchable:
        return {"should_surface": True, "reason": "safety_floor"}
    if maintenance:
        return {"should_surface": False, "reason": "suppressed_maintenance"}
    if tier in ("CRITICAL", "HIGH"):
        return {"should_surface": True, "reason": "above_threshold"}
    if tier == "MEDIUM":
        return {"should_surface": True, "reason": "above_threshold"}
    return {"should_surface": False, "reason": "below_threshold"}


def main():
    rows = []
    for tier, p, m in product(TIERS, PATCHABLE_OPTIONS, MAINTENANCE_OPTIONS):
        result = evaluate_cell(tier, p, m)
        rows.append({
            "risk_tier": tier,
            "patchable": p,
            "maintenance_active": m,
            "should_surface": result["should_surface"],
            "reason": result["reason"],
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Markdown version
    lines = [
        "# RQ1 / RQ3 — Tier × Surfacing Truth Table",
        "",
        "| risk_tier | patchable | maintenance | should_surface | reason |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['risk_tier']} | {r['patchable']} | {r['maintenance_active']} "
            f"| {r['should_surface']} | {r['reason']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"Wrote {OUT_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

**DO NOT GUESS:** the actual `score_alert` import and signature need to be confirmed against `src/risk_scorer.py`. The placeholder logic in `evaluate_cell` mirrors ARCHITECTURE.md Step 10 but should be replaced with a real call to `score_alert` once the signature is known.

---

## 9. Execution order — full sequence

Run these in order. Stop at any failed verification.

```bash
# ─── PHASE 1: REFACTOR ──────────────────────────────────────
# Create common/context_mappings.py
# Update module6_evaluation.py imports
pytest tests/
# All previously-passing tests must still pass.

# ─── PHASE 2: EXTEND MODULE 3 ───────────────────────────────
# Modify module3_risk_scoring/module3_risk_scores.py
# Add tests/test_step9_composite_risk.py::test_risk_scores_npz_schema_v1_1
pytest tests/test_step9_composite_risk.py::test_risk_scores_npz_schema_v1_1
# Expected: FAIL (npz not regenerated)
python -m module3_risk_scoring.module3_risk_scores
pytest tests/test_step9_composite_risk.py::test_risk_scores_npz_schema_v1_1
# Expected: PASS

# ─── PHASE 3: HEADLINE METRICS ──────────────────────────────
# Resolve Phase-2.5 question first (per-model probas).
# Create module6_evaluation/compute_rq1_metrics.py
python -m module6_evaluation.compute_rq1_metrics
# Add tests/acceptance_tests.py::test_rq1_targets_met
pytest tests/acceptance_tests.py::test_rq1_targets_met
# Expected: PASS

# ─── PHASE 4: SUPPORTING ANALYSES (after questions resolved) ─
# 4a. analysis/compute_weight_sensitivity.py        [SPEC PENDING]
# 4b. analysis/compute_track_b_cascade_ablation.py  [verify model availability]
# 4c. module6_evaluation/compute_rq1_metrics_medsec.py  [verify device context]
# 4d. analysis/merge_rq1_metrics.py
python -m analysis.merge_rq1_metrics

# ─── PHASE 5: FIGURES & TRUTH TABLE ─────────────────────────
python -m module6_evaluation.make_rq1_figures
python -m module6_evaluation.make_rq1_truth_table

# ─── PHASE 6: FINAL VERIFICATION ────────────────────────────
pytest tests/
# All green.
ls results/rq1_metrics.json \
   results/rq1_weight_sensitivity.json \
   results/rq1_cascade_ablation.json \
   results/rq1_metrics_medsec.json \
   results/rq1_tier_surfacing_truth_table.csv \
   results/figures/{roc_curves,pr_curves,confusion_matrix,tier_boundary_histogram}.pdf
```

---

## 10. Open questions to surface before implementation

Claude Code should pause and ask the developer about each of these. Do not guess.

1. **Phase 2.5 — per-model probabilities.** Are `P_xgb`, `P_rf`, `P_dt` already in `risk_scores.npz`, or only the max as `c_track_a`? If only max, extend Module 3 to persist all three before completing `compute_track_a_ablation`.
2. **Phase 4 — Stage 5B design.** Weight sensitivity protocol is not finalized. Confirm perturbation strategy, agreement metric, and whether to include the multiplicative R alternative.
3. **Phase 4 — Stage 5C model availability.** Are there two trained DAEs (raw-input and cascade-input)? If not, training is a Module 2 task that must precede cascade ablation.
4. **Phase 4 — Stage 5D MedSec viability.** Does MedSec-25 data include `device_class` annotations? Are Module 1 and Module 3 runnable on it? If device context is missing, scope this stage to per-class Track B AUC only.
5. **Phase 5 — `score_alert` signature.** Confirm the exact `src/risk_scorer.score_alert` import and parameter signature so the truth table calls real code, not synthesized logic.
6. **Module 3 row filtering.** Does the current Module 3 implementation filter or shuffle rows between loading the parquet and emitting the npz? If yes, `row_id` must be the *original* parquet row index, not the post-filter position.

---

## 11. Coverage map — every RQ1 expected output to its stage

| RQ1_expected_outputs.md section | Pipeline stage |
|---|---|
| 1.1 — `results/rq1_metrics.json` | 3 (5A) + merged from 4 |
| 1.1 — `results/reports/risk_scores.npz` | 2 |
| 1.1 — `results/figures/*.pdf` (3 PDFs) | 5 |
| 1.2 — All headline metrics | 3 → `headline` |
| 2.1 — Track A ablation (Table) | 3 → `track_a_ablation` (pending Phase 2.5) |
| 2.2 — Track B cascade (Table) | 4 (5C) → merged into 3 |
| 2.3 — Composite risk sensitivity (Table) | 4 (5B) → merged into 3 |
| 2.4 — Per-class Track B EHMS | 3 → `track_b_per_class` |
| 2.4 — Per-class Track B MedSec | 4 (5D) → `rq1_metrics_medsec.json` |
| 3.1 — Spoofing failure mode | 3 → `documented_failure_modes` |
| 3.2 — Compensatory effects | 4 (5B) → `weight_sensitivity` |
| 3.3 — L1–L4 limitations | 3 → `limitations_acknowledged` |
| 5.2 — Tier × surfacing truth table | 5 (truth table script) |
| 5.2 — Tier boundary calibration histogram | 5 (figures script) — `tier_boundary_histogram.pdf` |
| 7 — D_crit / D_clinical_tier correlation | 3 → `correlation_diagnostics` |
| 6 — Test coverage | Existing files + new tests in Phases 2 and 3 |

Every numbered item is traceable to an implemented or pending stage. Pending stages are explicitly flagged in `_status` fields in the JSON, so the artifact is valid even before they complete.

---

## End of spec

Implementation order: Phases 1 → 2 → 3 → 5 → 4 → 6. Phases 4 and 5 are independent of each other; Phase 5 only needs Phase 3 done, so it can ship while Phase 4 questions are being resolved.