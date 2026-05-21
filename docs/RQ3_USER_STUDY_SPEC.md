# RQ3 Track 5 — User Study (RQ3 Lens — Escalation Chi-square)

**Project:** XAI-IDS-Healthcare
**Scope:** RQ3.5 — Verify role-distributed responsibility via the shared user-study dataset, with a new RQ3-specific metric (**appropriate escalation rate**, Chi-square). Reuses the RQ2.c per-role analysis as the substrate; layers escalation on top.
**Purpose:** Single, self-contained spec for the RQ3-lens user-study pipeline. Hand to Claude Code.
**Status of design:** All decisions locked under the answers to §10. Four `DO NOT GUESS` checkpoints (escalation action set, per-row severity threshold, Chi-square shape, Path C framing).
**Status of data:** Data-gated — depends on the same `survey/study_responses_*.json` files RQ2.c consumes. Under Path C those files are LLM-persona simulation, not human study, so the methodology disclosure carries forward.

---

## 0. How to use this spec

1. Phase 0 is mandatory — Claude Code must (a) confirm the RQ2.c per-role JSON exists, and (b) inventory the actual `response.action` values in the persona files before locking the escalation taxonomy.
2. Phases 1–5 are sequential.
3. Markers:
   - **DO NOT GUESS** — stop and ask the developer
   - **DEFENSE-CRITICAL** — directly defends the role-distributed-responsibility claim
   - **CROSS-REF** — relies on an artifact produced by RQ2 Track 3 (`compute_rq2c_per_role.py`)
4. Total expected size: 1 YAML manifest, 2 analysis scripts, 1 test file. Runtime: sub-second.

---

## 1. Background: what Track 5 produces

| Component | Question | Output |
|---|---|---|
| Escalation taxonomy | What action codes count as "appropriate escalation"? | `configs/rq3_escalation_definition.yaml` (pre-registered) |
| Per-role escalation analysis | Does condition B (with MVE) produce higher appropriate-escalation rates than A, per role? | `analysis/compute_rq3_escalation.py` |
| RQ3-lens aggregator | Wrap RQ2c per-role analysis with the RQ3 framing + add escalation block | `analysis/compute_rq3_per_role.py` |
| Final analysis JSON | Single artifact for thesis §5.6 / RQ3 results | `analysis/outputs/rq3_user_study.json` |
| Methodology test gates | Schema + methodology disclosure tests (matches RQ2.c pattern) | `tests/test_rq3_user_study.py` |

The defining property of Track 5: it answers a **role-distribution** question ("does MVE help personas in each role escalate appropriately?"), not a per-explanation-quality question. RQ2.c asks "does MVE change accuracy/confidence per role?"; RQ3 asks "does MVE change appropriate-action distribution per role?"

---

## 2. Locked design decisions

| Decision | Resolution |
|---|---|
| Data source | Same as RQ2.c — `survey/study_responses_LLM_*.json` (Path C — LLM-persona simulation) |
| Escalation definition | **Severity-aware ground-truth conjunction.** "Appropriate escalation" = `persona.action ∈ ESCALATION_ACTIONS` AND `row.correct_action ∈ ESCALATION_ACTIONS`. Pre-registered in `configs/rq3_escalation_definition.yaml` |
| ESCALATION_ACTIONS set (default) | `{"escalate", "escalate_clinical", "isolate"}` — Phase 0 must confirm against real persona action vocabulary; YAML editable post-Phase-0 |
| Severity threshold | None as a separate field — severity is implicit in `correct_action`. An alert is "severe enough to warrant escalation" iff its `correct_action` is in the escalation set |
| Statistical test | Chi-square 2×2 per role (escalated/not × condition A/B), with Fisher's exact fallback when any expected cell count < 5 |
| Effect size | Cramér's V (φ for 2×2); also report observed escalation rates per cell and odds ratio |
| Cross-role rollup | 3×2 contingency (role × escalation rate per condition) reported as secondary aggregate, NOT the primary test |
| Multiple-comparisons | Raw p-values; no correction; disclosed in `methodology_notes` (same policy as RQ2.c) |
| Sample-size threshold | Per-cell `n_warning` flag when any cell n < 10; insufficient-data flag when any expected cell count < 5 (Fisher's fallback) |
| Aggregation | Persona-level — one escalation flag per persona (computed from their successful rows); Chi-square across personas, NOT raw rows. Matches RQ2.c independence assumption |
| Path C framing | LLM-persona disclosure mandatory in every output JSON's `methodology_notes` + `limitations` blocks |

---

## 3. Phase 0 — Discovery (DO NOT GUESS)

### 3.1 Discovery script

```python
# scripts/discover_rq3_escalation_inputs.py — TRANSIENT, delete after Phase 0
"""Inventory the inputs Track 5 needs:
  1. RQ2.c per-role JSON exists?
  2. survey/study_responses_LLM_*.json action vocabulary
  3. correct_action vocabulary across all rows
  4. Existing compute_rq2c_per_role.py — usable as substrate?
"""
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
findings = {}

# 1. RQ2.c artifact
rq2c_path = REPO_ROOT / "analysis/outputs/rq2c_per_role.json"
findings["rq2c_per_role"] = {"path": str(rq2c_path),
                              "exists": rq2c_path.exists()}
if rq2c_path.exists():
    doc = json.loads(rq2c_path.read_text())
    findings["rq2c_per_role"]["roles_present"] = list(
        (doc.get("per_role") or {}).keys()
    )

# 2. Action vocabulary
survey_dir = REPO_ROOT / "survey"
action_counts = Counter()
correct_action_counts = Counter()
n_files = 0
for path in sorted(survey_dir.glob("study_responses_LLM_*.json")):
    n_files += 1
    rec = json.loads(path.read_text())
    for r in rec.get("rows", []):
        if r.get("error") is not None:
            continue
        resp = r.get("response") or {}
        action = resp.get("action")
        if action:
            action_counts[action] += 1
        ca = r.get("correct_action")
        if ca:
            correct_action_counts[ca] += 1

findings["action_inventory"] = {
    "n_files": n_files,
    "action_values": dict(action_counts),
    "correct_action_values": dict(correct_action_counts),
}

print(json.dumps(findings, indent=2, default=str))
print("\n" + "=" * 60)
print("DEVELOPER ACTION:")
print("  1. Confirm rq2c_per_role.json exists with 3 roles.")
print("  2. Review action_values + correct_action_values. Lock the")
print("     ESCALATION_ACTIONS set in configs/rq3_escalation_definition.yaml.")
print("  3. Confirm Path C framing applies.")
print("=" * 60)
```

### 3.2 What to confirm before Phase 1

1. **`analysis/outputs/rq2c_per_role.json` exists** with the 3 roles (`biomed_engineer`, `IT_generalist`, `nurse_manager`).
2. **Real action vocabulary** in `response.action`. Locks the `ESCALATION_ACTIONS` set in the YAML.
3. **Real `correct_action` vocabulary** — determines what "severe enough to warrant escalation" means in practice.
4. **`analysis/compute_rq2c_per_role.py` usable as substrate.** Track 5 reads its JSON output; it does not re-run the per-role Mann-Whitney.

---

## 4. Phase 1 — Escalation definition manifest

### 4.1 Create `configs/rq3_escalation_definition.yaml`

```yaml
# configs/rq3_escalation_definition.yaml
# Pre-registered escalation taxonomy for RQ3 Track 5.

schema_version: "1.0"
taxonomy_locked_on: "<YYYY-MM-DD>"
taxonomy_predates_data: false
taxonomy_source: |
  RQ3_USER_STUDY_SPEC.md §4.1, derived from RQ3_pipeline.md §10
  (Chi-square for escalation rate; new for RQ3 vs RQ2.c's Mann-Whitney).

# Action codes that count as "escalation-class" responses.
# Verified against the real persona action vocabulary in Phase 0.
escalation_actions:
  - escalate
  - escalate_clinical
  - isolate

# Per-row "appropriate escalation" rule:
#   row.response.action IS escalation-class AND
#   row.correct_action  IS escalation-class
appropriate_escalation_rule: |
  persona_action in escalation_actions AND
  correct_action in escalation_actions

# Per-persona aggregation threshold.
min_appropriate_escalation_proportion: 0.5

# Sample-size guards.
chi_square_min_expected_cell_count: 5
n_warning_threshold: 10
```

---

## 5. Phase 2 — Escalation Chi-square (DEFENSE-CRITICAL)

### 5.1 Create `analysis/compute_rq3_escalation.py`

**Contract:**
- **Inputs:** `survey/study_responses_LLM_*.json`, `configs/rq3_escalation_definition.yaml`, `survey/rq2c_exclusions.json`
- **Output:** `analysis/outputs/rq3_escalation.json`
- **Algorithm:**
  1. Load YAML; extract `escalation_actions` set + threshold
  2. Load excluded persona IDs
  3. For each included persona:
     - `n_warranted` = rows where `correct_action ∈ esc_set` (excluding error rows)
     - `n_appropriate` = rows where BOTH persona action AND correct_action ∈ esc_set
     - `escalated_appropriately = (n_appropriate / n_warranted) >= threshold` when `n_warranted >= 1`
     - Record (role, condition)
  4. Per role, build 2×2 contingency: `escalated_yes/no × condition_A/B`
  5. Run `chi2_contingency` (correction=False); if any expected cell <5, fall back to `fisher_exact`
  6. Compute Cramér's V (= sqrt(χ²/n) for 2×2 = |φ|), odds ratio, per-cell rates
  7. Emit `per_role` block + `overall` (3 roles collapsed) + `cell_diagnostics`

### 5.2 Output schema

```json
{
  "_meta": {
    "schema_version": "1.0",
    "generated_at": "<ISO-8601>",
    "generated_by": "analysis/compute_rq3_escalation.py",
    "taxonomy_path": "configs/rq3_escalation_definition.yaml",
    "data_source": "LLM-persona simulation (gpt-4o-mini); not human study"
  },
  "methodology_notes": [
    "Appropriate escalation = persona action AND correct_action both in [escalate, escalate_clinical, isolate].",
    "Chi-square 2x2 per role; Fisher's exact fallback when expected cell count < 5.",
    "Persona-level aggregation: escalated_appropriately = (n_appropriate / n_warranted) >= 0.5.",
    "Raw p-values; NO multiple-comparisons correction across the 3 role tests.",
    "Cramer's V (= |phi| for 2x2) as effect size; observed rates + odds ratio also reported."
  ],
  "limitations": [
    "LLM-persona simulation, not human study.",
    "Small per-cell N; Chi-square assumptions marginally satisfied.",
    "Single dimension of escalation (containment-class actions).",
    "Multiple-comparisons inflation (~0.14 false positives at alpha=0.05 across 3 role tests)."
  ],
  "overall": {
    "_scope": "All included personas (3 roles collapsed)",
    "n_A": 50, "n_B": 50,
    "contingency_2x2": {"A_escalated": 5, "A_not": 45, "B_escalated": 22, "B_not": 28},
    "test": "chi_square",
    "statistic": 0.0, "p_value": 0.0,
    "cramers_v": 0.0, "odds_ratio": 0.0,
    "rate_A": 0.10, "rate_B": 0.44,
    "n_warning": false, "fisher_fallback": false
  },
  "per_role": {
    "biomed_engineer": {...},
    "IT_generalist": {...},
    "nurse_manager": {...}
  },
  "cell_diagnostics": {
    "min_n_per_cell": 10, "max_n_per_cell": 25,
    "cells_with_warning": 0, "fisher_fallback_count": 0,
    "warning_threshold": 10
  }
}
```

### 5.3 Implementation outline

See full code template in `RQ3_USER_STUDY_SPEC_DRAFT_v1.md` (this section is preserved as outline-only in the locked spec; the implementation script ports the template with the standard `_role_from_pid`, `_persona_escalation_flag`, `_build_2x2`, `_run_test`, `_cramers_v`, `_odds_ratio` helpers from compute_rq2c_per_role.py's surrounding pattern). Key invariants:
- Use `scipy.stats.chi2_contingency(arr, correction=False)` — Fisher fallback when `expected.min() < min_expected`
- Cramér's V uncorrected variant: `sqrt(chi2 / n)` for 2×2
- Honest disclosure block (matches RQ2.c methodology_notes / limitations pattern)

---

## 6. Phase 3 — RQ3-lens wrapper

### 6.1 Create `analysis/compute_rq3_per_role.py`

Wraps `analysis/outputs/rq2c_per_role.json` with RQ3 framing + folds in the escalation block. **Does NOT recompute the per-role Mann-Whitney** — just relabels and extends.

**Output:** `analysis/outputs/rq3_user_study.json`

Schema:
```json
{
  "_meta": {
    "schema_version": "1.0",
    "research_question": "RQ3 — Does the system support distributed security responsibility across hospital roles while preserving clinical safety constraints?",
    "rq3_lens": "Reframes RQ2.c per-role accuracy/confidence + adds RQ3-specific appropriate-escalation Chi-square.",
    "data_source": "LLM-persona simulation (gpt-4o-mini); not human study"
  },
  "methodology_notes": [...],
  "limitations": [...],
  "per_role_accuracy_confidence": <rq2c_per_role.json::per_role>,
  "overall_accuracy_confidence":  <rq2c_per_role.json::overall>,
  "per_role_escalation":          <rq3_escalation.json::per_role>,
  "overall_escalation":           <rq3_escalation.json::overall>,
  "rq2c_cell_diagnostics":        <rq2c_per_role.json::cell_diagnostics>,
  "rq3_cell_diagnostics":         <rq3_escalation.json::cell_diagnostics>
}
```

---

## 7. Phase 4 — Tests

### 7.1 Create `tests/test_rq3_user_study.py`

7 test functions:
- `test_escalation_schema_complete` — top-level keys present
- `test_escalation_methodology_discloses_llm_persona` — Path C disclosure
- `test_escalation_methodology_discloses_no_correction` — multiple-comparisons disclosure
- `test_escalation_all_three_roles_present` — biomed_engineer / IT_generalist / nurse_manager
- `test_escalation_p_values_in_valid_range` — 0 ≤ p ≤ 1
- `test_escalation_cramers_v_in_valid_range` — 0 ≤ V ≤ 1 (for 2×2)
- `test_wrapper_schema_complete` — RQ3-lens wrapper structure

### 7.2 CI gate in `tests/acceptance_tests.py`

```python
def test_rq3_user_study_outputs_exist():
    """RQ3 Track 5: escalation JSON + RQ3-lens wrapper must exist."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    required = [
        "configs/rq3_escalation_definition.yaml",
        "analysis/outputs/rq3_escalation.json",
        "analysis/outputs/rq3_user_study.json",
    ]
    missing = [p for p in required if not (repo / p).exists()]
    assert not missing, f"Missing RQ3 Track 5 artifact(s): {missing}"
```

---

## 8. Execution order

```bash
# ─── PHASE 0: DISCOVERY ────────────────────────────────────────
python scripts/discover_rq3_escalation_inputs.py > /tmp/rq3_us_inventory.json
# DEVELOPER CONFIRMS action vocabulary + Path C framing.

# ─── PHASE 1: TAXONOMY ─────────────────────────────────────────
# Create configs/rq3_escalation_definition.yaml; set taxonomy_locked_on.

# ─── PHASE 2: ESCALATION CHI-SQUARE ────────────────────────────
python -m analysis.compute_rq3_escalation
cat analysis/outputs/rq3_escalation.json | python -m json.tool | head -40

# ─── PHASE 3: RQ3-LENS WRAPPER ─────────────────────────────────
python -m analysis.compute_rq3_per_role
cat analysis/outputs/rq3_user_study.json | python -m json.tool | head -30

# ─── PHASE 4: TESTS ────────────────────────────────────────────
pytest tests/test_rq3_user_study.py -v
pytest tests/acceptance_tests.py::test_rq3_user_study_outputs_exist -v
```

---

## 9. Integration with future `compute_rq3_metrics.py`

```python
def _load_user_study_subfile():
    p = REPO_ROOT / "analysis/outputs/rq3_user_study.json"
    if not p.exists():
        return {"_status": "pending — data-gated"}
    data = json.loads(p.read_text())
    return {
        "_status": "complete",
        "_merged_at": datetime.now(timezone.utc).isoformat(),
        "subfile_path": "analysis/outputs/rq3_user_study.json",
        "headline": {
            "overall_escalation": data["overall_escalation"],
            "rq2c_overall_accuracy": data["overall_accuracy_confidence"].get("accuracy"),
        },
    }
```

In the aggregator: `out["user_study"] = _load_user_study_subfile()`.

---

## 10. Open questions to surface (DO NOT GUESS)

Claude Code must pause and ask:

1. **Phase 0 — ESCALATION_ACTIONS set.** Default is `{escalate, escalate_clinical, isolate}`. Confirm against the real persona action vocabulary. Drop/add as needed.
2. **Phase 1 — taxonomy_locked_on date.** Honest disclosure: set to commit date with `taxonomy_predates_data: false` (same pattern as RQ2.d / RQ3 Track 1).
3. **Phase 2 — Persona-level threshold.** Default `min_appropriate_escalation_proportion: 0.5`. Sensitivity check (e.g., 0.3, 0.7) is future work.
4. **Phase 2 — Chi-square contingency shape.** Spec locks 2×2 per role as primary, 3×2 (collapsed) as secondary. Confirm or flip.

---

## 11. Coverage map — RQ3_pipeline.md §3 Track 5 → pipeline phase

| RQ3_pipeline.md item | Phase | Output |
|---|---|---|
| Track 5: appropriate escalation rate (Chi-square) | 2 | `analysis/outputs/rq3_escalation.json` |
| Track 5: reuses RQ2c per-role analysis | 3 | wrapper at `analysis/outputs/rq3_user_study.json` |
| Track 5: data-gated on RQ2.c | (existing) | `analysis/outputs/rq2c_per_role.json` |
| §10.3 cross-RQ overlap (RQ2 ∩ RQ3 invariants) | (Track 1) | `configs/invariants_manifest.yaml` |

---

## 12. Defense talking points this enables

- **"Why a Chi-square test for escalation when RQ2.c uses Mann-Whitney?"**
  *"Escalation is a binary outcome (appropriate vs not) per persona; Mann-Whitney is for continuous/ordinal data. Chi-square 2×2 per role tests whether condition changes the proportion of personas who escalate appropriately. Same data, different question."*

- **"How do you define 'appropriate escalation'?"**
  *"Pre-registered in `configs/rq3_escalation_definition.yaml`. A persona-row counts as appropriate escalation when BOTH the persona's chosen action AND the row's `correct_action` are in the escalation-class set. The conjunction captures 'persona recognized severity AND took a containment action.' A persona is flagged 'escalated_appropriately' overall if they passed that test on ≥50% of their escalation-warranted rows."*

- **"What about small cell counts?"**
  *"Chi-square requires expected cell counts ≥5. The script falls back to Fisher's exact when that's violated, flagging `fisher_fallback: true`. Per-cell N<10 also triggers `n_warning`."*

- **"Did MVE help appropriate escalation?"**
  *"Per-role rates + p-values + Cramér's V are in `per_role`. Overall 3-role-collapsed result is in `overall`. methodology_notes discloses no multiple-comparisons correction across the 3 role tests. Effect direction (B > A with positive Cramér's V) is the substantive signal; statistical significance at this N is exploratory."*

- **"How does this address RQ3 vs RQ2.c?"**
  *"RQ2.c: 'does MVE change how each role rates explanations?' (accuracy/confidence). RQ3: 'does MVE change the action distribution per role?' (appropriate-escalation rate). Same data, different framings. The wrapper carries both side-by-side."*

---

## 13. What this track deliberately does NOT do

- **Re-run the per-role Mann-Whitney.** That's RQ2.c's job.
- **Define escalation severity gradients.** Single binary outcome. Graded escalation is future work.
- **Apply a multiple-comparisons correction.** Same policy as RQ2.c — raw p-values, disclosed.
- **Run a human study.** Path C: LLM personas only.

---

## End of spec

Implementation order: Phase 0 (discovery + vocabulary inventory) → Phase 1 (YAML taxonomy) → Phase 2 (Chi-square) → Phase 3 (wrapper) → Phase 4 (tests). Each phase independently verifiable.

After Track 5 is implemented:
- `analysis/outputs/rq3_user_study.json` is the single reference for RQ3 user-study results
- All 5 RQ3 tracks (Invariant Evidence, Audit Integrity, No-Auto-Execution, Truth Table, User Study) are shippable
- Each has artifacts gated by CI tests in `tests/acceptance_tests.py`
