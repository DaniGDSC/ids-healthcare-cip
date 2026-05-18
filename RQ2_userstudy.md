# RQ2 User Study Pipeline — Per-Role × Per-Metric Analysis

**Project:** XAI-IDS-Healthcare
**Scope:** RQ2.c — Does MVE differentiate triage support across stakeholder roles?
**Purpose:** Single, self-contained spec for the user study analysis pipeline: schema validation, M5 Mann-Whitney per role × metric, exclusion criteria, effect sizes, manual qualitative coding, paper-ready outputs. Hand to Claude Code.
**Status of design:** All decisions locked. Six `DO NOT GUESS` checkpoints (study_loader.py contents, AlertScenario schema, participant JSON schema, attention check mechanism, existing analysis output format, role enum values).

---

## 0. How to use this spec

1. Phase 0 is mandatory — Claude Code must read existing `study_loader.py` and `study_analysis.py` before writing any new code. **This track is more "extend + verify" than "build from scratch."**
2. Phases 1–5 are sequential; do not skip.
3. Markers:
   - **DO NOT GUESS** — stop and ask the developer
   - **DEFENSE-CRITICAL** — this directly defends statistical methodology
   - **DATA-GATED** — runs only after user study data collection completes
   - **TARGET** — from `RQ2_expected_outputs.md`

---

## 1. Background: what Track 4 produces

| Component | Question | Output | Status |
|---|---|---|---|
| Schema validator | Are participant JSONs structurally complete? | Pytest test | New |
| Exclusion auditor | Which participants are excluded and why? | `rq2c_exclusions.json` | New |
| M5 Mann-Whitney aggregator | Group A vs Group B, overall + per role × per metric | `survey/m5_result.yaml` | Existing — verify |
| Per-role × per-metric breakdown | 9-cell table with effect sizes | `analysis/outputs/rq2c_per_role.json` | New |
| Qualitative theme manifest | Free-text themes per role | `survey/qualitative_themes.yaml` | New (manual) |
| Methodology documentation | Multiple-comparisons disclosure, exclusion rules | Embedded in JSON `_meta` and `methodology_notes` | New |

---

## 2. Locked design decisions

| Decision | Resolution |
|---|---|
| Role assignment | Self-select at signup; role is a field in each participant's JSON |
| Multiple comparisons | Report raw p-values, no correction; document in methodology_notes |
| Sample size threshold | Report all cells regardless of N; flag low-N (<10) with `n_warning: true` |
| Decision accuracy | Pre-recorded ground truth + reasonable alternatives per AlertScenario |
| Effect size | Cliff's delta (primary); also report direction and magnitude |
| Qualitative analysis | Manual theme coding written into YAML manifest |
| Participant exclusion | Attention check fail OR duration outlier (<30s or >30min total) |
| Counterbalancing | Existing MD5 seeding in `study_loader.py` — not modified |
| Statistical test | Mann-Whitney U (two-sided), as already implemented in `study_analysis.py` |
| Methodology transparency | Mandatory `methodology_notes` block in every output JSON listing all choices |

---

## 3. Phase 0 — Verify existing infrastructure (DO NOT GUESS)

This is **not optional**. Track 4 builds on existing files; getting the schema wrong corrupts every downstream metric silently.

### 3.1 Discovery script

```python
# scripts/discover_study_artifacts.py — TRANSIENT, delete after Phase 0
"""
Inventory existing user study code and any collected data.
"""
import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
findings = {}

# 1. study_loader.py
loader_path = REPO_ROOT / "module6_evaluation/study_loader.py"
findings["study_loader"] = {"path": str(loader_path), "exists": loader_path.exists()}
if loader_path.exists():
    text = loader_path.read_text()
    # Look for expected symbols
    for sym in [
        "AlertScenario", "load_scenarios", "assign_group",
        "MD5", "md5", "correct_action", "reasonable_alternatives",
        "attention_check", "shuffle",
    ]:
        findings["study_loader"][f"has_{sym}"] = sym in text

# 2. study_analysis.py
analysis_path = REPO_ROOT / "module6_evaluation/study_analysis.py"
findings["study_analysis"] = {"path": str(analysis_path), "exists": analysis_path.exists()}
if analysis_path.exists():
    text = analysis_path.read_text()
    for sym in [
        "mannwhitneyu", "Mann-Whitney", "m5_result", "decision_time",
        "accuracy", "confidence", "per_role", "cliff",
    ]:
        findings["study_analysis"][f"has_{sym}"] = sym in text

# 3. Existing participant responses
survey_dir = REPO_ROOT / "survey"
findings["survey_dir"] = {"path": str(survey_dir), "exists": survey_dir.exists()}
if survey_dir.exists():
    responses = sorted(survey_dir.glob("study_responses_*.json"))
    findings["survey_dir"]["n_responses"] = len(responses)
    if responses:
        # Sample first response to show schema
        sample = json.loads(responses[0].read_text())
        findings["survey_dir"]["sample_top_keys"] = list(sample.keys())
        # If there's a per-scenario records array, show its keys
        for k, v in sample.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                findings["survey_dir"][f"{k}_record_keys"] = list(v[0].keys())
                break

# 4. m5_result.yaml (existing output)
m5_path = REPO_ROOT / "survey/m5_result.yaml"
findings["m5_result"] = {"path": str(m5_path), "exists": m5_path.exists()}
if m5_path.exists():
    import yaml
    try:
        doc = yaml.safe_load(m5_path.read_text())
        findings["m5_result"]["top_keys"] = list(doc.keys()) if isinstance(doc, dict) else "NOT DICT"
    except Exception as e:
        findings["m5_result"]["parse_error"] = str(e)

# 5. AlertScenario class definition
if loader_path.exists():
    text = loader_path.read_text()
    # Find class block
    if "class AlertScenario" in text:
        # Extract attributes (heuristic — look for assignment-like lines after class def)
        idx = text.find("class AlertScenario")
        block = text[idx:idx + 2000]
        findings["AlertScenario_excerpt"] = block.split("\nclass ")[0][:800]

print(json.dumps(findings, indent=2, default=str))
print("\n" + "="*60)
print("DEVELOPER ACTION:")
print("  1. Confirm AlertScenario has correct_action + reasonable_alternatives fields")
print("  2. Confirm per-participant JSON has: participant_id, role, group, responses[]")
print("  3. Confirm per-response record has: alert_id, action_taken, decision_time_sec,")
print("     confidence (1-5), rationale, attention_check_passed (or similar)")
print("  4. Confirm role values: IT_GENERALIST / BIOMED_ENGINEER / NURSE_MANAGER")
print("  5. Confirm existing m5_result.yaml structure (overall A vs B only? per-role too?)")
print("="*60)
```

### 3.2 Six things to confirm before Phase 1

1. **`AlertScenario` schema** — has `correct_action`, has `reasonable_alternatives`, has an `attention_check` flag or scenario-type marker?
2. **Per-participant JSON top-level fields** — at minimum: `participant_id`, `role`, `group` (A or B), `started_at`, `completed_at`, `responses` (list).
3. **Per-response record fields** — at minimum: `alert_id`, `action_taken`, `decision_time_sec`, `confidence`, `rationale`, `attention_check_passed` (if applicable).
4. **Attention check mechanism** — is there an attention-check scenario (e.g., one with explicit "select 'isolate device' to confirm you're reading")? Or is it response-pattern based (e.g., flat-line confidence)? Or none?
5. **Existing `m5_result.yaml` structure** — does the current `study_analysis.py` already produce per-role breakdown, or only overall A vs B?
6. **Role enum values** — the exact strings used (`IT_GENERALIST` vs `IT Generalist` vs `it_generalist`?).

### 3.3 Verification

```bash
python scripts/discover_study_artifacts.py > /tmp/study_discovery.json
# DEVELOPER REVIEWS and confirms the 6 items above
```

If any of the 6 items reveals a mismatch (e.g., AlertScenario lacks `reasonable_alternatives`), the spec adapts:
- Missing `reasonable_alternatives` → Phase 2 falls back to strict-only accuracy
- Missing attention check → Phase 1 exclusion uses duration-only criterion
- Existing m5_result.yaml is overall-only → Phase 3 extends; if already per-role → Phase 3 wraps/reformats

---

## 4. Phase 1 — Schema validator and exclusion auditor

### 4.1 Create `analysis/audit_study_data.py`

**Contract:**
- **Input:** all `survey/study_responses_*.json` files.
- **Outputs:**
  - `survey/study_data_audit.json` — schema validation findings.
  - `survey/rq2c_exclusions.json` — list of excluded participants with reasons.
- **Runtime:** sub-second.

```python
"""
Audit user study data: validate schemas + apply exclusion criteria.

Exclusion rules (locked):
  - EX-1 Attention check failed (if scenario provides one)
  - EX-2 Total duration <30s (speedrunner) or >30min (distracted)
  - EX-3 Schema-invalid response (missing required fields)

Writes:
  - survey/study_data_audit.json
  - survey/rq2c_exclusions.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SURVEY_DIR = REPO_ROOT / "survey"
AUDIT_OUT = SURVEY_DIR / "study_data_audit.json"
EXCL_OUT = SURVEY_DIR / "rq2c_exclusions.json"

DURATION_MIN_SEC = 30
DURATION_MAX_SEC = 30 * 60  # 30 minutes

REQUIRED_TOP_LEVEL = ["participant_id", "role", "group", "responses"]
REQUIRED_PER_RESPONSE = [
    "alert_id", "action_taken", "decision_time_sec", "confidence"
]
VALID_ROLES = {"IT_GENERALIST", "BIOMED_ENGINEER", "NURSE_MANAGER"}
VALID_GROUPS = {"A", "B"}


def _validate_participant(record: dict) -> list:
    """Return list of validation errors for one participant JSON."""
    errors = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in record:
            errors.append({"check": "missing_top_level", "field": key})

    if record.get("role") not in VALID_ROLES:
        errors.append({"check": "invalid_role", "value": record.get("role")})
    if record.get("group") not in VALID_GROUPS:
        errors.append({"check": "invalid_group", "value": record.get("group")})

    responses = record.get("responses", [])
    if not isinstance(responses, list) or not responses:
        errors.append({"check": "responses_empty_or_invalid"})
    else:
        for i, r in enumerate(responses):
            for key in REQUIRED_PER_RESPONSE:
                if key not in r:
                    errors.append({
                        "check": "missing_response_field",
                        "response_index": i, "field": key
                    })
                    break  # one error per response is enough

    return errors


def _compute_total_duration(record: dict) -> float:
    """Sum decision_time_sec across responses. Falls back to nan if missing."""
    responses = record.get("responses", [])
    times = [r.get("decision_time_sec", 0) for r in responses
             if isinstance(r.get("decision_time_sec"), (int, float))]
    return sum(times) if times else 0.0


def _check_attention_failure(record: dict) -> bool:
    """
    DO NOT GUESS — adapt to the actual attention-check mechanism in AlertScenario.

    Possibilities:
      - Each response has 'attention_check_passed' (only set for attention-check scenarios)
      - Each scenario has 'is_attention_check' + 'expected_action'; compare to action_taken
      - No explicit mechanism — return False (no exclusion via this path)
    """
    responses = record.get("responses", [])
    for r in responses:
        # Pattern 1: explicit flag
        if r.get("is_attention_check") and not r.get("attention_check_passed", True):
            return True
    return False


def main():
    if not SURVEY_DIR.exists():
        print(f"No survey directory at {SURVEY_DIR} — nothing to audit")
        return

    response_files = sorted(SURVEY_DIR.glob("study_responses_*.json"))
    audit_records = []
    exclusions = []

    for path in response_files:
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            audit_records.append({
                "file": path.name,
                "valid": False,
                "errors": [{"check": "json_parse_error", "msg": str(e)}],
            })
            exclusions.append({
                "participant_id": path.stem,
                "reason": "EX-3 schema invalid (JSON parse error)",
            })
            continue

        errors = _validate_participant(rec)
        duration = _compute_total_duration(rec)
        attention_failed = _check_attention_failure(rec)
        duration_outlier = (
            duration < DURATION_MIN_SEC or duration > DURATION_MAX_SEC
        )

        excluded = bool(errors) or attention_failed or duration_outlier
        reasons = []
        if errors:
            reasons.append("EX-3 schema invalid")
        if attention_failed:
            reasons.append("EX-1 attention check failed")
        if duration_outlier:
            reasons.append(
                f"EX-2 duration outlier ({duration:.0f}s; "
                f"valid range {DURATION_MIN_SEC}-{DURATION_MAX_SEC}s)"
            )

        audit_records.append({
            "file": path.name,
            "participant_id": rec.get("participant_id"),
            "role": rec.get("role"),
            "group": rec.get("group"),
            "n_responses": len(rec.get("responses", [])),
            "total_duration_sec": duration,
            "attention_check_failed": attention_failed,
            "duration_outlier": duration_outlier,
            "schema_errors": errors,
            "excluded": excluded,
            "exclusion_reasons": reasons,
        })

        if excluded:
            exclusions.append({
                "participant_id": rec.get("participant_id", path.stem),
                "file": path.name,
                "reasons": reasons,
            })

    audit = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/audit_study_data.py",
            "n_participant_files": len(response_files),
            "n_excluded": len(exclusions),
            "exclusion_rules": {
                "EX-1": "Attention check failed",
                "EX-2": f"Total duration < {DURATION_MIN_SEC}s or > {DURATION_MAX_SEC}s",
                "EX-3": "Schema validation failed",
            },
        },
        "summary": {
            "n_total": len(response_files),
            "n_included": len(response_files) - len(exclusions),
            "n_excluded": len(exclusions),
            "exclusion_rate": (
                len(exclusions) / len(response_files) if response_files else 0
            ),
            "by_role": _summarize_by_field(audit_records, "role"),
            "by_group": _summarize_by_field(audit_records, "group"),
            "by_role_x_group": _summarize_role_x_group(audit_records),
        },
        "audit_records": audit_records,
    }

    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, default=str))

    EXCL_OUT.write_text(json.dumps({
        "_meta": audit["_meta"],
        "exclusions": exclusions,
    }, indent=2, default=str))

    print(f"Wrote {AUDIT_OUT.relative_to(REPO_ROOT)}")
    print(f"Wrote {EXCL_OUT.relative_to(REPO_ROOT)}")
    print(f"Total participants: {len(response_files)}  "
          f"Included: {len(response_files) - len(exclusions)}  "
          f"Excluded: {len(exclusions)}")


def _summarize_by_field(records, field):
    out = {}
    for r in records:
        if r.get("excluded"):
            continue
        key = str(r.get(field) or "UNKNOWN")
        out[key] = out.get(key, 0) + 1
    return out


def _summarize_role_x_group(records):
    """Count of included participants per (role, group) cell."""
    out = {}
    for r in records:
        if r.get("excluded"):
            continue
        key = f"{r.get('role') or 'UNKNOWN'}|{r.get('group') or 'UNKNOWN'}"
        out[key] = out.get(key, 0) + 1
    return out


if __name__ == "__main__":
    main()
```

### 4.2 Create `tests/test_study_data_schema.py`

```python
"""Schema validation tests for user study data."""
import json
from pathlib import Path

import pytest

AUDIT_OUT = Path("survey/study_data_audit.json")


@pytest.fixture(scope="module")
def audit():
    if not AUDIT_OUT.exists():
        pytest.skip("Run analysis/audit_study_data.py first")
    return json.loads(AUDIT_OUT.read_text())


def test_some_participants_collected(audit):
    """Sanity: at least one participant file exists."""
    assert audit["summary"]["n_total"] > 0, \
        "No participant responses found in survey/"


def test_exclusion_rate_reasonable(audit):
    """Soft check: exclusion rate should be < 30%. Hard fail at 60%+."""
    rate = audit["summary"]["exclusion_rate"]
    assert rate < 0.60, (
        f"Exclusion rate {rate:.1%} is alarmingly high. "
        f"Check survey/rq2c_exclusions.json for patterns."
    )


def test_all_roles_represented(audit):
    """At least one included participant per role (for analysis to be meaningful)."""
    by_role = audit["summary"]["by_role"]
    missing = [
        r for r in ["IT_GENERALIST", "BIOMED_ENGINEER", "NURSE_MANAGER"]
        if by_role.get(r, 0) == 0
    ]
    if missing:
        pytest.skip(f"Roles missing from data (recruitment incomplete): {missing}")


def test_both_groups_represented(audit):
    """Both A and B groups must be present."""
    by_group = audit["summary"]["by_group"]
    assert by_group.get("A", 0) > 0 and by_group.get("B", 0) > 0, \
        f"Group imbalance: {by_group}"
```

### 4.3 Verification

```bash
python -m analysis.audit_study_data
pytest tests/test_study_data_schema.py -v
```

---

## 5. Phase 2 — Per-role × per-metric analysis (DEFENSE-CRITICAL)

### 5.1 Create `analysis/compute_rq2c_per_role.py`

**Contract:**
- **Input:** all `survey/study_responses_*.json` files + `survey/rq2c_exclusions.json` (to filter out excluded participants).
- **Output:** `analysis/outputs/rq2c_per_role.json`.
- **Runtime:** sub-second.

This is the script that produces the 9-cell table from `RQ2_expected_outputs.md §3.1`.

### 5.2 Output schema

`analysis/outputs/rq2c_per_role.json`:

```json
{
  "_meta": {
    "schema_version": "1.0",
    "generated_at": "<ISO-8601>",
    "generated_by": "analysis/compute_rq2c_per_role.py",
    "inputs": {
      "n_participant_files": 32,
      "n_included": 28,
      "n_excluded": 4,
      "exclusion_audit": "survey/rq2c_exclusions.json"
    }
  },
  "methodology_notes": [
    "Raw p-values reported; no multiple-comparisons correction applied (9 cells).",
    "Mann-Whitney U is two-sided.",
    "Effect size: Cliff's delta; |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.",
    "Cells with n<10 per group flagged with n_warning: true.",
    "Accuracy uses pre-recorded correct_action + reasonable_alternatives per AlertScenario.",
    "Exclusion criteria: attention check fail OR duration <30s OR >30min."
  ],
  "limitations": [
    "Multiple comparisons (9 cells × 3 metrics) inflates Type I error rate;",
    "  with α=0.05 and no correction, expect ~0.45 false positives by chance.",
    "Role assignment is self-selected, producing cell-size imbalance.",
    "Sample size may yield underpowered cells; see power_analysis section."
  ],
  "overall": {
    "_scope": "All included participants",
    "n_A": 14, "n_B": 14,
    "decision_time": {
      "median_A": 47.2, "median_B": 38.6,
      "mannwhitney_u": 65.0, "p_value": 0.032,
      "cliffs_delta": -0.36, "magnitude": "medium",
      "direction": "B faster than A",
      "n_warning": false
    },
    "accuracy": { ... },
    "confidence": { ... }
  },
  "per_role": {
    "IT_GENERALIST": {
      "n_A": 5, "n_B": 5,
      "n_warning": true,
      "decision_time": {
        "median_A": 51.0, "median_B": 39.3,
        "mannwhitney_u": 11.0, "p_value": 0.421,
        "cliffs_delta": -0.20, "magnitude": "small",
        "direction": "B nominally faster",
        "n_warning": true
      },
      "accuracy": { ... },
      "confidence": { ... }
    },
    "BIOMED_ENGINEER": { ... },
    "NURSE_MANAGER": { ... }
  },
  "cell_diagnostics": {
    "_description": "Per-cell sample sizes for the 9-cell × 3-metric table",
    "min_n_per_cell": 5,
    "max_n_per_cell": 5,
    "cells_with_warning": 3
  }
}
```

### 5.3 Implementation outline

```python
"""
compute_rq2c_per_role.py
Mann-Whitney U + Cliff's delta for Group A vs Group B,
overall and per role × per metric.

Inputs:
  survey/study_responses_*.json  (per-participant responses)
  survey/rq2c_exclusions.json    (list of excluded participants)

Output:
  analysis/outputs/rq2c_per_role.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[1]
SURVEY_DIR = REPO_ROOT / "survey"
EXCL_PATH = SURVEY_DIR / "rq2c_exclusions.json"
SCENARIOS_PATH = REPO_ROOT / "module6_evaluation/study_loader.py"  # for AlertScenario imports
OUT_DIR = REPO_ROOT / "analysis/outputs"
OUT_PATH = OUT_DIR / "rq2c_per_role.json"

ROLES = ["IT_GENERALIST", "BIOMED_ENGINEER", "NURSE_MANAGER"]
GROUPS = ["A", "B"]
N_WARNING_THRESHOLD = 10
NEGLIGIBLE_DELTA = 0.147
SMALL_DELTA = 0.33
MEDIUM_DELTA = 0.474


# ─── Cliff's delta ─────────────────────────────────────────────

def cliffs_delta(a, b):
    """
    Cliff's delta: (#(a>b) - #(b>a)) / (n_a * n_b).
    Range [-1, 1]. Negative means B > A on average.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return None
    # Vectorized pairwise comparison
    diff = a[:, None] - b[None, :]
    greater = (diff > 0).sum()
    less = (diff < 0).sum()
    return float((greater - less) / (len(a) * len(b)))


def delta_magnitude(delta):
    if delta is None:
        return "undefined"
    abs_d = abs(delta)
    if abs_d < NEGLIGIBLE_DELTA:
        return "negligible"
    if abs_d < SMALL_DELTA:
        return "small"
    if abs_d < MEDIUM_DELTA:
        return "medium"
    return "large"


# ─── Per-metric computation ────────────────────────────────────

def compute_cell(group_a_values, group_b_values, metric_name):
    """
    Mann-Whitney U + Cliff's delta + diagnostics for one (subgroup, metric) cell.
    Higher-is-better metrics (accuracy, confidence): direction interpreted as "B has higher X."
    Lower-is-better (decision_time): direction interpreted as "B has lower X."
    """
    a = np.asarray(group_a_values, dtype=float)
    b = np.asarray(group_b_values, dtype=float)
    n_a, n_b = len(a), len(b)

    result = {
        "n_A": int(n_a), "n_B": int(n_b),
        "median_A": float(np.median(a)) if n_a else None,
        "median_B": float(np.median(b)) if n_b else None,
        "mean_A": float(np.mean(a)) if n_a else None,
        "mean_B": float(np.mean(b)) if n_b else None,
        "n_warning": n_a < N_WARNING_THRESHOLD or n_b < N_WARNING_THRESHOLD,
    }

    if n_a < 2 or n_b < 2:
        result.update({
            "mannwhitney_u": None, "p_value": None,
            "cliffs_delta": None, "magnitude": "undefined",
            "direction": "insufficient_data",
        })
        return result

    try:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        delta = cliffs_delta(a, b)
        result["mannwhitney_u"] = float(u)
        result["p_value"] = float(p)
        result["cliffs_delta"] = delta
        result["magnitude"] = delta_magnitude(delta)

        # Direction interpretation per metric semantics
        if metric_name == "decision_time":
            # Lower is better; negative delta means B has lower values
            if delta is None:
                direction = "undefined"
            elif delta < -NEGLIGIBLE_DELTA:
                direction = "B faster than A"
            elif delta > NEGLIGIBLE_DELTA:
                direction = "A faster than B"
            else:
                direction = "no meaningful difference"
        else:
            # Higher is better (accuracy, confidence)
            if delta is None:
                direction = "undefined"
            elif delta > NEGLIGIBLE_DELTA:
                direction = "A higher than B"
            elif delta < -NEGLIGIBLE_DELTA:
                direction = "B higher than A"
            else:
                direction = "no meaningful difference"
        result["direction"] = direction

    except ValueError as e:
        result.update({
            "mannwhitney_u": None, "p_value": None,
            "cliffs_delta": None, "magnitude": "undefined",
            "direction": f"test_failed: {e}",
        })

    return result


# ─── Loading + filtering ───────────────────────────────────────

def _load_excluded_ids():
    if not EXCL_PATH.exists():
        return set()
    excl = json.loads(EXCL_PATH.read_text())
    return {str(e["participant_id"]) for e in excl.get("exclusions", [])}


def _load_included_participants():
    excluded = _load_excluded_ids()
    out = []
    for path in sorted(SURVEY_DIR.glob("study_responses_*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if str(rec.get("participant_id")) in excluded:
            continue
        out.append(rec)
    return out


def _load_ground_truth():
    """
    Load correct_action + reasonable_alternatives per scenario.

    DO NOT GUESS — Phase 0 must confirm whether this is:
      (a) A dict in study_loader.py: SCENARIOS = {alert_id: AlertScenario(...)}
      (b) A YAML file: config/study_scenarios.yaml
      (c) Hardcoded inside study_loader.load_scenarios()

    Replace this with the actual import + access pattern.
    """
    # PLACEHOLDER — adapt to actual code structure
    from module6_evaluation.study_loader import load_scenarios
    scenarios = load_scenarios()
    return {
        s.alert_id: {
            "correct_action": s.correct_action,
            "reasonable_alternatives": getattr(s, "reasonable_alternatives", []),
        }
        for s in scenarios
    }


# ─── Per-participant aggregation ───────────────────────────────

def _participant_metrics(participant, ground_truth):
    """Aggregate one participant's responses into (mean accuracy, mean time, mean confidence)."""
    responses = participant.get("responses", [])
    if not responses:
        return None

    times, confs, correct = [], [], []
    for r in responses:
        # Skip attention-check scenarios from metric computation
        if r.get("is_attention_check"):
            continue
        t = r.get("decision_time_sec")
        c = r.get("confidence")
        if isinstance(t, (int, float)):
            times.append(t)
        if isinstance(c, (int, float)):
            confs.append(c)

        gt = ground_truth.get(r.get("alert_id"))
        if gt:
            taken = r.get("action_taken")
            ok = (
                taken == gt["correct_action"]
                or taken in gt.get("reasonable_alternatives", [])
            )
            correct.append(1 if ok else 0)

    if not times or not confs:
        return None

    return {
        "decision_time": float(np.median(times)),  # median per participant
        "confidence": float(np.mean(confs)),
        "accuracy": (float(np.mean(correct)) if correct else None),
    }


# ─── Main ──────────────────────────────────────────────────────

def main():
    included = _load_included_participants()
    ground_truth = _load_ground_truth()

    # Per-participant metric tuples
    by_role_group = defaultdict(lambda: defaultdict(list))  # by_role_group[role][group] = list of dicts
    overall_by_group = defaultdict(list)  # overall_by_group[group] = list of dicts

    for p in included:
        m = _participant_metrics(p, ground_truth)
        if m is None:
            continue
        role = p.get("role")
        group = p.get("group")
        by_role_group[role][group].append(m)
        overall_by_group[group].append(m)

    # Build output
    def extract(records, metric):
        return [r[metric] for r in records if r.get(metric) is not None]

    overall = {
        "_scope": "All included participants",
        "n_A": len(overall_by_group["A"]),
        "n_B": len(overall_by_group["B"]),
    }
    for metric in ["decision_time", "accuracy", "confidence"]:
        a = extract(overall_by_group["A"], metric)
        b = extract(overall_by_group["B"], metric)
        overall[metric] = compute_cell(a, b, metric)

    per_role = {}
    cell_warnings = 0
    cell_sizes = []
    for role in ROLES:
        if role not in by_role_group:
            per_role[role] = {"_status": "no participants in this role"}
            continue
        a_records = by_role_group[role]["A"]
        b_records = by_role_group[role]["B"]
        cell_sizes.extend([len(a_records), len(b_records)])
        entry = {
            "n_A": len(a_records), "n_B": len(b_records),
            "n_warning": (
                len(a_records) < N_WARNING_THRESHOLD or
                len(b_records) < N_WARNING_THRESHOLD
            ),
        }
        for metric in ["decision_time", "accuracy", "confidence"]:
            a = extract(a_records, metric)
            b = extract(b_records, metric)
            cell_result = compute_cell(a, b, metric)
            if cell_result.get("n_warning"):
                cell_warnings += 1
            entry[metric] = cell_result
        per_role[role] = entry

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/compute_rq2c_per_role.py",
            "inputs": {
                "n_participant_files": len(list(SURVEY_DIR.glob("study_responses_*.json"))),
                "n_included": len(included),
                "exclusion_audit": "survey/rq2c_exclusions.json",
            },
        },
        "methodology_notes": [
            "Raw p-values reported; NO multiple-comparisons correction applied "
            "across the 9 role × metric cells.",
            "Mann-Whitney U is two-sided.",
            f"Cliff's delta thresholds: negligible<{NEGLIGIBLE_DELTA}, "
            f"small<{SMALL_DELTA}, medium<{MEDIUM_DELTA}, else large.",
            f"Cells with n<{N_WARNING_THRESHOLD} per group flagged with n_warning=true.",
            "Accuracy uses pre-recorded correct_action + reasonable_alternatives "
            "per AlertScenario.",
            "Exclusion criteria: attention check fail OR duration <30s OR >30min.",
            "Attention-check scenarios excluded from per-metric aggregation.",
        ],
        "limitations": [
            "Multiple comparisons across 9 cells × 3 metrics inflates Type I error "
            "rate. With α=0.05 and no correction, ~0.45 false positives expected "
            "under the null hypothesis. Findings should be treated as exploratory.",
            "Role assignment is self-selected, producing potential cell-size "
            "imbalance and selection effects.",
            "Sample sizes may yield underpowered cells; n_warning flags identify "
            "these. Reviewers should weight findings by per-cell N.",
            "Single-round evaluation; iteration cycle is future work (see RQ2.d).",
        ],
        "overall": overall,
        "per_role": per_role,
        "cell_diagnostics": {
            "_description": "Per-cell sample sizes across 9 role × group cells",
            "min_n_per_cell": (min(cell_sizes) if cell_sizes else 0),
            "max_n_per_cell": (max(cell_sizes) if cell_sizes else 0),
            "cells_with_warning": cell_warnings,
            "warning_threshold": N_WARNING_THRESHOLD,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Overall: n_A={overall['n_A']} n_B={overall['n_B']}")
    for role in ROLES:
        r = per_role.get(role, {})
        print(f"  {role}: n_A={r.get('n_A', 0)} n_B={r.get('n_B', 0)}"
              f"{' (LOW-N WARNING)' if r.get('n_warning') else ''}")


if __name__ == "__main__":
    main()
```

### 5.4 Create `tests/test_rq2c_per_role.py`

```python
"""Schema + sanity tests for per-role analysis output."""
import json
from pathlib import Path

import pytest

OUT = Path("analysis/outputs/rq2c_per_role.json")


@pytest.fixture(scope="module")
def result():
    if not OUT.exists():
        pytest.skip("Run analysis/compute_rq2c_per_role.py first")
    return json.loads(OUT.read_text())


def test_schema_complete(result):
    for key in ["_meta", "methodology_notes", "limitations",
                "overall", "per_role", "cell_diagnostics"]:
        assert key in result


def test_methodology_notes_disclose_no_correction(result):
    text = " ".join(result["methodology_notes"]).lower()
    assert ("no multiple-comparisons correction" in text or
            "no multiple comparisons correction" in text), (
        "Methodology must explicitly disclose absence of multiple-comparisons "
        "correction (defense-critical transparency)."
    )


def test_limitations_disclose_multiple_comparisons(result):
    text = " ".join(result["limitations"]).lower()
    assert "multiple comparisons" in text, (
        "Limitations must call out the 9-cell multiple-comparisons issue."
    )


def test_overall_has_three_metrics(result):
    for m in ["decision_time", "accuracy", "confidence"]:
        assert m in result["overall"], f"Missing metric in overall: {m}"


def test_all_three_roles_present(result):
    for role in ["IT_GENERALIST", "BIOMED_ENGINEER", "NURSE_MANAGER"]:
        assert role in result["per_role"], f"Missing role: {role}"


def test_p_values_in_valid_range(result):
    """Sanity: every reported p-value must be in [0, 1]."""
    def _check_p(cell):
        p = cell.get("p_value")
        if p is not None:
            assert 0 <= p <= 1, f"Invalid p-value: {p}"

    for m in ["decision_time", "accuracy", "confidence"]:
        _check_p(result["overall"][m])
        for role in ["IT_GENERALIST", "BIOMED_ENGINEER", "NURSE_MANAGER"]:
            cell = result["per_role"].get(role, {}).get(m, {})
            _check_p(cell)


def test_cliffs_delta_in_valid_range(result):
    """Cliff's delta must be in [-1, 1]."""
    def _check_d(cell):
        d = cell.get("cliffs_delta")
        if d is not None:
            assert -1 <= d <= 1, f"Invalid Cliff's delta: {d}"

    for m in ["decision_time", "accuracy", "confidence"]:
        _check_d(result["overall"][m])
        for role in ["IT_GENERALIST", "BIOMED_ENGINEER", "NURSE_MANAGER"]:
            cell = result["per_role"].get(role, {}).get(m, {})
            _check_d(cell)
```

### 5.5 Verification

```bash
python -m analysis.compute_rq2c_per_role
pytest tests/test_rq2c_per_role.py -v
# Expected: 7 tests pass when data is sufficient; some may skip if data thin
```

---

## 6. Phase 3 — Qualitative theme coding (MANUAL)

This is a **human-in-the-loop step**. The script extracts rationales and presents them for the developer to read; the developer writes themes into a YAML file.

### 6.1 Create `analysis/extract_qualitative_rationales.py`

```python
"""
Extract free-text rationales from participant responses, grouped by role + group,
for manual theme coding by the developer.

Writes survey/qualitative_rationales_for_review.json — human-readable bundle.

Developer reads this, identifies themes, and writes them into
survey/qualitative_themes.yaml (template provided below).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SURVEY_DIR = REPO_ROOT / "survey"
EXCL_PATH = SURVEY_DIR / "rq2c_exclusions.json"
OUT = SURVEY_DIR / "qualitative_rationales_for_review.json"


def main():
    excluded = set()
    if EXCL_PATH.exists():
        excl = json.loads(EXCL_PATH.read_text())
        excluded = {str(e["participant_id"]) for e in excl.get("exclusions", [])}

    by_role_group = {}
    for path in sorted(SURVEY_DIR.glob("study_responses_*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if str(rec.get("participant_id")) in excluded:
            continue

        role = rec.get("role", "UNKNOWN")
        group = rec.get("group", "UNKNOWN")
        key = f"{role}|{group}"
        by_role_group.setdefault(key, [])

        for r in rec.get("responses", []):
            if r.get("is_attention_check"):
                continue
            rationale = (r.get("rationale") or "").strip()
            if rationale:
                by_role_group[key].append({
                    "participant_id": rec.get("participant_id"),
                    "alert_id": r.get("alert_id"),
                    "action_taken": r.get("action_taken"),
                    "rationale": rationale,
                })

    out = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/extract_qualitative_rationales.py",
            "instructions": (
                "Read each role|group bucket. Identify recurring themes. "
                "Write themes into survey/qualitative_themes.yaml using the "
                "template documented in the spec."
            ),
        },
        "by_role_group": by_role_group,
    }

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"Role × Group buckets: {len(by_role_group)}")
    for k, v in by_role_group.items():
        print(f"  {k}: {len(v)} rationales")
```

### 6.2 Create `survey/qualitative_themes.yaml` template

The developer fills this in after reading the bundle:

```yaml
# survey/qualitative_themes.yaml
# Manual theme coding of free-text rationales from user study.
# Updated by developer after reading qualitative_rationales_for_review.json.

schema_version: "1.0"
last_coded: ""              # ISO-8601 date when coding completed
coded_by: ""                 # Developer name/initials
methodology: |
  Single-coder thematic analysis. Read all rationales bundled by role+group.
  Identified recurring patterns; grouped into positive themes (helped decision)
  and confusion patterns (hindered decision). No inter-rater reliability
  computed (single coder); reported as exploratory.

themes_per_role:
  IT_GENERALIST:
    positive_themes:
      - theme: ""
        frequency: 0
        example_quote: ""
    confusion_patterns:
      - theme: ""
        frequency: 0
        example_quote: ""

  BIOMED_ENGINEER:
    positive_themes: []
    confusion_patterns: []

  NURSE_MANAGER:
    positive_themes: []
    confusion_patterns: []
```

### 6.3 Create `tests/test_qualitative_themes.py`

```python
"""Smoke tests for the manually-coded qualitative themes manifest."""
import yaml
from pathlib import Path

import pytest

YAML_PATH = Path("survey/qualitative_themes.yaml")


@pytest.fixture(scope="module")
def themes():
    if not YAML_PATH.exists():
        pytest.skip("Run analysis/extract_qualitative_rationales.py, "
                    "then manually code themes in survey/qualitative_themes.yaml")
    return yaml.safe_load(YAML_PATH.read_text())


def test_coded_metadata_present(themes):
    assert themes.get("last_coded"), \
        "last_coded date missing — fill in after manual coding"
    assert themes.get("coded_by"), \
        "coded_by missing — fill in after manual coding"


def test_all_three_roles_addressed(themes):
    for role in ["IT_GENERALIST", "BIOMED_ENGINEER", "NURSE_MANAGER"]:
        assert role in themes.get("themes_per_role", {}), \
            f"Missing role: {role}"


def test_each_role_has_at_least_one_theme(themes):
    """Skip until coded; once coded, every role should have themes."""
    if not themes.get("last_coded"):
        pytest.skip("Themes not yet coded")
    for role, entry in themes["themes_per_role"].items():
        total_themes = (
            len(entry.get("positive_themes", [])) +
            len(entry.get("confusion_patterns", []))
        )
        assert total_themes > 0, f"Role {role} has no coded themes"
```

### 6.4 Verification

```bash
python -m analysis.extract_qualitative_rationales
# DEVELOPER READS survey/qualitative_rationales_for_review.json
# DEVELOPER FILLS IN survey/qualitative_themes.yaml
pytest tests/test_qualitative_themes.py -v
```

---

## 7. Phase 4 — Verify existing `study_analysis.py` integration

If Phase 0 discovery showed that `study_analysis.py` already produces a per-role breakdown in `m5_result.yaml`, then `compute_rq2c_per_role.py` may be partially redundant.

Three integration patterns are acceptable:

### 7.1 Pattern A — `study_analysis.py` is overall-only

Then `compute_rq2c_per_role.py` adds the per-role extension. Both run; both outputs are merged by the master aggregator.

### 7.2 Pattern B — `study_analysis.py` already does per-role

Then `compute_rq2c_per_role.py` becomes a *wrapper* that reads `m5_result.yaml` and reformats it into the JSON schema in §5.2. Adds methodology_notes and limitations blocks that may be missing.

### 7.3 Pattern C — `study_analysis.py` does per-role but with different statistical choices

Spec says Mann-Whitney U + Cliff's delta. If existing code uses t-test, Bonferroni-corrected, etc., `compute_rq2c_per_role.py` *replaces* it and `study_analysis.py` is deprecated or extended to match.

**DO NOT GUESS** which pattern applies — Phase 0 discovery must confirm.

---

## 8. Phase 5 — Tests + CI gates

All test files were defined in earlier phases. Final assertions go into `tests/acceptance_tests.py`:

```python
def test_rq2c_pipeline_outputs_exist():
    """RQ2.c pipeline must produce all required artifacts before defense."""
    for path in [
        "survey/study_data_audit.json",
        "survey/rq2c_exclusions.json",
        "analysis/outputs/rq2c_per_role.json",
        "survey/qualitative_themes.yaml",
    ]:
        assert Path(path).exists(), (
            f"Missing RQ2.c artifact: {path}. "
            f"Run the relevant Track 4 analysis script."
        )
```

---

## 9. Execution order

```bash
# ─── PHASE 0: DISCOVERY ────────────────────────────────────────
python scripts/discover_study_artifacts.py > /tmp/study_discovery.json
# DEVELOPER CONFIRMS: AlertScenario fields, JSON schema, attention check
# mechanism, existing m5_result.yaml structure, role enum values.

# ─── PHASE 1: SCHEMA + EXCLUSION AUDIT ─────────────────────────
python -m analysis.audit_study_data
pytest tests/test_study_data_schema.py -v

# ─── PHASE 2: PER-ROLE × PER-METRIC ANALYSIS ───────────────────
python -m analysis.compute_rq2c_per_role
pytest tests/test_rq2c_per_role.py -v

# ─── PHASE 3: QUALITATIVE THEME CODING (HUMAN-IN-LOOP) ─────────
python -m analysis.extract_qualitative_rationales
# DEVELOPER READS survey/qualitative_rationales_for_review.json
# DEVELOPER FILLS IN survey/qualitative_themes.yaml
pytest tests/test_qualitative_themes.py -v

# ─── PHASE 4: RECONCILE WITH EXISTING study_analysis.py ────────
# Adapt per Pattern A/B/C from §7

# ─── PHASE 5: CI GATE ──────────────────────────────────────────
pytest tests/acceptance_tests.py::test_rq2c_pipeline_outputs_exist -v

# ─── FINAL VERIFICATION ────────────────────────────────────────
pytest tests/test_study_data_schema.py \
       tests/test_rq2c_per_role.py \
       tests/test_qualitative_themes.py -v
ls survey/study_data_audit.json \
   survey/rq2c_exclusions.json \
   survey/qualitative_themes.yaml \
   analysis/outputs/rq2c_per_role.json
```

---

## 10. Integration with `compute_rq2_metrics.py`

```python
def _load_user_study_subfiles():
    audit_p = REPO_ROOT / "survey/study_data_audit.json"
    per_role_p = REPO_ROOT / "analysis/outputs/rq2c_per_role.json"
    themes_p = REPO_ROOT / "survey/qualitative_themes.yaml"
    excl_p = REPO_ROOT / "survey/rq2c_exclusions.json"

    block = {"_status": "pending — data collection or analysis incomplete",
             "_merged_at": None}
    if audit_p.exists() and per_role_p.exists():
        block = {
            "_status": "complete" if themes_p.exists() else "partial — themes pending",
            "_merged_at": datetime.now(timezone.utc).isoformat(),
            "data_audit": json.loads(audit_p.read_text()),
            "exclusions": json.loads(excl_p.read_text()) if excl_p.exists() else None,
            "per_role_analysis": json.loads(per_role_p.read_text()),
            "qualitative_themes_path": str(themes_p.relative_to(REPO_ROOT))
                                       if themes_p.exists() else None,
        }
    return block
```

In the aggregator: `out["user_study"] = _load_user_study_subfiles()`.

---

## 11. Open questions to surface (DO NOT GUESS)

Claude Code must pause and ask:

1. **Phase 0 — `AlertScenario` schema.** Does the class have `correct_action` and `reasonable_alternatives` fields? If not, who maintains the ground truth table?
2. **Phase 0 — Per-participant JSON schema.** Confirm the exact top-level field names. Spec assumes `participant_id`, `role`, `group`, `responses`.
3. **Phase 0 — Per-response record schema.** Confirm `alert_id`, `action_taken`, `decision_time_sec`, `confidence`, `rationale`, `is_attention_check`, `attention_check_passed` field names.
4. **Phase 0 — Attention check mechanism.** Is there one? If so, how is failure flagged?
5. **Phase 0 — Existing `study_analysis.py` output format.** Does `m5_result.yaml` have per-role breakdown? Does it report effect sizes?
6. **Phase 0 — Role enum values.** Exact strings: `IT_GENERALIST` vs `IT Generalist` vs other.
7. **Phase 2 — Ground truth loader path.** Where does `_load_ground_truth()` import from?
8. **Phase 3 — Single-coder vs second coder for qualitative.** Spec assumes single coder (you). If a second coder is available, inter-rater reliability section is needed.

---

## 12. Coverage map — RQ2.c expected outputs → pipeline phase

| RQ2_expected_outputs.md §3 item | Phase | Output |
|---|---|---|
| §3.1 Per-role decision time Mann-Whitney | 2 | `rq2c_per_role.json.per_role.<ROLE>.decision_time` |
| §3.1 Per-role accuracy Mann-Whitney | 2 | `rq2c_per_role.json.per_role.<ROLE>.accuracy` |
| §3.1 Per-role confidence Mann-Whitney | 2 | `rq2c_per_role.json.per_role.<ROLE>.confidence` |
| §3.1 Effect size column | 2 | `cliffs_delta` + `magnitude` per cell |
| §3.2 `survey/study_responses_<PID>.json` | (existing) | per-participant data |
| §3.2 `survey/m5_result.yaml` | (existing — verify) | aggregated overall |
| §3.2 `analysis/outputs/rq2c_per_role.json` | 2 | per-role breakdown |
| §3.3 Per-role qualitative themes | 3 | `survey/qualitative_themes.yaml` |
| §3.4 Bedside nurse role question | (future work) | acknowledged in limitations |

Every numbered RQ2.c item is traceable to a phase except §3.4 (bedside nurse), which is acknowledged future work per the overview.

---

## 13. Defense talking points this enables

When a defense reviewer asks RQ2.c questions, you can answer:

- **"How do you handle multiple comparisons?"**
  *"We report raw p-values across all 9 cells (3 roles × 3 metrics) and explicitly disclose this in the methodology notes embedded in the output JSON. With α=0.05 and no correction, ~0.45 false positives are expected by chance. Findings are framed as exploratory. The decision to skip correction was deliberate: it lets reviewers apply their own preferred correction (Bonferroni divides by 9; Holm-Bonferroni provides sequential rejection)."*

- **"How do you handle low-N cells?"**
  *"Every cell with n<10 per group is flagged with `n_warning: true` in the JSON. Cliff's delta gives a magnitude interpretation independent of statistical significance, so even low-N cells provide directional evidence without inflating false-positive risk through bare p-values."*

- **"Who decided which actions count as correct?"**
  *"Each AlertScenario has a pre-recorded `correct_action` plus a list of `reasonable_alternatives`. Both count as accurate. The full ground truth table is in `module6_evaluation/study_loader.py`; no post-hoc judgment of individual responses."*

- **"What about participants who didn't pay attention?"**
  *"Pre-registered exclusion criteria: failed attention check, or total duration <30s or >30min. All exclusions are logged in `survey/rq2c_exclusions.json` with explicit reasons. The exclusion rate is reported alongside the analysis."*

- **"Did you do qualitative analysis?"**
  *"Yes, single-coder thematic analysis of free-text rationales, bundled by role × group. Themes are documented in `survey/qualitative_themes.yaml`. We acknowledge the single-coder limitation; second-coder inter-rater reliability is future work."*

---

## End of spec

Implementation order: Phase 0 (discovery) → 1 (audit) → 2 (per-role analysis) → 3 (qualitative) → 4 (reconcile existing) → 5 (CI gate). Phases 1–3 can run in parallel after Phase 0 completes. The pipeline is DATA-GATED on user study completion.