# RQ3 Track 1 — Invariant Evidence Aggregator

**Project:** XAI-IDS-Healthcare
**Scope:** RQ3.1 — Verify all 9 architectural invariants pass; aggregate into a single, paper-renderable JSON.
**Purpose:** Single, self-contained spec for the invariant evidence pipeline. Hand to Claude Code.
**Status of design:** All decisions locked. Four `DO NOT GUESS` checkpoints (test file inventory, pytest-json-report output schema, invariant 3 grep target, invariant manifest pre-registration date).

---

## 0. How to use this spec

1. Phase 0 is mandatory — Claude Code must run the discovery script and confirm which invariant tests already exist before any code is written.
2. Phases 1–5 are sequential.
3. Markers:
   - **DO NOT GUESS** — stop and ask the developer
   - **STRICT-GATE** — relates to the Q4 strict completeness decision
   - **DEFENSE-CRITICAL** — directly defends an architectural claim
4. Total expected size: 1 YAML manifest, 2 new analysis scripts, 1 new test file, 4 modifications to existing test files. Runtime: pytest takes whatever your existing test suite takes; aggregator is sub-second.

---

## 1. Background: what Track 1 produces

| Component | Question | Output |
|---|---|---|
| Invariant manifest | What are the 9 invariants, where are they enforced, who tests them? | `config/invariants_manifest.yaml` |
| Test invariant tagging | Which test corresponds to which invariant? | Manifest field per invariant + pytest markers |
| Evidence aggregator | Did every invariant test pass in the last CI run? | `results/rq3_invariant_evidence.json` |
| Markdown renderer | Paper-ready table for §5.6 Safety Engineering | `results/rq3_invariant_evidence.md` |
| Strict CI gate | Hard-fail if any invariant test failed | `tests/acceptance_tests.py::test_invariant_evidence_complete` |

The defining property of Track 1 is that it produces **evidence** rather than performing new computation. The tests already encode the invariants; Track 1 collects their pass/fail status into a single auditable artifact.

---

## 2. Locked design decisions

| Decision | Resolution |
|---|---|
| Manifest format | YAML source of truth at `config/invariants_manifest.yaml`; Python reads it; markdown rendered from it |
| Pytest capture | `pytest-json-report` plugin; aggregator parses its output |
| Failure mode | Hard fail (CI-blocking) on any invariant test failure |
| Test mapping strictness | Strict: every invariant must have at least one test mapped |
| Manifest pre-registration | Manifest has `preregistered_date` field; matches the date invariants were locked |
| Cross-RQ overlap | Manifest documents which RQ each invariant serves; same test can serve multiple RQs |
| Invariant 3 special case | Grep-based; mapped via a YAML `verification_method: grep` instead of `verification_method: pytest` |
| Strictness escape valve | Per-invariant `status` field: `enforced` (full strict) / `documented` (no test required) / `pending` (test scheduled) |

The strictness escape valve resolves the dependency on Track 2 (audit integrity test) and the Invariant 3 grep special case. Strict completeness means "every invariant must have a mapping" — but the mapping can be to a grep audit or a `pending` status, not just to a pytest test.

---

## 3. Phase 0 — Test inventory discovery (DO NOT GUESS)

Before writing any code, Claude Code must enumerate which invariant tests currently exist. The manifest cannot validate without this.

### 3.1 Discovery script

```python
# scripts/discover_invariant_tests.py — TRANSIENT, delete after Phase 0
"""
Inventory existing invariant test files and report which invariants
have at least one test, which are pending, and which use grep verification.
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Per ARCHITECTURE.md + RQ3_expected_outputs.md, expected test files:
EXPECTED = {
    1: ("DAE only elevates",
        "tests/test_step9_composite_risk.py"),
    2: ("Safety floor",
        "tests/test_safe_failure.py"),
    3: ("No auto-execution",
        "tests/negative_tests.py"),
    4: ("Audit trail complete",
        "tests/test_step16_audit_integrity.py"),
    5: ("Layer 1 references SHAP top features",
        "tests/test_step12_mve_faithfulness.py"),
    6: ("Each role authorizes only role-appropriate actions",
        "tests/test_step13_cross_role_consistency.py"),
    7: ("DO_NOT present for CRITICAL on clinical devices",
        "tests/test_step12_mve_faithfulness.py"),
    8: ("Layer 2 severity matches risk tier",
        "tests/test_step12_mve_faithfulness.py"),
    9: ("Shared anchor identical across roles",
        "tests/test_step13_cross_role_consistency.py"),
}

INV3_GREP_TARGETS = [
    "pipeline/module5_response/",
    "module5_responses/",
    "src/mve_generator.py",
]

findings = {"invariants": {}, "grep_targets": {}}

for inv_id, (name, test_file) in EXPECTED.items():
    p = REPO_ROOT / test_file
    findings["invariants"][inv_id] = {
        "name": name,
        "expected_test_file": test_file,
        "test_file_exists": p.exists(),
        "test_file_size_bytes": p.stat().st_size if p.exists() else 0,
    }
    if p.exists():
        # Look for invariant marker comments inside
        content = p.read_text()
        # Match patterns like "Invariant 3" or "invariant_3" or "INV-3"
        markers = re.findall(
            rf"\b[Ii]nvariant[\s_-]+{inv_id}\b", content
        )
        findings["invariants"][inv_id]["explicit_invariant_refs"] = len(markers)
        # Look for pytest.mark.invariant() decorators
        marker_re = re.compile(rf"@pytest\.mark\.invariant\(\s*{inv_id}\b")
        findings["invariants"][inv_id]["pytest_marker_count"] = len(
            marker_re.findall(content)
        )

# Invariant 3 grep targets (these are directories, not files)
for target in INV3_GREP_TARGETS:
    p = REPO_ROOT / target
    findings["grep_targets"][target] = {
        "exists": p.exists(),
        "is_dir": p.is_dir() if p.exists() else False,
    }

# pytest-json-report availability check
try:
    import importlib
    importlib.import_module("pytest_jsonreport")
    findings["pytest_json_report_installed"] = True
except ImportError:
    findings["pytest_json_report_installed"] = False
    findings["_pytest_install_hint"] = (
        "pip install pytest-json-report"
    )

# Verify pytest config exists
for cfg in ["pytest.ini", "pyproject.toml", "setup.cfg"]:
    if (REPO_ROOT / cfg).exists():
        findings["pytest_config_file"] = cfg
        break
else:
    findings["pytest_config_file"] = None

print(json.dumps(findings, indent=2, default=str))
print("\n" + "=" * 60)
print("DEVELOPER ACTION:")
print("  1. Review which invariant test files exist and which are missing.")
print("  2. Confirm grep targets — which directory contains the response code?")
print("  3. If pytest-json-report not installed, run: pip install pytest-json-report")
print("  4. Decide for any missing test:")
print("     - 'pending' (test will be created later)")
print("     - 'documented' (no test required; e.g., Invariant 3 is grep-based)")
print("=" * 60)
```

### 3.2 What to confirm before Phase 1

1. **Test file existence:** which of the 9 invariant tests currently exist? Missing tests get `status: pending` in the manifest.
2. **Grep targets for Invariant 3:** confirm the actual directory path (`pipeline/module5_response/` vs `module5_responses/` — both appear in docs).
3. **pytest-json-report installed:** if not, this is a one-line `pip install` before Phase 2.
4. **Pre-registration date:** when were the invariants first formalized? Manifest's `preregistered_date` must reflect the actual date for defense purposes.

### 3.3 Verification

```bash
python scripts/discover_invariant_tests.py > /tmp/invariant_inventory.json
# Developer reviews; confirms test inventory + grep targets
```

**DO NOT GUESS** any of these. Inaccurate inventory propagates into the manifest, which is the single source of truth for the entire track.

---

## 4. Phase 1 — Invariant manifest

### 4.1 Create `config/invariants_manifest.yaml`

This is the **canonical list of 9 architectural invariants.** The manifest is pre-registered (carries `preregistered_date`) — it cannot be retroactively edited to make failing tests pass.

```yaml
# config/invariants_manifest.yaml
# Canonical list of 9 architectural invariants for XAI-IDS-Healthcare.
#
# This manifest is the SINGLE SOURCE OF TRUTH for invariant definitions.
# Python code reads it; markdown is rendered from it; tests are tagged against it.
#
# DEFENSE-CRITICAL: do not edit invariant DEFINITIONS post-evaluation.
# Edits to verification_method or test_files are acceptable as implementation
# evolves, but the invariant text itself is locked at preregistered_date.

schema_version: "1.0"
preregistered_date: "2025-08-14"   # DO NOT GUESS — set to actual lock date
last_implementation_update: "2026-MM-DD"

# Severity tier informs the CI gate behaviour (not used here — all are hard-fail —
# but documented for cross-RQ overlap).
severity_tiers:
  safety_critical: [1, 2, 3, 4]   # violations are safety incidents
  quality:         [5, 6, 7, 8, 9] # violations are quality regressions

invariants:
  - id: 1
    title: "DAE only elevates detection confidence; never suppresses"
    text: |
      Fusion uses c_detect = max(c_track_a, c_track_b). The DAE (Track B)
      can only increase detection confidence above Track A's, never below.
      This guarantees the supervised path's evidence is preserved.
    rationale: |
      Without this property, a noisy DAE could mask a real attack that
      the supervised classifier correctly flagged.
    serves_rqs: [1]
    severity: safety_critical
    enforced_by:
      - "src/risk_scorer.py: c_detect = max(c_track_a, c_track_b)"
      - "Module 3 batch scoring (module3_risk_scoring/module3_risk_scores.py)"
    verification_method: pytest
    test_files:
      - "tests/test_step9_composite_risk.py"
    status: enforced   # enforced | pending | documented

  - id: 2
    title: "Safety floor — CRITICAL+unpatchable always surfaces"
    text: |
      For any alert where device_criticality = CRITICAL AND patchable = False,
      should_surface = True regardless of risk score, maintenance window, or
      threshold. Operator override cannot suppress these.
    rationale: |
      Life-critical devices that cannot be patched are the worst-case
      threat surface. Even during maintenance windows or under quiet-mode
      configuration, the operator must see these.
    serves_rqs: [1, 3]
    severity: safety_critical
    enforced_by:
      - "src/risk_scorer.py: should_surface logic"
    verification_method: pytest
    test_files:
      - "tests/test_safe_failure.py"
      - "tests/test_step10_surfacing_logic.py"
    status: enforced

  - id: 3
    title: "No auto-execution — recommendation only"
    text: |
      The system must never execute mitigation actions automatically.
      Every alert produces a recommendation; the operator's explicit
      decision is required before any state-changing action.
    rationale: |
      Hospital networks contain life-critical devices. Automated quarantine
      of a ventilator is unacceptable. The system is HITL by design.
    serves_rqs: [3]
    severity: safety_critical
    enforced_by:
      - "module5_responses/module5_pipeline.py: recommend()"
      - "ResponseRecommendation.operator_decision_required = True always"
    verification_method: grep_and_pytest   # special case
    grep_audit:
      target_dirs:
        - "pipeline/module5_response/"   # DO NOT GUESS — confirm in Phase 0
        - "module5_responses/"
      forbidden_patterns:
        - 'subprocess'
        - 'os\.system'
        - 'iptables'
        - 'netcat'
        - r'\bnc\s'
        - 'curl'
        - 'wget'
        - 'ssh'
        - 'sudo'
        - 'eval'
        - r'exec\('
      forbidden_imports:
        - "^import subprocess"
        - "^from subprocess"
    test_files:
      - "tests/negative_tests.py"
    status: enforced

  - id: 4
    title: "Audit trail complete"
    text: |
      Every alert decision produces an audit log entry containing alert
      context, operator context, decision capture, explanation context,
      and tamper-evidence (SHA256 hash chain). Schema is non-negotiable.
    rationale: |
      Distributed responsibility requires that any operator action be
      traceable to its full context. Hash chain makes silent tampering
      detectable.
    serves_rqs: [3]
    severity: safety_critical
    enforced_by:
      - "src/audit_logger.py (per Step 16)"
      - "Hash chain: SHA256 of previous_hash || entry_body"
    verification_method: pytest
    test_files:
      - "tests/test_step16_audit_integrity.py"
    status: pending   # depends on RQ3 Track 2 implementation

  - id: 5
    title: "Layer 1 references actual SHAP top features"
    text: |
      MVE Layer 1 (WHY) must contain the top-3 SHAP features (raw names or
      human-readable mappings) that drove the detection. No top features
      → no faithful explanation.
    rationale: |
      Faithfulness requires explanations to reflect the actual model
      decision logic, not post-hoc plausible-sounding text.
    serves_rqs: [2]
    severity: quality
    enforced_by:
      - "src/mve_generator.py: build_layer1 + Mode B fallback"
      - "config/llm_data_flow.yaml: SHAP fields allowed in prompts"
    verification_method: pytest
    test_files:
      - "tests/test_step12_mve_faithfulness.py"
    status: pending   # created by RQ2 Track 1 / faithfulness spec

  - id: 6
    title: "Each role authorizes only role-appropriate actions"
    text: |
      Layer 3 immediate_action per role view must be among the actions
      authorized for that role in config/role_action_authorization.yaml.
      Cross-role severity (Layer 2) is invariant across roles.
    rationale: |
      Distributed responsibility requires clear authorization boundaries.
      An IT generalist must not see a "biomed-only" action and an alert
      cannot show one role a different severity than another.
    serves_rqs: [2, 3]
    severity: quality
    enforced_by:
      - "src/mve_generator.py: derive_role_view"
      - "config/role_action_authorization.yaml"
    verification_method: pytest
    test_files:
      - "tests/test_step13_cross_role_consistency.py"
      - "tests/test_step15_role_consistency.py"
      - "tests/test_safe_failure.py"
    status: pending   # created by RQ2 Track 3 / compliance spec

  - id: 7
    title: "DO_NOT present for CRITICAL on clinical devices"
    text: |
      For every alert with risk_tier=CRITICAL on a clinical device class
      (ventilator, patient_monitor, infusion_pump), Layer 3 must include
      a non-empty layer3_do_not clinical-safety constraint.
    rationale: |
      Clinical safety constraints are the floor of the recommendation.
      "Do not disconnect ventilator" must be visible regardless of role
      or LLM behaviour.
    serves_rqs: [2, 3]
    severity: quality
    enforced_by:
      - "src/mve_generator.py: clinical_constraint preservation"
    verification_method: pytest
    test_files:
      - "tests/test_step12_mve_faithfulness.py"
      - "tests/test_step13_cross_role_consistency.py"
    status: pending

  - id: 8
    title: "Layer 2 severity string matches risk_tier"
    text: |
      The severity adjective in Layer 2 (e.g., "critical", "high",
      "medium", "low") must lexically match the risk_tier of the alert.
      Layer 2 text containing "low" for a CRITICAL alert is a violation.
    rationale: |
      Audience-appropriate explanation must not contradict the underlying
      severity. Layer 2 is the role-invariant severity assertion.
    serves_rqs: [2]
    severity: quality
    enforced_by:
      - "src/mve_generator.py: build_layer2"
    verification_method: pytest
    test_files:
      - "tests/test_step12_mve_faithfulness.py"
    status: pending

  - id: 9
    title: "Shared anchor identical across roles"
    text: |
      For one alert rendered to multiple roles, the shared anchor
      (alert_id, risk_tier, device_class, timestamp_iso) must be
      byte-identical across all role views. Only Layer 1 (vocabulary)
      and Layer 3 (action) may vary.
    rationale: |
      During phone-based incident handling, two operators in different
      roles must agree on what alert they are discussing. The shared
      anchor is the unambiguous reference key.
    serves_rqs: [2, 3]
    severity: quality
    enforced_by:
      - "src/mve_generator.py: derive_role_view preserves anchor"
    verification_method: pytest
    test_files:
      - "tests/test_step13_cross_role_consistency.py"
    status: pending
```

### 4.2 Verification

```bash
python -c "
import yaml
from pathlib import Path
doc = yaml.safe_load(Path('config/invariants_manifest.yaml').read_text())
print(f\"Schema: {doc['schema_version']}\")
print(f\"Pre-registered: {doc['preregistered_date']}\")
print(f\"Invariants: {len(doc['invariants'])}\")
for inv in doc['invariants']:
    print(f\"  #{inv['id']} [{inv['status']}] {inv['title']}\")
"
# Expected: 9 invariants listed with status
```

---

## 5. Phase 2 — Manifest validator (STRICT-GATE)

### 5.1 Create `analysis/validate_invariant_manifest.py`

**Contract:**
- **Input:** `config/invariants_manifest.yaml`
- **Output:** `results/rq3_invariant_manifest_validation.json` (pass/fail per check)
- **Runtime:** sub-second.
- **Side effects:** writes one file.

**Validation checks (all must pass for STRICT-GATE):**

| Check ID | Description | Pass condition |
|---|---|---|
| V1 | Manifest parses | YAML valid |
| V2 | preregistered_date present | non-empty ISO-8601 |
| V3 | Exactly 9 invariants | `len(invariants) == 9` |
| V4 | All IDs unique 1-9 | `set(ids) == {1..9}` |
| V5 | Every invariant has title + text + rationale | non-empty strings |
| V6 | Every invariant has at least one test_file or grep_audit | mapping exists |
| V7 | Severity is one of {safety_critical, quality} | enum check |
| V8 | Status is one of {enforced, pending, documented} | enum check |
| V9 | If status=enforced, test_files must exist on disk | file existence check |
| V10 | Cross-RQ overlap consistent with serves_rqs | manual: serves_rqs ⊆ {1,2,3} |

```python
"""
validate_invariant_manifest.py
Validates config/invariants_manifest.yaml against 10 structural rules.

Writes results/rq3_invariant_manifest_validation.json.
Exit code 1 if any check fails (CI-blocking).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import re

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "config/invariants_manifest.yaml"
OUT = REPO_ROOT / "results/rq3_invariant_manifest_validation.json"

VALID_SEVERITY = {"safety_critical", "quality"}
VALID_STATUS = {"enforced", "pending", "documented"}
VALID_RQS = {1, 2, 3}


def main():
    findings = []

    # V1: parse
    try:
        doc = yaml.safe_load(MANIFEST.read_text())
        findings.append({"check_id": "V1", "severity": "PASS",
                         "description": "Manifest parsed"})
    except Exception as e:
        findings.append({"check_id": "V1", "severity": "FAIL",
                         "description": "Manifest failed to parse",
                         "details": {"error": str(e)}})
        _finalize(findings, [])
        sys.exit(1)

    # V2: preregistered_date
    prd = doc.get("preregistered_date")
    findings.append({
        "check_id": "V2",
        "severity": "PASS" if prd else "FAIL",
        "description": "preregistered_date present",
        "details": {"value": prd},
    })

    # V3: exactly 9 invariants
    invs = doc.get("invariants", [])
    findings.append({
        "check_id": "V3",
        "severity": "PASS" if len(invs) == 9 else "FAIL",
        "description": "Exactly 9 invariants",
        "details": {"count": len(invs)},
    })

    # V4: IDs unique 1-9
    ids = [inv.get("id") for inv in invs]
    findings.append({
        "check_id": "V4",
        "severity": "PASS" if set(ids) == {1, 2, 3, 4, 5, 6, 7, 8, 9} else "FAIL",
        "description": "IDs unique and complete 1-9",
        "details": {"ids": ids},
    })

    # V5-V10: per-invariant
    for inv in invs:
        inv_id = inv.get("id")
        prefix = f"Inv {inv_id}"

        # V5: title + text + rationale
        missing = [f for f in ["title", "text", "rationale"]
                   if not (inv.get(f) or "").strip()]
        findings.append({
            "check_id": f"V5-{inv_id}",
            "severity": "PASS" if not missing else "FAIL",
            "description": f"{prefix} has title/text/rationale",
            "details": {"missing_fields": missing},
        })

        # V6: at least one test_file or grep_audit
        has_test = bool(inv.get("test_files"))
        has_grep = bool(inv.get("grep_audit"))
        findings.append({
            "check_id": f"V6-{inv_id}",
            "severity": "PASS" if (has_test or has_grep) else "FAIL",
            "description": f"{prefix} has at least one test_file or grep_audit",
            "details": {"test_files": inv.get("test_files"),
                        "has_grep_audit": has_grep},
        })

        # V7: severity
        sev = inv.get("severity")
        findings.append({
            "check_id": f"V7-{inv_id}",
            "severity": "PASS" if sev in VALID_SEVERITY else "FAIL",
            "description": f"{prefix} severity valid",
            "details": {"value": sev},
        })

        # V8: status
        status = inv.get("status")
        findings.append({
            "check_id": f"V8-{inv_id}",
            "severity": "PASS" if status in VALID_STATUS else "FAIL",
            "description": f"{prefix} status valid",
            "details": {"value": status},
        })

        # V9: enforced → test files exist
        if status == "enforced":
            missing_files = [
                tf for tf in inv.get("test_files", [])
                if not (REPO_ROOT / tf).exists()
            ]
            findings.append({
                "check_id": f"V9-{inv_id}",
                "severity": "PASS" if not missing_files else "FAIL",
                "description": f"{prefix} test files exist (enforced)",
                "details": {"missing_files": missing_files},
            })

        # V10: serves_rqs subset of {1,2,3}
        rqs = set(inv.get("serves_rqs", []))
        findings.append({
            "check_id": f"V10-{inv_id}",
            "severity": "PASS" if rqs.issubset(VALID_RQS) else "FAIL",
            "description": f"{prefix} serves_rqs valid",
            "details": {"value": sorted(rqs)},
        })

    _finalize(findings, invs)


def _finalize(findings, invs):
    n_fail = sum(1 for f in findings if f["severity"] == "FAIL")
    audit = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/validate_invariant_manifest.py",
            "manifest_path": str(MANIFEST.relative_to(REPO_ROOT)),
        },
        "headline": {
            "validation_pass": n_fail == 0,
            "n_invariants": len(invs),
            "n_checks": len(findings),
            "n_fail": n_fail,
        },
        "findings": findings,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2, default=str))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"Validation: {'PASS' if n_fail == 0 else f'FAIL ({n_fail} checks)'}")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 5.2 Verification

```bash
python -m analysis.validate_invariant_manifest
# Expected: exits 0 if manifest is structurally valid
cat results/rq3_invariant_manifest_validation.json | python -m json.tool | head -20
```

---

## 6. Phase 3 — Evidence aggregator

### 6.1 Create `analysis/compile_invariant_evidence.py`

**Contract:**
- **Inputs:** `config/invariants_manifest.yaml` + `tests/_report.json` (pytest-json-report output)
- **Outputs:** `results/rq3_invariant_evidence.json`, `results/rq3_invariant_evidence.md`
- **Runtime:** sub-second after pytest completes.
- **CI usage:** typically run *after* `pytest --json-report --json-report-file=tests/_report.json`.

### 6.2 Output schema

```json
{
  "_meta": {
    "schema_version": "1.0",
    "generated_at": "<ISO-8601>",
    "generated_by": "analysis/compile_invariant_evidence.py",
    "manifest_path": "config/invariants_manifest.yaml",
    "preregistered_date": "2025-08-14",
    "pytest_report_path": "tests/_report.json",
    "pytest_run_at": "<ISO-8601>"
  },
  "headline": {
    "all_invariants_pass": false,
    "n_invariants_total": 9,
    "n_enforced": 4,
    "n_pending": 5,
    "n_documented": 0,
    "n_failed": 1,
    "_overall_status": "1 invariant failing — see invariants block"
  },
  "invariants": [
    {
      "id": 1,
      "title": "DAE only elevates detection confidence; never suppresses",
      "severity": "safety_critical",
      "serves_rqs": [1],
      "status_manifest": "enforced",
      "verification_method": "pytest",
      "test_results": {
        "test_files": ["tests/test_step9_composite_risk.py"],
        "n_tests_total": 5,
        "n_tests_passed": 5,
        "n_tests_failed": 0,
        "n_tests_skipped": 0,
        "outcome": "pass"
      },
      "_overall_status": "pass"
    },
    {
      "id": 3,
      "title": "No auto-execution — recommendation only",
      "verification_method": "grep_and_pytest",
      "grep_results": {
        "target_dirs": ["module5_responses/"],
        "pattern_matches": [],
        "import_matches": [],
        "outcome": "pass"
      },
      "test_results": {
        "test_files": ["tests/negative_tests.py"],
        "outcome": "pass"
      },
      "_overall_status": "pass"
    },
    {
      "id": 4,
      "title": "Audit trail complete",
      "status_manifest": "pending",
      "test_results": null,
      "_overall_status": "pending",
      "_note": "Test created by RQ3 Track 2 — Audit Integrity"
    }
  ]
}
```

### 6.3 Implementation outline

```python
"""
compile_invariant_evidence.py
Aggregates pytest results + grep checks for each invariant in the manifest.

Inputs:
  config/invariants_manifest.yaml
  tests/_report.json     (from `pytest --json-report --json-report-file=tests/_report.json`)

Outputs:
  results/rq3_invariant_evidence.json
  results/rq3_invariant_evidence.md
"""

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "config/invariants_manifest.yaml"
PYTEST_REPORT = REPO_ROOT / "tests/_report.json"
OUT_JSON = REPO_ROOT / "results/rq3_invariant_evidence.json"
OUT_MD = REPO_ROOT / "results/rq3_invariant_evidence.md"


def _load_manifest():
    if not MANIFEST.exists():
        raise SystemExit(f"Manifest missing: {MANIFEST}. Run Phase 1.")
    return yaml.safe_load(MANIFEST.read_text())


def _load_pytest_report():
    """Load pytest-json-report output. Return None if unavailable."""
    if not PYTEST_REPORT.exists():
        return None
    try:
        return json.loads(PYTEST_REPORT.read_text())
    except json.JSONDecodeError:
        return None


def _pytest_results_for_file(report, test_file):
    """
    Return tests from pytest report whose nodeid starts with test_file.
    pytest-json-report schema: report['tests'] is a list of dicts with
    'nodeid' (e.g., 'tests/test_x.py::TestClass::test_method') and 'outcome'.
    """
    if not report:
        return None
    matches = [
        t for t in report.get("tests", [])
        if t.get("nodeid", "").startswith(test_file)
    ]
    return matches


def _aggregate_test_results(report, test_files):
    """Aggregate pytest results across one invariant's test files."""
    if not report:
        return {
            "test_files": test_files,
            "n_tests_total": 0,
            "n_tests_passed": 0,
            "n_tests_failed": 0,
            "n_tests_skipped": 0,
            "outcome": "no_report",
        }

    all_tests = []
    for tf in test_files:
        tests = _pytest_results_for_file(report, tf) or []
        all_tests.extend(tests)

    n_total = len(all_tests)
    n_passed = sum(1 for t in all_tests if t.get("outcome") == "passed")
    n_failed = sum(1 for t in all_tests if t.get("outcome") == "failed")
    n_skipped = sum(1 for t in all_tests if t.get("outcome") == "skipped")

    if n_total == 0:
        outcome = "no_tests_found"
    elif n_failed > 0:
        outcome = "fail"
    elif n_passed == n_total:
        outcome = "pass"
    elif n_passed > 0 and n_skipped == n_total - n_passed:
        outcome = "partial_skip"
    else:
        outcome = "unknown"

    return {
        "test_files": test_files,
        "n_tests_total": n_total,
        "n_tests_passed": n_passed,
        "n_tests_failed": n_failed,
        "n_tests_skipped": n_skipped,
        "outcome": outcome,
    }


def _run_grep_audit(grep_audit):
    """
    Run the grep audit for Invariant 3 (or any invariant using grep_audit).
    Returns dict with pattern_matches, import_matches, outcome.
    """
    if not grep_audit:
        return None

    target_dirs = [
        REPO_ROOT / d for d in grep_audit.get("target_dirs", [])
        if (REPO_ROOT / d).exists()
    ]
    if not target_dirs:
        return {
            "target_dirs": [str(d) for d in grep_audit.get("target_dirs", [])],
            "pattern_matches": [],
            "import_matches": [],
            "outcome": "no_target_dirs_exist",
        }

    pattern_matches = []
    for pat in grep_audit.get("forbidden_patterns", []):
        for d in target_dirs:
            result = subprocess.run(
                ["grep", "-rnE", "--include=*.py", pat, str(d)],
                capture_output=True, text=True,
            )
            if result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    pattern_matches.append({"pattern": pat, "match": line[:200]})

    import_matches = []
    for pat in grep_audit.get("forbidden_imports", []):
        for d in target_dirs:
            result = subprocess.run(
                ["grep", "-rnE", "--include=*.py", pat, str(d)],
                capture_output=True, text=True,
            )
            if result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    import_matches.append({"pattern": pat, "match": line[:200]})

    outcome = "pass" if not (pattern_matches or import_matches) else "fail"

    return {
        "target_dirs": [str(d.relative_to(REPO_ROOT)) for d in target_dirs],
        "pattern_matches": pattern_matches[:20],  # truncate for readability
        "import_matches": import_matches[:20],
        "n_pattern_matches": len(pattern_matches),
        "n_import_matches": len(import_matches),
        "outcome": outcome,
    }


def _determine_overall_status(inv, test_results, grep_results):
    """Combine pytest + grep outcomes into an overall status."""
    manifest_status = inv.get("status", "enforced")

    if manifest_status == "pending":
        return "pending"
    if manifest_status == "documented":
        return "documented"

    # enforced
    method = inv.get("verification_method", "pytest")
    if method == "pytest":
        return "pass" if (test_results and test_results["outcome"] == "pass") else "fail"
    if method == "grep_and_pytest":
        pytest_ok = (test_results and test_results["outcome"] == "pass")
        grep_ok = (grep_results and grep_results["outcome"] == "pass")
        return "pass" if (pytest_ok and grep_ok) else "fail"
    return "unknown"


def main():
    doc = _load_manifest()
    report = _load_pytest_report()

    invariant_outputs = []
    for inv in doc.get("invariants", []):
        test_results = None
        grep_results = None
        manifest_status = inv.get("status", "enforced")

        if manifest_status != "pending":
            if inv.get("test_files"):
                test_results = _aggregate_test_results(report, inv["test_files"])
            if inv.get("grep_audit"):
                grep_results = _run_grep_audit(inv["grep_audit"])

        overall = _determine_overall_status(inv, test_results, grep_results)

        entry = {
            "id": inv["id"],
            "title": inv["title"],
            "severity": inv.get("severity"),
            "serves_rqs": inv.get("serves_rqs", []),
            "status_manifest": manifest_status,
            "verification_method": inv.get("verification_method"),
            "test_results": test_results,
            "grep_results": grep_results,
            "_overall_status": overall,
        }
        if manifest_status == "pending":
            entry["_note"] = inv.get("_note") or "Test creation scheduled"
        invariant_outputs.append(entry)

    n_total = len(invariant_outputs)
    n_enforced = sum(1 for i in invariant_outputs if i["status_manifest"] == "enforced")
    n_pending = sum(1 for i in invariant_outputs if i["status_manifest"] == "pending")
    n_documented = sum(1 for i in invariant_outputs
                       if i["status_manifest"] == "documented")
    n_failed = sum(1 for i in invariant_outputs if i["_overall_status"] == "fail")
    all_pass = (n_failed == 0) and all(
        i["_overall_status"] in {"pass", "pending", "documented"}
        for i in invariant_outputs
    )

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/compile_invariant_evidence.py",
            "manifest_path": str(MANIFEST.relative_to(REPO_ROOT)),
            "preregistered_date": doc.get("preregistered_date"),
            "pytest_report_path": str(PYTEST_REPORT.relative_to(REPO_ROOT)),
            "pytest_report_available": report is not None,
        },
        "headline": {
            "all_invariants_pass": all_pass,
            "n_invariants_total": n_total,
            "n_enforced": n_enforced,
            "n_pending": n_pending,
            "n_documented": n_documented,
            "n_failed": n_failed,
            "_overall_status": (
                "all enforced invariants pass"
                if all_pass else
                f"{n_failed} invariant(s) failing"
            ),
        },
        "invariants": invariant_outputs,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Headline: {out['headline']['_overall_status']}")
    for i in invariant_outputs:
        marker = {
            "pass": "✓", "fail": "✗",
            "pending": "○", "documented": "—",
            "no_tests_found": "?",
        }.get(i["_overall_status"], "?")
        print(f"  {marker} #{i['id']:<2} [{i['status_manifest']:<10}] {i['title']}")


if __name__ == "__main__":
    main()
```

### 6.4 Verification

```bash
# 1. Run pytest with the JSON reporter
pytest --json-report --json-report-file=tests/_report.json
# 2. Run the aggregator
python -m analysis.compile_invariant_evidence
# Inspect output
cat results/rq3_invariant_evidence.json | python -m json.tool | head -30
```

---

## 7. Phase 4 — Markdown renderer

### 7.1 Create `analysis/render_invariant_evidence_markdown.py`

**Purpose:** paper-ready markdown for §5.6 (Safety Engineering) of the thesis.

```python
"""
render_invariant_evidence_markdown.py
Render results/rq3_invariant_evidence.json into a paper-ready markdown table.

Writes results/rq3_invariant_evidence.md.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = REPO_ROOT / "results/rq3_invariant_evidence.json"
OUT_MD = REPO_ROOT / "results/rq3_invariant_evidence.md"


def main():
    if not JSON_PATH.exists():
        raise SystemExit(
            f"{JSON_PATH} missing — run analysis/compile_invariant_evidence.py first"
        )
    data = json.loads(JSON_PATH.read_text())

    lines = []
    lines.append("# RQ3 — Architectural Invariant Evidence")
    lines.append("")
    lines.append(f"*Generated by `analysis/compile_invariant_evidence.py` "
                 f"on {data['_meta']['generated_at']}.*")
    lines.append(f"*Manifest pre-registered: "
                 f"{data['_meta'].get('preregistered_date', 'unknown')}.*")
    lines.append("")

    h = data["headline"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Status:** {h['_overall_status']}")
    lines.append(f"- **Enforced (live tests):** {h['n_enforced']}/9")
    lines.append(f"- **Pending (tests scheduled):** {h['n_pending']}/9")
    lines.append(f"- **Documented (no test required):** {h['n_documented']}/9")
    lines.append(f"- **Failed:** {h['n_failed']}/9")
    lines.append("")

    # Summary table
    lines.append("## Invariant Status Table")
    lines.append("")
    lines.append("| # | Title | Severity | Serves RQ | Status | Verification |")
    lines.append("|---|---|---|---|---|---|")
    marker = {
        "pass": "✓ PASS", "fail": "✗ FAIL",
        "pending": "○ pending", "documented": "— documented",
        "no_tests_found": "? no tests",
    }
    for inv in data["invariants"]:
        rqs = ", ".join(f"RQ{r}" for r in inv["serves_rqs"])
        lines.append(
            f"| {inv['id']} | {inv['title']} | {inv['severity']} | {rqs} | "
            f"{marker.get(inv['_overall_status'], inv['_overall_status'])} | "
            f"{inv.get('verification_method', '—')} |"
        )
    lines.append("")

    # Per-invariant detail
    lines.append("## Per-Invariant Detail")
    lines.append("")
    for inv in data["invariants"]:
        lines.append(f"### Invariant {inv['id']} — {inv['title']}")
        lines.append("")
        lines.append(f"- **Severity:** {inv['severity']}")
        lines.append(f"- **Serves:** {', '.join(f'RQ{r}' for r in inv['serves_rqs'])}")
        lines.append(f"- **Status:** "
                     f"{marker.get(inv['_overall_status'], inv['_overall_status'])}")
        lines.append(f"- **Verification:** {inv.get('verification_method', '—')}")
        lines.append("")

        if inv.get("test_results"):
            tr = inv["test_results"]
            lines.append(f"**Pytest results:** {tr['n_tests_passed']}/"
                         f"{tr['n_tests_total']} passed across "
                         f"{len(tr['test_files'])} file(s)")
            for tf in tr["test_files"]:
                lines.append(f"  - `{tf}`")
            lines.append("")

        if inv.get("grep_results"):
            gr = inv["grep_results"]
            lines.append(f"**Grep audit:** {gr['outcome']}")
            lines.append(f"  - target dirs: {gr['target_dirs']}")
            lines.append(f"  - forbidden pattern matches: {gr['n_pattern_matches']}")
            lines.append(f"  - forbidden import matches: {gr['n_import_matches']}")
            lines.append("")

        if inv.get("_note"):
            lines.append(f"*Note: {inv['_note']}*")
            lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

### 7.2 Verification

```bash
python -m analysis.render_invariant_evidence_markdown
head -40 results/rq3_invariant_evidence.md
```

---

## 8. Phase 5 — CI strict gate

### 8.1 Add to `tests/acceptance_tests.py`

```python
def test_invariant_manifest_valid():
    """STRICT-GATE: manifest must validate structurally."""
    import json
    from pathlib import Path

    p = Path("results/rq3_invariant_manifest_validation.json")
    if not p.exists():
        import subprocess
        subprocess.run(
            ["python", "-m", "analysis.validate_invariant_manifest"],
            check=True
        )
    audit = json.loads(p.read_text())
    assert audit["headline"]["validation_pass"], (
        f"Invariant manifest validation failed: "
        f"{audit['headline']['n_fail']} check(s). "
        f"See {p} for details."
    )


def test_invariant_evidence_complete():
    """
    STRICT-GATE: all enforced invariants must pass their tests.
    Pending and documented invariants are acceptable but reported.
    """
    import json
    from pathlib import Path

    p = Path("results/rq3_invariant_evidence.json")
    if not p.exists():
        import pytest
        pytest.skip(
            "Run: pytest --json-report --json-report-file=tests/_report.json "
            "&& python -m analysis.compile_invariant_evidence"
        )

    data = json.loads(p.read_text())
    failures = [
        i for i in data["invariants"]
        if i["_overall_status"] == "fail"
    ]
    assert not failures, (
        f"{len(failures)} invariant(s) failing: "
        f"{[(i['id'], i['title']) for i in failures]}"
    )


def test_no_orphan_invariants():
    """
    STRICT-GATE: every invariant must be either enforced/pending/documented.
    No 'unknown' or 'no_tests_found' acceptable for safety_critical.
    """
    import json
    from pathlib import Path

    p = Path("results/rq3_invariant_evidence.json")
    if not p.exists():
        import pytest
        pytest.skip("Run compile_invariant_evidence first")

    data = json.loads(p.read_text())
    orphans = [
        i for i in data["invariants"]
        if i["severity"] == "safety_critical"
        and i["_overall_status"] in {"unknown", "no_tests_found"}
    ]
    assert not orphans, (
        f"{len(orphans)} safety-critical invariant(s) with no test results: "
        f"{[i['id'] for i in orphans]}"
    )
```

### 8.2 Verification

```bash
pytest tests/acceptance_tests.py::test_invariant_manifest_valid -v
pytest tests/acceptance_tests.py::test_invariant_evidence_complete -v
pytest tests/acceptance_tests.py::test_no_orphan_invariants -v
```

---

## 9. Execution order

```bash
# ─── PHASE 0: TEST INVENTORY DISCOVERY ─────────────────────────
python scripts/discover_invariant_tests.py > /tmp/invariant_inventory.json
# Developer reviews: which tests exist? Grep targets? pytest-json-report installed?
pip install pytest-json-report   # if not yet installed

# ─── PHASE 1: MANIFEST CREATION ────────────────────────────────
# Create config/invariants_manifest.yaml using template from §4.1
# Set preregistered_date to actual lock date
# Set status: enforced | pending | documented per Phase 0 findings

# ─── PHASE 2: MANIFEST VALIDATION ──────────────────────────────
python -m analysis.validate_invariant_manifest
# Expected: exits 0; results/rq3_invariant_manifest_validation.json written
cat results/rq3_invariant_manifest_validation.json | python -m json.tool | head -20

# ─── PHASE 3: EVIDENCE AGGREGATOR ──────────────────────────────
# Run pytest with JSON reporter first
pytest --json-report --json-report-file=tests/_report.json
# Then aggregate
python -m analysis.compile_invariant_evidence
cat results/rq3_invariant_evidence.json | python -m json.tool | head -40

# ─── PHASE 4: MARKDOWN RENDER ──────────────────────────────────
python -m analysis.render_invariant_evidence_markdown
head -40 results/rq3_invariant_evidence.md

# ─── PHASE 5: CI STRICT GATE ───────────────────────────────────
# Add three tests to tests/acceptance_tests.py per §8.1
pytest tests/acceptance_tests.py -k invariant -v

# ─── FINAL VERIFICATION ────────────────────────────────────────
ls config/invariants_manifest.yaml \
   results/rq3_invariant_manifest_validation.json \
   results/rq3_invariant_evidence.json \
   results/rq3_invariant_evidence.md
```

---

## 10. Integration with `compute_rq3_metrics.py`

When the Phase 6 (RQ3 merge) spec is written, it will fold Track 1 output in:

```python
def _load_invariant_evidence_subfile():
    p = REPO_ROOT / "results/rq3_invariant_evidence.json"
    if not p.exists():
        return {"_status": "pending"}
    evidence = json.loads(p.read_text())
    return {
        "_status": "complete" if evidence["headline"]["all_invariants_pass"]
                   else "partial",
        "_merged_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": "config/invariants_manifest.yaml",
        "evidence_json_path": "results/rq3_invariant_evidence.json",
        "evidence_md_path": "results/rq3_invariant_evidence.md",
        "headline": evidence["headline"],
        "invariants": [
            {k: v for k, v in inv.items()
             if k in {"id", "title", "severity", "serves_rqs",
                      "_overall_status", "status_manifest"}}
            for inv in evidence["invariants"]
        ],
    }
```

The aggregator only carries headline + per-invariant status — not full test results. Full results stay in `rq3_invariant_evidence.json` for paper rendering.

---

## 11. Open questions to surface (DO NOT GUESS)

Claude Code must pause and ask:

1. **Phase 0 — test file existence.** Which of the 9 expected test files actually exist today? `_overall_status: pending` per invariant maps to "test scheduled by another spec." Confirm which RQ2/RQ3 specs create each missing test.
2. **Phase 0 — grep target directory.** ARCHITECTURE.md says `module5_responses/`; `RQ3_expected_outputs.md §3.1` says `pipeline/module5_response/`. Which is correct?
3. **Phase 0 — pytest-json-report installation.** Is it already in `requirements.txt` / `pyproject.toml`? If not, add it.
4. **Phase 1 — pre-registration date.** When were the 9 invariants formally locked? The manifest's `preregistered_date` must be truthful and predate evaluation work.
5. **Phase 3 — grep audit Python version.** The `_run_grep_audit` function uses subprocess + system `grep`. If portability matters (Windows dev environment, etc.), consider using Python `re` instead. Recommend system grep for performance.

---

## 12. Coverage map — RQ3 §1.1 → pipeline phase

| RQ3_expected_outputs.md §1.1 row | Manifest invariant | Phase |
|---|---|---|
| Role-based explanation routing | 6 | 3 (referenced from RQ2 Track 3) |
| Tier recommendation routing | 2 | 3 |
| Action authorization per role | 6 | 3 |
| No auto-execution | 3 | 3 (grep + pytest) |
| Audit trail per role | 4 | 3 (pending, created by RQ3 Track 2) |
| Cross-role severity invariance | 6 | 3 |
| Shared anchor across roles | 9 | 3 |

Every numbered row in §1.1 has an invariant. The full 9-invariant catalog from `RQ3_expected_outputs.md §4.1` is similarly traceable.

---

## 13. Defense talking points this enables

- **"How do you prove every architectural invariant holds?"**
  *"The invariant manifest `config/invariants_manifest.yaml` lists all 9 invariants with their pre-registration date, enforcement code, and test files. The aggregator at `analysis/compile_invariant_evidence.py` runs every invariant test and produces `results/rq3_invariant_evidence.json` showing pass/fail per invariant. CI fails on any safety-critical failure."*

- **"How do you handle invariants where no pytest test exists (e.g., no-auto-execution)?"**
  *"Invariant 3 uses `verification_method: grep_and_pytest` — both a code grep (against `forbidden_patterns`) AND a pytest test (`negative_tests.py::test_no_automated_blocking`). The manifest declares the grep target directories and patterns explicitly. Both must pass for the invariant to count as enforced."*

- **"What if you change an invariant after collecting evaluation data?"**
  *"The manifest carries `preregistered_date`. CI test `test_invariant_manifest_valid` asserts the date is set; reviewers can verify this predates evaluation. Edits post-date are version-controlled and visible. The invariant text itself is locked."*

- **"What's the difference between 'enforced', 'pending', and 'documented'?"**
  *"Enforced = there is a live test and it passes. Pending = the test is scheduled (typically created by another RQ track). Documented = the invariant is enforced by code or config but doesn't have a pytest test (e.g., Invariant 3 has a grep audit instead). CI only hard-fails on enforced invariants failing — pending and documented don't gate the build."*

---

## 14. What this track deliberately does NOT do

- **Write new invariant tests.** It assumes tests are created by other RQ specs and that Phase 0 inventory tracks their status.
- **Modify pytest configuration.** Adds pytest-json-report as a dependency but doesn't change how pytest itself runs.
- **Aggregate non-invariant test results.** Other tests run, but only those tagged in the manifest contribute to invariant evidence.

---

## End of spec

Implementation order: Phase 0 (discovery) → Phase 1 (manifest) → Phase 2 (validator) → Phase 3 (aggregator) → Phase 4 (renderer) → Phase 5 (CI gate). Phases 2-5 each depend on the previous. Track 1 is the architectural backbone of RQ3; once complete, the catalog is the single defense reference for all 9 invariant claims.