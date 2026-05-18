# RQ2 Compliance & Cross-Role Pipeline — PHI Test + Word Budget + Cross-Role + Compliance Map

**Project:** XAI-IDS-Healthcare
**Scope:** RQ2.a — Does MVE satisfy formal explainability requirements? Plus the HIPAA/compliance-adjacent guarantees that any defense reviewer will ask about first.
**Purpose:** Single, self-contained spec for the four Track 3 components: (1) PHI flow control test, (2) word budget audit, (3) cross-role consistency test, (4) compliance mapping manifest. Hand to Claude Code.
**Status of design:** All decisions locked. Four `DO NOT GUESS` checkpoints (LLM data flow YAML, audit log location, MVE generator signature, role enum names).

---

## 0. How to use this spec

1. Implement Phase 0 first — schema discovery for `config/llm_data_flow.yaml` and Mode A audit logs.
2. Phases 1–4 are mostly independent and can be implemented in any order after Phase 0.
3. Each phase has a `verification` command. Do not proceed if it fails.
4. Markers:
   - **DO NOT GUESS** — stop and ask the developer
   - **DEFENSE-CRITICAL** — this check is what a HIPAA-aware reviewer will look at first
   - **TARGET** — a numeric goal from `RQ2_expected_outputs.md`
5. Four scripts + four test files produced. All are fast (sub-second to seconds); none requires model inference.

---

## 1. Background: what Track 3 establishes

| Component | Question | Output | Defense weight |
|---|---|---|---|
| **PHI flow control** | Does any PHI ever reach Mode A's LLM API? | Test result + audit-log scan report | DEFENSE-CRITICAL — highest stakes in RQ2 |
| **Word budget audit** | Do MVE outputs respect the 150-word total budget? | Pass/fail JSON per-layer per-output | Architecture contract |
| **Cross-role consistency** | Do roles share anchor + severity (Inv 6+9) while differing in actions? | Pass/fail test asserting same anchor, same Layer 2, different Layer 3 | Architecture invariants |
| **Compliance mapping** | Does every literature requirement have a code+test evidence trail? | Generated markdown + manifest validation test | Paper appendix material |

---

## 2. Locked design decisions

| Decision | Resolution |
|---|---|
| PHI test approach | Live (synthetic alerts through Mode A, mocked API, prompt scanned) + audit-log scan (existing entries) |
| PHI definition source | `config/llm_data_flow.yaml` — single source of truth |
| Mode coverage | Mode A: PHI content check. Mode B: no-external-calls assertion (different invariant). |
| Word budget criterion | Hard fail on any violation (per-layer AND total) |
| Word count method | `len(text.split())` — whitespace tokens |
| Cross-role assertions | (1) shared anchor identical, (2) Layer 2 severity identical, (3) Layer 3 actions DIFFERENT |
| Compliance mapping artifact | YAML manifest → generated markdown, with test verifying evidence files exist |
| Word-budget escape hatch | Post-generation truncation in `src/mve_generator.py` ensures user-facing output respects budget regardless of upstream variance |

---

## 3. Phase 0 — Schema discovery (DO NOT GUESS)

Before any test code, Claude Code must locate and inspect three artifacts. The spec's Phases 1–4 each depend on at least one.

### 3.1 Discovery script

```python
# scripts/discover_compliance_artifacts.py — TRANSIENT, delete after Phase 0
"""
Locate and summarize the compliance artifacts Track 3 depends on:
  1. config/llm_data_flow.yaml — PHI definition source
  2. Mode A audit log location and schema — historical PHI scan target
  3. src/mve_generator.py signatures — for live PHI test + cross-role test
  4. Role enum values — for cross-role test
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
findings = {}

# 1. LLM data flow YAML
yaml_path = REPO_ROOT / "config/llm_data_flow.yaml"
findings["llm_data_flow"] = {
    "expected_path": str(yaml_path.relative_to(REPO_ROOT)),
    "exists": yaml_path.exists(),
}
if yaml_path.exists():
    import yaml
    try:
        doc = yaml.safe_load(yaml_path.read_text())
        findings["llm_data_flow"]["top_level_keys"] = list(doc.keys()) \
            if isinstance(doc, dict) else "NOT A DICT"
        findings["llm_data_flow"]["sample"] = (
            {k: doc[k] for k in list(doc.keys())[:3]} if isinstance(doc, dict) else None
        )
    except Exception as e:
        findings["llm_data_flow"]["parse_error"] = str(e)

# 2. Mode A audit log
audit_candidates = [
    "logs/llm_audit.jsonl",
    "logs/mode_a_audit.jsonl",
    "results/llm_audit_log.jsonl",
    "audit/mode_a.jsonl",
    "results/reports/alert_responses.json",
]
for p in audit_candidates:
    full = REPO_ROOT / p
    if full.exists():
        findings["audit_log"] = {
            "path": p,
            "size_bytes": full.stat().st_size,
        }
        # Sample first record to show schema
        try:
            with open(full) as f:
                first_line = f.readline()
            findings["audit_log"]["sample_record_keys"] = list(
                json.loads(first_line).keys()
            ) if first_line.strip() else "empty"
        except Exception as e:
            findings["audit_log"]["sample_error"] = str(e)
        break
else:
    findings["audit_log"] = {"status": "NOT FOUND — confirm location with developer"}

# 3. mve_generator inspection
mve_gen = REPO_ROOT / "src/mve_generator.py"
findings["mve_generator"] = {
    "path": "src/mve_generator.py",
    "exists": mve_gen.exists(),
}
if mve_gen.exists():
    text = mve_gen.read_text()
    for sym in ["generate_mve", "derive_role_view", "OperatorRole",
                "MVEOutput", "Mode", "build_prompt", "_build_prompt"]:
        findings["mve_generator"][f"has_{sym}"] = sym in text

# 4. Role enum (look in src or common)
for p in ["src/mve_generator.py", "common/roles.py", "src/roles.py"]:
    full = REPO_ROOT / p
    if full.exists() and "OperatorRole" in full.read_text():
        findings["role_enum_file"] = p
        break
else:
    findings["role_enum_file"] = "NOT FOUND — confirm where OperatorRole lives"

# 5. role_action_authorization.yaml
auth_path = REPO_ROOT / "config/role_action_authorization.yaml"
findings["role_authorization"] = {
    "expected_path": "config/role_action_authorization.yaml",
    "exists": auth_path.exists(),
}

print(json.dumps(findings, indent=2, default=str))
print("\n" + "="*60)
print("Confirm with developer before proceeding to Phase 1:")
print("  1. llm_data_flow.yaml exists and has fields named appropriately")
print("  2. Mode A audit log path + record schema")
print("  3. mve_generator.py exports the functions/enum we need")
print("  4. OperatorRole enum values (IT_GENERALIST / BIOMED_ENGINEER / NURSE_MANAGER?)")
print("="*60)
```

### 3.2 What to confirm before proceeding

After running the discovery script, Claude Code must confirm four things with the developer:

1. **`config/llm_data_flow.yaml` exists** and contains either an `allowed_fields` allowlist OR a `forbidden_fields` blocklist (or both). The PHI test uses this as the source of truth. If the file doesn't exist, Phase 1 includes creating a minimal version (see §4.3 below).
2. **Mode A audit log location and schema.** Expected fields per `RQ3_expected_outputs.md`: `llm_provider`, `llm_model_version`, `full_prompt`, `full_response`. The historical-scan test reads from this. If no audit log exists yet (no past Mode A runs), the historical scan is a no-op until logs accumulate.
3. **`src/mve_generator.py` signatures.** Specifically: how `generate_mve` is invoked (what arguments?), how Mode A vs Mode B is selected (parameter? config flag? auto-fallback?), and how the LLM client is injected (so the test can mock it).
4. **`OperatorRole` enum values.** Per ARCHITECTURE.md the three roles are IT Generalist / Biomed Engineer / Nurse Manager. Confirm the exact enum identifier strings.

### 3.3 Verification

```bash
python scripts/discover_compliance_artifacts.py > /tmp/discovery.json
# Developer reviews; confirms or fills gaps
```

**DO NOT GUESS** any of the four artifacts. Phases 1–4 each fail differently and silently if these are wrong.

---

## 4. Phase 1 — PHI flow control (DEFENSE-CRITICAL)

This is the highest-stakes test in all of RQ2. It must demonstrate two things:

- **Current code is safe:** synthetic alerts go through Mode A's prompt construction; the prompt contains no PHI.
- **Historical record is clean:** prior Mode A audit-log entries contain no PHI.

Plus, a separate test for Mode B: no external network calls of any kind.

### 4.1 Create `config/llm_data_flow.yaml` (if missing)

If Phase 0 discovery found this file, skip this step. If not, create a minimal version:

```yaml
# config/llm_data_flow.yaml
# Single source of truth for what data may cross the Mode A LLM boundary.
# Read by tests/test_phi_not_in_llm_prompt.py to verify compliance.

mitre_framework_version: v14.1   # not used here but kept for consistency
last_validated: "2025-08-14"

# Fields that ARE permitted in Mode A prompts.
# Anything not in this list is forbidden by default.
allowed_fields:
  - alert_id              # opaque identifier, no PHI
  - device_class          # category (e.g., "infusion_pump"), not device serial
  - device_criticality    # CRITICAL / HIGH / MEDIUM / LOW
  - attack_category       # category label only
  - risk_tier             # CRITICAL / HIGH / MEDIUM / LOW
  - fusion_class          # KNOWN_ATTACK / etc.
  - shap_top_features     # raw feature names (e.g., "fwd_pkts_tot")
  - shap_top_values       # numeric SHAP magnitudes
  - mitre_technique_id    # T1565, etc.
  - mitre_technique_name  # "Data Manipulation", etc.
  - timestamp_iso         # alert timestamp; not patient encounter timestamp

# Field names that are EXPLICITLY forbidden, even if they appear by accident.
# Used by the historical audit-log scan.
forbidden_field_names:
  - patient_name
  - patient_id
  - mrn               # medical record number
  - dob               # date of birth
  - bed_number
  - room_number
  - encounter_id
  - admission_date
  - clinician_name
  - clinician_id
  - device_serial     # unlike device_class, serial number is uniquely identifying
  - device_hostname   # may encode location/department
  - src_ip            # network identifier with potential location inference
  - dst_ip
  - mac_address

# Regex patterns for content-shape detection
# (catches PHI that arrives via free-text fields).
forbidden_patterns:
  - name: ssn
    regex: '\b\d{3}-\d{2}-\d{4}\b'
  - name: mrn_numeric
    regex: '\b(?:MRN|mrn)[\s:]*\d{6,10}\b'
  - name: phone_us
    regex: '\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
  - name: email
    regex: '\b[\w.+-]+@[\w-]+\.[\w.-]+\b'
  - name: dob_iso
    regex: '\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b'
  - name: bed_pattern
    regex: '\b(?:Bed|bed|Room|room)[\s:]*\d+[-\d]*\b'
```

**DO NOT GUESS** the contents of `allowed_fields` — these must reflect what `src/mve_generator.py` *actually* sends. If Phase 0 discovery already located an existing `llm_data_flow.yaml`, use that; do not overwrite.

### 4.2 Create `tests/test_phi_not_in_llm_prompt.py`

```python
"""
DEFENSE-CRITICAL: verifies Mode A's LLM prompt never contains PHI.

Two complementary checks:
  1. Live prompt scan: synthesize a realistic alert dict containing PHI markers,
     run it through Mode A prompt construction with a mocked LLM client,
     assert no forbidden field name or pattern appears in the captured prompt.
  2. Historical audit-log scan: read past Mode A audit log entries,
     scan full_prompt + full_response for forbidden patterns.

Plus a Mode B no-external-calls check (different invariant).
"""

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FLOW_CONFIG = REPO_ROOT / "config/llm_data_flow.yaml"
AUDIT_LOG = REPO_ROOT / "logs/llm_audit.jsonl"   # DO NOT GUESS — verify in Phase 0


# ─── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def flow_config():
    """Load the PHI/data-flow contract."""
    if not FLOW_CONFIG.exists():
        pytest.fail(
            f"{FLOW_CONFIG} missing. Create per Phase 1 spec before running."
        )
    return yaml.safe_load(FLOW_CONFIG.read_text())


@pytest.fixture
def phi_laden_alert():
    """
    A synthetic alert dict containing PHI markers that should NEVER cross to LLM.

    The forbidden-content fields below are deliberately included to catch any
    code path that accidentally serializes the whole alert dict into the prompt.
    """
    return {
        # Allowed fields (these SHOULD appear in the prompt)
        "alert_id": "alert_00042",
        "device_class": "infusion_pump",
        "device_criticality": "CRITICAL",
        "attack_category": "Data Alteration",
        "risk_tier": "CRITICAL",
        "fusion_class": "KNOWN_ATTACK",
        "shap_top_features": ["fwd_pkts_tot", "flow_duration", "fwd_pkts_per_sec"],
        "shap_top_values": [0.42, 0.31, 0.18],
        "mitre_technique_id": "T1565",
        "mitre_technique_name": "Data Manipulation",
        "timestamp_iso": "2026-05-19T14:32:00Z",

        # FORBIDDEN — these are PHI honeypots. If any reach the prompt,
        # the test fails. Values chosen to also match regex patterns.
        "patient_name": "John Doe",
        "patient_id": "PT-998877",
        "mrn": "MRN: 9988776",
        "dob": "1947-03-12",
        "bed_number": "Bed 4-2",
        "room_number": "Room 312A",
        "encounter_id": "ENC-2026-051901",
        "clinician_name": "Dr. Sarah Smith",
        "device_serial": "INF-PUMP-SN-44872",
        "device_hostname": "ICU-PUMP-BED4-2",
        "src_ip": "10.0.1.42",
        "dst_ip": "192.168.10.99",
        "mac_address": "AA:BB:CC:DD:EE:FF",
        "free_text_note": "Patient John Doe in Bed 4-2 has device serial INF-PUMP-SN-44872.",
        "phone": "555-123-4567",
        "email": "smith@hospital.example",
    }


# ─── Helpers ───────────────────────────────────────────────────

def _scan_for_phi(text: str, flow_config: dict) -> list:
    """
    Return a list of findings; empty list means PHI-free.

    Checks:
      - forbidden_field_names appear as case-insensitive substring
      - forbidden_patterns regex match
    """
    findings = []
    if not isinstance(text, str):
        return findings  # nothing to scan
    lower = text.lower()

    for name in flow_config.get("forbidden_field_names", []):
        if name.lower() in lower:
            findings.append({
                "type": "forbidden_field_name",
                "name": name,
                "snippet": _snippet_around(lower, name.lower()),
            })

    for pat in flow_config.get("forbidden_patterns", []):
        regex = re.compile(pat["regex"])
        match = regex.search(text)
        if match:
            findings.append({
                "type": "forbidden_pattern",
                "name": pat["name"],
                "match": match.group(0),
            })

    return findings


def _snippet_around(text: str, needle: str, width: int = 40) -> str:
    idx = text.find(needle)
    start = max(0, idx - width)
    end = min(len(text), idx + len(needle) + width)
    return text[start:end]


# ─── Test 1: Live prompt scan (Mode A) ─────────────────────────

def test_mode_a_prompt_excludes_phi(flow_config, phi_laden_alert):
    """
    DEFENSE-CRITICAL: synthesize an alert with PHI markers, run Mode A's
    prompt construction with a mocked LLM client, scan the captured prompt.
    """
    # Import dynamically to avoid module-load issues if generator isn't ready
    from src.mve_generator import generate_mve

    # Mock the LLM client — capture the prompt, return a stub response
    captured_prompts = []

    def fake_llm_call(prompt, **kwargs):
        captured_prompts.append(prompt)
        return {
            "layer1_why": "Stub: traffic pattern matches T1565 Data Manipulation.",
            "layer2_impact": "Stub: critical device.",
            "layer3_action": "Stub: isolate device.",
            "layer3_do_not": "Stub: do not disconnect.",
        }

    # DO NOT GUESS — the exact way to inject the mocked client depends on
    # how src/mve_generator.py is structured. Adapt this patch target.
    with patch("src.mve_generator._llm_call", side_effect=fake_llm_call):
        result = generate_mve(phi_laden_alert, mode="A")

    assert len(captured_prompts) >= 1, \
        "Mode A did not invoke LLM — check mock target path"

    for i, prompt in enumerate(captured_prompts):
        findings = _scan_for_phi(prompt, flow_config)
        assert not findings, (
            f"PHI detected in Mode A prompt #{i}: "
            f"{json.dumps(findings, indent=2)[:500]}"
        )


def test_mode_a_prompt_only_uses_allowed_fields(flow_config, phi_laden_alert):
    """
    Stricter complement: not only should forbidden fields be absent —
    only fields in allowed_fields should appear in the prompt.

    Checks that no field NAME outside allowed_fields appears as a label in
    the prompt. This is a heuristic; not all field names are also labels.
    """
    from src.mve_generator import generate_mve

    allowed = set(flow_config.get("allowed_fields", []))
    captured = []

    def fake_llm_call(prompt, **kwargs):
        captured.append(prompt)
        return {
            "layer1_why": "Stub.", "layer2_impact": "Stub.",
            "layer3_action": "Stub.", "layer3_do_not": "Stub.",
        }

    with patch("src.mve_generator._llm_call", side_effect=fake_llm_call):
        generate_mve(phi_laden_alert, mode="A")

    # Check all alert dict keys that are NOT in allowed_fields
    for key in phi_laden_alert.keys():
        if key in allowed:
            continue
        # The key name itself should not appear as a label in the prompt
        # (this catches `f"patient_name: {value}"` patterns)
        for prompt in captured:
            assert key not in prompt, (
                f"Disallowed field '{key}' appears in Mode A prompt — "
                f"check src/mve_generator.py prompt construction."
            )


# ─── Test 2: Historical audit-log scan ─────────────────────────

def test_historical_audit_log_phi_free(flow_config):
    """
    Scan past Mode A audit-log entries for PHI in full_prompt + full_response.
    Skips if no audit log exists yet (acceptable — no prior runs).
    """
    if not AUDIT_LOG.exists():
        pytest.skip(f"No audit log at {AUDIT_LOG} — no prior Mode A runs to scan")

    violations = []
    with AUDIT_LOG.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                violations.append({"line": line_no, "error": "invalid JSON"})
                continue

            for field in ("full_prompt", "full_response"):
                text = rec.get(field, "")
                findings = _scan_for_phi(text, flow_config)
                if findings:
                    violations.append({
                        "line": line_no,
                        "field": field,
                        "findings": findings,
                        "alert_id": rec.get("alert_id"),
                    })

    assert not violations, (
        f"PHI found in {len(violations)} audit log entries. "
        f"Sample: {json.dumps(violations[:3], indent=2)[:1000]}"
    )


# ─── Test 3: Mode B — no external calls ─────────────────────────

def test_mode_b_makes_no_external_calls(phi_laden_alert):
    """
    Mode B is the local fallback. It must NEVER make a network call.
    This is a different invariant from PHI exclusion.
    """
    from src.mve_generator import generate_mve

    network_calls = []

    # Patch every plausible HTTP entry point
    with patch("requests.request", side_effect=lambda *a, **kw:
               network_calls.append(("requests.request", a, kw))), \
         patch("requests.post",    side_effect=lambda *a, **kw:
               network_calls.append(("requests.post", a, kw))), \
         patch("requests.get",     side_effect=lambda *a, **kw:
               network_calls.append(("requests.get", a, kw))), \
         patch("urllib.request.urlopen", side_effect=lambda *a, **kw:
               network_calls.append(("urllib.urlopen", a, kw))):
        # Also patch anthropic / openai clients if used
        try:
            with patch("anthropic.Anthropic") as anth_mock:
                anth_mock.side_effect = AssertionError(
                    "Mode B invoked anthropic.Anthropic — should never happen"
                )
                generate_mve(phi_laden_alert, mode="B")
        except (ImportError, ModuleNotFoundError):
            generate_mve(phi_laden_alert, mode="B")

    assert not network_calls, (
        f"Mode B made {len(network_calls)} network call(s): {network_calls[:3]}"
    )
```

### 4.3 Verification

```bash
pytest tests/test_phi_not_in_llm_prompt.py -v
# Expected: 4 tests, all pass (or skip historical if no audit log yet)
```

**DO NOT GUESS** the patch targets — `src.mve_generator._llm_call` is a placeholder. Phase 0 must confirm the actual function or method being patched.

---

## 5. Phase 2 — Word budget audit

### 5.1 Architecture contract

Per ARCHITECTURE.md Step 12:

| Layer | Word budget |
|---|---|
| Layer 1 (WHY) | ≤ 40 words |
| Layer 2 (IMPACT) | ≤ 50 words |
| Layer 3 action | ≤ 60 words |
| Layer 3 DO_NOT | ≤ 30 words |
| **TOTAL** | **≤ 180 words** (Layer 1 + 2 + 3 action + 3 DO_NOT) |

Note: the RQ2_expected_outputs.md mentions "Word budget ≤150 total" but the architecture sums to 40+50+60+30 = 180. **DO NOT GUESS** — confirm with developer which is the binding constraint. The spec below uses per-layer budgets as the binding contract; the total is the sum.

### 5.2 Word-budget escape hatch (recommended)

To make the hard-fail contract stable under Mode A non-determinism, add post-generation truncation in `src/mve_generator.py`:

```python
# Suggested addition to src/mve_generator.py
LAYER_WORD_BUDGETS = {
    "layer1_why": 40,
    "layer2_impact": 50,
    "layer3_action": 60,
    "layer3_do_not": 30,
}

def _enforce_word_budget(mve_output):
    """Truncate any layer that exceeds its budget. Returns truncated output."""
    out = dict(mve_output)
    for field, budget in LAYER_WORD_BUDGETS.items():
        text = out.get(field, "")
        words = text.split()
        if len(words) > budget:
            out[field] = " ".join(words[:budget])
            out.setdefault("_truncated", []).append(field)
    return out
```

This makes the user-facing contract deterministic regardless of upstream LLM variance. The audit then verifies the post-truncation output. The `_truncated` list lets you monitor drift without breaking CI.

### 5.3 Create `analysis/audit_word_budgets.py`

```python
"""
Audit MVE outputs for word-budget compliance.

Per-layer budgets:
  layer1_why:     ≤ 40 words
  layer2_impact:  ≤ 50 words
  layer3_action:  ≤ 60 words
  layer3_do_not:  ≤ 30 words
  TOTAL:          ≤ 180 words

Hard fail on any violation. The matching test (test_word_budgets) reads
this script's output and fails CI if any violation is reported.

Writes results/rq2_word_budget_audit.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MVE_OUTPUTS = REPO_ROOT / "results/mve_outputs.jsonl"   # DO NOT GUESS
OUT = REPO_ROOT / "results/rq2_word_budget_audit.json"

LAYER_BUDGETS = {
    "layer1_why": 40,
    "layer2_impact": 50,
    "layer3_action": 60,
    "layer3_do_not": 30,
}
TOTAL_BUDGET = sum(LAYER_BUDGETS.values())   # 180


def _word_count(text):
    return len((text or "").split())


def _load_mve_outputs():
    """DO NOT GUESS — adapt to actual MVE output schema."""
    out = []
    with MVE_OUTPUTS.open() as f:
        for line in f:
            rec = json.loads(line)
            out.append(rec)
    return out


def main():
    records = _load_mve_outputs()

    violations = []
    per_layer_stats = {layer: {"max": 0, "mean": 0, "n_over": 0}
                       for layer in LAYER_BUDGETS}
    total_stats = {"max": 0, "mean": 0, "n_over": 0}
    all_totals = []

    for rec in records:
        layers = {}
        total = 0
        rec_violations = []

        for layer, budget in LAYER_BUDGETS.items():
            wc = _word_count(rec.get(layer, ""))
            layers[layer] = wc
            total += wc

            per_layer_stats[layer]["max"] = max(per_layer_stats[layer]["max"], wc)
            if wc > budget:
                per_layer_stats[layer]["n_over"] += 1
                rec_violations.append({
                    "layer": layer,
                    "count": wc,
                    "budget": budget,
                    "over_by": wc - budget,
                })

        all_totals.append(total)
        total_stats["max"] = max(total_stats["max"], total)
        if total > TOTAL_BUDGET:
            total_stats["n_over"] += 1
            rec_violations.append({
                "layer": "TOTAL",
                "count": total,
                "budget": TOTAL_BUDGET,
                "over_by": total - TOTAL_BUDGET,
            })

        if rec_violations:
            violations.append({
                "row_id": rec.get("row_id"),
                "mode": rec.get("mode"),
                "per_layer_counts": layers,
                "total": total,
                "violations": rec_violations,
            })

    n = len(records) or 1
    for layer in LAYER_BUDGETS:
        per_layer_stats[layer]["mean"] = sum(
            _word_count(r.get(layer, "")) for r in records
        ) / n
    total_stats["mean"] = sum(all_totals) / n if all_totals else 0

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/audit_word_budgets.py",
            "inputs": {
                "mve_outputs": str(MVE_OUTPUTS.relative_to(REPO_ROOT)),
                "n_records": len(records),
            },
            "config": {
                "word_count_method": "len(text.split())",
                "layer_budgets": LAYER_BUDGETS,
                "total_budget": TOTAL_BUDGET,
                "pass_criterion": "Hard fail on any per-layer or total violation",
            },
        },
        "headline": {
            "n_records": len(records),
            "n_records_with_violations": len(violations),
            "audit_pass": len(violations) == 0,
            "violation_rate": len(violations) / n,
        },
        "per_layer_stats": per_layer_stats,
        "total_stats": total_stats,
        "violations": violations[:50],   # truncate for readability
        "violations_truncated_at": 50,
        "violations_total_count": len(violations),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"Audit: {'PASS' if out['headline']['audit_pass'] else 'FAIL'} "
          f"({len(violations)} violations across {len(records)} records)")


if __name__ == "__main__":
    main()
```

### 5.4 Create `tests/test_word_budgets.py`

```python
"""Hard-fail CI test for word budget audit results."""
import json
from pathlib import Path

import pytest

OUT = Path("results/rq2_word_budget_audit.json")


@pytest.fixture(scope="module")
def audit():
    if not OUT.exists():
        pytest.skip("Run analysis/audit_word_budgets.py first")
    return json.loads(OUT.read_text())


def test_audit_pass(audit):
    h = audit["headline"]
    assert h["audit_pass"], (
        f"Word budget audit failed: {h['n_records_with_violations']} "
        f"records exceeded budget out of {h['n_records']}. "
        f"See results/rq2_word_budget_audit.json for details."
    )


def test_no_total_budget_overflow(audit):
    """Even with per-layer slack, the total must never exceed."""
    assert audit["total_stats"]["n_over"] == 0, (
        f"{audit['total_stats']['n_over']} records exceed TOTAL word budget. "
        "User-facing output overflow — investigate truncation logic in mve_generator.py."
    )
```

### 5.5 Verification

```bash
python -m analysis.audit_word_budgets
pytest tests/test_word_budgets.py -v
# Expected: audit_pass true, 2 tests pass
```

---

## 6. Phase 3 — Cross-role consistency

### 6.1 The three invariants asserted

For one synthetic alert generated under each of three roles:

| Assertion | Rationale | Architecture ref |
|---|---|---|
| Shared anchor (alert_id, risk_tier, device_class, timestamp) IDENTICAL across roles | Invariant 9 — prevents miscommunication during phone-based incident handling | Senior reviewer's concern |
| Layer 2 severity string IDENTICAL across roles | Invariant 6 — cross-role severity invariance | ARCHITECTURE.md |
| Layer 3 immediate_action DIFFERENT between at least two roles | Positive proof of role adaptation; catches the bug where role-scoping silently fails | Senior reviewer's role-scoping concern |

### 6.2 Create `tests/test_step13_cross_role_consistency.py`

```python
"""
Cross-role consistency tests (Invariants 6 + 9 + positive role differentiation).

For one synthetic alert, generate MVE views under 3 roles and assert:
  1. Shared anchor (alert_id, tier, device, timestamp) identical across roles
  2. Layer 2 impact (severity) identical across roles
  3. Layer 3 immediate_action differs between at least two roles
"""

import pytest

# DO NOT GUESS the import paths — Phase 0 confirms these
from src.mve_generator import generate_mve, derive_role_view, OperatorRole


@pytest.fixture
def synthetic_alert():
    return {
        "alert_id": "alert_cross_role_test",
        "device_class": "infusion_pump",
        "device_criticality": "CRITICAL",
        "attack_category": "Data Alteration",
        "risk_tier": "CRITICAL",
        "fusion_class": "KNOWN_ATTACK",
        "shap_top_features": ["fwd_pkts_tot", "flow_duration", "fwd_pkts_per_sec"],
        "shap_top_values": [0.42, 0.31, 0.18],
        "mitre_technique_id": "T1565",
        "mitre_technique_name": "Data Manipulation",
        "timestamp_iso": "2026-05-19T14:32:00Z",
    }


@pytest.fixture
def three_role_views(synthetic_alert):
    """
    Generate one MVE, then derive role views for all three operator roles.
    Uses Mode B to keep the test deterministic.
    """
    base_mve = generate_mve(synthetic_alert, mode="B")
    roles = [
        OperatorRole.IT_GENERALIST,
        OperatorRole.BIOMED_ENGINEER,
        OperatorRole.NURSE_MANAGER,
    ]
    return {role: derive_role_view(base_mve, role) for role in roles}


# ─── Invariant 9: Shared anchor identical across roles ─────────

@pytest.mark.parametrize("field", [
    "alert_id", "risk_tier", "device_class", "timestamp_iso",
])
def test_invariant_9_shared_anchor_identical(three_role_views, field):
    """Shared anchor fields must be identical across all three role views."""
    values = {
        role: getattr(view, field, None) or view.get(field)
        for role, view in three_role_views.items()
    }
    distinct = set(values.values())
    assert len(distinct) == 1, (
        f"Invariant 9 violated: field '{field}' differs across roles: {values}"
    )


# ─── Invariant 6: Layer 2 (severity) identical across roles ─────

def test_invariant_6_layer2_severity_identical(three_role_views):
    """Layer 2 (clinical impact / severity) must be string-equal across roles."""
    layer2_values = {
        role: getattr(view, "layer2_impact", None) or view.get("layer2_impact")
        for role, view in three_role_views.items()
    }
    distinct = set(layer2_values.values())
    assert len(distinct) == 1, (
        f"Invariant 6 violated: Layer 2 differs across roles. "
        f"Samples: {[(r, (v or '')[:80]) for r, v in layer2_values.items()]}"
    )


# ─── Positive role differentiation: Layer 3 actions differ ──────

def test_role_differentiation_layer3_actions_differ(three_role_views):
    """
    Layer 3 immediate_action must differ between at least two roles.
    If all three roles produce the same action, role-adaptation has silently failed.
    """
    actions = {
        role: getattr(view, "layer3_action", None) or view.get("layer3_action")
        for role, view in three_role_views.items()
    }
    distinct = set(actions.values())
    assert len(distinct) >= 2, (
        f"Role adaptation failed: all three roles produced identical Layer 3 "
        f"actions. {actions}"
    )


# ─── Invariant 7 sanity: DO_NOT present for CRITICAL clinical device ──

def test_invariant_7_do_not_present_for_critical(three_role_views):
    """
    For a CRITICAL alert on a clinical device, every role view must include
    a non-empty layer3_do_not (clinical safety constraint).
    """
    for role, view in three_role_views.items():
        do_not = getattr(view, "layer3_do_not", None) or view.get("layer3_do_not")
        assert do_not and do_not.strip(), (
            f"Invariant 7 violated for role {role}: layer3_do_not is empty "
            f"on CRITICAL clinical device alert."
        )
```

### 6.3 Verification

```bash
pytest tests/test_step13_cross_role_consistency.py -v
# Expected: 6 tests pass (4 anchor fields parametrized + 1 Layer 2 + 1 differentiation + 1 DO_NOT)
```

**DO NOT GUESS:**
- Whether MVE output objects are dicts or dataclasses — the tests use both `getattr` and `.get()` defensively, but Phase 0 should confirm.
- Exact `OperatorRole` enum values — adapt the imports if Phase 0 revealed different names.

---

## 7. Phase 4 — Compliance mapping manifest

### 7.1 Create `config/rq2_compliance_manifest.yaml`

The manifest is the source of truth for the literature-requirement ↔ MVE-implementation mapping. The paper appendix renders from it; CI verifies evidence files exist.

```yaml
# config/rq2_compliance_manifest.yaml
# Maps formal explainability requirements (literature) to the MVE
# implementation choices that satisfy them, with evidence file pointers.
# Verified by tests/test_compliance_manifest.py.

schema_version: "1.0"
last_validated: "2026-05-19"

requirements:
  - id: REQ-FAITHFULNESS
    literature_term: Faithfulness
    description: >
      Explanations must reflect the actual decision logic of the underlying model,
      not post-hoc plausible-sounding text.
    mve_implementation: >
      Invariant 5: Layer 1 must reference SHAP top-3 features (as raw names
      or their human-readable mappings).
    evidence_files:
      - tests/test_step12_mve_faithfulness.py
      - analysis/compute_mve_shap_alignment.py
      - results/rq2_mve_shap_alignment.json

  - id: REQ-STABILITY
    literature_term: Stability
    description: >
      Explanations should not change drastically under small input perturbations.
    mve_implementation: SHAP stability score ≥0.90 on top-3 feature overlap.
    evidence_files:
      - tests/test_step11_shap_stability.py
      - analysis/compute_shap_stability.py
      - results/rq2_shap_stability.json

  - id: REQ-COMPLETENESS
    literature_term: Completeness
    description: >
      Explanations should cover the why, the impact, and the recommended action.
    mve_implementation: 3-layer structure (WHY / IMPACT / ACTION + DO_NOT).
    evidence_files:
      - src/mve_generator.py
      - tests/test_coverage_mve.py

  - id: REQ-BREVITY
    literature_term: Brevity
    description: >
      Explanations must be concise enough for time-pressured triage decisions.
    mve_implementation: Per-layer word budgets enforced (40/50/60/30; total 180).
    evidence_files:
      - analysis/audit_word_budgets.py
      - results/rq2_word_budget_audit.json
      - tests/test_word_budgets.py

  - id: REQ-AUDIENCE_APPROPRIATENESS
    literature_term: Audience appropriateness
    description: >
      Same alert should be communicated differently to different stakeholders
      based on role, while preserving shared facts.
    mve_implementation: >
      Role views (IT Generalist / Biomed Engineer / Nurse Manager) with
      Invariant 6 (Layer 2 cross-role identical) and Invariant 9 (shared anchor).
    evidence_files:
      - src/mve_generator.py
      - tests/test_step13_cross_role_consistency.py
      - tests/test_safe_failure.py
      - config/role_action_authorization.yaml

  - id: REQ-PROVENANCE
    literature_term: Provenance
    description: >
      For LLM-generated explanations, the prompt, model version, and response
      must be auditable for reproducibility and accountability.
    mve_implementation: >
      Mode A audit log captures llm_provider, llm_model_version,
      full_prompt, full_response per alert. Hash-chained for tamper evidence.
    evidence_files:
      - logs/llm_audit.jsonl
      - tests/test_phi_not_in_llm_prompt.py

  - id: REQ-FALLBACK
    literature_term: Fallback / availability
    description: >
      The explanation system must degrade gracefully when external dependencies
      fail (e.g., LLM API unavailable).
    mve_implementation: >
      Mode B rule-based generator activates on Mode A failure; UI badge
      "Rule-based fallback" visible to operator.
    evidence_files:
      - src/mve_generator.py
      - tests/test_safe_failure.py

  - id: REQ-PHI_CONTROL
    literature_term: Data minimization / HIPAA boundary
    description: >
      For deployments with patient context, no PHI must cross to external
      LLM providers; data flow contract must be auditable.
    mve_implementation: >
      config/llm_data_flow.yaml defines allowed/forbidden fields and content
      patterns. Tests verify both live prompts and historical audit logs.
    evidence_files:
      - config/llm_data_flow.yaml
      - tests/test_phi_not_in_llm_prompt.py
```

### 7.2 Create `analysis/make_rq2_compliance_table.py`

```python
"""
Render config/rq2_compliance_manifest.yaml into a markdown table for paper appendix.

Writes results/rq2_compliance_mapping.md.

Also performs a structural sanity check: every evidence_files entry must point
to an existing file (or be marked as intentionally missing).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "config/rq2_compliance_manifest.yaml"
OUT_MD = REPO_ROOT / "results/rq2_compliance_mapping.md"
OUT_AUDIT = REPO_ROOT / "results/rq2_compliance_audit.json"


def main():
    if not MANIFEST.exists():
        print(f"Manifest not found: {MANIFEST}", file=sys.stderr)
        sys.exit(1)

    doc = yaml.safe_load(MANIFEST.read_text())
    reqs = doc.get("requirements", [])

    # Evidence file existence audit
    audit = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(MANIFEST.relative_to(REPO_ROOT)),
            "schema_version": doc.get("schema_version"),
            "last_validated": doc.get("last_validated"),
        },
        "requirements_total": len(reqs),
        "evidence_audit": [],
    }

    for req in reqs:
        missing = []
        for ev in req.get("evidence_files", []):
            full = REPO_ROOT / ev
            if not full.exists():
                missing.append(ev)
        audit["evidence_audit"].append({
            "id": req["id"],
            "evidence_total": len(req.get("evidence_files", [])),
            "missing_count": len(missing),
            "missing_files": missing,
        })

    audit["all_evidence_present"] = all(
        e["missing_count"] == 0 for e in audit["evidence_audit"]
    )

    # Render markdown
    lines = [
        "# RQ2 — Compliance Mapping (literature ↔ MVE)",
        "",
        f"*Generated from `{MANIFEST.relative_to(REPO_ROOT)}` "
        f"on {audit['_meta']['generated_at']}.*",
        f"*Manifest last validated: {doc.get('last_validated', 'unknown')}.*",
        "",
        "| Requirement | Literature Term | MVE Implementation | Evidence |",
        "|---|---|---|---|",
    ]
    for req in reqs:
        ev_lines = "<br>".join(f"`{e}`" for e in req.get("evidence_files", []))
        lines.append(
            f"| **{req['id']}** | {req['literature_term']} | "
            f"{req['mve_implementation'].strip()} | {ev_lines} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Descriptions")
    lines.append("")
    for req in reqs:
        lines.append(f"### {req['id']} — {req['literature_term']}")
        lines.append("")
        lines.append(req['description'].strip())
        lines.append("")
        lines.append(f"**MVE Implementation:** {req['mve_implementation'].strip()}")
        lines.append("")
        lines.append("**Evidence:**")
        for e in req.get("evidence_files", []):
            lines.append(f"- `{e}`")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, default=str))

    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_AUDIT.relative_to(REPO_ROOT)}")
    if not audit["all_evidence_present"]:
        missing = [(e["id"], e["missing_files"])
                   for e in audit["evidence_audit"] if e["missing_count"] > 0]
        print(f"WARN: missing evidence files: {missing}")


if __name__ == "__main__":
    main()
```

### 7.3 Create `tests/test_compliance_manifest.py`

```python
"""
Tests for the RQ2 compliance manifest.

Verifies that every requirement has at least one evidence file, and
every listed evidence file exists on disk.
"""
import json
from pathlib import Path

import pytest

OUT = Path("results/rq2_compliance_audit.json")


@pytest.fixture(scope="module")
def audit():
    if not OUT.exists():
        pytest.skip("Run analysis/make_rq2_compliance_table.py first")
    return json.loads(OUT.read_text())


def test_all_evidence_files_exist(audit):
    """Every evidence file listed in the manifest must exist on disk."""
    missing = [
        (e["id"], e["missing_files"])
        for e in audit["evidence_audit"]
        if e["missing_count"] > 0
    ]
    assert not missing, (
        f"Compliance manifest references missing files: {missing}"
    )


def test_every_requirement_has_evidence(audit):
    """No requirement may have zero evidence files."""
    empty = [
        e["id"] for e in audit["evidence_audit"]
        if e["evidence_total"] == 0
    ]
    assert not empty, f"Requirements with no evidence: {empty}"


def test_manifest_last_validated_present(audit):
    """The manifest's last_validated date must be set."""
    assert audit["_meta"]["last_validated"], (
        "rq2_compliance_manifest.yaml needs last_validated set"
    )
```

### 7.4 Verification

```bash
python -m analysis.make_rq2_compliance_table
pytest tests/test_compliance_manifest.py -v
# Expected: 3 tests pass, markdown table generated
```

---

## 8. Execution order

```bash
# ─── PHASE 0: SCHEMA DISCOVERY ──────────────────────────────────
python scripts/discover_compliance_artifacts.py > /tmp/discovery.json
# DEVELOPER CONFIRMS: llm_data_flow.yaml schema, audit log path,
# mve_generator.py signatures, OperatorRole enum values

# ─── PHASE 1: PHI FLOW CONTROL (DEFENSE-CRITICAL) ──────────────
# Create config/llm_data_flow.yaml if missing
# Create tests/test_phi_not_in_llm_prompt.py
pytest tests/test_phi_not_in_llm_prompt.py -v
# Expected: 4 tests pass (1 may skip if no audit log exists yet)

# ─── PHASE 2: WORD BUDGET AUDIT ────────────────────────────────
# Optional: add _enforce_word_budget() to src/mve_generator.py
# Create analysis/audit_word_budgets.py
python -m analysis.audit_word_budgets
# Create tests/test_word_budgets.py
pytest tests/test_word_budgets.py -v

# ─── PHASE 3: CROSS-ROLE CONSISTENCY ───────────────────────────
# Create tests/test_step13_cross_role_consistency.py
pytest tests/test_step13_cross_role_consistency.py -v
# Expected: 6 tests pass

# ─── PHASE 4: COMPLIANCE MANIFEST ──────────────────────────────
# Create config/rq2_compliance_manifest.yaml
# Create analysis/make_rq2_compliance_table.py
# Create tests/test_compliance_manifest.py
python -m analysis.make_rq2_compliance_table
pytest tests/test_compliance_manifest.py -v

# ─── FINAL VERIFICATION ────────────────────────────────────────
pytest tests/test_phi_not_in_llm_prompt.py \
       tests/test_word_budgets.py \
       tests/test_step13_cross_role_consistency.py \
       tests/test_compliance_manifest.py -v
ls config/llm_data_flow.yaml \
   config/rq2_compliance_manifest.yaml \
   results/rq2_word_budget_audit.json \
   results/rq2_compliance_mapping.md \
   results/rq2_compliance_audit.json
```

---

## 9. Integration with `compute_rq2_metrics.py`

When the master aggregator is built, fold Track 3 outputs in under a `compliance` block:

```python
def _load_compliance_subfiles():
    word_budget_p = REPO_ROOT / "results/rq2_word_budget_audit.json"
    manifest_p = REPO_ROOT / "results/rq2_compliance_audit.json"

    block = {"_status": "pending", "_merged_at": None}
    if word_budget_p.exists() and manifest_p.exists():
        block = {
            "_status": "complete",
            "_merged_at": datetime.now(timezone.utc).isoformat(),
            "word_budget_audit": json.loads(word_budget_p.read_text()),
            "compliance_manifest_audit": json.loads(manifest_p.read_text()),
            "_note": (
                "PHI and cross-role checks are pytest-only (no JSON artifact). "
                "Look at the CI badge for those — they're hard-fail gates."
            ),
        }
    return block
```

In the aggregator: `out["compliance"] = _load_compliance_subfiles()`.

The PHI test and cross-role test deliberately don't produce JSON outputs — they're CI-gating tests. Either they pass or they don't; there's nothing to aggregate.

---

## 10. Open questions to surface (DO NOT GUESS)

Claude Code must pause and ask:

1. **Phase 0 — `config/llm_data_flow.yaml` existence and schema.** Does this file exist? If yes, what's its structure (top-level keys, field name conventions)? If no, is Phase 1's template above acceptable as a starting point?
2. **Phase 0 — Mode A audit log location and record schema.** Per `RQ3_expected_outputs.md` the fields should include `llm_provider`, `llm_model_version`, `full_prompt`, `full_response`. Confirm path and schema.
3. **Phase 0 — `src/mve_generator.py` signatures.** Specifically: what's the public `generate_mve` signature? How is mode (A/B) selected? How is the LLM client injected (so we can mock it)?
4. **Phase 0 — `OperatorRole` enum.** Confirm enum identifier names (IT_GENERALIST vs IT_Generalist vs it_generalist?).
5. **Phase 2 — Word budget total.** ARCHITECTURE.md per-layer budgets sum to 180; `RQ2_expected_outputs.md` says ≤150. Which is binding? Likely 180 since it's the explicit per-layer sum, but confirm.
6. **Phase 1 — `_llm_call` mock target.** The internal function name in `src/mve_generator.py` that actually invokes the API. May be named differently (`_call_anthropic`, `_invoke_llm`, etc.). Adjust the `patch()` target accordingly.

---

## 11. Coverage map — RQ2.a + compliance items → pipeline phase

| RQ2_expected_outputs.md item | Phase | Output |
|---|---|---|
| §1.1 Mapping table (Faithfulness, Stability, Completeness, Brevity, Audience appropriateness, Provenance, Fallback) | 4 | `results/rq2_compliance_mapping.md` |
| §1.2 Word budgets enforced (compliance checklist) | 2 | `results/rq2_word_budget_audit.json` + test |
| §1.2 Mode A audit log captures prompt+response (provenance) | 4 | manifest evidence_files (logs/llm_audit.jsonl) |
| §1.2 Mode B fallback (compliance checklist) | 1 (Mode B no-network test) | pytest gate |
| Senior reviewer: PHI exposure question | 1 | `tests/test_phi_not_in_llm_prompt.py` |
| Senior reviewer: cross-role mental model | 3 | `tests/test_step13_cross_role_consistency.py` |
| Senior reviewer: Mode A auditability | 4 | manifest REQ-PROVENANCE |
| `tests/test_phi_not_in_llm_prompt.py` (RQ2 §8) | 1 | created |
| `tests/test_step13_cross_role_consistency.py` (RQ2 §8) | 3 | created |
| Invariant 6 (cross-role severity) | 3 | parametrized test |
| Invariant 7 (DO_NOT present for CRITICAL clinical) | 3 | sanity test |
| Invariant 9 (shared anchor) | 3 | parametrized test |

Every Track 3 item from the expected-outputs doc is traceable to a phase.

---

## 12. What this track deliberately does NOT cover

- **Hash-chained audit log integrity** — that's RQ3 territory (`verify_audit_log_integrity()`).
- **Mode A vs Mode B comparison metrics** — that's Track 1 (alignment metric breaks down by mode).
- **MITRE grounding evidence** — that's Track 2 (compliance manifest references Track 2's outputs as evidence for REQ-FAITHFULNESS).

These adjacencies matter: Track 3 doesn't duplicate work, it links to it via the compliance manifest.

---

## End of spec

Implementation order: Phase 0 → 1 → 2/3/4 (parallel). Phase 1 is DEFENSE-CRITICAL; finish it before user study data is collected so the HIPAA story is solid before any human subjects research is documented.