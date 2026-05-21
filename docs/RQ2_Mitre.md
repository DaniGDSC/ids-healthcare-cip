# RQ2 MITRE Grounding Pipeline — Config Audit + Layer 1 Reference Rate

**Project:** XAI-IDS-Healthcare
**Scope:** RQ2.e — Does MVE ground explanations in MITRE ATT&CK?
**Purpose:** Single, self-contained spec for (1) auditing `config/attack_to_mitre_mapping.yaml` and (2) measuring how often MVE Layer 1 outputs reference the mapped MITRE technique. Hand to Claude Code.
**Status of design:** All decisions locked. Two `DO NOT GUESS` checkpoints (Phase 0 schema discovery, Phase 2 MVE output structure).

---

## 0. How to use this spec

1. Implement in the order given. Phase 0 is a **schema discovery step** — Claude Code reads the existing YAML and adapts the rest of the spec to it. Do not write Phases 1–3 until Phase 0 has produced a known schema.
2. Each phase has a `verification` command. Do not proceed if it fails.
3. Markers:
   - **DO NOT GUESS** — stop and ask the developer
   - **TARGET** — a numeric goal from `RQ2_expected_outputs.md`
4. Two scripts and two test files are produced. Both scripts are fast (seconds); neither requires model inference.

---

## 1. Background: what these metrics measure

| Deliverable | Question | Output | Target |
|---|---|---|---|
| **Config audit** | Is `attack_to_mitre_mapping.yaml` structurally sound and orphan-free? | Pass/fail JSON + structured findings | 100% attack categories mapped; `mitre_framework_version` + `last_validated` set |
| **Grounding rate** | Does MVE Layer 1 actually reference the mapped MITRE technique? | % overall + per-attack-category + by mode | > 90% surfaced alerts reference MITRE |

The audit is a structural CI check (runs every commit). The grounding rate is an output analysis (runs after a Module 5 MVE batch).

---

## 2. Locked design decisions

| Decision | Resolution |
|---|---|
| Match rule | Either T-ID OR human name appears as case-insensitive substring in `layer1_why` |
| Mapping cardinality | Read existing YAML structure; adapt grounding logic to whatever convention it uses (1-to-1, 1-to-many, or primary+secondary) |
| Audit strictness | Strict orphan-free + requires `mitre_framework_version` (top-level) + `last_validated` (per mapping or top-level) |
| Grounding sample sets | Three reported: surfaced as headline, per-attack-category as breakdown, all-MVE as appendix |
| Match scope | `layer1_why` field only (consistent with alignment metric) |
| Output integration | Two separate JSONs, merged into `rq2_metrics.json` via aggregator (RQ1 pattern) |
| Strict-match variant | Also report "both T-ID AND human name present" as appendix metric (zero extra cost) |

---

## 3. Phase 0 — YAML schema discovery (DO NOT GUESS)

Before writing any audit code, Claude Code must inspect the existing YAML and document its actual structure. The spec's Phase 1 and Phase 2 logic depends on what's found.

### 3.1 Discovery script

```python
# scripts/discover_mitre_yaml_schema.py — TRANSIENT, delete after Phase 0
"""
Inspect config/attack_to_mitre_mapping.yaml and emit a schema summary
for the developer to confirm before audit logic is written.
"""
import json
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = REPO_ROOT / "config/attack_to_mitre_mapping.yaml"

if not YAML_PATH.exists():
    print(f"YAML not found at {YAML_PATH}")
    print("DO NOT PROCEED — confirm with developer where the mapping lives.")
    raise SystemExit(1)

doc = yaml.safe_load(YAML_PATH.read_text())

summary = {
    "yaml_path": str(YAML_PATH),
    "top_level_keys": list(doc.keys()) if isinstance(doc, dict) else "NOT A DICT",
    "has_mitre_framework_version": "mitre_framework_version" in (doc or {}),
    "has_top_level_last_validated": "last_validated" in (doc or {}),
}

# Identify the mappings block — likely "mappings" or "attack_categories"
mappings_candidates = [
    k for k in (doc or {}).keys()
    if k.lower() in {"mappings", "attack_categories", "attacks", "categories"}
]
summary["mappings_key_candidates"] = mappings_candidates

# Sample 2-3 entries to show structure
if mappings_candidates:
    block = doc[mappings_candidates[0]]
    if isinstance(block, dict):
        sample_keys = list(block.keys())[:3]
        summary["sample_entries"] = {
            k: block[k] for k in sample_keys
        }
    elif isinstance(block, list):
        summary["sample_entries"] = block[:3]

print(json.dumps(summary, indent=2, default=str))
print("\n" + "="*60)
print("REVIEW THIS SCHEMA BEFORE PROCEEDING TO PHASE 1.")
print("Confirm with developer:")
print("  1. The mappings key name (e.g., 'mappings' vs 'attack_categories')")
print("  2. The per-entry structure: list of techniques? primary+secondary?")
print("  3. Per-entry vs top-level last_validated convention")
print("="*60)
```

### 3.2 What to do with the output

The discovery script prints a JSON summary. Claude Code must show this to the developer and **confirm three things before writing Phase 1**:

1. **Mappings key:** what's the dict key containing the actual category-to-technique mappings? (Examples: `mappings`, `attack_categories`, `categories`.)
2. **Cardinality pattern per entry:** does each category map to one technique, a list, or a primary+secondary structure? Concrete patterns Claude Code should detect:

   **Pattern A — flat 1-to-1:**
   ```yaml
   Spoofing:
     technique_id: T1556
     technique_name: "Modify Authentication Process"
     confidence: MEDIUM
   ```

   **Pattern B — 1-to-many (list):**
   ```yaml
   Data Alteration:
     techniques:
       - {id: T1565, name: "Data Manipulation", confidence: HIGH}
       - {id: T1565.001, name: "Stored Data Manipulation", confidence: MEDIUM}
   ```

   **Pattern C — primary + secondary:**
   ```yaml
   Data Alteration:
     primary: {id: T1565, name: "Data Manipulation", confidence: HIGH}
     secondary:
       - {id: T1565.001, name: "Stored Data Manipulation", confidence: MEDIUM}
   ```

3. **`last_validated` location:** top-level only, per-mapping only, or both?

### 3.3 Verification

```bash
python scripts/discover_mitre_yaml_schema.py
# DEVELOPER REVIEWS OUTPUT, CONFIRMS PATTERN
# Then write Phase 1 against the confirmed pattern.
```

**DO NOT GUESS** the pattern. Phase 1's audit logic and Phase 2's grounding logic both depend on it.

---

## 4. Phase 1 — Config audit

### 4.1 Create `analysis/audit_mitre_config.py`

**Contract:**
- **Input:** `config/attack_to_mitre_mapping.yaml` + the set of attack categories present in `risk_scores.npz`.
- **Output:** `results/rq2_mitre_audit.json`.
- **Runtime:** sub-second.
- **Side effects:** writes one JSON file. Idempotent.

### 4.2 Audit checks (in order)

| Check ID | Description | Pass condition | Failure severity |
|---|---|---|---|
| A1 | YAML parses without error | `yaml.safe_load` succeeds | FAIL |
| A2 | Top-level `mitre_framework_version` present and non-empty | string, not empty | FAIL |
| A3 | Top-level OR per-entry `last_validated` present | ISO-8601 date present somewhere | FAIL |
| A4 | All attack categories in test data have a mapping | for each `attack_category` in `risk_scores.npz` (excluding "normal"), key exists in YAML mappings block | FAIL — list orphans |
| A5 | Every mapped entry has at least one technique | per entry: technique_id (or techniques list, or primary) is set | FAIL — list orphans |
| A6 | Every technique has T-ID matching MITRE pattern | regex `^T\d{4}(\.\d{3})?$` per id | WARN |
| A7 | Every technique has a human name (`name` or `technique_name`) | non-empty string | WARN |
| A8 | Every technique has confidence in {HIGH, MEDIUM, LOW} | enum match | WARN |
| A9 | No mapped categories absent from data (the inverse of A4) | listed as `unused_mappings` — informational, not a fail | INFO |

The audit emits **FAIL / WARN / INFO** findings. Headline `pass` is true iff zero FAIL findings.

### 4.3 Output schema

`results/rq2_mitre_audit.json`:

```json
{
  "_meta": {
    "schema_version": "1.0",
    "generated_at": "<ISO-8601>",
    "generated_by": "analysis/audit_mitre_config.py",
    "inputs": {
      "yaml_path": "config/attack_to_mitre_mapping.yaml",
      "yaml_sha256": "<hash>",
      "risk_scores_npz": "results/reports/risk_scores.npz",
      "n_attack_categories_in_data": 4
    },
    "config": {
      "required_top_level_fields": ["mitre_framework_version", "last_validated"],
      "tid_regex": "^T\\d{4}(\\.\\d{3})?$",
      "valid_confidence_levels": ["HIGH", "MEDIUM", "LOW"]
    }
  },
  "headline": {
    "audit_pass": true,
    "n_fail": 0,
    "n_warn": 1,
    "n_info": 2,
    "mitre_framework_version": "v14.1",
    "last_validated_top_level": "2025-08-14",
    "n_categories_mapped": 6,
    "n_categories_in_data": 4,
    "orphan_categories": [],
    "unused_mappings": ["Reconnaissance", "Exfiltration"]
  },
  "findings": [
    {
      "check_id": "A1", "severity": "PASS",
      "description": "YAML parsed successfully",
      "details": null
    },
    {
      "check_id": "A2", "severity": "PASS",
      "description": "mitre_framework_version present",
      "details": {"value": "v14.1"}
    },
    {
      "check_id": "A4", "severity": "PASS",
      "description": "All in-data categories have mappings",
      "details": {"categories_in_data": ["Data Alteration", "Spoofing"]}
    },
    {
      "check_id": "A8", "severity": "WARN",
      "description": "Confidence level outside enum",
      "details": {"category": "Spoofing", "technique": "T1556", "confidence": "Medium"}
    }
  ],
  "mappings_summary": {
    "Data Alteration": {
      "n_techniques": 2,
      "technique_ids": ["T1565", "T1565.001"],
      "technique_names": ["Data Manipulation", "Stored Data Manipulation"],
      "confidence_set": ["HIGH", "MEDIUM"],
      "last_validated": "2025-08-14"
    },
    "Spoofing": {
      "n_techniques": 1,
      "technique_ids": ["T1556"],
      "technique_names": ["Modify Authentication Process"],
      "confidence_set": ["MEDIUM"],
      "last_validated": "2025-08-14"
    }
  }
}
```

### 4.4 Implementation outline

```python
# analysis/audit_mitre_config.py
"""
Audit config/attack_to_mitre_mapping.yaml for structural completeness.

Checks:
  A1 YAML parses
  A2 mitre_framework_version present
  A3 last_validated present (top-level or per-entry)
  A4 Every attack_category in data has a mapping (orphan check)
  A5 Every mapped entry has at least one technique
  A6 T-IDs match MITRE regex
  A7 Each technique has a human name
  A8 Confidence levels in enum
  A9 Mappings present but unused in data (informational)

Writes results/rq2_mitre_audit.json.
Runtime: sub-second.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = REPO_ROOT / "config/attack_to_mitre_mapping.yaml"
NPZ_PATH = REPO_ROOT / "results/reports/risk_scores.npz"
OUT = REPO_ROOT / "results/rq2_mitre_audit.json"

TID_RE = re.compile(r"^T\d{4}(\.\d{3})?$")
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml():
    """Load YAML and return (doc, parse_finding)."""
    try:
        doc = yaml.safe_load(YAML_PATH.read_text())
        return doc, {
            "check_id": "A1", "severity": "PASS",
            "description": "YAML parsed successfully",
            "details": None,
        }
    except Exception as e:
        return None, {
            "check_id": "A1", "severity": "FAIL",
            "description": "YAML failed to parse",
            "details": {"error": str(e)},
        }


def _mappings_block(doc):
    """
    Locate the mappings dict in the YAML.

    DO NOT GUESS — Phase 0 discovery confirmed the key name.
    Replace 'mappings' with whatever the actual key is.
    """
    return doc.get("mappings", {})


def _extract_techniques(entry):
    """
    Normalize an entry to a list of (tid, name, confidence) tuples,
    regardless of whether the YAML uses Pattern A, B, or C.

    DO NOT GUESS — Phase 0 discovery confirmed the pattern.
    Implement accordingly. Example for Pattern B (1-to-many list):

        techniques = entry.get("techniques", [])
        return [
            (t.get("id"), t.get("name"), t.get("confidence"))
            for t in techniques
        ]

    For Pattern A (flat 1-to-1):

        return [(entry.get("technique_id"),
                 entry.get("technique_name"),
                 entry.get("confidence"))]

    For Pattern C (primary + secondary):

        out = []
        p = entry.get("primary", {})
        out.append((p.get("id"), p.get("name"), p.get("confidence")))
        for s in entry.get("secondary", []):
            out.append((s.get("id"), s.get("name"), s.get("confidence")))
        return out
    """
    raise NotImplementedError("Phase 0 schema discovery must precede implementation")


def _attack_categories_in_data() -> set:
    """Set of non-benign attack categories present in test data."""
    data = np.load(NPZ_PATH, allow_pickle=False)
    cats = set(str(c) for c in np.unique(data["attack_category"]))
    return cats - {"normal", ""}


def main():
    findings = []
    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/audit_mitre_config.py",
            "inputs": {
                "yaml_path": str(YAML_PATH.relative_to(REPO_ROOT)),
                "yaml_sha256": _sha256(YAML_PATH),
                "risk_scores_npz": str(NPZ_PATH.relative_to(REPO_ROOT)),
            },
            "config": {
                "required_top_level_fields": [
                    "mitre_framework_version", "last_validated"
                ],
                "tid_regex": TID_RE.pattern,
                "valid_confidence_levels": sorted(VALID_CONFIDENCE),
            },
        },
        "headline": {},
        "findings": [],
        "mappings_summary": {},
    }

    # A1: parse
    doc, a1 = _load_yaml()
    findings.append(a1)
    if doc is None:
        _finalize(out, findings)
        return

    # A2: mitre_framework_version
    framework_version = doc.get("mitre_framework_version")
    findings.append({
        "check_id": "A2",
        "severity": "PASS" if framework_version else "FAIL",
        "description": "mitre_framework_version present",
        "details": {"value": framework_version},
    })

    # A3: last_validated (top-level OR per-entry)
    top_lv = doc.get("last_validated")
    mappings = _mappings_block(doc)
    has_per_entry_lv = any(
        isinstance(v, dict) and v.get("last_validated")
        for v in mappings.values()
    )
    findings.append({
        "check_id": "A3",
        "severity": "PASS" if (top_lv or has_per_entry_lv) else "FAIL",
        "description": "last_validated present (top-level or per-entry)",
        "details": {"top_level": top_lv, "per_entry_present": has_per_entry_lv},
    })

    # A4 + A9: orphan check
    in_data = _attack_categories_in_data()
    in_yaml = set(mappings.keys())
    orphans = sorted(in_data - in_yaml)
    unused = sorted(in_yaml - in_data)
    findings.append({
        "check_id": "A4",
        "severity": "FAIL" if orphans else "PASS",
        "description": "Every in-data category has a mapping",
        "details": {"orphans": orphans, "categories_in_data": sorted(in_data)},
    })
    if unused:
        findings.append({
            "check_id": "A9",
            "severity": "INFO",
            "description": "Mappings exist but no data uses them",
            "details": {"unused_mappings": unused},
        })

    # A5–A8: per-entry technique validation
    mappings_summary = {}
    for category, entry in mappings.items():
        try:
            techniques = _extract_techniques(entry)
        except Exception as e:
            findings.append({
                "check_id": "A5",
                "severity": "FAIL",
                "description": "Failed to extract techniques from entry",
                "details": {"category": category, "error": str(e)},
            })
            continue

        if not techniques:
            findings.append({
                "check_id": "A5",
                "severity": "FAIL",
                "description": "Entry has no techniques",
                "details": {"category": category},
            })
            continue

        tids, names, confs = zip(*techniques)
        for tid, name, conf in techniques:
            if tid and not TID_RE.match(str(tid)):
                findings.append({
                    "check_id": "A6",
                    "severity": "WARN",
                    "description": "Technique ID does not match MITRE pattern",
                    "details": {"category": category, "tid": tid},
                })
            if not name:
                findings.append({
                    "check_id": "A7",
                    "severity": "WARN",
                    "description": "Technique missing human name",
                    "details": {"category": category, "tid": tid},
                })
            if conf and str(conf).upper() not in VALID_CONFIDENCE:
                findings.append({
                    "check_id": "A8",
                    "severity": "WARN",
                    "description": "Confidence level outside enum",
                    "details": {"category": category, "tid": tid, "confidence": conf},
                })

        mappings_summary[category] = {
            "n_techniques": len(techniques),
            "technique_ids": list(tids),
            "technique_names": list(names),
            "confidence_set": sorted({str(c).upper() for c in confs if c}),
            "last_validated": (
                entry.get("last_validated") if isinstance(entry, dict) else None
            ),
        }

    out["mappings_summary"] = mappings_summary
    out["headline"] = {
        "audit_pass": not any(f["severity"] == "FAIL" for f in findings),
        "n_fail": sum(1 for f in findings if f["severity"] == "FAIL"),
        "n_warn": sum(1 for f in findings if f["severity"] == "WARN"),
        "n_info": sum(1 for f in findings if f["severity"] == "INFO"),
        "mitre_framework_version": framework_version,
        "last_validated_top_level": top_lv,
        "n_categories_mapped": len(in_yaml),
        "n_categories_in_data": len(in_data),
        "orphan_categories": orphans,
        "unused_mappings": unused,
    }

    _finalize(out, findings)


def _finalize(out, findings):
    out["findings"] = findings
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    h = out.get("headline", {})
    print(f"Audit: {'PASS' if h.get('audit_pass') else 'FAIL'} "
          f"(fail={h.get('n_fail', '?')}, warn={h.get('n_warn', '?')})")


if __name__ == "__main__":
    main()
```

### 4.5 Verification

```bash
python -m analysis.audit_mitre_config
cat results/rq2_mitre_audit.json | python -m json.tool | head -30
# Expected: audit_pass: true, no FAIL findings
```

---

## 5. Phase 2 — MVE Layer 1 grounding rate

### 5.1 Create `analysis/compute_mitre_grounding.py`

**Contract:**
- **Inputs:** `results/rq2_mitre_audit.json` (for the mapping), MVE outputs file, `risk_scores.npz` (for surfaced mask + attack_category alignment).
- **Output:** `results/rq2_mitre_grounding.json`.
- **Runtime:** seconds.

**DO NOT GUESS** the MVE output file path and field names. Same as the alignment script in `RQ2_FAITHFULNESS_SPEC.md` §7.1, this script needs to know:
- Where MVE outputs live (e.g., `results/mve_outputs.jsonl`)
- The field name for Layer 1 text (`layer1_why`, `layer_1_why`, `why`?)
- The field name for mode tagging (`mode`, `degradation_badge`?)

If alignment was already implemented, reuse the same loader to keep them in sync.

### 5.2 Algorithm

```
STEP 1 — Load mapping from audit JSON:
  audit = load(results/rq2_mitre_audit.json)
  category_to_techniques = {}
  for category, entry in audit["mappings_summary"].items():
      # Each technique contributes T-IDs + human names to the match set
      match_terms = set()
      for tid in entry["technique_ids"]:
          if tid: match_terms.add(tid.lower())
      for name in entry["technique_names"]:
          if name: match_terms.add(name.lower())
      category_to_techniques[category] = match_terms

STEP 2 — Load alerts:
  data = np.load(NPZ_PATH)
  row_ids = data["row_id"]
  attack_categories = data["attack_category"]
  fusion_classes = data["fusion_class"]
  surfaced_mask = fusion_classes != "BENIGN"

STEP 3 — Load MVE outputs aligned by row_id:
  mve = load_mve_outputs(MVE_PATH)  # dict keyed by row_id

STEP 4 — Per-alert grounding decision:
  For each alert i:
      cat = attack_categories[i]
      if cat == "normal" or row_ids[i] not in mve:
          continue
      layer1 = mve[row_ids[i]]["layer1_why"].lower()

      # Expected technique terms for THIS category
      expected_terms = category_to_techniques.get(cat, set())

      # Match: ANY mapped term appears in layer1
      tid_hits = [
          t for t in expected_terms
          if t.startswith("t") and len(t) <= 10 and t in layer1
      ]
      name_hits = [
          t for t in expected_terms
          if not t.startswith("t") and t in layer1
      ]

      grounded = bool(tid_hits or name_hits)
      strict_grounded = bool(tid_hits and name_hits)

      record = {
          row_id, category, mode, fusion_class,
          surfaced, grounded, strict_grounded,
          tid_hits_count, name_hits_count,
      }

STEP 5 — Aggregate three views:
  - Headline: rate over surfaced alerts only
  - Breakdown: rate per attack_category (paired with expected technique)
  - Appendix: rate over all MVE outputs
  - Both: strict_grounded rate over surfaced (appendix metric)

STEP 6 — Mode A vs Mode B breakdown:
  Same headline metric, filtered by mode.
```

### 5.3 Output schema

`results/rq2_mitre_grounding.json`:

```json
{
  "_meta": {
    "schema_version": "1.0",
    "generated_at": "<ISO-8601>",
    "generated_by": "analysis/compute_mitre_grounding.py",
    "inputs": {
      "audit_json": "results/rq2_mitre_audit.json",
      "mve_outputs": "results/mve_outputs.jsonl",
      "risk_scores_npz": "results/reports/risk_scores.npz",
      "n_surfaced": 384,
      "n_total_mve": 412
    },
    "config": {
      "match_rule": "case-insensitive substring; T-ID OR human name accepted",
      "search_scope": "layer1_why field only",
      "strict_appendix_metric": "T-ID AND human name both present"
    }
  },
  "headline": {
    "_scope": "surfaced alerts (fusion_class != BENIGN)",
    "n_evaluated": 384,
    "grounded_pct": 0.92,
    "target": 0.90,
    "pass": true,
    "strict_grounded_pct": 0.78
  },
  "by_attack_category": {
    "Data Alteration": {
      "n_evaluated": 144,
      "expected_terms": ["T1565", "T1565.001", "data manipulation",
                         "stored data manipulation"],
      "grounded_pct": 0.94,
      "strict_grounded_pct": 0.82,
      "_pair_validity": "paired against mapped technique for this category"
    },
    "Spoofing": {
      "n_evaluated": 240,
      "expected_terms": ["T1556", "modify authentication process"],
      "grounded_pct": 0.90,
      "strict_grounded_pct": 0.74
    }
  },
  "by_mode": {
    "A_llm": {
      "n_evaluated": 312,
      "grounded_pct": 0.88,
      "strict_grounded_pct": 0.72
    },
    "B_rule": {
      "n_evaluated": 72,
      "grounded_pct": 1.00,
      "strict_grounded_pct": 0.99,
      "_note": "Rule-based MVE injects MITRE terms by construction"
    }
  },
  "appendix_all_mve": {
    "_scope": "all alerts with an MVE output (including non-surfaced if any)",
    "n_evaluated": 412,
    "grounded_pct": 0.93
  },
  "failure_examples": [
    {
      "row_id": 1789,
      "category": "Spoofing",
      "mode": "A",
      "expected_terms": ["T1556", "modify authentication process"],
      "layer1_excerpt": "...detected unusual authentication anomalies on the device...",
      "_note": "LLM described concept without naming technique."
    }
  ]
}
```

### 5.4 Implementation outline

```python
# analysis/compute_mitre_grounding.py
"""
Measure how often MVE Layer 1 outputs reference the mapped MITRE technique.

Reads:
  - results/rq2_mitre_audit.json     (for the category-to-technique map)
  - <mve_outputs_path>               (DO NOT GUESS — confirm)
  - results/reports/risk_scores.npz  (for row_id + attack_category + fusion_class)

Writes results/rq2_mitre_grounding.json.
Runtime: seconds.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPO_ROOT / "results/rq2_mitre_audit.json"
NPZ = REPO_ROOT / "results/reports/risk_scores.npz"
MVE_OUTPUTS = REPO_ROOT / "results/mve_outputs.jsonl"   # DO NOT GUESS — verify
OUT = REPO_ROOT / "results/rq2_mitre_grounding.json"

GROUNDED_TARGET = 0.90


def _load_match_terms_from_audit():
    """Build category -> set of match terms (T-IDs + human names, lowercase)."""
    audit = json.loads(AUDIT.read_text())
    out = {}
    for cat, entry in audit["mappings_summary"].items():
        terms = set()
        for tid in entry.get("technique_ids", []):
            if tid:
                terms.add(str(tid).lower())
        for name in entry.get("technique_names", []):
            if name:
                terms.add(str(name).lower())
        out[cat] = terms
    return out


def _load_mve(path):
    """
    Load MVE outputs keyed by row_id.

    DO NOT GUESS — adapt to the actual MVE output schema. Confirm field names:
      - row_id field (int)
      - layer1_why field (str)
      - mode field ("A" / "B" / similar)
      - fusion_class field
    """
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[int(rec["row_id"])] = rec
    return out


def _split_terms(terms):
    """Separate T-IDs from human names. T-IDs match regex T followed by digits."""
    tids, names = set(), set()
    for t in terms:
        if t.startswith("t") and any(c.isdigit() for c in t[:6]):
            tids.add(t)
        else:
            names.add(t)
    return tids, names


def _grounding_for_alert(layer1_text, expected_terms):
    """Return (grounded, strict_grounded, tid_hits, name_hits)."""
    if not layer1_text:
        return False, False, 0, 0
    txt = layer1_text.lower()
    tids, names = _split_terms(expected_terms)
    tid_hits = sum(1 for t in tids if t in txt)
    name_hits = sum(1 for n in names if n in txt)
    grounded = (tid_hits + name_hits) > 0
    strict = (tid_hits > 0) and (name_hits > 0)
    return grounded, strict, tid_hits, name_hits


def _aggregate(records, scope_name):
    n = len(records)
    if n == 0:
        return {"_scope": scope_name, "n_evaluated": 0}
    grounded = sum(1 for r in records if r["grounded"])
    strict = sum(1 for r in records if r["strict_grounded"])
    return {
        "_scope": scope_name,
        "n_evaluated": n,
        "grounded_pct": grounded / n,
        "strict_grounded_pct": strict / n,
    }


def main():
    category_to_terms = _load_match_terms_from_audit()
    mve = _load_mve(MVE_OUTPUTS)
    data = np.load(NPZ, allow_pickle=False)

    row_ids = data["row_id"]
    attack_cats = data["attack_category"]
    fusion = data["fusion_class"]

    records = []
    failure_examples = []
    for i in range(len(row_ids)):
        rid = int(row_ids[i])
        cat = str(attack_cats[i])
        if cat == "normal" or rid not in mve:
            continue

        mve_rec = mve[rid]
        layer1 = mve_rec.get("layer1_why", "") or ""
        expected = category_to_terms.get(cat, set())
        grounded, strict, tid_h, name_h = _grounding_for_alert(layer1, expected)

        rec = {
            "row_id": rid,
            "category": cat,
            "mode": mve_rec.get("mode", "unknown"),
            "fusion_class": str(fusion[i]),
            "surfaced": str(fusion[i]) != "BENIGN",
            "grounded": grounded,
            "strict_grounded": strict,
            "tid_hits": tid_h,
            "name_hits": name_h,
        }
        records.append(rec)

        if not grounded and len(failure_examples) < 10:
            failure_examples.append({
                "row_id": rid,
                "category": cat,
                "mode": rec["mode"],
                "expected_terms": sorted(expected),
                "layer1_excerpt": layer1[:200],
            })

    # Headline: surfaced only
    surfaced = [r for r in records if r["surfaced"]]
    headline = _aggregate(surfaced, "surfaced alerts (fusion_class != BENIGN)")
    headline["target"] = GROUNDED_TARGET
    headline["pass"] = headline.get("grounded_pct", 0) >= GROUNDED_TARGET

    # By attack category (paired with mapped technique)
    by_cat = {}
    for cat in sorted({r["category"] for r in surfaced}):
        cat_records = [r for r in surfaced if r["category"] == cat]
        agg = _aggregate(cat_records, f"surfaced alerts of category={cat}")
        agg["expected_terms"] = sorted(category_to_terms.get(cat, set()))
        agg["_pair_validity"] = "paired against mapped technique for this category"
        by_cat[cat] = agg

    # By mode
    by_mode = {}
    for m in sorted({r["mode"] for r in surfaced}):
        m_records = [r for r in surfaced if r["mode"] == m]
        agg = _aggregate(m_records, f"surfaced alerts, mode={m}")
        if m == "B":
            agg["_note"] = "Rule-based MVE injects MITRE terms by construction"
        by_mode[m] = agg

    # Appendix: all MVE
    appendix = _aggregate(records, "all alerts with an MVE output")

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/compute_mitre_grounding.py",
            "inputs": {
                "audit_json": str(AUDIT.relative_to(REPO_ROOT)),
                "mve_outputs": str(MVE_OUTPUTS.relative_to(REPO_ROOT)),
                "risk_scores_npz": str(NPZ.relative_to(REPO_ROOT)),
                "n_surfaced": len(surfaced),
                "n_total_mve": len(records),
            },
            "config": {
                "match_rule": "case-insensitive substring; T-ID OR human name accepted",
                "search_scope": "layer1_why field only",
                "strict_appendix_metric": "T-ID AND human name both present",
            },
        },
        "headline": headline,
        "by_attack_category": by_cat,
        "by_mode": by_mode,
        "appendix_all_mve": appendix,
        "failure_examples": failure_examples,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"Grounding rate (surfaced): {headline.get('grounded_pct', 0):.3f} "
          f"(target ≥ {GROUNDED_TARGET})")


if __name__ == "__main__":
    main()
```

### 5.5 Verification

```bash
python -m analysis.compute_mitre_grounding
cat results/rq2_mitre_grounding.json | python -m json.tool | head -40
# Expected: headline grounded_pct >= 0.90
```

---

## 6. Phase 3 — Tests + CI gates

### 6.1 Add `tests/test_mitre_audit.py`

```python
"""Smoke + invariant tests for the MITRE config audit."""
import json
from pathlib import Path

import pytest

OUT = Path("results/rq2_mitre_audit.json")


@pytest.fixture(scope="module")
def audit():
    if not OUT.exists():
        pytest.skip("Run analysis/audit_mitre_config.py first")
    return json.loads(OUT.read_text())


def test_audit_passed(audit):
    assert audit["headline"]["audit_pass"], (
        f"MITRE audit failed with {audit['headline']['n_fail']} FAIL findings. "
        f"See results/rq2_mitre_audit.json for details."
    )


def test_no_orphan_categories(audit):
    orphans = audit["headline"]["orphan_categories"]
    assert not orphans, f"Attack categories without MITRE mapping: {orphans}"


def test_framework_version_pinned(audit):
    assert audit["headline"]["mitre_framework_version"], \
        "mitre_framework_version is required (RQ2_expected_outputs.md §5.2)"


def test_last_validated_present(audit):
    h = audit["headline"]
    has_some = h.get("last_validated_top_level") or any(
        v.get("last_validated") for v in audit["mappings_summary"].values()
    )
    assert has_some, "last_validated must be set top-level or per-mapping"
```

### 6.2 Add `tests/test_mitre_grounding.py`

```python
"""Smoke + invariant tests for MVE-MITRE grounding rate."""
import json
from pathlib import Path

import pytest

OUT = Path("results/rq2_mitre_grounding.json")


@pytest.fixture(scope="module")
def grounding():
    if not OUT.exists():
        pytest.skip("Run analysis/compute_mitre_grounding.py first")
    return json.loads(OUT.read_text())


def test_schema_complete(grounding):
    for key in ["_meta", "headline", "by_attack_category",
                "by_mode", "appendix_all_mve"]:
        assert key in grounding


def test_grounding_target(grounding):
    h = grounding["headline"]
    assert h["pass"], (
        f"MITRE grounding {h['grounded_pct']:.3f} below target {h['target']}"
    )


def test_mode_b_near_perfect(grounding):
    """Mode B rule-based should be ≥99% grounded by construction."""
    by_mode = grounding["by_mode"]
    if "B" in by_mode and by_mode["B"]["n_evaluated"] > 0:
        assert by_mode["B"]["grounded_pct"] >= 0.99, (
            f"Mode B grounding {by_mode['B']['grounded_pct']:.3f} below 0.99"
        )


def test_per_category_no_zero_columns(grounding):
    """No mapped category should have 0% grounding — likely a mapping bug."""
    for cat, stats in grounding["by_attack_category"].items():
        if stats.get("n_evaluated", 0) >= 10:
            assert stats["grounded_pct"] > 0, (
                f"Category '{cat}' has 0% grounding — investigate mapping."
            )
```

### 6.3 Verification

```bash
pytest tests/test_mitre_audit.py tests/test_mitre_grounding.py -v
# Expected: all pass once both scripts have run
```

---

## 7. Execution order

```bash
# ─── PHASE 0: SCHEMA DISCOVERY (HUMAN REVIEW) ─────────────────
python scripts/discover_mitre_yaml_schema.py
# DEVELOPER CONFIRMS: mappings key, cardinality pattern, last_validated location
# Claude Code adapts _mappings_block() and _extract_techniques() accordingly

# ─── PHASE 1: CONFIG AUDIT ────────────────────────────────────
python -m analysis.audit_mitre_config
pytest tests/test_mitre_audit.py -v
# Expected: audit_pass: true, 4 tests pass

# ─── PHASE 2: GROUNDING RATE ──────────────────────────────────
# Verify mve_outputs.jsonl path + schema first
python -m analysis.compute_mitre_grounding
pytest tests/test_mitre_grounding.py -v
# Expected: grounded_pct >= 0.90, 4 tests pass

# ─── PHASE 3: INTEGRATE INTO compute_rq2_metrics.py ───────────
# Add _load_mitre_subfiles() reading both JSONs into a "mitre_grounding" block
```

---

## 8. Integration with `compute_rq2_metrics.py`

When the master aggregator (Phase 18 of the RQ2 overview) is built, it should fold these two JSONs in under a `mitre_grounding` block:

```python
def _load_mitre_subfiles():
    audit_p = REPO_ROOT / "results/rq2_mitre_audit.json"
    grounding_p = REPO_ROOT / "results/rq2_mitre_grounding.json"

    block = {"_status": "pending", "_merged_at": None}
    if audit_p.exists() and grounding_p.exists():
        block = {
            "_status": "complete",
            "_merged_at": datetime.now(timezone.utc).isoformat(),
            "config_audit": json.loads(audit_p.read_text()),
            "layer1_grounding": json.loads(grounding_p.read_text()),
        }
    elif audit_p.exists():
        block = {
            "_status": "partial — grounding pending",
            "config_audit": json.loads(audit_p.read_text()),
        }
    return block
```

In the aggregator, add `out["mitre_grounding"] = _load_mitre_subfiles()`.

---

## 9. Open questions to surface (DO NOT GUESS)

Claude Code must pause and ask:

1. **Phase 0 — YAML location.** Is `config/attack_to_mitre_mapping.yaml` the correct path? If not, where?
2. **Phase 0 — YAML structure.** What is the actual mappings-block key name? What cardinality pattern (A/B/C)?
3. **Phase 0 — `last_validated` convention.** Top-level only, per-entry only, or both supported?
4. **Phase 2 — MVE output path.** Same question from `RQ2_FAITHFULNESS_SPEC.md` §11 — confirm path and field names. If alignment script already loads it, reuse that loader.
5. **Phase 2 — Mode tagging.** Same as above — confirm how Mode A vs B is recorded.
6. **MedSec attack categories.** This spec focuses on EHMS test data per `risk_scores.npz`. If MedSec coverage is needed (per `RQ2_expected_outputs.md §5.1`), a sibling `compute_mitre_grounding_medsec.py` is the right pattern (mirrors `compute_rq1_metrics_medsec.py` from `RQ1_PIPELINE_SPEC.md`).

---

## 10. Coverage map — RQ2.e expected outputs → pipeline phase

| RQ2_expected_outputs.md §5 item | Phase | Status |
|---|---|---|
| Per-attack-category MITRE coverage table | 2 | `by_attack_category` block |
| `config/attack_to_mitre_mapping.yaml` audit | 1 | `rq2_mitre_audit.json` |
| Framework version pinned | 1 | A2 check |
| `last_validated` per mapping | 1 | A3 check |
| 100% attack categories with mapping (no orphans) | 1 | A4 check + `orphan_categories` |
| > 90% MVE Layer 1 references MITRE | 2 | `headline.grounded_pct` |
| Confidence-weighted accuracy | — | NOT IN SCOPE — qualitative review per the doc |

Every numbered RQ2.e item is traceable to a phase except "confidence-weighted accuracy," which the expected-outputs doc explicitly marks as qualitative.

---

## End of spec

Implementation order: Phase 0 (discovery) → Phase 1 (audit) → Phase 2 (grounding) → Phase 3 (tests). Phases 1 and 2 are independent of Tracks 1, 3, 4, 5 of the RQ2 overview, so this track can be implemented in parallel with the others.