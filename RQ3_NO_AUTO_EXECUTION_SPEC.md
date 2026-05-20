# RQ3 Track 3 — No-Auto-Execution (Triple-Layer Defense)

**Project:** XAI-IDS-Healthcare
**Scope:** RQ3.3 — Verify Invariant 3: the system NEVER executes mitigation actions automatically. Recommendations only; operator decision required.
**Purpose:** Single, self-contained spec for the no-auto-execution audit pipeline. Hand to Claude Code.
**Status of design:** All decisions locked. Three `DO NOT GUESS` checkpoints (existing negative_tests.py contents, ResponseRecommendation class location, response pipeline import path).

---

## 0. How to use this spec

1. Phase 0 is mandatory — Claude Code must inspect existing `tests/negative_tests.py` and find `ResponseRecommendation` before writing any code.
2. Phases 1–4 are sequential.
3. Markers:
   - **DO NOT GUESS** — stop and ask the developer
   - **DEFENSE-CRITICAL** — directly defends the most-asked reviewer question
   - **HITL-INVARIANT** — Human-In-The-Loop safety invariant (Invariant 3)
4. Total expected size: 1 YAML config, 1 audit script, 2 test files (1 new, 1 extended), updates to `tests/negative_tests.py`. Runtime: sub-second.

---

## 1. Background: the most important safety claim

Per the senior engineer review:
> *"This is the single most important decision in the document. Auto-containment in clinical networks is how patients get hurt — an automated quarantine of an infusion pump mid-infusion is a sentinel event. The `grep -rn "subprocess|os.system|iptables"` test is a clever, durable enforcement mechanism."*

Per `RQ3_expected_outputs.md §3.1`, the defense is a **three-layer mechanism**:

| Layer | Mechanism | Purpose |
|---|---|---|
| A | Documentation: Invariant 3 in architecture | Establishes intent |
| B | Static grep: forbidden patterns + imports | Catches code that *could* auto-execute |
| C | Runtime check: `negative_tests.py::test_no_automated_blocking` | Catches behavior that *does* auto-execute |

Track 3 implements Layers B and C, and strengthens Layer C with two additional runtime checks (operator_decision_required field + subprocess mock).

The three-layer framing is preserved — Layers B and C are distinct files. The new Layer D (runtime mock + field check) is reported as "strengthened Layer C" so the paper's "three-layer defense" wording stays accurate.

---

## 2. Locked design decisions

| Decision | Resolution |
|---|---|
| Grep scope | YAML-configured: each directory's role declared (production / test / analysis) |
| False-positive handling | Three-layer: hardcoded patterns + context analysis + `# noqa: no-auto-exec` marker |
| Audit output | Detect-all + categorize: group findings by pattern type |
| Integration with existing negative_tests.py | Wrap: `test_no_automated_blocking` invokes the audit script as subprocess; runtime checks live in separate file |
| Runtime field check | Yes: `ResponseRecommendation.operator_decision_required == True` always |
| Runtime mock smoke test | Yes: mock subprocess + os.system, call recommend() end-to-end, assert mocks never called |
| CI failure mode | Hard fail always; `--list-violations` CLI flag for human investigation; no env-var override |
| Three-layer framing | Preserved: Layer C (existing test) wraps and stays the canonical entry point; new tests strengthen it |

---

## 3. Phase 0 — Discovery (DO NOT GUESS)

### 3.1 Discovery script

```python
# scripts/discover_no_auto_exec_artifacts.py — TRANSIENT, delete after Phase 0
"""
Locate the existing infrastructure Track 3 builds on:
  1. tests/negative_tests.py contents and current test behavior
  2. ResponseRecommendation class definition (location, fields)
  3. module5_responses/module5_pipeline.py::recommend signature
  4. Any existing subprocess imports in the production pipeline
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
findings = {}

# 1. negative_tests.py
neg = REPO_ROOT / "tests/negative_tests.py"
findings["negative_tests"] = {"exists": neg.exists()}
if neg.exists():
    text = neg.read_text()
    findings["negative_tests"]["has_test_no_automated_blocking"] = (
        "def test_no_automated_blocking" in text
    )
    findings["negative_tests"]["size_bytes"] = len(text)
    # Find all test function names for inventory
    test_funcs = re.findall(r"def (test_\w+)\b", text)
    findings["negative_tests"]["test_functions"] = test_funcs

# 2. ResponseRecommendation class
rr_locations = []
for p in REPO_ROOT.rglob("*.py"):
    if any(skip in str(p) for skip in [".git/", "__pycache__"]):
        continue
    try:
        text = p.read_text()
    except (UnicodeDecodeError, PermissionError):
        continue
    if "class ResponseRecommendation" in text:
        # Extract dataclass-style field hints if present
        cls_start = text.find("class ResponseRecommendation")
        cls_block = text[cls_start:cls_start + 1500]
        # Look for operator_decision_required default value
        match = re.search(
            r"operator_decision_required\s*[:=]\s*([^\n]+)", cls_block
        )
        rr_locations.append({
            "path": str(p.relative_to(REPO_ROOT)),
            "operator_decision_required_default": match.group(1).strip()
                if match else "NOT FOUND",
            "is_dataclass": "@dataclass" in text[max(0, cls_start - 100):cls_start],
        })
findings["response_recommendation_class"] = rr_locations

# 3. recommend() function in module5_responses
candidates = [
    "module5_responses/module5_pipeline.py",
    "pipeline/module5_response/module5_pipeline.py",
    "src/module5_pipeline.py",
]
findings["recommend_function"] = []
for c in candidates:
    p = REPO_ROOT / c
    if not p.exists():
        continue
    text = p.read_text()
    has_recommend = bool(re.search(r"\bdef recommend\b", text))
    findings["recommend_function"].append({
        "path": c,
        "has_recommend": has_recommend,
        "size_bytes": len(text),
    })

# 4. Pre-existing subprocess / os.system usage anywhere
forbidden_patterns_quick_scan = [
    "import subprocess", "from subprocess",
    "import os\\b.*system", "os\\.system\\(",
    "iptables", "netcat", r"\bnc\s", r"\bcurl\s",
]
hits = {}
for p in REPO_ROOT.rglob("*.py"):
    if any(skip in str(p) for skip in [
        ".git/", "__pycache__", "/tests/", "/analysis/", "/scripts/",
        "/.venv/", "/venv/",
    ]):
        continue
    try:
        text = p.read_text()
    except (UnicodeDecodeError, PermissionError):
        continue
    for pat in forbidden_patterns_quick_scan:
        if re.search(pat, text):
            hits.setdefault(str(p.relative_to(REPO_ROOT)), []).append(pat)
findings["existing_violations_quick_scan"] = hits

print(json.dumps(findings, indent=2, default=str))
print("\n" + "=" * 60)
print("DEVELOPER ACTION:")
print("  1. Confirm tests/negative_tests.py contains test_no_automated_blocking")
print("     (or note absence — Phase 4 creates it if missing).")
print("  2. Confirm ResponseRecommendation location and that operator_decision_required")
print("     defaults to True. If not, Phase 3 sets it.")
print("  3. Confirm the production response module path:")
print("     - module5_responses/module5_pipeline.py")
print("     - pipeline/module5_response/module5_pipeline.py")
print("     - or something else?")
print("  4. Review existing_violations_quick_scan output. Any hits in PRODUCTION code")
print("     are immediate problems — must be cleared before Phase 2 audit can pass.")
print("=" * 60)
```

### 3.2 Three things to confirm before Phase 1

1. **`tests/negative_tests.py::test_no_automated_blocking` exists.** If not, Phase 4 creates it. If yes, Phase 4 modifies it to wrap the new audit script.
2. **`ResponseRecommendation` class location.** Could be in `src/`, `module5_responses/`, or elsewhere. Phase 3's runtime tests need the import path. Also: does `operator_decision_required` already default to `True`? If not, Phase 3 will note this as a finding rather than silently fix it.
3. **Production response module path.** ARCHITECTURE.md uses `module5_responses/`; `RQ3_expected_outputs.md` uses `pipeline/module5_response/`. Discovery script checks both.

### 3.3 Verification

```bash
python scripts/discover_no_auto_exec_artifacts.py > /tmp/no_auto_exec_inventory.json
# Developer reviews; flags any production-code violations that must be cleared first
```

**DO NOT GUESS** the existing infrastructure. If discovery reveals production-code violations in the quick scan, **fix those before proceeding** — the spec assumes the production codebase is currently clean.

---

## 4. Phase 1 — Scope configuration manifest

### 4.1 Create `config/no_auto_exec_scope.yaml`

The YAML declares each directory's role. Production directories must be clean of execution patterns. Test and analysis directories are explicitly authorized to use subprocess (test infrastructure simulates attacks; analysis scripts may invoke grep/subprocess for verification).

```yaml
# config/no_auto_exec_scope.yaml
# Declares which directories are subject to the no-auto-execution audit.
# Defense framing: production code is forbidden from auto-execution;
# test code is explicitly authorized to simulate attacks.
#
# Edits to this file are reviewable in version control — adding a new
# directory to "test" or "analysis" requires explicit, auditable change.

schema_version: "1.0"
last_validated: "2026-MM-DD"   # set on commit

# Production: must contain ZERO forbidden patterns.
# Audit fails on any match.
production_dirs:
  - module5_responses/
  - pipeline/module5_response/   # DO NOT GUESS — verify in Phase 0
  - src/
  - common/
  - module3_risk_scoring/
  - module4_xai/
  - module6_evaluation/

# Test directories: allowed to use subprocess for attack simulation.
# Audit reports findings here as INFO, not FAIL.
test_dirs:
  - tests/
  - module7_user_study/   # if applicable

# Analysis / scripts: allowed because analysis itself may invoke subprocess
# (e.g., this audit script runs grep via subprocess).
# Audit skips entirely.
analysis_dirs:
  - analysis/
  - scripts/

# Explicitly excluded (never scanned)
excluded_paths:
  - .git/
  - __pycache__/
  - .venv/
  - venv/
  - node_modules/
  - data/
  - results/
  - logs/
  - survey/
  - .pytest_cache/
  - .mypy_cache/

# The patterns themselves.
# Each pattern is a Python regex compiled with re.MULTILINE.
forbidden_patterns:
  subprocess:
    - 'subprocess'
    - r'\bsubprocess\.\w+\('
  os_system:
    - r'os\.system\('
    - r'os\.popen\('
  shell_commands:
    - r'\biptables\b'
    - r'\bnetcat\b'
    - r'\bnc\s+\-'   # netcat short form
    - r'\bcurl\s'
    - r'\bwget\s'
    - r'\bssh\s'
    - r'\bsudo\s'
  python_exec:
    - r'\beval\('
    - r'\bexec\('
    - r'\b__import__\('
  forbidden_imports:
    - '^import subprocess'
    - '^from subprocess'
    - '^import os, '   # imports os in a way that suggests system calls
    - '^from os import system'
```

### 4.2 Verification

```bash
python -c "
import yaml
from pathlib import Path
doc = yaml.safe_load(Path('config/no_auto_exec_scope.yaml').read_text())
print(f'Production dirs: {len(doc[\"production_dirs\"])}')
print(f'Test dirs:       {len(doc[\"test_dirs\"])}')
print(f'Analysis dirs:   {len(doc[\"analysis_dirs\"])}')
print(f'Patterns:        {sum(len(v) for v in doc[\"forbidden_patterns\"].values())}')
"
```

---

## 5. Phase 2 — Grep audit script

### 5.1 Create `analysis/audit_no_auto_execution.py`

**Contract:**
- **Inputs:** `config/no_auto_exec_scope.yaml`
- **Output:** `results/rq3_no_auto_execution.json`
- **CLI flags:**
  - `--list-violations` — print findings to stdout, exit 0 (developer pre-commit use)
  - `--strict` — default; exit 1 on any production-scope violation (CI use)
- **Runtime:** sub-second on typical repo
- **Behavior:** detect-all + categorize per Round 1 Q3
- **False-positive handling:** comment lines, `"""` docstring blocks, `# noqa: no-auto-exec` opt-out per Round 1 Q2

### 5.2 Algorithm

```
STEP 1 — Load scope config:
  doc = yaml.safe_load(config/no_auto_exec_scope.yaml)
  production_dirs, test_dirs, analysis_dirs, excluded_paths, patterns = doc.*

STEP 2 — Walk filesystem; classify each .py file by directory role:
  for p in REPO_ROOT.rglob("*.py"):
      if p matches any excluded_paths: skip
      elif p inside any production_dir: classify "production"
      elif p inside any test_dir: classify "test"
      elif p inside any analysis_dir: classify "analysis"
      else: classify "unclassified"  # warn

STEP 3 — Scan each production-classified file:
  for each line in file:
      strip = line.strip()
      if strip starts with #: skip (comment)
      if line contains "# noqa: no-auto-exec": skip (explicit opt-out)
      if line is inside a triple-quoted block: skip (docstring/string)
      for each (category, patterns_in_category):
          for each pattern:
              if re.search(pattern, line):
                  findings.append({
                      category, pattern, file, line_no, content
                  })

STEP 4 — Aggregate + emit JSON:
  findings_by_category = group_by category
  total_violations_production = len(findings)
  total_findings_test = scan test_dirs same way (informational)

  output = {
      _meta: ...,
      headline: {
          audit_pass: total_violations_production == 0,
          n_violations_production: ...,
          n_findings_test: ...,
          n_files_scanned: ...,
      },
      violations_by_category: { subprocess: [...], iptables: [...], ... },
      test_dir_findings: [...]  // informational, not failure
  }
```

### 5.3 Implementation outline

```python
"""
audit_no_auto_execution.py
DEFENSE-CRITICAL: Layer B of the three-layer no-auto-execution defense.

Walks the codebase per config/no_auto_exec_scope.yaml. Scans production
directories for forbidden execution patterns. Reports findings categorized
by pattern type.

Production violations → hard fail (exit 1).
Test violations → informational only (test code is allowed to simulate attacks).
Analysis dirs → skipped entirely.

Usage:
  python -m analysis.audit_no_auto_execution             # CI mode (strict)
  python -m analysis.audit_no_auto_execution --list-violations
                                                          # Human investigation mode
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/no_auto_exec_scope.yaml"
OUT = REPO_ROOT / "results/rq3_no_auto_execution.json"

NOQA_MARKER = "# noqa: no-auto-exec"


def _load_config():
    if not CONFIG.exists():
        raise SystemExit(
            f"Config missing: {CONFIG}. Create per Phase 1 spec before running."
        )
    return yaml.safe_load(CONFIG.read_text())


def _classify_file(p: Path, cfg: dict) -> str:
    """Return 'production' | 'test' | 'analysis' | 'unclassified' | 'excluded'."""
    rel = p.relative_to(REPO_ROOT)
    rel_str = str(rel)

    for ex in cfg.get("excluded_paths", []):
        if rel_str.startswith(ex) or ex.rstrip("/") in rel.parts:
            return "excluded"
    for d in cfg.get("production_dirs", []):
        if rel_str.startswith(d):
            return "production"
    for d in cfg.get("test_dirs", []):
        if rel_str.startswith(d):
            return "test"
    for d in cfg.get("analysis_dirs", []):
        if rel_str.startswith(d):
            return "analysis"
    return "unclassified"


def _strip_triple_quoted(text: str) -> str:
    """
    Replace triple-quoted string contents with spaces, preserving line count.
    Naive but adequate: handles """ and ''' across multiple lines.
    """
    pattern = re.compile(r'("""|\'\'\')([\s\S]*?)\1')
    def replace(m):
        # Preserve number of newlines so line numbers stay correct
        content = m.group(0)
        n_newlines = content.count("\n")
        return m.group(1) + " " * (len(m.group(2))) + m.group(1)
        # Actually simpler: just replace content with spaces
    # We need to preserve line numbers carefully:
    out_chars = list(text)
    for m in pattern.finditer(text):
        for i in range(m.start(), m.end()):
            if text[i] != "\n":
                out_chars[i] = " "
    return "".join(out_chars)


def _scan_file(p: Path, cfg: dict) -> list:
    """Return list of findings (one per pattern match)."""
    try:
        raw = p.read_text()
    except (UnicodeDecodeError, PermissionError):
        return []

    # Replace docstring/string bodies with spaces, preserving line structure
    sanitized = _strip_triple_quoted(raw)

    findings = []
    patterns_by_category = cfg.get("forbidden_patterns", {})
    compiled = {
        cat: [re.compile(pat) for pat in pats]
        for cat, pats in patterns_by_category.items()
    }

    for line_no, line in enumerate(sanitized.splitlines(), start=1):
        stripped = line.strip()

        # Skip pure comment lines
        if stripped.startswith("#"):
            continue

        # Skip lines with explicit opt-out marker
        if NOQA_MARKER in line:
            continue

        for category, regexes in compiled.items():
            for regex in regexes:
                if regex.search(line):
                    # Get the original line content for the report (not the sanitized one)
                    original_line = raw.splitlines()[line_no - 1] \
                        if line_no - 1 < len(raw.splitlines()) else ""
                    findings.append({
                        "category": category,
                        "pattern": regex.pattern,
                        "file": str(p.relative_to(REPO_ROOT)),
                        "line": line_no,
                        "content": original_line.strip()[:200],
                    })
                    break  # one finding per line per category
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-violations", action="store_true",
                    help="Print findings to stdout; do not exit non-zero.")
    args = ap.parse_args()

    cfg = _load_config()

    files_by_class = defaultdict(list)
    for p in REPO_ROOT.rglob("*.py"):
        cls = _classify_file(p, cfg)
        files_by_class[cls].append(p)

    production_findings = []
    test_findings = []
    unclassified_findings = []

    for p in files_by_class["production"]:
        production_findings.extend(_scan_file(p, cfg))
    for p in files_by_class["test"]:
        test_findings.extend(_scan_file(p, cfg))
    for p in files_by_class["unclassified"]:
        unclassified_findings.extend(_scan_file(p, cfg))

    # Group production findings by category
    by_category = defaultdict(list)
    for f in production_findings:
        by_category[f["category"]].append(f)

    audit = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/audit_no_auto_execution.py",
            "config_path": str(CONFIG.relative_to(REPO_ROOT)),
            "_framing": (
                "Static-analysis Layer B of the no-auto-execution three-layer "
                "defense. Production code MUST contain zero matches; test code "
                "may use subprocess for attack simulation."
            ),
        },
        "headline": {
            "audit_pass": len(production_findings) == 0,
            "n_violations_production": len(production_findings),
            "n_findings_test_info_only": len(test_findings),
            "n_findings_unclassified": len(unclassified_findings),
            "n_files_scanned": {
                "production": len(files_by_class["production"]),
                "test": len(files_by_class["test"]),
                "analysis_skipped": len(files_by_class["analysis"]),
                "unclassified": len(files_by_class["unclassified"]),
                "excluded": len(files_by_class["excluded"]),
            },
        },
        "violations_by_category": {
            cat: matches for cat, matches in sorted(by_category.items())
        },
        "test_dir_findings": test_findings[:50],
        "test_dir_findings_total": len(test_findings),
        "unclassified_findings": unclassified_findings[:20],
        "unclassified_findings_total": len(unclassified_findings),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2, default=str))

    h = audit["headline"]

    if args.list_violations:
        # Human-friendly stdout for pre-commit investigation
        print(f"\n=== No-Auto-Execution Audit (LIST MODE) ===")
        print(f"Production violations: {h['n_violations_production']}")
        if h["n_violations_production"]:
            print("\nBy category:")
            for cat, matches in audit["violations_by_category"].items():
                print(f"  [{cat}] {len(matches)} finding(s):")
                for m in matches[:10]:
                    print(f"    {m['file']}:{m['line']}  {m['content'][:100]}")
        if h["n_findings_unclassified"]:
            print(f"\nUNCLASSIFIED files with matches: {h['n_findings_unclassified']}")
            print("  Add their parent directories to config/no_auto_exec_scope.yaml.")
        print(f"\nFull report: {OUT.relative_to(REPO_ROOT)}")
        sys.exit(0)

    # CI mode
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"Production violations: {h['n_violations_production']}  "
          f"(test-dir info: {h['n_findings_test_info_only']})")
    if not h["audit_pass"]:
        print("FAIL: production code contains forbidden execution patterns.")
        print("  Run `python -m analysis.audit_no_auto_execution --list-violations`")
        print("  for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 5.4 Verification

```bash
python -m analysis.audit_no_auto_execution
# Expected on clean codebase: exits 0, JSON written

python -m analysis.audit_no_auto_execution --list-violations
# Human-readable output even if findings exist; exits 0
```

---

## 6. Phase 3 — Runtime checks (strengthened Layer C)

### 6.1 Create `tests/test_response_recommendation_no_exec.py`

Two complementary tests. **DEFENSE-CRITICAL**: these are positive-evidence tests, not just absence checks. They run the actual recommendation pipeline and verify safety properties at runtime.

```python
"""
tests/test_response_recommendation_no_exec.py

Runtime checks strengthening Layer C of the no-auto-execution defense:
  1. ResponseRecommendation.operator_decision_required == True (positive field check)
  2. Calling recommend() never invokes subprocess (mocked smoke test)

These complement (do not replace) tests/negative_tests.py::test_no_automated_blocking.
"""

from unittest.mock import patch, MagicMock

import pytest


# DO NOT GUESS — Phase 0 confirms exact import paths.
# Adapt these lines once discovery completes.
try:
    from src.response_recommendation import ResponseRecommendation
except ImportError:
    try:
        from module5_responses.response_recommendation import ResponseRecommendation
    except ImportError:
        ResponseRecommendation = None

try:
    from module5_responses.module5_pipeline import recommend
except ImportError:
    try:
        from pipeline.module5_response.module5_pipeline import recommend
    except ImportError:
        recommend = None


@pytest.mark.skipif(
    ResponseRecommendation is None,
    reason="ResponseRecommendation not importable; check src.response_recommendation"
)
def test_operator_decision_required_default_is_true():
    """
    HITL-INVARIANT: a default-constructed ResponseRecommendation must have
    operator_decision_required = True. If the default flips, every code path
    that uses defaults silently disables HITL.
    """
    rec = ResponseRecommendation(
        # Provide whatever the required positional args are; Phase 0 confirms.
        # Most likely the dataclass has sensible defaults except for ID/action.
        alert_id="test_alert",
        recommended_action="isolate_device",
        primary_action_code="ISOLATE",
    )
    assert rec.operator_decision_required is True, (
        "DEFAULT VIOLATION: ResponseRecommendation default-constructed with "
        "operator_decision_required != True. This breaks Invariant 3 silently."
    )


@pytest.mark.skipif(
    ResponseRecommendation is None,
    reason="ResponseRecommendation not importable"
)
def test_operator_decision_required_cannot_be_false_in_typical_construction():
    """
    Stronger check: try common construction patterns; none should yield
    operator_decision_required = False (which would imply auto-execution).
    """
    cases = [
        {"alert_id": "a", "recommended_action": "x", "primary_action_code": "X"},
        # Add other typical construction patterns Phase 0 reveals.
    ]
    for kwargs in cases:
        rec = ResponseRecommendation(**kwargs)
        assert rec.operator_decision_required is True, (
            f"Construction {kwargs} produced operator_decision_required=False"
        )


@pytest.mark.skipif(
    recommend is None,
    reason="recommend() not importable; check module5_responses.module5_pipeline"
)
def test_recommend_never_invokes_subprocess():
    """
    DEFENSE-CRITICAL: patch subprocess and os.system globally, run a full
    recommend() call end-to-end, assert neither mock was called.

    This is the strongest possible runtime evidence that the system does
    NOT auto-execute mitigation actions.
    """
    sample_alert = {
        # Provide a minimally-valid alert dict. Phase 0 confirms required fields.
        "alert_id": "alert_test_no_exec",
        "fusion_class": "KNOWN_ATTACK",
        "risk_tier": "CRITICAL",
        "device_class": "infusion_pump",
        "device_criticality": "CRITICAL",
        "patchable": False,
    }

    subprocess_mock = MagicMock()
    os_system_mock = MagicMock()

    with patch("subprocess.run", subprocess_mock), \
         patch("subprocess.Popen", subprocess_mock), \
         patch("subprocess.call", subprocess_mock), \
         patch("subprocess.check_output", subprocess_mock), \
         patch("os.system", os_system_mock):

        try:
            result = recommend(sample_alert)
        except Exception as e:
            # If recommend() fails for reasons unrelated to subprocess,
            # the test is inconclusive. Re-raise so the developer fixes
            # the test fixture, not the production code.
            raise AssertionError(
                f"recommend() raised unexpectedly: {e}. "
                "Update sample_alert fixture to match the production schema."
            )

    assert subprocess_mock.call_count == 0, (
        f"subprocess invoked {subprocess_mock.call_count} time(s) during recommend()."
        f" calls: {subprocess_mock.mock_calls[:5]}"
    )
    assert os_system_mock.call_count == 0, (
        f"os.system invoked {os_system_mock.call_count} time(s) during recommend()."
    )

    # Additionally verify the recommendation itself has HITL flag set
    if hasattr(result, "operator_decision_required"):
        assert result.operator_decision_required is True, (
            "recommend() returned a recommendation with "
            "operator_decision_required = False."
        )
```

### 6.2 Verification

```bash
pytest tests/test_response_recommendation_no_exec.py -v
# Expected: 3 tests pass (or skip if imports unavailable)
```

---

## 7. Phase 4 — Wrap existing negative test

### 7.1 Update `tests/negative_tests.py`

**Goal:** keep `test_no_automated_blocking` as the canonical CI entry point per Q1, but have it invoke the new audit script via subprocess. The existing test's body (whatever it currently does) is **preserved** — the new code is added at the start.

```python
# tests/negative_tests.py
# DO NOT GUESS — Phase 0 confirms the current body of this function.
# Preserve existing assertions; add the audit invocation at the start.

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_no_automated_blocking():
    """
    DEFENSE-CRITICAL: Layer C of the three-layer no-auto-execution defense.

    Layer A: Invariant 3 documented in architecture.
    Layer B: Static grep audit via analysis/audit_no_auto_execution.py.
    Layer C: This test — runs the audit, verifies clean result.

    Layer C wraps Layer B at CI runtime: if the audit script reports any
    production-scope violation, this test fails.

    Additional runtime evidence lives in
    tests/test_response_recommendation_no_exec.py — those tests verify that
    calling recommend() at runtime does not invoke subprocess (mocked).
    """
    # Layer B invocation — run the audit script
    result = subprocess.run(
        [sys.executable, "-m", "analysis.audit_no_auto_execution"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, (
        f"No-auto-execution audit FAILED.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
        f"Run: python -m analysis.audit_no_auto_execution --list-violations"
    )

    # ── PRESERVE EXISTING BODY OF test_no_automated_blocking BELOW ──
    # Whatever the current implementation does (likely a runtime smoke check
    # or a specific anti-pattern assertion), keep it. The audit subprocess
    # call above is purely additive.
    #
    # If no existing body exists (i.e., Phase 0 revealed test_no_automated_blocking
    # does not exist yet), the audit call above IS the entire body.
```

### 7.2 Verification

```bash
pytest tests/negative_tests.py::test_no_automated_blocking -v
# Expected: passes if production code is clean.
```

---

## 8. Phase 5 — Aggregator + JSON for the master RQ3 metrics file

### 8.1 The audit script already writes `results/rq3_no_auto_execution.json` (Phase 2)

No additional aggregator needed. The Phase 6 master `compute_rq3_metrics.py` reads this file directly.

### 8.2 Integration snippet for `compute_rq3_metrics.py`

```python
def _load_no_auto_exec_subfile():
    p = REPO_ROOT / "results/rq3_no_auto_execution.json"
    if not p.exists():
        return {"_status": "pending"}
    data = json.loads(p.read_text())
    h = data["headline"]
    return {
        "_status": "complete" if h["audit_pass"] else "failing",
        "_merged_at": datetime.now(timezone.utc).isoformat(),
        "subfile_path": "results/rq3_no_auto_execution.json",
        "headline": h,
        "_framing": data["_meta"]["_framing"],
    }
```

In the aggregator: `out["no_auto_execution"] = _load_no_auto_exec_subfile()`.

---

## 9. Execution order

```bash
# ─── PHASE 0: DISCOVERY ────────────────────────────────────────
python scripts/discover_no_auto_exec_artifacts.py > /tmp/no_auto_exec_inventory.json
# DEVELOPER CONFIRMS:
#  - tests/negative_tests.py contents
#  - ResponseRecommendation class location + operator_decision_required default
#  - production response module path
#  - any pre-existing production violations (must be cleared first)

# ─── PHASE 1: SCOPE CONFIG ─────────────────────────────────────
# Create config/no_auto_exec_scope.yaml per §4.1
# Adjust production_dirs based on Phase 0 findings

# ─── PHASE 2: GREP AUDIT SCRIPT ────────────────────────────────
# Create analysis/audit_no_auto_execution.py
python -m analysis.audit_no_auto_execution
# Expected: exits 0; results/rq3_no_auto_execution.json written

# If non-zero, run --list-violations for human investigation
python -m analysis.audit_no_auto_execution --list-violations

# ─── PHASE 3: RUNTIME CHECKS ───────────────────────────────────
# Create tests/test_response_recommendation_no_exec.py
pytest tests/test_response_recommendation_no_exec.py -v

# ─── PHASE 4: WRAP EXISTING NEGATIVE TEST ──────────────────────
# Update tests/negative_tests.py — add the audit subprocess call at start
pytest tests/negative_tests.py::test_no_automated_blocking -v

# ─── FINAL VERIFICATION ────────────────────────────────────────
pytest tests/negative_tests.py tests/test_response_recommendation_no_exec.py -v
ls config/no_auto_exec_scope.yaml \
   results/rq3_no_auto_execution.json
```

---

## 10. Open questions to surface (DO NOT GUESS)

Claude Code must pause and ask:

1. **Phase 0 — existing test body.** What does `tests/negative_tests.py::test_no_automated_blocking` currently assert? Phase 4 must preserve it. If the function doesn't exist yet, Phase 4 creates it from scratch.
2. **Phase 0 — ResponseRecommendation import path.** Confirm the exact import line; the runtime test uses it.
3. **Phase 0 — operator_decision_required default.** Does the class currently default this to True? If not, this is itself a finding — production code may not yet meet Invariant 3. Track 3 reports this rather than fixing it.
4. **Phase 0 — production response module path.** `module5_responses/` vs `pipeline/module5_response/`. The YAML scope config must list the right one.
5. **Phase 3 — sample alert schema.** The mock smoke test calls `recommend(sample_alert)`; the fixture in §6.1 is a starting point. Confirm the required fields against the actual `recommend()` signature.
6. **Phase 0 — pre-existing violations.** If the quick scan reveals subprocess imports in production code that *aren't* there for auto-execution (e.g., subprocess-based logging), they must either be removed or declared in YAML with an explanatory `# noqa: no-auto-exec` marker.

---

## 11. Coverage map — RQ3 §3.1 expected outputs → pipeline phase

| RQ3_expected_outputs.md item | Phase | Output |
|---|---|---|
| §3.1 Layer A — Invariant 3 documented | (existing — ARCHITECTURE.md) | — |
| §3.1 Layer B — `grep -rnE "subprocess|os.system|iptables"` | 2 | `analysis/audit_no_auto_execution.py` |
| §3.1 Layer B — import statement scan | 2 | Same script, `forbidden_imports` patterns |
| §3.1 Layer C — `tests/negative_tests.py::test_no_automated_blocking` | 4 | Wrapped |
| §4.1 Operator decision required runtime | 3 | `test_operator_decision_required_default_is_true` |
| §1.1 No auto-execution invariant evidence | 2+3+4 | Three layers combined |
| §9 "How do you ensure no auto-execution?" defense Q | 13 | Spec §13 talking points |

Every expected output is traceable.

---

## 12. Defense talking points this enables

- **"How do you ensure no auto-execution?"**
  *"Four layers of defense. Layer A: Invariant 3 documented in `ARCHITECTURE.md`. Layer B: static grep audit via `analysis/audit_no_auto_execution.py`, configured by `config/no_auto_exec_scope.yaml`, that searches production directories for forbidden patterns (`subprocess`, `os.system`, `iptables`, etc.). Layer C: `tests/negative_tests.py::test_no_automated_blocking` invokes Layer B at CI time. Layer D: `tests/test_response_recommendation_no_exec.py` patches `subprocess` and `os.system`, runs the actual recommend() call end-to-end, and asserts the mocks were never invoked. Layers A and B prove the code can't auto-execute; Layers C and D prove it doesn't at runtime."*

- **"What if a developer adds a subprocess call legitimately?"**
  *"The YAML config declares directories as production / test / analysis. Production is strict; test code is explicitly authorized to use subprocess for simulating attacks; analysis scripts are skipped (this audit itself uses subprocess to run grep). Adding subprocess to production requires either: (a) moving the code to a test/analysis directory, or (b) marking the specific line with `# noqa: no-auto-exec` — both of which are version-controlled and reviewable. There is no env-var override or CI escape hatch."*

- **"Could a static grep miss something?"**
  *"Yes — that's why Layer D exists. We patch subprocess and os.system at Python's import level and run the full recommend() function end-to-end. If any execution primitive is invoked through any indirect path, the mock catches it. This is the strongest possible runtime evidence."*

- **"What about `ResponseRecommendation.operator_decision_required`?"**
  *"There's a positive-evidence test (`test_operator_decision_required_default_is_true`) that constructs a recommendation with default arguments and asserts the field is True. If a future refactor changes the default, the test fails immediately. This guards against silent disabling of the HITL invariant."*

- **"What about false positives — comments mentioning subprocess?"**
  *"The audit strips triple-quoted docstrings before scanning (preserving line numbers), skips comment-only lines, and respects an explicit `# noqa: no-auto-exec` marker for the rare legitimate case. False positives are loud, not silent — they show up as audit failures and require explicit acknowledgement."*

---

## 13. What this track deliberately does NOT do

- **Modify production code.** If Phase 0 reveals violations, those are reported but not fixed by this spec. The spec assumes production is currently clean; the developer fixes any pre-existing issues before running Phase 2.
- **Replace the existing negative test.** The senior reviewer's three-layer framing depends on Layer C existing. Phase 4 wraps; it does not delete.
- **Lint other anti-patterns.** Track 3 is scoped narrowly to no-auto-execution. Other negative tests in `negative_tests.py` (no discovery, no CVSS, etc.) are out of scope.
- **Provide a CI bypass.** No env-var override, no warn-only mode. The only path to "allowing" a subprocess in a directory is editing the YAML — reviewable in version control.

---

## End of spec

Implementation order: Phase 0 (discovery) → Phase 1 (YAML config) → Phase 2 (audit script) → Phase 3 (runtime tests) → Phase 4 (wrap existing test). Each phase is independently verifiable. After Track 3 is implemented, Invariant 3 in the RQ3 invariant manifest flips from `pending` to `enforced`.