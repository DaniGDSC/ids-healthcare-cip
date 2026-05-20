"""Layer B of the no-auto-execution three-layer defense (Invariant 3).

Walks the codebase per configs/no_auto_exec_scope.yaml. Scans production
directories for forbidden execution patterns. Reports findings grouped
by category.

  - Production violations -> hard fail (exit 1).
  - Test-dir findings    -> informational only.
  - Analysis dirs        -> skipped.

Usage:
  python -m analysis.audit_no_auto_execution             # CI mode (strict)
  python -m analysis.audit_no_auto_execution --list-violations
                                                          # Human investigation; exit 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "no_auto_exec_scope.yaml"
OUT = REPO_ROOT / "results" / "rq3_no_auto_execution.json"

NOQA_MARKER = "# noqa: no-auto-exec"


def _load_config() -> dict:
    if not CONFIG.exists():
        raise SystemExit(
            f"Config missing: {CONFIG.relative_to(REPO_ROOT)}. "
            "Create per RQ3_NO_AUTO_EXECUTION_SPEC.md Phase 1."
        )
    return yaml.safe_load(CONFIG.read_text())


def _classify_file(p: Path, cfg: dict) -> str:
    """Return 'production' | 'test' | 'analysis' | 'unclassified' | 'excluded'."""
    rel = p.relative_to(REPO_ROOT)
    rel_str = str(rel)
    for ex in cfg.get("excluded_paths") or []:
        ex_clean = ex.rstrip("/")
        if rel_str.startswith(ex) or ex_clean in rel.parts:
            return "excluded"
    for d in cfg.get("production_dirs") or []:
        if rel_str.startswith(d):
            return "production"
    for d in cfg.get("test_dirs") or []:
        if rel_str.startswith(d):
            return "test"
    for d in cfg.get("analysis_dirs") or []:
        if rel_str.startswith(d):
            return "analysis"
    return "unclassified"


def _strip_triple_quoted(text: str) -> str:
    """Blank out triple-quoted string bodies, preserving line count.

    Matches both ``\"\"\"`` and ``'''`` blocks (greedy minimal across
    multiple lines). Newlines are preserved so line numbers stay
    correct in the output report.
    """
    out_chars = list(text)
    pattern = re.compile(r'("""|\'\'\')([\s\S]*?)\1', re.MULTILINE)
    for m in pattern.finditer(text):
        for i in range(m.start(), m.end()):
            if text[i] != "\n":
                out_chars[i] = " "
    return "".join(out_chars)


def _scan_file(p: Path, compiled: dict[str, list[re.Pattern]]) -> list[dict]:
    """Scan one file. Return findings (one per matched pattern per line)."""
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")
    except (UnicodeDecodeError, PermissionError, OSError):
        return []

    sanitized = _strip_triple_quoted(raw)
    raw_lines = raw.splitlines()
    findings: list[dict] = []

    for line_no, line in enumerate(sanitized.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if NOQA_MARKER in line:
            continue
        for category, regexes in compiled.items():
            matched = False
            for regex in regexes:
                if regex.search(line):
                    original = raw_lines[line_no - 1] if line_no - 1 < len(raw_lines) else ""
                    findings.append({
                        "category": category,
                        "pattern": regex.pattern,
                        "file": str(p.relative_to(REPO_ROOT)),
                        "line": line_no,
                        "content": original.strip()[:200],
                    })
                    matched = True
                    break  # one finding per (category, line) is enough
            if matched:
                # Continue to next category — a line can match multiple categories.
                continue
    return findings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-violations", action="store_true",
                    help="Print findings to stdout; exit 0 (human investigation).")
    args = ap.parse_args()

    cfg = _load_config()
    compiled: dict[str, list[re.Pattern]] = {
        cat: [re.compile(p) for p in pats]
        for cat, pats in (cfg.get("forbidden_patterns") or {}).items()
    }

    files_by_class: dict[str, list[Path]] = defaultdict(list)
    for p in REPO_ROOT.rglob("*.py"):
        cls = _classify_file(p, cfg)
        files_by_class[cls].append(p)

    production_findings: list[dict] = []
    test_findings: list[dict] = []
    unclassified_findings: list[dict] = []

    for p in files_by_class["production"]:
        production_findings.extend(_scan_file(p, compiled))
    for p in files_by_class["test"]:
        test_findings.extend(_scan_file(p, compiled))
    for p in files_by_class["unclassified"]:
        unclassified_findings.extend(_scan_file(p, compiled))

    by_category: dict[str, list[dict]] = defaultdict(list)
    for f in production_findings:
        by_category[f["category"]].append(f)

    audit = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/audit_no_auto_execution.py",
            "config_path": str(CONFIG.relative_to(REPO_ROOT)),
            "taxonomy_locked_on": cfg.get("taxonomy_locked_on"),
            "_framing": (
                "Static-analysis Layer B of the no-auto-execution three-layer "
                "defense (Invariant 3). Production code MUST contain zero "
                "matches; test code may use subprocess for attack simulation; "
                "analysis scripts are skipped."
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
        print("\n=== No-Auto-Execution Audit (LIST MODE) ===")
        print(f"Production violations: {h['n_violations_production']}")
        if h["n_violations_production"]:
            print("\nBy category:")
            for cat, matches in audit["violations_by_category"].items():
                print(f"  [{cat}] {len(matches)} finding(s):")
                for m in matches[:10]:
                    print(f"    {m['file']}:{m['line']}  {m['content'][:100]}")
        if h["n_findings_unclassified"]:
            print(f"\nUNCLASSIFIED files with matches: "
                  f"{h['n_findings_unclassified']}")
            print("  Consider adding their parent dirs to "
                  "configs/no_auto_exec_scope.yaml.")
        print(f"\nFull report: {OUT.relative_to(REPO_ROOT)}")
        sys.exit(0)

    # CI mode
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"Production violations: {h['n_violations_production']}  "
          f"(test-dir info: {h['n_findings_test_info_only']})")
    if not h["audit_pass"]:
        print("FAIL: production code contains forbidden execution patterns.")
        print("  Run `python -m analysis.audit_no_auto_execution "
              "--list-violations` for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
