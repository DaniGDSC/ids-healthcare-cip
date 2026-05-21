"""RQ3 Track 5 — appropriate-escalation Chi-square per role.

Inputs:
  survey/study_responses_LLM_*.json
  configs/rq3_escalation_definition.yaml
  survey/rq2c_exclusions.json (Path C exclusions — same as RQ2.c)

Output:
  analysis/outputs/rq3_escalation.json

Persona-level aggregation; Chi-square 2x2 per role; Fisher's exact
fallback when any expected cell count < 5. Cramér's V (= |φ| for 2×2)
as effect size; observed cell rates + odds ratio also reported.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy.stats import chi2_contingency, fisher_exact

REPO_ROOT = Path(__file__).resolve().parents[1]
SURVEY_DIR = REPO_ROOT / "survey"
TAXONOMY = REPO_ROOT / "configs" / "rq3_escalation_definition.yaml"
EXCL = SURVEY_DIR / "rq2c_exclusions.json"
OUT = REPO_ROOT / "analysis" / "outputs" / "rq3_escalation.json"

ROLES = ["biomed_engineer", "IT_generalist", "nurse_manager"]


def _role_from_pid(pid: str) -> str:
    """Strip a trailing ``_P\\d+`` suffix to recover the role string."""
    parts = pid.split("_")
    if parts and re.match(r"^P\d+$", parts[-1]):
        parts = parts[:-1]
    return "_".join(parts) if parts else pid


def _persona_escalation_flag(rec: dict, esc_set: set[str],
                             threshold: float) -> Optional[dict]:
    """Aggregate one persona's rows into a single escalation flag.

    Returns dict with role, condition, n_warranted, n_appropriate,
    escalated_appropriately. Returns None if the persona has zero
    escalation-warranted rows (cannot contribute to the test).
    """
    pid = rec.get("persona_id", "")
    role = _role_from_pid(pid)
    rows = rec.get("rows") or []
    cond: Optional[str] = None
    n_warranted = 0
    n_appropriate = 0
    for r in rows:
        if r.get("error") is not None:
            continue
        resp = r.get("response")
        if not isinstance(resp, dict):
            continue
        cond = cond or r.get("condition")
        ca = r.get("correct_action")
        if ca not in esc_set:
            continue
        n_warranted += 1
        if resp.get("action") in esc_set:
            n_appropriate += 1
    if n_warranted == 0 or cond not in ("A", "B"):
        return None
    return {
        "persona_id": pid,
        "role": role,
        "condition": cond,
        "n_warranted": n_warranted,
        "n_appropriate": n_appropriate,
        "escalated_appropriately": (n_appropriate / n_warranted) >= threshold,
    }


def _build_2x2(persona_flags: list[dict],
               role: Optional[str] = None) -> list[list[int]]:
    """Build the 2x2 contingency: [[A_yes, A_no], [B_yes, B_no]]."""
    counts = {"A": {"yes": 0, "no": 0}, "B": {"yes": 0, "no": 0}}
    for p in persona_flags:
        if role is not None and p["role"] != role:
            continue
        k = "yes" if p["escalated_appropriately"] else "no"
        counts[p["condition"]][k] += 1
    return [[counts["A"]["yes"], counts["A"]["no"]],
            [counts["B"]["yes"], counts["B"]["no"]]]


def _run_test(contingency: list[list[int]],
              min_expected: int) -> tuple[str, Optional[float],
                                          Optional[float], bool]:
    """Run Chi-square; fall back to Fisher's exact on low expected counts.

    Returns (test_name, statistic, p_value, fisher_fallback).
    statistic semantics: chi2 for Chi-square; odds_ratio for Fisher.
    """
    arr = np.array(contingency, dtype=float)
    if arr.sum() == 0:
        return "no_data", None, None, False
    row_sums = arr.sum(axis=1)
    col_sums = arr.sum(axis=0)
    if (row_sums == 0).any() or (col_sums == 0).any():
        # Degenerate margin — Chi-square undefined; use Fisher
        odds, p_fisher = fisher_exact(arr)
        return "fisher_exact", float(odds), float(p_fisher), True
    try:
        chi2, p, _, expected = chi2_contingency(arr, correction=False)
        if expected.min() < min_expected:
            odds, p_fisher = fisher_exact(arr)
            return "fisher_exact", float(odds), float(p_fisher), True
        return "chi_square", float(chi2), float(p), False
    except ValueError as exc:
        return f"test_failed:{exc}", None, None, False


def _cramers_v(contingency: list[list[int]]) -> Optional[float]:
    """Cramer's V for 2x2 = |phi| = sqrt(chi2 / n)."""
    arr = np.array(contingency, dtype=float)
    n = arr.sum()
    if n == 0:
        return None
    row_sums = arr.sum(axis=1)
    col_sums = arr.sum(axis=0)
    if (row_sums == 0).any() or (col_sums == 0).any():
        return None
    chi2, _, _, _ = chi2_contingency(arr, correction=False)
    return float(np.sqrt(chi2 / n))


def _odds_ratio(contingency: list[list[int]]) -> Optional[float]:
    """Sample odds ratio: (a*d) / (b*c). None if denominator is zero."""
    a, b = contingency[0]
    c, d = contingency[1]
    if b == 0 or c == 0:
        return None
    return float((a * d) / (b * c))


def _build_cell(contingency: list[list[int]], min_expected: int,
                n_warn: int, scope: str) -> dict:
    n_a = contingency[0][0] + contingency[0][1]
    n_b = contingency[1][0] + contingency[1][1]
    test, stat, p, fisher = _run_test(contingency, min_expected)
    v = _cramers_v(contingency) if test == "chi_square" else None
    od = _odds_ratio(contingency)
    return {
        "_scope": scope,
        "n_A": n_a,
        "n_B": n_b,
        "contingency_2x2": {
            "A_escalated": contingency[0][0], "A_not": contingency[0][1],
            "B_escalated": contingency[1][0], "B_not": contingency[1][1],
        },
        "test": test,
        "statistic": round(stat, 4) if stat is not None else None,
        "p_value": float(p) if p is not None else None,
        "cramers_v": round(v, 4) if v is not None else None,
        "odds_ratio": round(od, 4) if od is not None else None,
        "rate_A": round(contingency[0][0] / n_a, 4) if n_a else None,
        "rate_B": round(contingency[1][0] / n_b, 4) if n_b else None,
        "n_warning": (n_a < n_warn) or (n_b < n_warn),
        "fisher_fallback": fisher,
    }


def main() -> None:
    taxonomy = yaml.safe_load(TAXONOMY.read_text())
    esc_set = set(taxonomy.get("escalation_actions") or [])
    threshold = float(taxonomy.get("min_appropriate_escalation_proportion", 0.5))
    n_warn = int(taxonomy.get("n_warning_threshold", 10))
    min_expected = int(taxonomy.get("chi_square_min_expected_cell_count", 5))

    excluded: set[str] = set()
    if EXCL.exists():
        d = json.loads(EXCL.read_text())
        excluded = {str(e.get("persona_id")) for e in d.get("exclusions", [])}

    persona_flags: list[dict] = []
    n_files = 0
    n_no_warranted_rows = 0
    for p in sorted(SURVEY_DIR.glob("study_responses_LLM_*.json")):
        n_files += 1
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if str(rec.get("persona_id")) in excluded:
            continue
        flag = _persona_escalation_flag(rec, esc_set, threshold)
        if flag is None:
            n_no_warranted_rows += 1
            continue
        persona_flags.append(flag)

    per_role: dict[str, dict] = {}
    cells_with_warning = 0
    fisher_fallback_count = 0
    cell_sizes: list[int] = []
    for role in ROLES:
        cont = _build_2x2(persona_flags, role=role)
        n_a = cont[0][0] + cont[0][1]
        n_b = cont[1][0] + cont[1][1]
        cell_sizes.extend([n_a, n_b])
        cell = _build_cell(cont, min_expected, n_warn,
                           f"role={role}")
        if cell["n_warning"]:
            cells_with_warning += 1
        if cell["fisher_fallback"]:
            fisher_fallback_count += 1
        per_role[role] = cell

    overall_cont = _build_2x2(persona_flags, role=None)
    overall = _build_cell(overall_cont, min_expected, n_warn,
                         "All included personas (3 roles collapsed)")
    if overall["fisher_fallback"]:
        fisher_fallback_count += 1

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/compute_rq3_escalation.py",
            "taxonomy_path": str(TAXONOMY.relative_to(REPO_ROOT)),
            "taxonomy_locked_on": taxonomy.get("taxonomy_locked_on"),
            "data_source": (
                "LLM-persona simulation (gpt-4o-mini); not human study"
            ),
            "escalation_actions": sorted(esc_set),
            "min_appropriate_escalation_proportion": threshold,
            "inputs": {
                "n_persona_files": n_files,
                "n_excluded": len(excluded),
                "n_included_with_warranted_rows": len(persona_flags),
                "n_no_warranted_rows": n_no_warranted_rows,
                "exclusion_source": str(EXCL.relative_to(REPO_ROOT)),
            },
        },
        "methodology_notes": [
            f"Appropriate escalation = persona action AND correct_action "
            f"both in {sorted(esc_set)}. Pre-registered in "
            f"configs/rq3_escalation_definition.yaml.",
            "Chi-square 2x2 per role; Fisher's exact fallback when any "
            f"expected cell count < {min_expected}.",
            f"Persona-level aggregation: escalated_appropriately = "
            f"(n_appropriate / n_warranted) >= {threshold}.",
            "Raw p-values; NO multiple-comparisons correction across the "
            "3 role tests.",
            "Cramer's V (= |phi| for 2x2) reported as effect size; "
            "observed cell rates + odds ratio also reported.",
            "Data source: LLM-persona simulation; behavioural fidelity to "
            "human operators not established.",
        ],
        "limitations": [
            "Persona simulation, not human study.",
            "Small per-cell N (typical ~10-25); Chi-square assumptions "
            "marginally satisfied.",
            "Single dimension of escalation (containment-class actions). "
            "Future work: graded escalation severity.",
            "Multiple-comparisons inflation across 3 role tests "
            f"(~{0.05 * 3:.2f} false positives at alpha=0.05 under null).",
        ],
        "overall": overall,
        "per_role": per_role,
        "cell_diagnostics": {
            "_description": (
                "Per-role cell sizes (persona counts in each condition)."
            ),
            "min_n_per_cell": min(cell_sizes) if cell_sizes else 0,
            "max_n_per_cell": max(cell_sizes) if cell_sizes else 0,
            "cells_with_warning": cells_with_warning,
            "fisher_fallback_count": fisher_fallback_count,
            "warning_threshold": n_warn,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"Overall (n_A={overall['n_A']}, n_B={overall['n_B']}): "
          f"rate_A={overall['rate_A']} rate_B={overall['rate_B']} "
          f"p={overall['p_value']} V={overall['cramers_v']}")
    for role in ROLES:
        r = per_role[role]
        warn = " [LOW-N]" if r["n_warning"] else ""
        fb = " [Fisher]" if r["fisher_fallback"] else ""
        print(f"  {role}: A={r['rate_A']} B={r['rate_B']} "
              f"p={r['p_value']} V={r['cramers_v']}{warn}{fb}")


if __name__ == "__main__":
    main()
