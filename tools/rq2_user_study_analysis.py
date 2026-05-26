#!/usr/bin/env python3
"""RQ2.c / RQ2.d — User-study aggregation for the MVE faithfulness paper.

Reuses the same study data RQ3 used (`results/reports/participant_responses.json`
+ `survey/study_responses_A/B.json`) but slices it by RQ2 perspective:

  RQ2.c per-role:
    • Mann-Whitney U on decision_time, accuracy, confidence
    • Split by role (analyst / clinician / administrator)
    • Compare with_xai vs without_xai per role

  RQ2.d failure modes:
    • Catalog observation-level failure patterns from the survey
    • Single-round only — claim observation, NOT improvement
    • Themes derived from reasoning_summary text + scoring fields

Writes:
  • analysis/outputs/rq2c_per_role.json
  • analysis/outputs/rq2d_failure_modes.json
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "results" / "reports"
SURVEY = PROJECT_ROOT / "survey"
OUT = PROJECT_ROOT / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def compute_rq2c_per_role() -> dict:
    """RQ2.c per-role Mann-Whitney U (with_xai vs without_xai).

    The M6 study (`participant_responses.json`) has role labels —
    the M5 study (`survey/`) does not, so M5 is summarized at the
    population level only.
    """
    with open(REPORTS / "participant_responses.json") as f:
        records = json.load(f)

    roles = sorted({r["participant_role"] for r in records})
    rows = []
    role_tests = {}

    for role in roles:
        role_records = [r for r in records if r["participant_role"] == role]
        with_xai = [r for r in role_records if r["condition"] == "with_xai"]
        without_xai = [r for r in role_records if r["condition"] == "without_xai"]

        def _vec(recs, field):
            return [r[field] for r in recs if r.get(field) is not None]

        # Per-participant aggregates (within the role)
        def _per_participant(recs, field):
            by_p = defaultdict(list)
            for r in recs:
                by_p[r["participant_id"]].append(r[field])
            return [np.mean(v) for v in by_p.values()]

        # Accuracy MW
        acc_with = _per_participant(with_xai, "decision_correct")
        acc_without = _per_participant(without_xai, "decision_correct")
        acc_mw = stats.mannwhitneyu(acc_with, acc_without, alternative="greater") \
            if acc_with and acc_without else None

        # Decision time MW (B faster)
        t_with = _per_participant(with_xai, "decision_time_sec")
        t_without = _per_participant(without_xai, "decision_time_sec")
        time_mw = stats.mannwhitneyu(t_with, t_without, alternative="less") \
            if t_with and t_without else None

        # Confidence MW (B more confident)
        c_with = _per_participant(with_xai, "confidence")
        c_without = _per_participant(without_xai, "confidence")
        conf_mw = stats.mannwhitneyu(c_with, c_without, alternative="greater") \
            if c_with and c_without else None

        rows.append({
            "role": role,
            "n_with_xai": len(with_xai),
            "n_without_xai": len(without_xai),
            "accuracy_with_xai_mean": float(np.mean(_vec(with_xai, "decision_correct"))) if with_xai else None,
            "accuracy_without_xai_mean": float(np.mean(_vec(without_xai, "decision_correct"))) if without_xai else None,
            "decision_time_with_xai_mean": float(np.mean(_vec(with_xai, "decision_time_sec"))) if with_xai else None,
            "decision_time_without_xai_mean": float(np.mean(_vec(without_xai, "decision_time_sec"))) if without_xai else None,
            "confidence_with_xai_mean": float(np.mean(_vec(with_xai, "confidence"))) if with_xai else None,
            "confidence_without_xai_mean": float(np.mean(_vec(without_xai, "confidence"))) if without_xai else None,
            "likert_trust_with_xai_mean": float(np.mean(_vec(with_xai, "likert_trust"))) if with_xai else None,
            "likert_trust_without_xai_mean": float(np.mean(_vec(without_xai, "likert_trust"))) if without_xai else None,
            "likert_usefulness_with_xai_mean": float(np.mean(_vec(with_xai, "likert_usefulness"))) if with_xai else None,
            "likert_usefulness_without_xai_mean": float(np.mean(_vec(without_xai, "likert_usefulness"))) if without_xai else None,
        })

        role_tests[role] = {
            "accuracy_mw_p_value": float(acc_mw.pvalue) if acc_mw else None,
            "decision_time_mw_p_value": float(time_mw.pvalue) if time_mw else None,
            "confidence_mw_p_value": float(conf_mw.pvalue) if conf_mw else None,
            "accuracy_significant": bool(acc_mw.pvalue < 0.05) if acc_mw else None,
            "decision_time_significant": bool(time_mw.pvalue < 0.05) if time_mw else None,
            "confidence_significant": bool(conf_mw.pvalue < 0.05) if conf_mw else None,
        }

    return {
        "_meta": {
            "description": "RQ2.c — per-role MWU on accuracy / decision_time / confidence",
            "data_source": "results/reports/participant_responses.json",
            "n_total_records": len(records),
            "test": "Mann-Whitney U per role (with_xai vs without_xai)",
            "alternatives": {
                "accuracy": "with_xai > without_xai",
                "decision_time": "with_xai < without_xai (faster)",
                "confidence": "with_xai > without_xai",
            },
        },
        "per_role_means": rows,
        "per_role_mann_whitney": role_tests,
    }


# ──────────────────────────────────────────────────────────────────────
# RQ2.d failure mode catalog (observation-level)
# ──────────────────────────────────────────────────────────────────────


_FAILURE_PATTERNS = {
    "layer1_too_technical": [
        r"too technical", r"jargon", r"don't understand",
        r"unclear what.*shap", r"mve.*confusing",
    ],
    "mitre_not_understood": [
        r"mitre", r"att.ck", r"technique id",
        r"don't know what.*t1\d{3,4}",
    ],
    "do_not_constraint_ignored": [
        r"isolat", r"power.cycle", r"disconnect",
        r"shut.down", r"quarantine",
    ],
    "role_mismatch": [
        r"not my role", r"wrong queue", r"should.*nurse",
        r"should.*engineer", r"wrong person",
    ],
    "feature_too_abstract": [
        r"need more detail", r"vague", r"not specific",
        r"which feature", r"why flagged",
    ],
}


def _classify_failure(rationale: str) -> list:
    if not rationale:
        return []
    text = rationale.lower()
    matches = []
    for mode, patterns in _FAILURE_PATTERNS.items():
        for p in patterns:
            if re.search(p, text):
                matches.append(mode)
                break
    return matches


def compute_rq2d_failure_modes() -> dict:
    """Catalog failure modes observed in the user study.

    Sources:
      • survey/study_responses_A.json / _B.json — has reasoning_summary
      • results/reports/participant_responses.json — has feedback field
    """
    mode_counts = Counter()
    mode_examples = defaultdict(list)
    n_total_responses = 0
    n_with_failure_signal = 0

    for path in [SURVEY / "study_responses_A.json", SURVEY / "study_responses_B.json"]:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data:
            n_total_responses += 1
            rationale = r.get("response", {}).get("reasoning_summary", "")
            modes = _classify_failure(rationale)
            if modes:
                n_with_failure_signal += 1
                for m in modes:
                    mode_counts[m] += 1
                    if len(mode_examples[m]) < 3:
                        mode_examples[m].append({
                            "source": path.name,
                            "response_id": r.get("response_id"),
                            "alert_id": r.get("alert_id"),
                            "rationale_excerpt": rationale[:160],
                        })

    # Also scan participant_responses.json feedback field
    pr_path = REPORTS / "participant_responses.json"
    if pr_path.exists():
        with open(pr_path) as f:
            pr = json.load(f)
        for r in pr:
            n_total_responses += 1
            fb = r.get("feedback", "")
            modes = _classify_failure(fb)
            if modes:
                n_with_failure_signal += 1
                for m in modes:
                    mode_counts[m] += 1
                    if len(mode_examples[m]) < 3:
                        mode_examples[m].append({
                            "source": "participant_responses.json",
                            "response_id": r.get("participant_id"),
                            "alert_id": r.get("alert_id"),
                            "rationale_excerpt": fb[:160],
                        })

    # Per-mode iteration plan (observation-only — single round of study)
    iteration_catalog = {
        "layer1_too_technical": {
            "iteration": "Simplify Layer 1 vocabulary (Mode A LLM prompt + Mode B template). Drop SHAP value numerics from clinician view.",
            "metric": "Comprehension % (post-iteration study)",
            "status": "OBSERVATION ONLY — single round; improvement not measured",
        },
        "mitre_not_understood": {
            "iteration": "Inline plain-language gloss for each MITRE ID ('T1565 — Data Manipulation: unauthorized changes to data in transit').",
            "metric": "Recognition % (post-iteration study)",
            "status": "OBSERVATION ONLY — single round",
        },
        "do_not_constraint_ignored": {
            "iteration": "Visual emphasis in MVE Layer 3 (bold DO NOT + warning icon). Already partially implemented in Sentinel theme — extend to clinician + admin views.",
            "metric": "Action-compliance % (operator chose safe alternative)",
            "status": "OBSERVATION ONLY — single round",
        },
        "role_mismatch": {
            "iteration": "Better role inference at routing (Module 5) + role-switch button in UI when operator finds themselves on wrong queue.",
            "metric": "Match rate %",
            "status": "OBSERVATION ONLY — single round",
        },
        "feature_too_abstract": {
            "iteration": "Increase Mode B feature injection from top-1 to top-3 in `src.mve_generator.py` Layer 1 'Primary signal' suffix.",
            "metric": "MVE-SHAP alignment %≥2 (target 95%)",
            "status": "OBSERVATION ONLY — improvement gated on study re-run",
        },
    }

    return {
        "_meta": {
            "description": "RQ2.d — Failure mode catalog from user study",
            "data_sources": [
                "survey/study_responses_A.json",
                "survey/study_responses_B.json",
                "results/reports/participant_responses.json",
            ],
            "scope": (
                "Single-round study — claim is OBSERVATIONAL (failure "
                "modes identified). Improvement claims require a second "
                "round of evaluation post-iteration. See spec §RQ2.d "
                "scope note."
            ),
            "n_total_responses_scanned": n_total_responses,
            "n_responses_with_failure_signal": n_with_failure_signal,
            "pct_responses_with_signal": round(
                n_with_failure_signal / n_total_responses * 100, 2
            ) if n_total_responses else 0.0,
        },
        "failure_mode_counts": dict(mode_counts.most_common()),
        "examples_per_mode": dict(mode_examples),
        "iteration_catalog": iteration_catalog,
    }


def main():
    print("[8] Computing RQ2.c per-role analysis...")
    per_role = compute_rq2c_per_role()
    out1 = OUT / "rq2c_per_role.json"
    with open(out1, "w") as f:
        json.dump(per_role, f, indent=2, default=float)
    print(f"  → {out1.relative_to(PROJECT_ROOT)}")
    for row, tests in zip(per_role["per_role_means"], per_role["per_role_mann_whitney"].values()):
        print(f"  {row['role']:14s} "
              f"acc {row['accuracy_without_xai_mean']:.3f} → {row['accuracy_with_xai_mean']:.3f}  "
              f"p={tests['accuracy_mw_p_value']:.4f}  "
              f"time {row['decision_time_without_xai_mean']:.1f}s → {row['decision_time_with_xai_mean']:.1f}s")

    print()
    print("[9] Cataloging failure modes (RQ2.d)...")
    fm = compute_rq2d_failure_modes()
    out2 = OUT / "rq2d_failure_modes.json"
    with open(out2, "w") as f:
        json.dump(fm, f, indent=2, default=str)
    print(f"  → {out2.relative_to(PROJECT_ROOT)}")
    print(f"  scanned {fm['_meta']['n_total_responses_scanned']} responses, "
          f"{fm['_meta']['pct_responses_with_signal']}% with failure signal")
    print("  modes observed:")
    for mode, count in fm["failure_mode_counts"].items():
        print(f"    {mode:30s} {count}")


if __name__ == "__main__":
    main()
