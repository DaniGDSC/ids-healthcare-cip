#!/usr/bin/env python3
"""Compute RQ3 user-study analysis artifacts.

Two data sources:
  * survey/study_responses_{A,B}.json — M5 study (25 participants/group),
    Group A = baseline (no MVE), Group B = with MVE.
    This is the primary RQ3 hypothesis test source — m5_result.yaml is
    derived from it.
  * results/reports/participant_responses.json — M6 study with role labels
    (analyst / clinician / administrator) under with_xai vs without_xai.
    Used for per-role breakdown (the M5 study has no role metadata).

Outputs:
  * analysis/outputs/rq3_primary.json    — Mann-Whitney + secondary metrics
  * analysis/outputs/rq3_per_role.json   — role × condition breakdown
  * analysis/outputs/rq3_escalation_chi2.json — chi-square on escalation
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SURVEY = PROJECT_ROOT / "survey"
REPORTS = PROJECT_ROOT / "results" / "reports"
OUT = PROJECT_ROOT / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _participant_accuracy(responses: list) -> dict[str, float]:
    """Per-participant accuracy from a flat list of responses."""
    by_pid = defaultdict(list)
    for r in responses:
        pid = r.get("response_id", "")[:3]
        by_pid[pid].append(int(bool(r.get("scoring", {}).get("composite_correct"))))
    return {pid: float(np.mean(vals)) for pid, vals in by_pid.items()}


def _participant_metric(responses: list, accessor) -> dict[str, float]:
    """Per-participant mean of a metric extracted by `accessor(r) -> float`."""
    by_pid = defaultdict(list)
    for r in responses:
        pid = r.get("response_id", "")[:3]
        v = accessor(r)
        if v is not None:
            by_pid[pid].append(float(v))
    return {pid: float(np.mean(vals)) for pid, vals in by_pid.items() if vals}


def compute_primary():
    """Mann-Whitney U comparing Group A (no MVE) vs Group B (with MVE).

    Re-derives the result captured in survey/m5_result.yaml from the raw
    response lists, so future audits can replay the computation against
    a frozen JSON artifact rather than the YAML summary.
    """
    with open(SURVEY / "study_responses_A.json") as f:
        A = json.load(f)
    with open(SURVEY / "study_responses_B.json") as f:
        B = json.load(f)

    acc_A = _participant_accuracy(A)
    acc_B = _participant_accuracy(B)

    # Composite accuracy is the headline metric (severity + action both correct)
    mwu = stats.mannwhitneyu(list(acc_B.values()), list(acc_A.values()),
                              alternative="greater")
    # Cohen's d
    pooled_sd = np.sqrt(
        (np.var(list(acc_A.values()), ddof=1) + np.var(list(acc_B.values()), ddof=1)) / 2
    )
    d = (np.mean(list(acc_B.values())) - np.mean(list(acc_A.values()))) / pooled_sd if pooled_sd > 0 else float("nan")

    # Secondary metrics
    def _confidence(r):
        return r.get("response", {}).get("confidence_rating")
    def _decision_time(r):
        return r.get("response", {}).get("time_to_decision_seconds")
    def _severity_acc(r):
        return 1 if r.get("scoring", {}).get("severity_correct") else 0
    def _action_acc(r):
        return 1 if r.get("scoring", {}).get("action_correct") else 0

    def _agg(responses, accessor):
        vals = [accessor(r) for r in responses if accessor(r) is not None]
        return float(np.mean(vals)) if vals else None

    # Catastrophic miss: ground truth CRITICAL but operator chose LOW/dismiss-tier
    def _is_catastrophic_miss(r):
        if r.get("ground_truth_severity") != "CRITICAL":
            return None
        action = r.get("response", {}).get("action_chosen")
        # Catastrophic if dismissed or marked LOW severity
        sev = r.get("response", {}).get("severity_chosen")
        return 1 if (action == "dismiss" or sev == "LOW") else 0

    # Over/under reaction
    SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    def _over_reaction(r):
        chosen = SEVERITY_ORDER.get(r.get("response", {}).get("severity_chosen"))
        truth = SEVERITY_ORDER.get(r.get("ground_truth_severity"))
        if chosen is None or truth is None:
            return None
        return 1 if chosen > truth else 0
    def _under_reaction(r):
        chosen = SEVERITY_ORDER.get(r.get("response", {}).get("severity_chosen"))
        truth = SEVERITY_ORDER.get(r.get("ground_truth_severity"))
        if chosen is None or truth is None:
            return None
        return 1 if chosen < truth else 0

    def _summarize(responses):
        return {
            "n_responses": len(responses),
            "n_participants": len(set(r.get("response_id", "")[:3] for r in responses)),
            "mean_composite_accuracy": _agg(responses, lambda r: 1 if r["scoring"]["composite_correct"] else 0),
            "mean_severity_accuracy": _agg(responses, _severity_acc),
            "mean_action_accuracy": _agg(responses, _action_acc),
            "mean_confidence": _agg(responses, _confidence),
            "mean_decision_time_sec": _agg(responses, _decision_time),
            "catastrophic_miss_rate": _agg(responses, _is_catastrophic_miss),
            "over_reaction_rate": _agg(responses, _over_reaction),
            "under_reaction_rate": _agg(responses, _under_reaction),
        }

    # Mann-Whitney on confidence and decision time (paired by participant)
    conf_A = _participant_metric(A, _confidence)
    conf_B = _participant_metric(B, _confidence)
    time_A = _participant_metric(A, _decision_time)
    time_B = _participant_metric(B, _decision_time)

    mwu_conf = stats.mannwhitneyu(list(conf_B.values()), list(conf_A.values()), alternative="greater")
    mwu_time = stats.mannwhitneyu(list(time_B.values()), list(time_A.values()), alternative="less")

    return {
        "_meta": {
            "description": "RQ3 primary user-study analysis (M5 study, Group A baseline vs Group B with MVE)",
            "data_sources": [
                "survey/study_responses_A.json (n=500 responses, 25 participants)",
                "survey/study_responses_B.json (n=500 responses, 25 participants)",
            ],
            "hypothesis": "Group B (with MVE) > Group A (baseline) on composite accuracy",
            "test": "Mann-Whitney U (one-sided, greater)",
        },
        "primary_metric_composite_accuracy": {
            "group_A_mean": float(np.mean(list(acc_A.values()))),
            "group_B_mean": float(np.mean(list(acc_B.values()))),
            "relative_improvement": float(
                (np.mean(list(acc_B.values())) - np.mean(list(acc_A.values())))
                / np.mean(list(acc_A.values()))
            ),
            "mann_whitney_U": float(mwu.statistic),
            "mann_whitney_p_value": float(mwu.pvalue),
            "cohens_d": float(d),
            "effect_size": (
                "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5
                else "small" if abs(d) >= 0.2 else "negligible"
            ),
            "passes_significance": bool(mwu.pvalue < 0.05),
            "target_improvement": 0.3,
            "passes_improvement_threshold": bool(
                (np.mean(list(acc_B.values())) - np.mean(list(acc_A.values())))
                / np.mean(list(acc_A.values())) >= 0.3
            ),
            "verdict": "PASS" if (
                mwu.pvalue < 0.05
                and (np.mean(list(acc_B.values())) - np.mean(list(acc_A.values())))
                / np.mean(list(acc_A.values())) >= 0.3
            ) else "FAIL",
        },
        "secondary_confidence": {
            "group_A_mean": float(np.mean(list(conf_A.values()))),
            "group_B_mean": float(np.mean(list(conf_B.values()))),
            "mann_whitney_U": float(mwu_conf.statistic),
            "mann_whitney_p_value": float(mwu_conf.pvalue),
            "alternative": "B greater than A",
        },
        "secondary_decision_time_sec": {
            "group_A_mean": float(np.mean(list(time_A.values()))),
            "group_B_mean": float(np.mean(list(time_B.values()))),
            "mann_whitney_U": float(mwu_time.statistic),
            "mann_whitney_p_value": float(mwu_time.pvalue),
            "alternative": "B less than A (faster decisions)",
        },
        "summary_group_A": _summarize(A),
        "summary_group_B": _summarize(B),
    }


def compute_per_role():
    """Per-role breakdown from participant_responses.json (M6 study).

    The M5 study (survey/) has no role labels; the M6 study does (analyst /
    clinician / administrator). This is the closest available proxy for the
    spec's IT Generalist / Biomed / Nurse Manager triad.
    """
    with open(REPORTS / "participant_responses.json") as f:
        records = json.load(f)

    by_role_cond = defaultdict(list)
    for r in records:
        key = (r["participant_role"], r["condition"])
        by_role_cond[key].append(r)

    rows = []
    for (role, cond), recs in sorted(by_role_cond.items()):
        accuracies = [int(r["decision_correct"]) for r in recs]
        decision_times = [r["decision_time_sec"] for r in recs]
        confs = [r["confidence"] for r in recs]
        rows.append({
            "role": role,
            "condition": cond,
            "n": len(recs),
            "n_correct": int(sum(accuracies)),
            "accuracy": float(np.mean(accuracies)),
            "mean_decision_time_sec": float(np.mean(decision_times)),
            "mean_confidence": float(np.mean(confs)),
            "mean_likert_trust": float(np.mean([r["likert_trust"] for r in recs])),
            "mean_likert_usefulness": float(np.mean([r["likert_usefulness"] for r in recs])),
        })

    # Per-role with vs without XAI comparison (paired by role)
    role_comparison = {}
    for role in sorted(set(r["participant_role"] for r in records)):
        with_xai = [int(r["decision_correct"]) for r in records
                    if r["participant_role"] == role and r["condition"] == "with_xai"]
        without_xai = [int(r["decision_correct"]) for r in records
                       if r["participant_role"] == role and r["condition"] == "without_xai"]
        if not with_xai or not without_xai:
            continue
        mwu = stats.mannwhitneyu(with_xai, without_xai, alternative="greater")
        role_comparison[role] = {
            "n_with_xai": len(with_xai),
            "n_without_xai": len(without_xai),
            "accuracy_with_xai": float(np.mean(with_xai)),
            "accuracy_without_xai": float(np.mean(without_xai)),
            "delta_pp": float((np.mean(with_xai) - np.mean(without_xai)) * 100),
            "mann_whitney_U": float(mwu.statistic),
            "mann_whitney_p_value": float(mwu.pvalue),
        }

    return {
        "_meta": {
            "description": "RQ3 per-role breakdown (M6 study, with_xai vs without_xai)",
            "data_source": "results/reports/participant_responses.json",
            "note": (
                "Role labels are analyst / clinician / administrator — the "
                "closest proxy available for the spec's IT Generalist / "
                "Biomed Engineer / Nurse Manager triad. M5 study (survey/) "
                "is unrolled and has no role metadata."
            ),
        },
        "rows": rows,
        "role_comparison_with_vs_without_xai": role_comparison,
    }


def compute_escalation_chi2():
    """Chi-square on escalation behavior — escalate vs not, by group."""
    with open(SURVEY / "study_responses_A.json") as f:
        A = json.load(f)
    with open(SURVEY / "study_responses_B.json") as f:
        B = json.load(f)

    def _is_escalate(r):
        return 1 if r.get("response", {}).get("action_chosen") == "escalate" else 0

    # Per-ground-truth-severity escalation behavior
    SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    cells = {}
    for sev in SEVERITIES:
        A_sev = [r for r in A if r.get("ground_truth_severity") == sev]
        B_sev = [r for r in B if r.get("ground_truth_severity") == sev]
        A_esc = sum(_is_escalate(r) for r in A_sev)
        B_esc = sum(_is_escalate(r) for r in B_sev)
        cells[sev] = {
            "n_A": len(A_sev), "A_escalated": A_esc, "A_rate": A_esc / max(1, len(A_sev)),
            "n_B": len(B_sev), "B_escalated": B_esc, "B_rate": B_esc / max(1, len(B_sev)),
        }

    # Overall escalation rate chi-square (2×2 contingency: group × escalated/not)
    A_esc_total = sum(_is_escalate(r) for r in A)
    B_esc_total = sum(_is_escalate(r) for r in B)
    contingency = np.array([
        [A_esc_total,        len(A) - A_esc_total],
        [B_esc_total,        len(B) - B_esc_total],
    ])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Appropriate escalation: escalated when sev is CRITICAL or HIGH
    def _appropriate_escalation(responses):
        flags = []
        for r in responses:
            sev = r.get("ground_truth_severity")
            if sev not in ("CRITICAL", "HIGH"):
                continue
            flags.append(_is_escalate(r))
        return float(np.mean(flags)) if flags else None

    return {
        "_meta": {
            "description": "Chi-square on escalation rate (A baseline vs B with MVE)",
            "data_sources": ["survey/study_responses_A.json", "survey/study_responses_B.json"],
        },
        "overall_chi2": {
            "contingency_table": contingency.tolist(),
            "row_labels": ["Group A", "Group B"],
            "col_labels": ["Escalated", "Not escalated"],
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "expected_frequencies": expected.tolist(),
            "passes_significance": bool(p_value < 0.05),
        },
        "appropriate_escalation_rate": {
            "group_A": _appropriate_escalation(A),
            "group_B": _appropriate_escalation(B),
            "definition": "Fraction of CRITICAL+HIGH alerts where operator chose `escalate`",
        },
        "by_ground_truth_severity": cells,
    }


def main():
    print("[1] Computing primary stats (Mann-Whitney)...")
    primary = compute_primary()
    with open(OUT / "rq3_primary.json", "w") as f:
        json.dump(primary, f, indent=2, default=float)
    print(f"  → {OUT / 'rq3_primary.json'}")
    p = primary["primary_metric_composite_accuracy"]
    print(f"  composite accuracy A={p['group_A_mean']:.4f}  B={p['group_B_mean']:.4f}  "
          f"Δ={p['relative_improvement']*100:+.1f}%  p={p['mann_whitney_p_value']:.5f}  "
          f"d={p['cohens_d']:.3f}  verdict={p['verdict']}")

    print()
    print("[2] Computing per-role breakdown...")
    per_role = compute_per_role()
    with open(OUT / "rq3_per_role.json", "w") as f:
        json.dump(per_role, f, indent=2, default=float)
    print(f"  → {OUT / 'rq3_per_role.json'}")
    for role, comp in per_role["role_comparison_with_vs_without_xai"].items():
        print(f"  {role}: w/XAI={comp['accuracy_with_xai']:.3f}  "
              f"w/o={comp['accuracy_without_xai']:.3f}  "
              f"Δ={comp['delta_pp']:+.1f}pp  p={comp['mann_whitney_p_value']:.4f}")

    print()
    print("[3] Computing escalation chi-square...")
    chi2 = compute_escalation_chi2()
    with open(OUT / "rq3_escalation_chi2.json", "w") as f:
        json.dump(chi2, f, indent=2, default=float)
    print(f"  → {OUT / 'rq3_escalation_chi2.json'}")
    o = chi2["overall_chi2"]
    print(f"  chi2={o['chi2_statistic']:.3f}  p={o['p_value']:.5f}  "
          f"passes={'YES' if o['passes_significance'] else 'NO'}")
    ar = chi2["appropriate_escalation_rate"]
    print(f"  appropriate-escalation A={ar['group_A']:.3f}  B={ar['group_B']:.3f}")


if __name__ == "__main__":
    main()
