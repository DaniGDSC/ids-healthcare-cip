#!/usr/bin/env python3
"""
Compute M5 metric from user study responses.
Input:  results/reports/study_responses_*.json
Output: results/reports/m5_result.yaml
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import mannwhitneyu

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

TARGET_IMPROVEMENT = 0.30   # 30% relative improvement
MINIMUM_IMPROVEMENT = 0.15  # 15% minimum
P_VALUE_THRESHOLD = 0.05


def load_all_responses() -> list[dict]:
    """Load all participant response files."""
    responses = []
    for path in REPORTS_DIR.glob("study_responses_*.json"):
        with open(path) as f:
            responses.extend(json.load(f))
    logger.info("Loaded %d responses from %d participants",
                len(responses),
                len(list(REPORTS_DIR.glob("study_responses_*.json"))))
    return responses


def compute_participant_accuracy(
    responses: list[dict],
    condition: str
) -> dict[str, float]:
    """
    Compute composite accuracy per participant for a condition.
    Returns: {participant_id: mean_composite_score}
    """
    scores: dict[str, list[float]] = defaultdict(list)
    for r in responses:
        if r["condition"] == condition:
            scores[r["participant_id"]].append(r["composite_score"])
    return {pid: float(np.mean(s)) for pid, s in scores.items()}


def run_m5_analysis(responses: list[dict]) -> dict:
    """
    Primary M5 analysis:
    Mann-Whitney U test on composite accuracy
    with_mve group vs without_mve group.
    """
    with_mve = compute_participant_accuracy(responses, "with_mve")
    without_mve = compute_participant_accuracy(responses, "without_mve")

    if not with_mve or not without_mve:
        return {"error": "Insufficient data — need responses from both conditions"}

    a_scores = list(without_mve.values())
    b_scores = list(with_mve.values())

    mean_a = float(np.mean(a_scores))
    mean_b = float(np.mean(b_scores))

    relative_improvement = (mean_b - mean_a) / mean_a if mean_a > 0 else 0.0

    # Mann-Whitney U (one-tailed: B > A)
    stat, p_value = mannwhitneyu(b_scores, a_scores, alternative="greater")

    # Cohen's d
    pooled_std = float(np.sqrt(
        (np.std(a_scores, ddof=1)**2 + np.std(b_scores, ddof=1)**2) / 2
    ))
    cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0

    # Pass/fail (cast to Python bool to avoid numpy bool in YAML)
    passes_improvement = bool(relative_improvement >= TARGET_IMPROVEMENT)
    passes_significance = bool(p_value < P_VALUE_THRESHOLD)
    passes_minimum = bool(relative_improvement >= MINIMUM_IMPROVEMENT)

    verdict = (
        "PASS" if passes_improvement and passes_significance
        else "WARN" if passes_minimum and passes_significance
        else "FAIL"
    )

    return {
        "n_participants_with_mve": len(b_scores),
        "n_participants_without_mve": len(a_scores),
        "mean_accuracy_with_mve": round(mean_b, 4),
        "mean_accuracy_without_mve": round(mean_a, 4),
        "relative_improvement": round(relative_improvement, 4),
        "target_improvement": TARGET_IMPROVEMENT,
        "passes_improvement_threshold": passes_improvement,
        "mann_whitney_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "passes_significance": passes_significance,
        "cohens_d": round(cohens_d, 4),
        "effect_size": (
            "large" if abs(cohens_d) > 0.8
            else "medium" if abs(cohens_d) > 0.5
            else "small"
        ),
        "verdict": verdict,
    }


def run_secondary_analyses(responses: list[dict]) -> dict:
    """
    Secondary metrics:
    - Severity accuracy
    - Action accuracy
    - Over-reaction rate (CRITICAL/HIGH on false positives)
    - Under-reaction rate (LOW/dismiss on true positives)
    - Decision time comparison
    - Catastrophic miss rate (CRITICAL↔LOW)
    """
    results = {}

    for condition in ["with_mve", "without_mve"]:
        cond = [r for r in responses if r["condition"] == condition]
        if not cond:
            continue

        # M6-S1: single pass over cond — replaces 3 separate O(N) list comps
        over = under = catast = 0
        for r in cond:
            gt = r["ground_truth_label"]
            action = r["chosen_action"]
            if gt == "false_positive" and action in ("isolate", "escalate"):
                over += 1
            if (gt == "true_positive"
                    and r["correct_severity"] == "CRITICAL"
                    and action == "dismiss"):
                under += 1
            if r.get("catastrophic_miss", False):
                catast += 1

        n_cond = len(cond)
        results[condition] = {
            "severity_accuracy": round(
                float(np.mean([r["severity_score"] for r in cond])), 4
            ),
            "action_accuracy": round(
                float(np.mean([r["action_correct"] for r in cond])), 4
            ),
            "over_reaction_rate": round(over / n_cond, 4),
            "under_reaction_rate": round(under / n_cond, 4),
            "catastrophic_miss_rate": round(catast / n_cond, 4),
            "mean_decision_time_sec": round(
                float(np.mean([r["decision_time_sec"] for r in cond])), 1
            ),
            "mean_confidence": round(
                float(np.mean([r["confidence"] for r in cond])), 2
            ),
        }

    # Q21 + Q22 proxy analysis
    proxy_responses = [
        r for r in responses
        if "q21_clinical_clarity" in r
    ]

    if proxy_responses:
        # M6-S2: Counter replaces manual get/+1 dict accumulation
        q21_counts = Counter(r.get("q21_clinical_clarity", "") for r in proxy_responses)
        q22_counts = Counter(r.get("q22_management_justification", "") for r in proxy_responses)
        n_proxy = len(proxy_responses)

        results["proxy_stakeholder"] = {
            "n_respondents": n_proxy,
            "q21_clinical_clarity": dict(q21_counts),
            "q21_yes_rate": round(
                sum("Yes" in r.get("q21_clinical_clarity", "") for r in proxy_responses)
                / n_proxy, 4
            ),
            "q22_management_justification": dict(q22_counts),
            "q22_yes_rate": round(
                sum("Yes" in r.get("q22_management_justification", "") for r in proxy_responses)
                / n_proxy, 4
            ),
            "q21_notes": [r["q21_note"] for r in proxy_responses
                          if r.get("q21_note")],
            "q22_notes": [r["q22_note"] for r in proxy_responses
                          if r.get("q22_note")],
        }

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    responses = load_all_responses()
    if not responses:
        logger.error("No response files found in %s", REPORTS_DIR)
        return

    logger.info("Running M5 primary analysis...")
    m5 = run_m5_analysis(responses)

    logger.info("Running secondary analyses...")
    secondary = run_secondary_analyses(responses)

    result = {
        "claim": "C4 — enabling correct triage from non-specialist operators",
        "metric": "M5 — Triage Decision Accuracy",
        "m5_primary": m5,
        "secondary_metrics": secondary,
        "interpretation": {
            "PASS": "C4 SUPPORTED — MVE significantly improves triage accuracy",
            "WARN": "C4 PARTIAL — improvement present but below target threshold",
            "FAIL": "C4 NOT SUPPORTED — MVE did not improve accuracy significantly",
        }.get(m5.get("verdict", "FAIL"), "Unknown"),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("M5 RESULT SUMMARY")
    print("=" * 60)
    print(f"  Without MVE accuracy: {m5.get('mean_accuracy_without_mve', 0):.1%}")
    print(f"  With MVE accuracy:    {m5.get('mean_accuracy_with_mve', 0):.1%}")
    print(f"  Relative improvement: {m5.get('relative_improvement', 0):.1%}")
    print(f"  p-value:              {m5.get('p_value', 1):.4f}")
    print(f"  Cohen's d:            {m5.get('cohens_d', 0):.2f} ({m5.get('effect_size', 'N/A')})")
    print(f"  VERDICT:              {m5.get('verdict', 'FAIL')}")
    print("=" * 60)

    # Save
    out_path = REPORTS_DIR / "m5_result.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, allow_unicode=True)

    logger.info("Saved: %s", out_path)
    return result


if __name__ == "__main__":
    main()
