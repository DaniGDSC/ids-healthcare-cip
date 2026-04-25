#!/usr/bin/env python3
from __future__ import annotations

"""Analyze the final RQ3 A/B study from flat study response exports.

Canonical input:
    results/reports/study_responses_*.json

This script intentionally targets the current dashboard export format from
module6_evaluation/module6_app.py. Legacy prototype files under `survey/`
are out of scope for this analysis and should be converted separately if
they ever need to be reused.
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - environment dependent
    sns = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOB = "results/reports/study_responses_*.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "outputs"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "analysis" / "plots"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "results" / "reports" / "audit_trail.jsonl"


def _normalize_condition(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"with_mve", "with_xai"}:
        return "with_mve"
    if text in {"without_mve", "without_xai"}:
        return "without_mve"
    return text or None


def _is_proxy_row(row: dict[str, Any]) -> bool:
    return "q21_clinical_clarity" in row or "q22_management_justification" in row


def load_all_responses(glob_pattern: str = DEFAULT_GLOB) -> tuple[pd.DataFrame, pd.DataFrame]:
    response_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []

    for fname in sorted(glob.glob(glob_pattern)):
        with open(fname, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            continue

        for row in data:
            if not isinstance(row, dict):
                continue
            if _is_proxy_row(row):
                proxy_rows.append(dict(row))
                continue

            condition = _normalize_condition(row.get("condition"))
            if condition not in {"with_mve", "without_mve"}:
                continue

            normalized = dict(row)
            normalized["condition"] = condition
            response_rows.append(normalized)

    return pd.DataFrame(response_rows), pd.DataFrame(proxy_rows)


def load_demographics_from_audit(
    audit_path: Path = DEFAULT_AUDIT_PATH,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not audit_path.exists():
        return pd.DataFrame()

    with open(audit_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") != "study_start":
                continue
            rows.append(
                {
                    "participant_id": event.get("participant_id"),
                    "participant_role": event.get("role"),
                    "participant_years": event.get("years"),
                    "participant_ids_exp": event.get("ids_exp"),
                    "timestamp": event.get("timestamp"),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").drop_duplicates("participant_id", keep="last")
    return df


def compute_metrics(df: pd.DataFrame, correctness_mode: str = "composite_gt_0_5") -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed = df.copy()
    if detailed.empty:
        return pd.DataFrame(), detailed

    default_values: dict[str, Any] = {
        "composite_score": np.nan,
        "severity_score": np.nan,
        "severity_correct": False,
        "action_correct": False,
        "catastrophic_miss": False,
        "confidence": np.nan,
        "decision_time_sec": np.nan,
        "chosen_action": None,
        "correct_action": None,
        "participant_role": None,
    }
    for col, default in default_values.items():
        if col not in detailed.columns:
            detailed[col] = default

    numeric_cols = [
        "composite_score",
        "severity_score",
        "confidence",
        "decision_time_sec",
    ]
    bool_cols = [
        "severity_correct",
        "action_correct",
        "catastrophic_miss",
    ]
    for col in numeric_cols:
        if col in detailed.columns:
            detailed[col] = pd.to_numeric(detailed[col], errors="coerce")
    for col in bool_cols:
        if col in detailed.columns:
            detailed[col] = detailed[col].fillna(False).astype(bool)

    if correctness_mode == "exact_match":
        detailed["correct"] = detailed["severity_correct"] & detailed["action_correct"]
    else:
        detailed["correct"] = detailed["composite_score"] > 0.5

    detailed["over_reaction"] = (
        detailed["chosen_action"].isin(["isolate", "escalate"])
        & detailed["correct_action"].isin(["monitor", "dismiss"])
    )
    detailed["under_reaction"] = (
        detailed["chosen_action"].isin(["monitor", "dismiss"])
        & detailed["correct_action"].isin(["isolate", "escalate"])
    )

    group_stats = (
        detailed.groupby("condition")
        .agg(
            n_responses=("participant_id", "size"),
            n_participants=("participant_id", "nunique"),
            accuracy=("correct", "mean"),
            severity_accuracy=("severity_correct", "mean"),
            action_accuracy=("action_correct", "mean"),
            mean_composite_score=("composite_score", "mean"),
            over_reaction_rate=("over_reaction", "mean"),
            under_reaction_rate=("under_reaction", "mean"),
            catastrophic_miss_rate=("catastrophic_miss", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_response_time=("decision_time_sec", "mean"),
        )
        .reset_index()
    )

    return group_stats, detailed


def compute_participant_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby(["participant_id", "condition"])
        .agg(
            participant_role=("participant_role", "first"),
            accuracy=("correct", "mean"),
            severity_accuracy=("severity_correct", "mean"),
            action_accuracy=("action_correct", "mean"),
            mean_composite_score=("composite_score", "mean"),
            over_reaction=("over_reaction", "mean"),
            under_reaction=("under_reaction", "mean"),
            catastrophic_miss=("catastrophic_miss", "mean"),
            confidence=("confidence", "mean"),
            response_time=("decision_time_sec", "mean"),
        )
        .reset_index()
    )


def _cohens_dz(with_values: pd.Series, without_values: pd.Series) -> float | None:
    diffs = with_values - without_values
    if len(diffs) < 2:
        return None
    std = diffs.std(ddof=1)
    if std == 0 or np.isnan(std):
        return None
    return float(diffs.mean() / std)


def statistical_tests(participant_stats: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    if participant_stats.empty:
        return {}, pd.DataFrame()

    pivot = participant_stats.pivot(index="participant_id", columns="condition")
    pivot.columns = ["{}_{}".format(metric, condition) for metric, condition in pivot.columns]

    metrics = [
        "accuracy",
        "severity_accuracy",
        "action_accuracy",
        "mean_composite_score",
        "over_reaction",
        "under_reaction",
        "catastrophic_miss",
        "confidence",
        "response_time",
    ]

    results: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        col_with = f"{metric}_with_mve"
        col_without = f"{metric}_without_mve"
        if col_with not in pivot.columns or col_without not in pivot.columns:
            continue

        paired = pivot[[col_with, col_without]].dropna()
        if len(paired) < 2:
            results[metric] = {
                "n_pairs": int(len(paired)),
                "p_value": None,
                "statistic": None,
                "mean_with_mve": float(paired[col_with].mean()) if len(paired) else None,
                "mean_without_mve": float(paired[col_without].mean()) if len(paired) else None,
                "mean_difference": float((paired[col_with] - paired[col_without]).mean()) if len(paired) else None,
                "cohens_dz": None,
                "note": "Need at least 2 paired participants for Wilcoxon.",
            }
            continue

        diffs = paired[col_with] - paired[col_without]
        if np.allclose(diffs.values, 0.0, equal_nan=False):
            results[metric] = {
                "n_pairs": int(len(paired)),
                "p_value": 1.0,
                "statistic": 0.0,
                "mean_with_mve": float(paired[col_with].mean()),
                "mean_without_mve": float(paired[col_without].mean()),
                "mean_difference": 0.0,
                "cohens_dz": None,
            }
            continue

        stat, p_value = wilcoxon(
            paired[col_with],
            paired[col_without],
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        results[metric] = {
            "n_pairs": int(len(paired)),
            "p_value": float(p_value),
            "statistic": float(stat),
            "mean_with_mve": float(paired[col_with].mean()),
            "mean_without_mve": float(paired[col_without].mean()),
            "mean_difference": float((paired[col_with] - paired[col_without]).mean()),
            "cohens_dz": _cohens_dz(paired[col_with], paired[col_without]),
        }

    return results, pivot.reset_index()


def summarize_proxy_questions(proxy_df: pd.DataFrame) -> dict[str, Any]:
    if proxy_df.empty:
        return {}

    summary: dict[str, Any] = {
        "n_respondents": int(proxy_df["participant_id"].nunique()) if "participant_id" in proxy_df else 0,
    }

    if "q21_clinical_clarity" in proxy_df:
        q21 = proxy_df["q21_clinical_clarity"].dropna().astype(str)
        summary["q21_counts"] = q21.value_counts().to_dict()
        summary["q21_notes"] = [
            note for note in proxy_df.get("q21_note", pd.Series(dtype=object)).dropna().astype(str)
            if note.strip()
        ]
    if "q22_management_justification" in proxy_df:
        q22 = proxy_df["q22_management_justification"].dropna().astype(str)
        summary["q22_counts"] = q22.value_counts().to_dict()
        summary["q22_notes"] = [
            note for note in proxy_df.get("q22_note", pd.Series(dtype=object)).dropna().astype(str)
            if note.strip()
        ]
    return summary


def build_demographics_table(
    responses_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> pd.DataFrame:
    base = (
        responses_df.groupby("participant_id", as_index=False)
        .agg(participant_role=("participant_role", "first"))
        if not responses_df.empty else pd.DataFrame(columns=["participant_id", "participant_role"])
    )

    if audit_df.empty:
        if "participant_years" not in base.columns:
            base["participant_years"] = pd.NA
        if "participant_ids_exp" not in base.columns:
            base["participant_ids_exp"] = pd.NA
        return base

    merged = base.merge(audit_df, on="participant_id", how="outer", suffixes=("", "_audit"))
    if "participant_role_audit" in merged.columns:
        merged["participant_role"] = merged["participant_role"].fillna(merged["participant_role_audit"])
        merged = merged.drop(columns=["participant_role_audit"])
    return merged


def plot_results(participant_stats: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if participant_stats.empty:
        return

    if sns is not None:
        sns.set_theme(style="whitegrid")

        plt.figure(figsize=(6, 4))
        sns.barplot(data=participant_stats, x="condition", y="accuracy", errorbar="se")
        plt.title("Decision Accuracy by Condition")
        plt.ylabel("Accuracy")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_by_condition.png", dpi=150)
        plt.close()

        reactions = participant_stats.melt(
            id_vars=["participant_id", "condition"],
            value_vars=["over_reaction", "under_reaction"],
            var_name="reaction_type",
            value_name="rate",
        )
        plt.figure(figsize=(7, 4))
        sns.barplot(data=reactions, x="condition", y="rate", hue="reaction_type", errorbar="se")
        plt.title("Over-reaction and Under-reaction Rates")
        plt.ylabel("Rate")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(output_dir / "reaction_rates.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=participant_stats, x="condition", y="confidence")
        plt.title("Confidence by Condition")
        plt.ylabel("Confidence")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(output_dir / "confidence_boxplot.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=participant_stats, x="condition", y="response_time")
        plt.title("Response Time by Condition")
        plt.ylabel("Seconds")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(output_dir / "response_time_boxplot.png", dpi=150)
        plt.close()
        return

    grouped = participant_stats.groupby("condition")
    mean_accuracy = grouped["accuracy"].mean()
    se_accuracy = grouped["accuracy"].sem().fillna(0.0)
    plt.figure(figsize=(6, 4))
    plt.bar(mean_accuracy.index, mean_accuracy.values, yerr=se_accuracy.values, capsize=4)
    plt.title("Decision Accuracy by Condition")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_condition.png", dpi=150)
    plt.close()

    reactions = participant_stats.groupby("condition")[["over_reaction", "under_reaction"]].mean()
    x = np.arange(len(reactions.index))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, reactions["over_reaction"], width, label="over_reaction")
    plt.bar(x + width / 2, reactions["under_reaction"], width, label="under_reaction")
    plt.xticks(x, reactions.index)
    plt.ylabel("Rate")
    plt.title("Over-reaction and Under-reaction Rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reaction_rates.png", dpi=150)
    plt.close()

    confidence_data = [
        participant_stats.loc[participant_stats["condition"] == cond, "confidence"].dropna().values
        for cond in ["with_mve", "without_mve"]
        if cond in participant_stats["condition"].values
    ]
    confidence_labels = [
        cond for cond in ["with_mve", "without_mve"] if cond in participant_stats["condition"].values
    ]
    if confidence_data:
        plt.figure(figsize=(6, 4))
        plt.boxplot(confidence_data, tick_labels=confidence_labels)
        plt.title("Confidence by Condition")
        plt.ylabel("Confidence")
        plt.tight_layout()
        plt.savefig(output_dir / "confidence_boxplot.png", dpi=150)
        plt.close()

    time_data = [
        participant_stats.loc[participant_stats["condition"] == cond, "response_time"].dropna().values
        for cond in ["with_mve", "without_mve"]
        if cond in participant_stats["condition"].values
    ]
    time_labels = [
        cond for cond in ["with_mve", "without_mve"] if cond in participant_stats["condition"].values
    ]
    if time_data:
        plt.figure(figsize=(6, 4))
        plt.boxplot(time_data, tick_labels=time_labels)
        plt.title("Response Time by Condition")
        plt.ylabel("Seconds")
        plt.tight_layout()
        plt.savefig(output_dir / "response_time_boxplot.png", dpi=150)
        plt.close()


def save_outputs(
    *,
    group_stats: pd.DataFrame,
    participant_stats: pd.DataFrame,
    demographics: pd.DataFrame,
    proxy_summary: dict[str, Any],
    statistical_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    group_stats.to_csv(output_dir / "rq3_group_stats.csv", index=False)
    participant_stats.to_csv(output_dir / "rq3_participant_stats.csv", index=False)
    demographics.to_csv(output_dir / "rq3_demographics.csv", index=False)

    summary_payload = {
        "group_stats": group_stats.to_dict(orient="records"),
        "statistical_tests": statistical_results,
        "proxy_summary": proxy_summary,
    }
    with open(output_dir / "rq3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RQ3 user-study responses.")
    parser.add_argument("--glob", default=DEFAULT_GLOB, help="Glob pattern for study response JSON files.")
    parser.add_argument(
        "--correctness-mode",
        choices=["composite_gt_0_5", "exact_match"],
        default="composite_gt_0_5",
        help="How to convert per-alert responses into a binary correctness measure.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV/JSON outputs.")
    parser.add_argument("--plot-dir", default=str(DEFAULT_PLOT_DIR), help="Directory for plots.")
    args = parser.parse_args()

    responses_df, proxy_df = load_all_responses(args.glob)
    if responses_df.empty:
        print(f"No study responses found for pattern: {args.glob}")
        print("Expected files like results/reports/study_responses_P01.json")
        return

    n_participants = responses_df["participant_id"].nunique()
    print(f"Loaded {len(responses_df)} alert responses from {n_participants} participants.")

    group_stats, detailed_df = compute_metrics(responses_df, correctness_mode=args.correctness_mode)
    participant_stats = compute_participant_stats(detailed_df)
    statistical_results, _ = statistical_tests(participant_stats)
    proxy_summary = summarize_proxy_questions(proxy_df)
    audit_demographics = load_demographics_from_audit()
    demographics = build_demographics_table(responses_df, audit_demographics)

    plot_results(participant_stats, Path(args.plot_dir))
    save_outputs(
        group_stats=group_stats,
        participant_stats=participant_stats,
        demographics=demographics,
        proxy_summary=proxy_summary,
        statistical_results=statistical_results,
        output_dir=Path(args.output_dir),
    )

    print("\n=== Group Statistics ===")
    print(group_stats.to_string(index=False))

    print("\n=== Statistical Tests (Wilcoxon, paired by participant) ===")
    if not statistical_results:
        print("No paired-condition metrics available.")
    else:
        for metric, vals in statistical_results.items():
            p_val = vals.get("p_value")
            stat = vals.get("statistic")
            diff = vals.get("mean_difference")
            dz = vals.get("cohens_dz")
            print(
                f"{metric}: n={vals.get('n_pairs')}, "
                f"mean_diff={diff if diff is not None else 'NA'}, "
                f"p={p_val if p_val is not None else 'NA'}, "
                f"stat={stat if stat is not None else 'NA'}, "
                f"dz={dz if dz is not None else 'NA'}"
            )

    if proxy_summary:
        print("\n=== Proxy Questions ===")
        print(json.dumps(proxy_summary, indent=2))

    missing_years = demographics["participant_years"].isna().all() if "participant_years" in demographics else True
    missing_ids = demographics["participant_ids_exp"].isna().all() if "participant_ids_exp" in demographics else True
    if missing_years or missing_ids:
        print("\nNote: participant_years / participant_ids_exp were not recoverable for all participants.")
        print("The analyzer saves whatever it can recover from results/reports/audit_trail.jsonl.")

    print(f"\nPlots saved to: {args.plot_dir}")
    print(f"Tables and JSON summary saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
