"""Module 6 batch-pipeline entrypoint (Task 6.2 + 6.6-6.8 + 6D.5 + 6D.7)."""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .alerts import _curate_split_paths, curate_evaluation_alerts
from .feedback import analyze_feedback
from .figures import CHARTS_DIR, generate_thesis_figures
from .irr import compute_inter_rater_reliability
from .metrics import compute_evaluation_metrics
from .simulated_responses import generate_simulated_responses
from .statistical import statistical_analysis

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results/reports"


def _strict_json_default(obj):
    """Strict JSON encoder replacing the prior ``default=str`` (C4 fix).

    Converts datetime/Path/numpy scalars to native JSON types. Anything else
    raises ``TypeError`` so producer bugs surface immediately rather than
    silently stringifying upstream.
    """
    if isinstance(obj, (datetime, Path)):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(
        f"evaluation_results.json contains a non-JSON-serialisable value "
        f"of type {type(obj).__name__!r}: {obj!r}"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Build Module 6 evaluation artefacts. Default mode runs the full "
            "thesis pipeline (curate + simulate + stats + IRR + figures) for "
            "the test split. --curate-only is a thin mode that produces ONLY "
            "evaluation_alerts{suffix}.json."
        )
    )
    parser.add_argument(
        "--split", choices=("test", "demo"), default="test",
        help="Frozen split (test=paper-clean, demo=operator-clean). Default: test.",
    )
    parser.add_argument(
        "--curate-only", action="store_true",
        help="Only run 6.2 curation; skip simulated responses, stats, IRR, figures.",
    )
    args = parser.parse_args()

    paths = _curate_split_paths(args.split)
    suffix = paths["suffix"]

    sep = "=" * 72
    t0 = time.perf_counter()

    logger.info(sep)
    logger.info(
        "MODULE 6 — BUILD EVALUATION ARTIFACTS — split=%s%s",
        args.split, " [curate-only]" if args.curate_only else "",
    )
    logger.info(sep)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.curate_only:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    alerts = curate_evaluation_alerts(args.split)
    alerts_path = OUTPUT_DIR / f"evaluation_alerts{suffix}.json"
    alerts_path.write_text(json.dumps(alerts, indent=2), encoding="utf-8")
    logger.info("6.2 Saved: %s (%d alerts)", alerts_path.name, len(alerts))

    if args.curate_only:
        elapsed = round(time.perf_counter() - t0, 1)
        logger.info("")
        logger.info(sep)
        logger.info("CURATE-ONLY COMPLETE — %.1fs (split=%s)", elapsed, args.split)
        logger.info(sep)
        logger.info("  Alerts : %s (%d)", alerts_path.name, len(alerts))
        logger.info(sep)
        return

    logger.info("")
    logger.info("Generating simulated participant responses (for thesis validation)...")
    responses = generate_simulated_responses(alerts)
    resp_path = OUTPUT_DIR / "participant_responses.json"
    resp_path.write_text(json.dumps(responses, indent=2), encoding="utf-8")
    logger.info("  Saved: participant_responses.json (%d responses)", len(responses))

    logger.info("")
    logger.info("── 6.6 Evaluation Metrics ──")
    metrics = compute_evaluation_metrics(responses)
    logger.info("  With XAI:    accuracy=%.1f%%, trust=%.1f, time=%.0fs",
                metrics["with_xai"]["decision_accuracy"] * 100,
                metrics["with_xai"]["likert_trust"],
                metrics["with_xai"]["mean_decision_time_sec"])
    logger.info("  Without XAI: accuracy=%.1f%%, trust=%.1f, time=%.0fs",
                metrics["without_xai"]["decision_accuracy"] * 100,
                metrics["without_xai"]["likert_trust"],
                metrics["without_xai"]["mean_decision_time_sec"])

    logger.info("")
    logger.info("── 6.7 Statistical Analysis ──")
    stats = statistical_analysis(responses)
    for measure, result in stats.items():
        sig = "***" if result.get("significant") else "n.s."
        logger.info("  %s: Δ=%.3f, p=%.4f %s (d=%.2f, %s)",
                    measure, result.get("difference", 0), result.get("p_value", 1),
                    sig, result.get("cohens_d", 0), result.get("effect_size", ""))

    logger.info("")
    logger.info("── 6D.5 Inter-Rater Reliability ──")
    irr = compute_inter_rater_reliability(responses)
    for measure, result in irr.items():
        logger.info("  %s: alpha=%.4f (%s)", measure, result["alpha"],
                    result.get("interpretation", ""))

    logger.info("")
    logger.info("── 6D.7 Feedback Analysis ──")
    feedback = analyze_feedback(responses)
    logger.info("  Feedback texts: %d, Reclassifications: %d",
                feedback["total_feedback_texts"], feedback["n_reclassifications"])
    logger.info("  Themes: %s", feedback["thematic_counts"])

    eval_results = {
        "metrics": metrics,
        "statistical_tests": stats,
        "inter_rater_reliability": irr,
        "feedback_analysis": feedback,
    }
    (OUTPUT_DIR / "evaluation_results.json").write_text(
        json.dumps(eval_results, indent=2, default=_strict_json_default),
        encoding="utf-8",
    )
    logger.info("  Saved: evaluation_results.json")

    (OUTPUT_DIR / "feedback_recommendations.json").write_text(
        json.dumps({
            "corrections": feedback["corrections_for_modules_3_5"],
            "thematic_recommendations": feedback["recommendations"],
            "reclassification_summary": feedback["reclassification_counts"],
        }, indent=2, default=_strict_json_default),
        encoding="utf-8",
    )
    logger.info("  Saved: feedback_recommendations.json")

    logger.info("")
    logger.info("── 6.8 Thesis Figures ──")
    generate_thesis_figures(metrics, stats, responses)

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("EVALUATION BUILD COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    logger.info("  Protocol   : data/phase2/evaluation/evaluation_protocol.md")
    logger.info("  Alerts     : evaluation_alerts.json (%d)", len(alerts))
    logger.info("  Responses  : participant_responses.json (%d)", len(responses))
    logger.info("  Results    : evaluation_results.json")
    logger.info("  Charts     : %s", CHARTS_DIR)
    logger.info(sep)


__all__ = ["main", "_strict_json_default"]
