"""Module 5 pipeline CLI — worked examples + audit log management."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .audit import AuditLogger
from .audit.logger import DEFAULT_RETENTION_DAYS
from .audit.signing import OUTPUT_DIR
from .executor import ActionExecutor, NotificationService
from .loaders import (
    PROJECT_ROOT,
    load_attack_categories,
    load_explanations,
    load_risk_scores,
)
from .feedback import FeedbackLoop
from .policy import PolicyEngine, export_response_policy
from .worked_examples import run_worked_examples

logger = logging.getLogger(__name__)


def _strict_json_default(obj):
    """Coerce datetime/Path-like values for the worked_examples artifact.

    Y1 follow-up: replaces the old ``default=str`` blanket call. Only
    converts known datetime/Path types — any unexpected non-serialisable
    value raises a TypeError so producer bugs surface immediately.
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
        f"worked_examples.json contains a non-JSON-serialisable value "
        f"of type {type(obj).__name__!r}: {obj!r}"
    )


def _load_pipeline_inputs() -> tuple[dict, dict, dict, np.ndarray]:
    """Load the canonical Module 5 inputs via shared loaders.

    Centralising this keeps the CLI aligned with the signed risk-score
    pair and the split-aware path conventions already used elsewhere in
    Module 5.
    """
    risk_data = load_risk_scores(PROJECT_ROOT / "results/reports/risk_scores.npz")
    analyst_by_idx, clinician_by_idx = load_explanations(
        PROJECT_ROOT / "results/reports/analyst_report.json",
        PROJECT_ROOT / "results/reports/clinician_summaries.json",
    )
    attack_cats = load_attack_categories(
        PROJECT_ROOT / "data/processed/test_phase1.parquet",
    )
    return risk_data, analyst_by_idx, clinician_by_idx, attack_cats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sep = "=" * 72
    t0 = time.perf_counter()

    logger.info(sep)
    logger.info("MODULE 5 — RESPONSE PIPELINE INTEGRATION (Tasks 5.1-5.8)")
    logger.info(sep)

    risk_data, analyst_by_idx, clinician_by_idx, attack_cats = _load_pipeline_inputs()

    n_samples = len(risk_data["R"])
    logger.info("Loaded: %d samples", n_samples)

    export_response_policy()

    logger.info("")
    logger.info("── 5.7 End-to-End Worked Examples ──")
    scenarios = run_worked_examples(
        risk_data, attack_cats, analyst_by_idx, clinician_by_idx
    )
    (OUTPUT_DIR / "worked_examples.json").write_text(
        json.dumps(scenarios, indent=2, default=_strict_json_default),
        encoding="utf-8",
    )
    logger.info("  Saved: worked_examples.json (%d scenarios)", len(scenarios))

    logger.info("")
    logger.info("── 5.6/5.8 Full Pipeline Run (audit + feedback) ──")

    engine = PolicyEngine()
    executor = ActionExecutor()
    notifier = NotificationService()
    audit = AuditLogger(OUTPUT_DIR / "audit_log.jsonl")
    feedback = FeedbackLoop()

    R = risk_data["R"]
    levels = risk_data["risk_levels"]
    y_true = risk_data["y_true"]

    alert_count = 0
    for idx in range(n_samples):
        tier = str(levels[idx])
        if tier == "LOW" and R[idx] < 0.25:
            continue

        cat = str(attack_cats[idx])
        gt = "attack" if y_true[idx] == 1 else "benign"
        a_pat = float(risk_data["d_clinical_tier"][idx])

        rec = engine.recommend(tier, "vital_monitoring", cat, a_pat)
        ts = datetime(2026, 4, 3, 12, 0, 0) + timedelta(seconds=idx)
        alert_id = f"ALERT-{idx:05d}"

        exec_result = executor.execute(alert_id, idx, rec["actions"], rec, gt, ts)
        audit.log(exec_result)
        feedback.record(alert_id, gt, tier, float(R[idx]), rec["actions"])
        alert_count += 1

    logger.info("  Processed %d alerts through pipeline", alert_count)
    logger.info(
        "  Audit log: %s (%d records)", OUTPUT_DIR / "audit_log.jsonl", alert_count
    )

    adjustments = feedback.compute_adjustments()
    (OUTPUT_DIR / "feedback_analysis.json").write_text(
        json.dumps(adjustments, indent=2), encoding="utf-8"
    )
    logger.info("")
    logger.info("── 5.8 Feedback Loop Analysis ──")
    logger.info(
        "  TP=%d, FP=%d, FN=%d",
        adjustments["true_positives"],
        adjustments["false_positives"],
        adjustments["false_negatives"],
    )
    logger.info(
        "  FP rate: %.1f%%, FN rate: %.1f%%",
        adjustments["fpr"] * 100,
        adjustments["fnr"] * 100,
    )
    logger.info("  Current thresholds: %s", adjustments.get("current_thresholds"))
    logger.info(
        "  Suggested thresholds: %s", adjustments.get("suggested_threshold_change")
    )
    for adj in adjustments.get("adjustments", []):
        logger.info("  Adjustment: %s", adj)
    logger.info("  Saved: feedback_analysis.json")

    logger.info("")
    logger.info("  Notifications generated: %d", len(notifier.notifications))

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("RESPONSE PIPELINE COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    logger.info("  5.1 response_policy.json")
    logger.info("  5.6 audit_log.jsonl (%d records)", alert_count)
    logger.info("  5.7 worked_examples.json (%d scenarios)", len(scenarios))
    logger.info("  5.8 feedback_analysis.json")
    logger.info("  Output: %s", OUTPUT_DIR)
    logger.info(sep)


def _cli_verify(args: argparse.Namespace) -> int:
    path = Path(args.path or (OUTPUT_DIR / "audit_log.jsonl"))
    pubkey = Path(args.public_key) if args.public_key else None
    report = AuditLogger.verify(path, pubkey, legacy_ok=not args.strict)
    print(json.dumps(report, indent=2))
    return 0 if report["first_break_at"] is None else 1


def _cli_rotate(args: argparse.Namespace) -> int:
    path = Path(args.path or (OUTPUT_DIR / "audit_log.jsonl"))
    audit = AuditLogger(
        path,
        retention_days=args.retention_days,
        sign=not args.no_sign,
    )
    report = audit.rotate_and_purge(retention_days=args.retention_days)
    print(json.dumps(report, indent=2))
    return (
        0
        if report["verify_before_rotate"] is None
        or report["verify_before_rotate"]["first_break_at"] is None
        else 2
    )


def cli_entry() -> None:
    """Entry point for ``python -m module5_responses.module5_pipeline``."""
    parser = argparse.ArgumentParser(
        prog="python -m module5_responses.module5_pipeline",
        description="Module 5 — response pipeline + audit log management",
    )
    parser.add_argument(
        "--verify-audit-log",
        dest="verify_audit_log",
        action="store_true",
        help="Verify hash chain + signatures of an audit log JSONL file.",
    )
    parser.add_argument(
        "--rotate-audit-log",
        dest="rotate_audit_log",
        action="store_true",
        help="Rotate the active audit log if its oldest record is "
        "older than the retention window. Refuses to rotate a "
        "tampered log.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Audit log path (default: results/reports/audit_log.jsonl)",
    )
    parser.add_argument(
        "--public-key",
        default=None,
        help="Public key PEM for verification "
        "(default: results/reports/audit_signing_key.pub.pem)",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help=f"Retention window in days (default: {DEFAULT_RETENTION_DAYS}; "
        f"env: IOMT_AUDIT_RETENTION_DAYS)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat unsigned (legacy) records as verification failures.",
    )
    parser.add_argument(
        "--no-sign",
        action="store_true",
        help="Disable signing for the rotate marker (testing only).",
    )

    args = parser.parse_args()

    if args.verify_audit_log:
        sys.exit(_cli_verify(args))
    if args.rotate_audit_log:
        sys.exit(_cli_rotate(args))

    main()


__all__ = ["main", "cli_entry", "_cli_verify", "_cli_rotate"]
