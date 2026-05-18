"""Module 5 batch — produce ``results/reports/mve_outputs.jsonl``
(RQ2 prerequisite per RQ2_pipeline.md §3).

Iterates over the test split's **surfaced** alerts
(``fusion_class != "BENIGN"`` in ``risk_scores.npz`` schema v1.1), invokes
``src.mve_generator.generate_mve`` for each row, and serialises the
resulting ``MVEOutput`` as one JSON object per line.

Downstream consumers (RQ2 tracks 1, 2, 3, 5) read this file as their
canonical MVE corpus.

Usage:
    python -m module5_responses.module5_mve_batch
    python -m module5_responses.module5_mve_batch --include-benign  # for debugging
    python -m module5_responses.module5_mve_batch --limit 50         # smoke test

Outputs:
    results/reports/mve_outputs.jsonl       — one MVE per line
    results/reports/mve_outputs.meta.json   — provenance + counts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ``mve_generator`` and ``data_models`` use absolute imports rooted at
# the project; inserting the project root onto sys.path keeps direct
# script invocation working from any cwd.
from src.mve_generator import generate_mve, _generate_rule_based  # noqa: E402

logger = logging.getLogger(__name__)

NPZ_PATH = PROJECT_ROOT / "results/reports/risk_scores.npz"
META_PATH = PROJECT_ROOT / "results/reports/risk_scores.meta.json"
PARQUET_PATH = PROJECT_ROOT / "data/processed/test_phase1.parquet"
OUT_JSONL = PROJECT_ROOT / "results/reports/mve_outputs.jsonl"
OUT_META = PROJECT_ROOT / "results/reports/mve_outputs.meta.json"


def _assert_npz_schema_v1_1() -> None:
    """Refuse to run on the legacy npz — surfacing requires the v1.1
    extension arrays (device_class, device_criticality, patchable)."""
    if not META_PATH.exists():
        raise RuntimeError(
            f"{META_PATH} missing — run Module 3 first to regenerate "
            "risk_scores.npz under schema v1.1 (RQ1_pipeline.md §4)."
        )
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if meta.get("schema_version") != "1.1":
        raise RuntimeError(
            f"risk_scores.npz schema is {meta.get('schema_version')!r}, "
            "expected '1.1'. Re-run Module 3."
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _build_device_context(
    device_class: str, device_criticality: str, patchable: bool
) -> dict:
    """Synthesise the per-alert device_context dict expected by
    ``generate_mve``.  Field names match the contract in
    ``src/mve_generator.py:generate_mve`` and ``src/risk_scorer.py:score_alert``.
    """
    return {
        "device_type": device_class,
        "criticality": str(device_criticality).upper(),
        "patchable": bool(patchable),
        # The fields below have no per-row source in the EHMS schema;
        # leave them empty so the generator picks safe defaults.
        "clinical_function": "",
        "location": "",
    }


def _build_event_context() -> dict:
    """No per-row maintenance / vendor-IP flags in the EHMS test
    parquet, so the event_context is always benign-by-default for this
    batch.  Tracks that need maintenance-window MVEs synthesise them
    via the truth-table script instead."""
    return {
        "is_maintenance_window": False,
        "is_known_vendor_ip": False,
        "similar_events_past_30d": 0,
        "baseline_days": 90,
    }


def _generate_one(
    raw_dict: dict, device_context: dict, event_context: dict
):
    """Call generate_mve, falling back to the rule-based path if the
    LLM path raises or returns None.  Mirrors the resilience pattern in
    ``module6_evaluation.py:_process_and_append``."""
    try:
        mve = generate_mve(
            raw_dict, device_context, {}, None,
            event_context=event_context,
        )
        if mve is None:
            mve = _generate_rule_based(
                raw_dict, device_context, {}, None, "T1"
            )
    except Exception as exc:  # noqa: BLE001 — generator must not abort batch
        logger.warning(
            "generate_mve raised %s — falling back to rule-based",
            type(exc).__name__,
        )
        mve = _generate_rule_based(raw_dict, device_context, {}, None, "T1")
    return mve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-benign", action="store_true",
        help="Generate MVEs for benign-classified rows too (default: skip).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of MVEs generated (smoke-test use).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sep = "=" * 72
    t0 = time.perf_counter()
    logger.info(sep)
    logger.info("MODULE 5 BATCH — MVE OUTPUTS (RQ2 prerequisite)")
    logger.info(sep)

    _assert_npz_schema_v1_1()

    # Load v1.1 npz with allow_pickle=True because fusion_class and
    # data_quality are stored as object arrays; the schema-v1.1 required
    # arrays are fixed-width unicode and load cleanly under either mode.
    data = np.load(NPZ_PATH, allow_pickle=True)
    test_df = pd.read_parquet(PARQUET_PATH)

    n_rows = len(data["y_true"])
    if len(test_df) != n_rows:
        raise RuntimeError(
            f"row count mismatch: npz={n_rows} parquet={len(test_df)}"
        )

    fusion_class = np.asarray(data["fusion_class"]).astype(str)
    surfaced_mask = fusion_class != "BENIGN" if not args.include_benign \
        else np.ones(n_rows, dtype=bool)

    # row_id IS the parquet identity range (asserted in
    # tests/test_step9_composite_risk.py::test_risk_scores_npz_schema_v1_1).
    row_ids = data["row_id"]
    device_class_arr = np.asarray(data["device_class"]).astype(str)
    device_criticality_arr = np.asarray(data["device_criticality"]).astype(str)
    patchable_arr = np.asarray(data["patchable"]).astype(bool)
    risk_tier_arr = np.asarray(data["risk_levels"]).astype(str)
    R_arr = np.asarray(data["R"]).astype(float)
    y_true_arr = np.asarray(data["y_true"]).astype(int)
    attack_cat_arr = np.asarray(data["attack_category"]).astype(str)
    true_severity_arr = np.asarray(data["true_severity"]).astype(str)
    c_detect_arr = np.asarray(data["c_detect"]).astype(float)

    targets = np.flatnonzero(surfaced_mask)
    if args.limit is not None:
        targets = targets[: args.limit]
        logger.info("  --limit %d applied", args.limit)

    n_total = int(len(targets))
    n_surfaced_full = int(surfaced_mask.sum())
    logger.info(
        "  Surfaced alerts in test split: %d / %d (writing %d)",
        n_surfaced_full, n_rows, n_total,
    )

    mode_counts: dict[str, int] = {}
    fallback_count = 0

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for i in targets:
            row = test_df.iloc[int(i)]
            raw_dict = row.to_dict()

            device_context = _build_device_context(
                device_class_arr[i],
                device_criticality_arr[i],
                patchable_arr[i],
            )
            event_context = _build_event_context()

            mve = _generate_one(raw_dict, device_context, event_context)
            mode = getattr(mve, "mode_used", "B_rule")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            if mode == "B_rule":
                fallback_count += 1

            alert_id = f"ALERT-{int(row_ids[i]):05d}"
            record = mve.to_dict(alert_id=alert_id)
            # Enrich with the npz-side context the RQ2 tracks need.
            record.update({
                "row_id": int(row_ids[i]),
                "fusion_class": str(fusion_class[i]),
                "risk_tier": str(risk_tier_arr[i]),
                "risk_score_R": float(R_arr[i]),
                "c_detect": float(c_detect_arr[i]),
                "device_class": str(device_class_arr[i]),
                "device_criticality": str(device_criticality_arr[i]),
                "patchable": bool(patchable_arr[i]),
                "attack_category": str(attack_cat_arr[i]),
                "true_severity": str(true_severity_arr[i]),
                "y_true": int(y_true_arr[i]),
                "mode_used": mode,
                "llm_provider": getattr(mve, "llm_provider", None),
                "llm_model_version": getattr(mve, "llm_model_version", None),
            })
            fh.write(json.dumps(record, default=str))
            fh.write("\n")

    elapsed = round(time.perf_counter() - t0, 1)

    # Sidecar meta so RQ2 tracks can fingerprint the corpus they consumed.
    meta = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "module5_responses/module5_mve_batch.py",
        "inputs": {
            "risk_scores_npz": str(NPZ_PATH.relative_to(PROJECT_ROOT)),
            "risk_scores_sha256": _sha256_file(NPZ_PATH),
            "test_parquet": str(PARQUET_PATH.relative_to(PROJECT_ROOT)),
            "test_parquet_sha256": _sha256_file(PARQUET_PATH),
        },
        "selection": {
            "rule": ("all rows" if args.include_benign
                     else "fusion_class != 'BENIGN' (surfaced alerts)"),
            "limit": args.limit,
            "n_rows_in_split": int(n_rows),
            "n_surfaced_in_split": n_surfaced_full,
            "n_written": n_total,
        },
        "mode_counts": mode_counts,
        "fallback_to_rule_based_count": int(fallback_count),
        "wall_time_seconds": elapsed,
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("")
    logger.info(sep)
    logger.info("MVE BATCH COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    logger.info("  Wrote: %s (%d records)",
                OUT_JSONL.relative_to(PROJECT_ROOT), n_total)
    logger.info("  Wrote: %s", OUT_META.relative_to(PROJECT_ROOT))
    logger.info("  Mode counts: %s", mode_counts)
    if fallback_count:
        logger.info("  Rule-based (fallback or no-LLM): %d", fallback_count)
    logger.info(sep)


if __name__ == "__main__":
    main()
