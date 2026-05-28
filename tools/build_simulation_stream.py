#!/usr/bin/env python3
"""Build per-split simulation_stream_<split>.json for the M6 dashboard's
"Full stream" Online Simulation mode.

Combines:
  - data/processed/<split>_phase1.parquet  → raw feature rows in arrival order
  - results/reports/<split>_scores.npz     → per-sample tier + R + components
  - results/reports/alert_responses_<split>.json → full alert payload (LOW+ only)

Output: a list of 1632 entries (demo) / 2448 (test) in arrival order. Each
entry carries the minimum fields the dashboard needs to render either an
alert card (LOW+) or a NORMAL placeholder; LOW+ entries embed the full M5
alert record under ``alert``, NORMAL entries leave ``alert`` as None.

The artifact is *simulation-only* — it does not replace alert_responses or
audit_trail (those keep their FDA-style LOW+ contract), it just gives the
dashboard a single stream the operator can tick through end-to-end.

Usage:
    python -m tools.build_simulation_stream --split demo
    python -m tools.build_simulation_stream --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS = PROJECT_ROOT / "results" / "reports"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# Anchored arrival start — matches module6_evaluation/module6_app.py:
# load_live_stream_source so the two are observationally identical.
_STREAM_START = datetime(2026, 4, 9, 8, 0, 0)


def _paths(split: str) -> dict:
    if split not in ("test", "demo"):
        raise ValueError(f"unknown split: {split!r} (expected test|demo)")
    suffix = "" if split == "test" else "_demo"
    scores_name = "risk_scores.npz" if split == "test" else "demo_scores.npz"
    return {
        "parquet":   PROCESSED / f"{split}_phase1.parquet",
        "scores":    REPORTS / scores_name,
        "responses": REPORTS / f"alert_responses{suffix}.json",
        "out":       REPORTS / f"simulation_stream{suffix}.json",
    }


def build_stream(split: str) -> dict:
    """Build the simulation_stream payload for one split.

    Returns the dict that gets written to disk (``_meta`` + ``stream``).
    The function is a pure transform — no side effects until the caller
    writes the result. That keeps the smoke test in
    tests/test_build_simulation_stream.py free of fs writes.
    """
    p = _paths(split)
    if not p["parquet"].exists():
        raise FileNotFoundError(f"missing parquet: {p['parquet']}")
    if not p["scores"].exists():
        raise FileNotFoundError(f"missing risk scores npz: {p['scores']}")
    # alert_responses missing is recoverable: the stream still has N entries,
    # just with alert=None for every row. Useful when running before M5.
    responses_exist = p["responses"].exists()

    df = pd.read_parquet(p["parquet"]).reset_index(drop=True)
    n_total = len(df)
    scores = np.load(p["scores"], allow_pickle=True)
    levels = scores["risk_levels"]
    R = scores["R"].astype(float)
    y_true = scores["y_true"].astype(int)
    c_detect = scores["c_detect"].astype(float)
    d_crit = scores["d_crit"].astype(float)
    s_data = scores["s_data"].astype(float)
    d_clinical_tier = scores["d_clinical_tier"].astype(float)

    if len(levels) != n_total:
        raise ValueError(
            f"scores/parquet length mismatch: scores={len(levels)} vs "
            f"parquet={n_total}. Re-run Module 3 for split={split!r}."
        )

    # Build alert lookup by sample_index for O(1) join.
    alerts_by_idx: dict[int, dict] = {}
    if responses_exist:
        with open(p["responses"]) as f:
            payload = json.load(f)
        records = payload.get("records") if isinstance(payload, dict) else payload
        for r in (records or []):
            idx = r.get("sample_index")
            if isinstance(idx, (int, np.integer)) and 0 <= int(idx) < n_total:
                alerts_by_idx[int(idx)] = r

    attack_cats = (
        df["Attack Category"].astype(str).fillna("normal").tolist()
        if "Attack Category" in df.columns
        else ["normal"] * n_total
    )

    stream: list[dict] = []
    n_surfaced = 0
    n_normal = 0
    for idx in range(n_total):
        arrived_at = (_STREAM_START + timedelta(seconds=idx)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        tier = str(levels[idx])
        gt = "attack" if int(y_true[idx]) == 1 else "benign"
        cat = attack_cats[idx]
        cat = cat if cat and cat.lower() not in ("nan", "none", "") else "normal"

        if tier == "NORMAL":
            n_normal += 1
            entry = {
                "sample_index": idx,
                "arrived_at": arrived_at,
                "risk_level": "NORMAL",
                "risk_score": round(float(R[idx]), 4),
                "ground_truth": gt,
                "attack_category": cat,
                "risk_components": {
                    "c_detect":        round(float(c_detect[idx]), 4),
                    "d_crit":          round(float(d_crit[idx]), 4),
                    "s_data":          round(float(s_data[idx]), 4),
                    "d_clinical_tier": round(float(d_clinical_tier[idx]), 4),
                },
                "alert": None,
            }
        else:
            n_surfaced += 1
            alert = alerts_by_idx.get(idx)
            entry = {
                "sample_index": idx,
                "arrived_at": arrived_at,
                "risk_level": tier,
                "risk_score": round(float(R[idx]), 4),
                "ground_truth": gt,
                "attack_category": cat,
                "risk_components": {
                    "c_detect":        round(float(c_detect[idx]), 4),
                    "d_crit":          round(float(d_crit[idx]), 4),
                    "s_data":          round(float(s_data[idx]), 4),
                    "d_clinical_tier": round(float(d_clinical_tier[idx]), 4),
                },
                "alert": alert,  # full M5 payload, or None if M5 hasn't run
            }
        stream.append(entry)

    return {
        "_meta": {
            "split": split,
            "n_total": n_total,
            "n_surfaced": n_surfaced,
            "n_normal": n_normal,
            "stream_start": _STREAM_START.isoformat(),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "responses_joined": responses_exist,
        },
        "stream": stream,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build simulation_stream_<split>.json for the M6 "
                    "dashboard Full-stream mode.",
    )
    parser.add_argument(
        "--split", choices=("test", "demo"), required=True,
        help="Frozen split to build (test=paper-clean, demo=operator-clean).",
    )
    args = parser.parse_args()

    payload = build_stream(args.split)
    out = _paths(args.split)["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out)
    meta = payload["_meta"]
    print(
        f"wrote {out.relative_to(PROJECT_ROOT)}  "
        f"(n_total={meta['n_total']} surfaced={meta['n_surfaced']} "
        f"normal={meta['n_normal']} responses_joined={meta['responses_joined']})"
    )


if __name__ == "__main__":
    main()
