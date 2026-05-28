#!/usr/bin/env python3
"""Phase 4.3 — Deterministic simulator for the user-study round-2 comparison.

The upgrade plan's Phase 4.3 calls for re-running ``rq2_user_study_analysis``
with 5-10 participants per role on the upgraded artifacts. That requires
live human raters which aren't available in this environment. Instead,
this tool produces a deterministic proxy: it scores each clinician
summary on the four user-study Likert dimensions (comprehensibility,
trust, usefulness, actionability) using verifiable on-page signals:

  - presence of a counterfactual clause     → +trust + +actionability
  - presence of an observation_phrase       → +comprehensibility
  - presence of a stability badge           → +trust (knowing uncertainty)
  - presence of a playbook checklist        → +actionability
  - presence of an extension + SLA          → +actionability
  - presence of a MITRE plain_gloss         → +comprehensibility
  - presence of a routing_warning mismatch  → +trust (system honesty)

Each signal carries a small additive contribution (0..5 scale clamp).
The baseline scores are anchored at the pre-Phase-1 Likert values
observed in the existing ``participant_responses.json`` (round-1
study) so the comparison is apples-to-apples.

Output: ``results/rq2_round2_simulation.json`` with per-role mean
scores before and after, plus the delta and a marker on whether the
delta exceeds the ≥0.3 Likert-point acceptance bar from the upgrade
plan.

This is a *simulation* — it scores the artefact, not the human
reaction to it. Treat the deltas as upper-bound estimates of what a
real round-2 study would report. The deterministic scoring is
reproducible and shows the room each upgrade unlocked.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "results" / "reports"
OUTPUT  = PROJECT_ROOT / "results" / "rq2_round2_simulation.json"


# ── Anchors from round-1 study ─────────────────────────────────────


# Pulled from ``results/reports/participant_responses.json`` — the
# pre-upgrade Likert means per role × condition (with_xai). These are
# the baseline against which the round-2 simulation reports delta.
ROUND1_BASELINES = {
    "analyst": {
        "comprehensibility": 3.84, "trust": 4.22,
        "usefulness": 4.44, "actionability": 4.18,
    },
    "clinician": {
        "comprehensibility": 4.14, "trust": 4.56,
        "usefulness": 4.62, "actionability": 4.56,
    },
    "administrator": {
        "comprehensibility": 4.10, "trust": 4.26,
        "usefulness": 4.42, "actionability": 4.20,
    },
}

# Acceptance bar from the upgrade plan: comprehensibility must improve
# ≥0.3 Likert points for at least clinician + admin in the round-2 study.
ACCEPTANCE_DELTA = 0.30


# ── Per-record signal extraction ──────────────────────────────────


_RE_OBSERVATION = re.compile(r"observed [+-]?\d+\.?\d*", re.IGNORECASE)
_RE_COUNTERFACTUAL = re.compile(r"would clear if", re.IGNORECASE)
_RE_STABILITY_BADGE = re.compile(r"Explanation: (STABLE|BORDERLINE|UNSTABLE)")
_RE_PLAYBOOK = re.compile(r"\*\*Playbook:", re.IGNORECASE)
_RE_EXT_SLA = re.compile(r"ext \d{3,5}, SLA", re.IGNORECASE)
_RE_MITRE_GLOSS = re.compile(r"MITRE T\d{4}.*?—\s*\w", re.IGNORECASE)


def _score_clinician_summary(summary: str, record: dict) -> dict:
    """Return contributions to each Likert axis from the artifact signals.

    Each axis is anchored at the round-1 mean (different per role); the
    returned dict carries per-axis *additive* deltas.

    Signals are scanned across the clinician summary (Phase 1.1/2/3/4
    enrichments live here) AND the MVE Layer 1/3 (Phase 1.2/1.4
    enrichments live there). This mirrors what an operator actually
    sees when opening the alert from the dashboard.
    """
    mve = (record.get("explanation") or {}).get("mve") or {}
    layer1 = (mve.get("layer_1") or {}).get("deviation_description", "")
    layer3 = (
        (mve.get("layer_3") or {}).get("immediate_action", "")
        + " "
        + (mve.get("layer_3") or {}).get("escalation_path", "")
    )
    full_text = f"{summary}\n{layer1}\n{layer3}"

    obs        = bool(_RE_OBSERVATION.search(full_text))
    cf         = bool(_RE_COUNTERFACTUAL.search(full_text))
    stab_badge = bool(_RE_STABILITY_BADGE.search(full_text))
    playbook   = bool(_RE_PLAYBOOK.search(full_text))
    ext_sla    = bool(_RE_EXT_SLA.search(full_text))
    mitre      = bool(_RE_MITRE_GLOSS.search(full_text))
    routing_mismatch = bool((record.get("response") or {})
                             .get("routing_warning", {}).get("mismatch"))

    return {
        "comprehensibility": 0.18 * obs + 0.16 * mitre + 0.10 * playbook,
        "trust":             0.20 * cf  + 0.18 * stab_badge + 0.10 * routing_mismatch,
        "usefulness":        0.15 * cf  + 0.12 * playbook  + 0.08 * obs,
        "actionability":     0.22 * cf  + 0.18 * playbook  + 0.15 * ext_sla,
        # Diagnostic flags for the report.
        "_signals": {
            "observation":       obs,
            "counterfactual":    cf,
            "stability_badge":   stab_badge,
            "playbook":          playbook,
            "ext_sla":           ext_sla,
            "mitre_gloss":       mitre,
            "routing_mismatch":  routing_mismatch,
        },
    }


def _aggregate_per_role(records: list[dict], clinician_summaries: list[dict]) -> dict:
    """Build per-role round-2 means from artifact signals.

    For the deterministic proxy each role consumes the *same* per-record
    signal vector but weights different axes:
      - clinician sees the badge + playbook in plain language → gains
        most on comprehensibility + actionability
      - analyst sees the raw stability + counterfactual numbers in the
        analyst report → gains on trust + comprehensibility
      - admin sees the aggregated playbook coverage → gains on
        actionability
    """
    clinician_by_idx = {c["sample_index"]: c for c in clinician_summaries}

    per_axis: dict[str, dict[str, list[float]]] = {
        role: {axis: [] for axis in ROUND1_BASELINES[role]}
        for role in ROUND1_BASELINES
    }
    signal_totals = {k: 0 for k in (
        "observation", "counterfactual", "stability_badge", "playbook",
        "ext_sla", "mitre_gloss", "routing_mismatch",
    )}

    for rec in records:
        idx = rec["sample_index"]
        summary = clinician_by_idx.get(idx, {}).get("summary", "")
        d = _score_clinician_summary(summary, rec)
        for sig, present in d["_signals"].items():
            if present:
                signal_totals[sig] += 1

        # Clinician — uses the per-record deltas directly
        for axis in per_axis["clinician"]:
            per_axis["clinician"][axis].append(d[axis])

        # Analyst — sees raw stability/counterfactual in their report.
        # We weight more on trust (a faithfulness-aware operator) and
        # comprehensibility (they can read the SHAP values themselves).
        for axis, weight in (
            ("comprehensibility", 1.10),
            ("trust",             1.20),
            ("usefulness",        1.00),
            ("actionability",     0.85),
        ):
            per_axis["analyst"][axis].append(d[axis] * weight)

        # Admin — sees only the aggregate playbook + escalation; gains
        # most on actionability when the action set is concrete.
        for axis, weight in (
            ("comprehensibility", 0.85),
            ("trust",             0.90),
            ("usefulness",        1.05),
            ("actionability",     1.15),
        ):
            per_axis["administrator"][axis].append(d[axis] * weight)

    summary: dict = {}
    for role, axes in per_axis.items():
        anchor = ROUND1_BASELINES[role]
        role_block: dict = {}
        for axis, deltas in axes.items():
            mean_delta = statistics.fmean(deltas) if deltas else 0.0
            r1 = anchor[axis]
            r2 = min(5.0, r1 + mean_delta)
            role_block[axis] = {
                "round1": r1,
                "round2": round(r2, 3),
                "delta":  round(r2 - r1, 3),
                "meets_acceptance": (r2 - r1) >= ACCEPTANCE_DELTA,
            }
        summary[role] = role_block

    return {
        "per_role":      summary,
        "signal_totals": signal_totals,
        "n_records":     len(records),
    }


# ── Driver ─────────────────────────────────────────────────────────


def main() -> int:
    alert_path = REPORTS / "alert_responses.json"
    clin_path  = REPORTS / "clinician_summaries.json"
    if not alert_path.exists() or not clin_path.exists():
        print("ERROR: missing alert_responses.json or clinician_summaries.json "
              "— run the regen tools first.", file=sys.stderr)
        return 2

    alerts = json.loads(alert_path.read_text())
    records = alerts.get("records", alerts) if isinstance(alerts, dict) else alerts
    clinician = json.loads(clin_path.read_text())

    result = _aggregate_per_role(records, clinician)
    result["_meta"] = {
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "acceptance_delta": ACCEPTANCE_DELTA,
        "scope_note":       (
            "DETERMINISTIC SIMULATION — deltas are upper-bound estimates "
            "based on artifact-level signal presence, not human ratings. "
            "Replace with a live round-2 study before quoting in the thesis."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2))

    print()
    print("=" * 72)
    print("ROUND-2 SIMULATION (per role, deterministic)")
    print("=" * 72)
    print(f"  n_records = {result['n_records']}")
    print()
    for role, axes in result["per_role"].items():
        meets_compr = axes["comprehensibility"]["meets_acceptance"]
        meets_act   = axes["actionability"]["meets_acceptance"]
        flag = "✓" if (meets_compr and meets_act) else "—"
        print(f"  [{flag}] {role:<14s} compr {axes['comprehensibility']['round1']:.2f} "
              f"→ {axes['comprehensibility']['round2']:.2f} "
              f"({axes['comprehensibility']['delta']:+.2f})  "
              f"action {axes['actionability']['round1']:.2f} "
              f"→ {axes['actionability']['round2']:.2f} "
              f"({axes['actionability']['delta']:+.2f})")
    print()
    print(f"  Signals present across {result['n_records']} records:")
    for sig, n in result["signal_totals"].items():
        pct = 100 * n / result["n_records"] if result["n_records"] else 0
        print(f"    {sig:<20s} {n:>5d}  ({pct:.1f}%)")
    print("=" * 72)
    print(f"  wrote {OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
