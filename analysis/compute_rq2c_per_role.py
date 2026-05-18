"""RQ2.c per-role × per-metric analysis (LLM-persona variant).

Mann-Whitney U + Cliff's delta for condition A vs B, overall and per role,
aggregated at the persona level (one composite value per persona, then a
between-group test). Adds the methodology_notes + limitations blocks that
make multiple-comparison handling and the LLM-persona caveat explicit.

Inputs:
  survey/study_responses_*.json   per-persona LLM responses
  survey/rq2c_exclusions.json     personas excluded by audit_study_data.py

Output:
  analysis/outputs/rq2c_per_role.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[1]
SURVEY_DIR = REPO_ROOT / "survey"
EXCL_PATH = SURVEY_DIR / "rq2c_exclusions.json"
OUT_DIR = REPO_ROOT / "analysis" / "outputs"
OUT_PATH = OUT_DIR / "rq2c_per_role.json"

ROLES = ["biomed_engineer", "IT_generalist", "nurse_manager"]
METRICS = ["accuracy", "confidence"]  # decision_time absent in LLM data
N_WARNING_THRESHOLD = 10
NEGLIGIBLE_DELTA = 0.147
SMALL_DELTA = 0.33
MEDIUM_DELTA = 0.474


def _role_from_pid(pid: str) -> str:
    parts = pid.split("_")
    if parts and re.match(r"^P\d+$", parts[-1]):
        parts = parts[:-1]
    return "_".join(parts) if parts else pid


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size == 0 or b.size == 0:
        return None
    diff = a[:, None] - b[None, :]
    greater = int((diff > 0).sum())
    less = int((diff < 0).sum())
    return float((greater - less) / (a.size * b.size))


def delta_magnitude(delta: float | None) -> str:
    if delta is None:
        return "undefined"
    d = abs(delta)
    if d < NEGLIGIBLE_DELTA:
        return "negligible"
    if d < SMALL_DELTA:
        return "small"
    if d < MEDIUM_DELTA:
        return "medium"
    return "large"


def _direction(delta: float | None, higher_is_better: bool = True) -> str:
    """Interpret a Cliff's delta for the spec's direction string.

    Cliff's delta here is computed as (count(a>b) - count(b>a)) / (n_a * n_b),
    so positive δ means group A tends to score higher.
    """
    if delta is None:
        return "undefined"
    if abs(delta) < NEGLIGIBLE_DELTA:
        return "no meaningful difference"
    a_higher = delta > 0
    if higher_is_better:
        return "A higher than B" if a_higher else "B higher than A"
    # lower-is-better not used now (no decision_time) — kept for parity
    return "A lower than B" if not a_higher else "B lower than A"


def compute_cell(a_vals: list[float], b_vals: list[float],
                 metric_name: str) -> dict:
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    n_a, n_b = int(a.size), int(b.size)
    cell: dict = {
        "n_A": n_a,
        "n_B": n_b,
        "median_A": float(np.median(a)) if n_a else None,
        "median_B": float(np.median(b)) if n_b else None,
        "mean_A": float(np.mean(a)) if n_a else None,
        "mean_B": float(np.mean(b)) if n_b else None,
        "n_warning": n_a < N_WARNING_THRESHOLD or n_b < N_WARNING_THRESHOLD,
    }
    if n_a < 2 or n_b < 2:
        cell.update({
            "mannwhitney_u": None, "p_value": None,
            "cliffs_delta": None, "magnitude": "undefined",
            "direction": "insufficient_data",
        })
        return cell
    try:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        delta = cliffs_delta(a, b)
        cell.update({
            "mannwhitney_u": float(u),
            "p_value": float(p),
            "cliffs_delta": round(delta, 4) if delta is not None else None,
            "magnitude": delta_magnitude(delta),
            "direction": _direction(delta, higher_is_better=True),
        })
    except ValueError as exc:
        cell.update({
            "mannwhitney_u": None, "p_value": None,
            "cliffs_delta": None, "magnitude": "undefined",
            "direction": f"test_failed: {exc}",
        })
    return cell


def _load_excluded_ids() -> set[str]:
    if not EXCL_PATH.exists():
        return set()
    doc = json.loads(EXCL_PATH.read_text())
    return {str(e.get("persona_id")) for e in doc.get("exclusions", [])}


def _persona_metrics(record: dict) -> tuple[float | None, float | None, str | None]:
    """Return (mean_accuracy, mean_confidence, condition) for one persona."""
    rows = record.get("rows", [])
    accs: list[int] = []
    confs: list[float] = []
    cond: str | None = None
    for r in rows:
        if r.get("error") is not None:
            continue
        resp = r.get("response")
        if not isinstance(resp, dict):
            continue
        if cond is None:
            cond = r.get("condition")
        acc = int(resp.get("action") == r.get("correct_action"))
        accs.append(acc)
        c = resp.get("confidence")
        if isinstance(c, (int, float)):
            confs.append(float(c))
    if not accs:
        return None, None, cond
    return float(np.mean(accs)), float(np.mean(confs)) if confs else None, cond


def main() -> None:
    excluded = _load_excluded_ids()

    # by_role[role][condition] = list of {metric: value}
    by_role: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    overall: dict[str, list[dict]] = defaultdict(list)

    files = sorted(SURVEY_DIR.glob("study_responses_*.json"))
    for path in files:
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        pid = rec.get("persona_id", path.stem)
        if str(pid) in excluded:
            continue
        role = _role_from_pid(pid)
        acc, conf, cond = _persona_metrics(rec)
        if cond not in ("A", "B") or acc is None:
            continue
        entry = {"accuracy": acc, "confidence": conf}
        by_role[role][cond].append(entry)
        overall[cond].append(entry)

    def extract(records: list[dict], metric: str) -> list[float]:
        return [r[metric] for r in records if r.get(metric) is not None]

    overall_block: dict = {
        "_scope": "All included personas",
        "n_A": len(overall["A"]),
        "n_B": len(overall["B"]),
    }
    for m in METRICS:
        overall_block[m] = compute_cell(
            extract(overall["A"], m), extract(overall["B"], m), m)

    per_role: dict = {}
    cell_sizes: list[int] = []
    cells_with_warning = 0
    for role in ROLES:
        if role not in by_role:
            per_role[role] = {"_status": "no personas in this role"}
            continue
        a_recs = by_role[role]["A"]
        b_recs = by_role[role]["B"]
        cell_sizes.extend([len(a_recs), len(b_recs)])
        entry: dict = {
            "n_A": len(a_recs),
            "n_B": len(b_recs),
            "n_warning": len(a_recs) < N_WARNING_THRESHOLD
                         or len(b_recs) < N_WARNING_THRESHOLD,
        }
        for m in METRICS:
            cell = compute_cell(extract(a_recs, m), extract(b_recs, m), m)
            if cell.get("n_warning"):
                cells_with_warning += 1
            entry[m] = cell
        per_role[role] = entry

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/compute_rq2c_per_role.py",
            "inputs": {
                "n_participant_files": len(files),
                "n_included": (sum(len(by_role[r]["A"]) + len(by_role[r]["B"])
                                   for r in by_role)),
                "exclusion_audit": "survey/rq2c_exclusions.json",
            },
        },
        "methodology_notes": [
            "Data source: LLM-persona simulation (gpt-4o-mini, 100 personas × "
            "20 alerts). Not a human study; complements but does not replace "
            "human user study results.",
            "Aggregation: one composite value per persona (mean accuracy, mean "
            "confidence over successful rows); Mann-Whitney U on persona-level "
            "values to satisfy independence.",
            "Raw p-values reported; NO multiple-comparisons correction applied "
            f"across the {len(ROLES) * len(METRICS)} role × metric cells.",
            "Mann-Whitney U is two-sided.",
            f"Cliff's delta thresholds: negligible<{NEGLIGIBLE_DELTA}, "
            f"small<{SMALL_DELTA}, medium<{MEDIUM_DELTA}, else large.",
            f"Cells with n<{N_WARNING_THRESHOLD} per group flagged with "
            "n_warning=true.",
            "Accuracy uses pre-recorded correct_action per AlertScenario "
            "(strict match). reasonable_alternatives field exists on "
            "AlertScenario but is currently empty.",
            "Exclusion criteria: EX-3 schema invalid OR zero successful rows. "
            "EX-1 (attention check) and EX-2 (duration) do not apply to LLM "
            "personas.",
        ],
        "limitations": [
            f"Multiple comparisons across {len(ROLES) * len(METRICS)} role × "
            f"metric cells inflates Type I error rate. With α=0.05 and no "
            f"correction, ~{0.05 * len(ROLES) * len(METRICS):.2f} false "
            "positives expected under the null. Findings are exploratory.",
            "Participants are LLM personas, not humans. Behavioral fidelity "
            "to real clinicians/IT staff is not established. Generalization "
            "to human operators requires the parallel human user study.",
            "Decision-time metric absent in LLM responses; the 9-cell × "
            "3-metric table in the spec collapses to a 6-cell × 2-metric "
            "table here.",
            "Self-selected role assignment is N/A for LLM personas; role is "
            "set at persona-generation time. Cell-size imbalance reflects "
            "the persona-generation budget, not recruitment bias.",
            "Some persona × alert calls failed with OpenAI 429 rate-limit "
            "errors; statistical analysis operates on successful rows only. "
            "Failures treated as MCAR per the rate-limit error class.",
        ],
        "overall": overall_block,
        "per_role": per_role,
        "cell_diagnostics": {
            "_description": "Per-cell sample sizes across role × condition cells",
            "min_n_per_cell": min(cell_sizes) if cell_sizes else 0,
            "max_n_per_cell": max(cell_sizes) if cell_sizes else 0,
            "cells_with_warning": cells_with_warning,
            "warning_threshold": N_WARNING_THRESHOLD,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Overall: n_A={overall_block['n_A']} n_B={overall_block['n_B']}")
    for role in ROLES:
        r = per_role.get(role, {})
        warn = " (LOW-N WARNING)" if r.get("n_warning") else ""
        print(f"  {role}: n_A={r.get('n_A', 0)} n_B={r.get('n_B', 0)}{warn}")


if __name__ == "__main__":
    main()
