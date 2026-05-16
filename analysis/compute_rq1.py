"""RQ1: Detection performance + sensitivity analysis.

Sub-tasks:
  RQ1.1 — Test-split detection metrics + ROC/PR figures
  RQ1.2 — Track B per-class breakdown from existing ablation YAMLs
  RQ1.3 — Composite-risk weight sensitivity (30 perturbations + 3 baselines)
  RQ1.4 — Tier×Surfacing truth table from `src.risk_scorer.score_alert()`
  RQ1.5 — D_crit vs D_clinical_tier correlation across device inventory
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analysis._common import (
    RANDOM_SEED,
    RESULTS_DIR,
    build_provenance,
    file_hashes,
    log,
    section_begin,
    section_end,
    write_json,
)

CONFIGS = REPO / "configs"
REPORTS = RESULTS_DIR / "reports"
FIGURES = RESULTS_DIR / "figures"


# --------------------------------------------------------------------------
# RQ1.1 — Baseline detection metrics on TEST split
# --------------------------------------------------------------------------
def compute_rq1_1() -> dict[str, Any]:
    section = "RQ1.1"
    start = section_begin(section, "test-split baseline metrics")

    # Track A — XGBoost calibrated test probas (matches frozen test split)
    test_npz_path = REPO / "results" / "models" / "xgboost_test_predictions.npz"
    if not test_npz_path.exists():
        log(section, f"MISSING {test_npz_path} — cannot compute")
        return {"status": "FAILED", "reason": f"missing {test_npz_path.name}"}
    data = np.load(test_npz_path)
    y_true = np.asarray(data["y_true"]).astype(int)
    y_proba = np.asarray(data["y_proba"]).astype(float)
    y_pred = (y_proba >= _xgb_optimal_threshold()).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0))
    f1 = float(f1_score(y_true, y_pred))
    auc_score = float(roc_auc_score(y_true, y_proba))

    # FNR over CRITICAL true label proxy: in EHMS test split the
    # "CRITICAL" label is not separately persisted, so we use the
    # binary attack rate as the closest available proxy (matches
    # ARCHITECTURE.md Step [14] R_CRIT definition where attack rows
    # already trigger CRITICAL when device_criticality high).
    fnr_critical = fn / (fn + tp) if (fn + tp) else 0.0

    # ROC + PR curves
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = float(auc(rec, prec))

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(fpr, tpr, label=f"Track A (XGBoost-like GB) AUC={auc_score:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC curve — test split (RQ1.1)")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(FIGURES / "roc_curves.pdf")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(rec, prec, label=f"PR AUC={pr_auc:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall curve — test split (RQ1.1)")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIGURES / "pr_curves.pdf")
    plt.close(fig2)

    # Confusion matrix figure
    fig3, ax3 = plt.subplots(figsize=(4, 3.5))
    im = ax3.imshow(cm, cmap="Blues")
    ax3.set_xticks([0, 1], ["Pred Benign", "Pred Attack"])
    ax3.set_yticks([0, 1], ["True Benign", "True Attack"])
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax3.set_title("Confusion matrix — test split")
    fig3.colorbar(im, ax=ax3, shrink=0.7)
    fig3.tight_layout()
    fig3.savefig(FIGURES / "confusion_matrix.pdf")
    plt.close(fig3)

    # Hard assertions
    assertions = {
        "fnr_critical_in_range": 0.0 <= fnr_critical < 0.50,
        "sensitivity_above_0_30": sensitivity > 0.30,
        "specificity_above_0_30": specificity > 0.30,
        "auc_above_0_50": auc_score > 0.50,
        "metrics_in_probability_range": all(
            0.0 <= v <= 1.0 for v in (sensitivity, specificity, fnr_critical, auc_score, f2)
        ),
    }
    if not all(assertions.values()):
        # Output failure file but do not stop the whole script (per task spec)
        write_json(
            RESULTS_DIR / "rq1_FAILED.json",
            {
                "provenance": build_provenance(input_files=file_hashes()),
                "failed_assertions": [k for k, v in assertions.items() if not v],
                "metrics": {
                    "fnr_critical": fnr_critical,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "auc": auc_score,
                    "f2_score": f2,
                },
            },
        )
        section_end(section, start, "assertions FAILED, see rq1_FAILED.json")

    payload = {
        "provenance": build_provenance(input_files=file_hashes()),
        "results": {
            "fnr_critical": round(fnr_critical, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "f2_score": round(f2, 4),
            "auc": round(auc_score, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
            "threshold_used": _xgb_optimal_threshold(),
            "n_test_samples": int(len(y_true)),
            "n_attacks_test": int(int(np.sum(y_true == 1))),
            "n_benign_test": int(int(np.sum(y_true == 0))),
            "assertions": assertions,
        },
    }
    out_path = RESULTS_DIR / "rq1_metrics.json"

    # Compare to existing
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            prev_auc = prev.get("results", {}).get("auc") or prev.get("auc")
            if prev_auc is not None:
                delta = abs(float(prev_auc) - auc_score)
                if delta > 0.01:
                    log(section, f"WARN: existing AUC={prev_auc} vs new={auc_score} delta={delta:.4f}")
                else:
                    log(section, f"baseline match (delta={delta:.4f})")
        except Exception:
            log(section, "prior rq1_metrics.json unparseable — overwriting")

    write_json(out_path, payload)
    log(section, f"OUTPUT: {out_path.name}")
    section_end(section, start, f"AUC={auc_score:.4f} F2={f2:.4f} sens={sensitivity:.4f} spec={specificity:.4f}")
    return payload


def _xgb_optimal_threshold() -> float:
    """Read the F2-tuned threshold from xgboost_final_report.json."""
    p = REPO / "results" / "models" / "xgboost_final_report.json"
    try:
        rep = json.loads(p.read_text())
        return float(rep.get("optimal_threshold", 0.5))
    except Exception:
        return 0.5


# --------------------------------------------------------------------------
# RQ1.2 — Track B per-class breakdown
# --------------------------------------------------------------------------
def compute_rq1_2() -> dict[str, Any]:
    section = "RQ1.2"
    start = section_begin(section, "Track B per-class AUC")

    out: dict[str, Any] = {}
    files = [
        ("ehms", REPORTS / "dae_ablation_loo.yaml"),
        ("medsec25", REPORTS / "dae_ablation_loo_medsec25.yaml"),
    ]
    for key, fpath in files:
        if not fpath.exists():
            out[key] = {"status": "pending", "reason": f"missing {fpath.name}"}
            log(section, f"MISSING {fpath.name}")
            continue
        with open(fpath) as f:
            doc = yaml.safe_load(f)
        # Schema: top-level "results" is a list of {holdout_class, config_results: [...]}
        # Each config_results entry has {config, auc_benign_vs_novel, ...}
        per_class_by_config: dict[str, dict[str, float]] = {
            "DAE-raw": {},
            "DAE-cascade": {},
            "DAE-probas-only": {},
        }
        for r in doc.get("results", []) or []:
            cls = r.get("holdout_class")
            for cfg in r.get("config_results", []) or []:
                cfg_name = cfg.get("config")
                auc_val = cfg.get("auc_benign_vs_novel")
                if cfg_name in per_class_by_config and auc_val is not None:
                    per_class_by_config[cfg_name][str(cls)] = round(float(auc_val), 4)
        # Primary view: DAE-raw (the production configuration per ARCHITECTURE.md)
        primary = per_class_by_config.get("DAE-raw", {})
        out[key] = primary if primary else {"status": "pending", "reason": "could not parse per-class AUC"}
        out[f"{key}_by_config"] = per_class_by_config

    payload = {
        "provenance": build_provenance(input_files={
            "ehms_ablation": "sha256:" + _sha(files[0][1]) if files[0][1].exists() else "MISSING",
            "medsec25_ablation": "sha256:" + _sha(files[1][1]) if files[1][1].exists() else "MISSING",
        }),
        "results": out,
    }
    out_path = RESULTS_DIR / "rq1_track_b_per_class.json"
    write_json(out_path, payload)
    log(section, f"OUTPUT: {out_path.name}")
    section_end(section, start, f"keys: {list(out.keys())}")
    return payload


def _sha(p: Path) -> str:
    from analysis._common import sha256_file
    return sha256_file(p)


# --------------------------------------------------------------------------
# RQ1.3 — Composite-risk weight sensitivity analysis
# --------------------------------------------------------------------------
def compute_rq1_3() -> dict[str, Any]:
    section = "RQ1.3"
    start = section_begin(section, "sensitivity analysis (30 perturbations + 3 baselines)")

    # Baseline weights
    base_w = np.array([0.40, 0.25, 0.15, 0.20])
    boundaries = (0.80, 0.60, 0.40)  # critical_min, high_min, medium_min

    # Load risk components from M3 risk_scores.npz (test-split sourced)
    rs_path = REPORTS / "risk_scores.npz"
    if not rs_path.exists():
        log(section, f"MISSING {rs_path.name}")
        section_end(section, start, "pending")
        payload = {
            "provenance": build_provenance(input_files=file_hashes()),
            "results": {"status": "pending", "reason": "risk_scores.npz missing"},
        }
        write_json(RESULTS_DIR / "rq1_sensitivity_analysis.json", payload)
        return payload

    data = np.load(rs_path, allow_pickle=True)
    keys = set(data.files)
    # Required components
    needed = {"c_detect", "d_crit", "s_data", "d_clinical_tier"}
    if not needed.issubset(keys):
        # Try alternative names (M3 may use "d_clin" or similar)
        alt_map = {"d_clinical_tier": ["d_clinical_tier", "d_clin"]}
        comp = {}
        for k in ("c_detect", "d_crit", "s_data"):
            if k in keys:
                comp[k] = np.asarray(data[k], dtype=float)
            else:
                missing = k
                log(section, f"MISSING component {missing}")
                payload = {
                    "provenance": build_provenance(input_files=file_hashes()),
                    "results": {"status": "pending", "reason": f"missing component {missing}"},
                }
                write_json(RESULTS_DIR / "rq1_sensitivity_analysis.json", payload)
                section_end(section, start, "pending")
                return payload
        for tgt, alts in alt_map.items():
            found = None
            for cand in alts:
                if cand in keys:
                    found = cand
                    break
            if found is None:
                log(section, f"MISSING component {tgt}")
                payload = {
                    "provenance": build_provenance(input_files=file_hashes()),
                    "results": {"status": "pending", "reason": f"missing component {tgt}"},
                }
                write_json(RESULTS_DIR / "rq1_sensitivity_analysis.json", payload)
                section_end(section, start, "pending")
                return payload
            comp[tgt] = np.asarray(data[found], dtype=float)
    else:
        comp = {k: np.asarray(data[k], dtype=float) for k in needed}

    X = np.column_stack([comp["c_detect"], comp["d_crit"], comp["s_data"], comp["d_clinical_tier"]])
    # baseline tier assignment
    R_base = X @ base_w
    tier_base = _assign_tier(R_base, boundaries)

    # 30 perturbations
    rng = np.random.default_rng(RANDOM_SEED)
    n_pert = 30
    agreements: list[float] = []
    for _ in range(n_pert):
        # multiplicative ±10% perturbation
        pert = base_w * (1.0 + rng.uniform(-0.10, 0.10, size=4))
        pert = pert / pert.sum()  # renormalize to 1.0
        R = X @ pert
        tier = _assign_tier(R, boundaries)
        agreements.append(float(np.mean(tier == tier_base)))

    agreements_arr = np.asarray(agreements)
    histogram_counts, histogram_edges = np.histogram(agreements_arr, bins=10, range=(0.0, 1.0))

    # Baselines
    def baseline_eval(w: np.ndarray, multiplicative: bool = False) -> dict[str, float]:
        if multiplicative:
            R = comp["c_detect"] * np.maximum.reduce(
                [comp["d_crit"], comp["s_data"], comp["d_clinical_tier"]]
            )
        else:
            R = X @ w
        tier = _assign_tier(R, boundaries)
        agreement = float(np.mean(tier == tier_base))
        # FNR on CRITICAL: rows that should be CRITICAL but assigned lower
        crit_base = (tier_base == 3)
        if int(np.sum(crit_base)) > 0:
            fnr_delta = float(np.mean(crit_base & (tier < 3)))
        else:
            fnr_delta = 0.0
        return {"agreement": round(agreement, 4), "fnr_critical_delta": round(fnr_delta, 4)}

    baselines = {
        "equal_weights": baseline_eval(np.array([0.25, 0.25, 0.25, 0.25])),
        "c_detect_only": baseline_eval(np.array([1.0, 0.0, 0.0, 0.0])),
        "multiplicative": baseline_eval(np.array([1.0, 0.0, 0.0, 0.0]), multiplicative=True),
    }

    # Sensitivity histogram figure
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(agreements_arr, bins=10, range=(0.0, 1.0), edgecolor="black", alpha=0.75)
    ax.axvline(float(np.mean(agreements_arr)), color="red", linestyle="--",
               label=f"mean = {np.mean(agreements_arr):.3f}")
    ax.set_xlabel("Tier agreement with baseline weights")
    ax.set_ylabel("Number of perturbations")
    ax.set_title(f"RQ1.3 — Sensitivity ({n_pert} perturbations, ±10%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "sensitivity_histogram.pdf")
    plt.close(fig)

    results = {
        "perturbation_results": {
            "n_perturbations": n_pert,
            "agreement_mean": round(float(np.mean(agreements_arr)), 4),
            "agreement_std": round(float(np.std(agreements_arr)), 4),
            "agreement_min": round(float(np.min(agreements_arr)), 4),
            "agreement_max": round(float(np.max(agreements_arr)), 4),
            "agreement_p25": round(float(np.percentile(agreements_arr, 25)), 4),
            "agreement_p50": round(float(np.percentile(agreements_arr, 50)), 4),
            "agreement_p75": round(float(np.percentile(agreements_arr, 75)), 4),
            "histogram_counts": histogram_counts.tolist(),
            "histogram_edges": histogram_edges.tolist(),
        },
        "baselines": baselines,
        "baseline_weights": {
            "detection_confidence": float(base_w[0]),
            "device_criticality": float(base_w[1]),
            "data_sensitivity": float(base_w[2]),
            "clinical_tier": float(base_w[3]),
        },
        "tier_boundaries": {
            "critical_min": boundaries[0],
            "high_min": boundaries[1],
            "medium_min": boundaries[2],
        },
        "perturbation_method": "multiplicative ±10% then L1 renormalize to sum=1.0",
        "n_alerts_evaluated": int(X.shape[0]),
    }
    payload = {
        "provenance": build_provenance(input_files=file_hashes()),
        "results": results,
    }
    write_json(RESULTS_DIR / "rq1_sensitivity_analysis.json", payload)
    log(section, f"OUTPUT: rq1_sensitivity_analysis.json")
    section_end(section, start, f"mean_agreement={results['perturbation_results']['agreement_mean']:.3f}")
    return payload


def _assign_tier(R: np.ndarray, boundaries: tuple[float, float, float]) -> np.ndarray:
    """0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL."""
    crit, hi, med = boundaries
    out = np.zeros_like(R, dtype=int)
    out[R >= med] = 1
    out[R >= hi] = 2
    out[R >= crit] = 3
    return out


# --------------------------------------------------------------------------
# RQ1.4 — Tier × Surfacing truth table
# --------------------------------------------------------------------------
def compute_rq1_4() -> dict[str, Any]:
    section = "RQ1.4"
    start = section_begin(section, "truth table from src.risk_scorer.score_alert()")

    # Probe the actual surfacing logic by calling determine_surfacing or
    # equivalent across all (tier, patchable, maintenance_active) combos.
    from src.risk_scorer import score_alert as _maybe_score_alert  # noqa: F401
    # We construct minimal inputs that exercise the surfacing decision branch
    # by directly invoking the function. The full ScoredAlert may need
    # additional fields — we'll build them.

    try:
        from src.risk_scorer import score_alert  # type: ignore
    except Exception as exc:
        log(section, f"IMPORT ERROR: {exc}")
        section_end(section, start, "IMPORT ERROR")
        return {"status": "FAILED", "reason": f"import: {exc}"}

    rows: list[dict[str, Any]] = []
    tiers = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    tier_to_score = {"CRITICAL": 0.85, "HIGH": 0.65, "MEDIUM": 0.45, "LOW": 0.20}
    crit_to_dcrit = {"CRITICAL": "HIGH", "HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW"}

    for tier in tiers:
        for patchable in (True, False):
            for maint in (True, False):
                try:
                    args = {
                        "c_detect": tier_to_score[tier],
                        "d_crit_label": crit_to_dcrit[tier],
                        "patchable": patchable,
                        "maintenance_active": maint,
                        "device_class": "patient_monitor",
                        "fusion_class": "BENIGN",
                        "data_quality": "OK",
                    }
                    # The actual signature varies — we attempt a few common forms.
                    result = _invoke_score_alert(score_alert, args, tier, patchable, maint)
                    rows.append(result)
                except Exception as exc:
                    rows.append({
                        "risk_tier": tier,
                        "patchable": patchable,
                        "maintenance_active": maint,
                        "should_surface": None,
                        "error": str(exc),
                    })

    # Compare to documented behavior in ARCHITECTURE.md / risk_adaptive_thresholds.yaml
    docs_expected = _truth_table_documented()

    discrepancies: list[dict[str, Any]] = []
    for r in rows:
        if r.get("error"):
            continue
        key = (r["risk_tier"], r["patchable"], r["maintenance_active"])
        doc_expected = docs_expected.get(key)
        if doc_expected is None:
            continue
        if r["should_surface"] != doc_expected:
            discrepancies.append({
                "key": list(key),
                "code_result": r["should_surface"],
                "doc_expected": doc_expected,
            })

    if discrepancies:
        log(section, f"FLAG: {len(discrepancies)} truth-table discrepancies between code and docs")

    # Output Markdown
    md_lines = [
        "# RQ1.4 — Tier × Patchable × Maintenance Truth Table",
        "",
        "Derived from `src.risk_scorer.score_alert()` (live function call).",
        "",
        "| Risk Tier | Patchable | Maintenance Active | should_surface (code) | should_surface (doc) | Match |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        key = (r["risk_tier"], r["patchable"], r["maintenance_active"])
        doc_expected = docs_expected.get(key)
        match = "—" if (r.get("error") or doc_expected is None) else (
            "✓" if r["should_surface"] == doc_expected else "✗ DISCREPANCY"
        )
        code_val = r.get("should_surface")
        code_val_str = str(code_val) if not r.get("error") else f"ERROR: {r['error'][:50]}"
        md_lines.append(
            f"| {r['risk_tier']} | {r['patchable']} | {r['maintenance_active']} | "
            f"{code_val_str} | {doc_expected} | {match} |"
        )
    md = "\n".join(md_lines) + "\n"
    (RESULTS_DIR / "rq1_truth_table.md").write_text(md)
    log(section, "OUTPUT: rq1_truth_table.md")

    # YAML
    yaml_doc = {
        "provenance": build_provenance(input_files=file_hashes()),
        "rows": rows,
        "discrepancies": discrepancies,
    }
    (RESULTS_DIR / "rq1_truth_table.yaml").write_text(yaml.safe_dump(yaml_doc, sort_keys=False))
    log(section, "OUTPUT: rq1_truth_table.yaml")

    section_end(section, start, f"rows={len(rows)} discrepancies={len(discrepancies)}")
    return {"results": {"rows": rows, "discrepancies": discrepancies}}


def _invoke_score_alert(score_alert_fn, args, tier, patchable, maint) -> dict[str, Any]:
    """Invoke ``src.risk_scorer.score_alert`` with the dict-shape interface."""
    device_context = {
        "criticality": args["d_crit_label"],
        "patchable": args["patchable"],
        "device_class": args["device_class"],
        "clinical_function": "test",
    }
    event_context = {
        "is_maintenance_window": bool(args["maintenance_active"]),
        "is_known_vendor_ip": False,
        "similar_events_past_30d": 0,
    }
    result = score_alert_fn(
        anomaly_score=args["c_detect"],
        device_context=device_context,
        event_context=event_context,
        fusion_class=args["fusion_class"],
        data_quality=args["data_quality"],
    )
    if hasattr(result, "should_surface"):
        ss = bool(result.should_surface)
    elif isinstance(result, dict):
        ss = bool(result.get("should_surface"))
    else:
        ss = None
    return {
        "risk_tier": tier,
        "patchable": patchable,
        "maintenance_active": maint,
        "should_surface": ss,
    }


def _truth_table_documented() -> dict[tuple[str, bool, bool], bool]:
    """ARCHITECTURE.md / risk_adaptive_thresholds.yaml documented behavior.

    Per INVARIANT 2 (safety floor): CRITICAL+unpatchable always surfaces.
    LOW alerts never surface above threshold under maintenance.
    The full mapping is defined in risk_adaptive_thresholds.yaml multipliers
    plus the safety floor. We expose the policy:
      - CRITICAL+unpatchable → always surface (safety floor)
      - CRITICAL+patchable + maintenance → may suppress
      - HIGH/MEDIUM/LOW + maintenance → suppress per multiplier
    For documentation we make a coarse expectation:
    """
    return {
        ("CRITICAL", False, True): True,    # safety floor
        ("CRITICAL", False, False): True,
        ("CRITICAL", True, False): True,
        ("CRITICAL", True, True): True,     # high score → still surfaces
        ("HIGH", False, False): True,
        ("HIGH", True, False): True,
        ("HIGH", False, True): True,
        ("HIGH", True, True): True,
        ("MEDIUM", False, False): True,
        ("MEDIUM", True, False): True,
        ("MEDIUM", False, True): False,
        ("MEDIUM", True, True): False,
        ("LOW", False, False): False,
        ("LOW", True, False): False,
        ("LOW", False, True): False,
        ("LOW", True, True): False,
    }


# --------------------------------------------------------------------------
# RQ1.5 — D_crit vs D_clinical_tier correlation
# --------------------------------------------------------------------------
def compute_rq1_5() -> dict[str, Any]:
    section = "RQ1.5"
    start = section_begin(section, "D_crit vs D_clinical_tier Pearson correlation")

    inv_path = CONFIGS / "device_inventory.yaml"
    clin_path = CONFIGS / "device_clinical_tier_mapping.yaml"
    if not inv_path.exists() or not clin_path.exists():
        payload = {
            "provenance": build_provenance(input_files=file_hashes()),
            "results": {"status": "pending", "reason": f"missing config(s): inv={inv_path.exists()} clin={clin_path.exists()}"},
        }
        write_json(RESULTS_DIR / "rq1_dcrit_dclinical_correlation.json", payload)
        section_end(section, start, "pending — missing configs")
        return payload

    inv = yaml.safe_load(inv_path.read_text())
    clin = yaml.safe_load(clin_path.read_text())

    # Label→numeric encodings
    crit_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    clin_map = {
        "tier_5_administrative": 1,
        "tier_4_supportive": 2,
        "tier_4_low_clinical": 2,
        "tier_3_moderate": 3,
        "tier_3_moderate_clinical": 3,
        "tier_2_high_clinical": 4,
        "tier_1_life_critical": 5,
        "tier_1_critical": 5,
    }

    # device_inventory.yaml shape: {"devices": [ {device_type, criticality, patchable, ...}, ... ]}
    inv_list = inv.get("devices", []) if isinstance(inv, dict) else []
    # device_clinical_tier_mapping.yaml shape: {"mappings": {<device_class>: {tier, weight, rationale}}}
    clin_mappings = (
        clin.get("mappings") if isinstance(clin, dict) else None
    ) or {}

    # Map device_type free-text → canonical device_class via substring rules
    def to_device_class(device_type: str) -> str | None:
        s = device_type.lower()
        rules = [
            ("infusion pump", "infusion_pump"),
            ("ventilator", "ventilator"),
            ("patient monitor", "patient_monitor"),
            ("pulse oximeter", "patient_monitor"),
            ("blood pressure", "patient_monitor"),
            ("ekg", "ekg_machine"),
            ("ecg", "ekg_machine"),
            ("ehr", "ehr_workstation"),
            ("admin workstation", "admin_workstation"),
            ("bedside terminal", "bedside_terminal"),
        ]
        for sub, cls in rules:
            if sub in s:
                return cls
        return None

    crit_vals: list[float] = []
    clin_vals: list[float] = []
    scatter: list[dict[str, Any]] = []
    devices_used: list[str] = []
    skipped: list[dict[str, str]] = []
    for d in inv_list:
        if not isinstance(d, dict):
            continue
        dtype = str(d.get("device_type", ""))
        crit_label = d.get("criticality")
        device_class = to_device_class(dtype)
        if device_class is None:
            skipped.append({"device_type": dtype, "reason": "no canonical device_class match"})
            continue
        clin_entry = clin_mappings.get(device_class) if isinstance(clin_mappings, dict) else None
        clin_label = (clin_entry or {}).get("tier") if isinstance(clin_entry, dict) else None
        if crit_label is None or clin_label is None:
            skipped.append({"device_type": dtype, "reason": "missing crit or tier"})
            continue
        if str(crit_label).upper() not in crit_map or str(clin_label) not in clin_map:
            skipped.append({"device_type": dtype, "reason": f"unmapped labels {crit_label}/{clin_label}"})
            continue
        crit_vals.append(crit_map[str(crit_label).upper()])
        clin_vals.append(clin_map[str(clin_label)])
        scatter.append({
            "device_type": dtype,
            "device_class": device_class,
            "criticality": str(crit_label).upper(),
            "clinical_tier": str(clin_label),
            "d_crit": crit_vals[-1],
            "d_clinical_tier": clin_vals[-1],
        })
        devices_used.append(dtype)

    if len(crit_vals) < 3:
        payload = {
            "provenance": build_provenance(input_files=file_hashes()),
            "results": {
                "status": "pending",
                "reason": f"only {len(crit_vals)} aligned devices — need >=3",
                "matched_devices": devices_used,
                "skipped": skipped,
            },
        }
        write_json(RESULTS_DIR / "rq1_dcrit_dclinical_correlation.json", payload)
        section_end(section, start, f"only {len(crit_vals)} devices aligned")
        return payload

    from scipy.stats import pearsonr
    r, p = pearsonr(crit_vals, clin_vals)
    interp = (
        "high correlation (>0.7) indicates double-counting of device importance"
        if abs(r) > 0.7 else
        ("moderate correlation (0.4–0.7) indicates partial overlap"
         if abs(r) > 0.4 else
         "low correlation — components capture distinct concepts")
    )
    payload = {
        "provenance": build_provenance(input_files=file_hashes()),
        "results": {
            "pearson_r": round(float(r), 4),
            "p_value": float(p),
            "interpretation": interp,
            "device_count": len(crit_vals),
            "matched_devices": devices_used,
            "scatter_data": scatter,
            "encoding": {"d_crit": crit_map, "d_clinical_tier": clin_map},
        },
    }
    write_json(RESULTS_DIR / "rq1_dcrit_dclinical_correlation.json", payload)
    log(section, "OUTPUT: rq1_dcrit_dclinical_correlation.json")
    section_end(section, start, f"r={r:.3f} n={len(crit_vals)}")
    return payload


def main() -> None:
    compute_rq1_1()
    compute_rq1_2()
    compute_rq1_3()
    compute_rq1_4()
    compute_rq1_5()


if __name__ == "__main__":
    main()
