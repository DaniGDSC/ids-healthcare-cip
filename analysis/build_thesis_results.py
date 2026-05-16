"""Generate ``results/THESIS_RESULTS.md`` from per-RQ JSON outputs."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analysis._common import RESULTS_DIR, git_commit, now_iso

INDENT = "  "


def load(name: str) -> dict | None:
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    try:
        if p.suffix == ".yaml":
            import yaml
            return yaml.safe_load(p.read_text())
        return json.loads(p.read_text())
    except Exception:
        return None


def section_status(payload: dict | None) -> str:
    if payload is None:
        return "missing"
    r = payload.get("results", {})
    if r.get("status") == "pending":
        return "pending"
    return "complete"


def fmt(x, digits=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def main() -> None:
    rq1_metrics = load("rq1_metrics.json")
    rq1_tb = load("rq1_track_b_per_class.json")
    rq1_sens = load("rq1_sensitivity_analysis.json")
    rq1_truth = load("rq1_truth_table.yaml")  # YAML not JSON
    rq1_corr = load("rq1_dcrit_dclinical_correlation.json")
    rq2_shap = load("rq2_shap_stability.json")
    rq2_align = load("rq2_mve_shap_alignment.json")
    rq2_mitre = load("rq2_mitre_coverage.json")
    rq2_us = load("rq2_user_study.json")
    rq3_tests = load("rq3_test_summary.json")
    rq3_grep = load("rq3_no_auto_execution.json")
    rq3_audit = load("rq3_audit_integrity.json")
    rq3_role = load("rq3_cross_role_consistency.json")
    rq3_us = load("rq3_user_study.json")

    # Build markdown
    lines: list[str] = []
    lines.append("# Thesis Results — Computed Outputs")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Code commit:** `{git_commit()}`")
    lines.append(f"**Random seed:** 42 (applied throughout)")
    lines.append("")
    lines.append("This document summarises the quantitative evidence produced by "
                 "`analysis/compute_rq*.py` for the three research questions.")
    lines.append("Every per-RQ JSON includes a `provenance` block (timestamp, "
                 "git commit, input file SHA-256s, schema version).")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    rq1_done = all(section_status(p) == "complete" for p in [rq1_metrics, rq1_tb, rq1_sens, rq1_corr])
    rq2_done = all(section_status(p) == "complete" for p in [rq2_shap, rq2_align, rq2_mitre, rq2_us])
    rq3_done = all(section_status(p) == "complete" for p in [rq3_tests, rq3_grep, rq3_audit, rq3_role, rq3_us])
    lines.append("| RQ | Status | Notes |")
    lines.append("|---|---|---|")
    lines.append(f"| RQ1 (Detection + Sensitivity) | {'OK' if rq1_done else 'partial'} | 5 subsections; baseline metrics, ablation extract, sensitivity analysis, truth table, correlation |")
    lines.append(f"| RQ2 (MVE Faithfulness + Study) | {'OK' if rq2_done else 'partial'} | 4 subsections; SHAP stability, MVE alignment, MITRE coverage, LLM-persona study |")
    lines.append(f"| RQ3 (Safety + HITL) | {'OK' if rq3_done else 'partial'} | 5 subsections; pytest summary, no-auto-execution, audit chain, role consistency, HITL study |")
    lines.append("")

    # ────────────────────────────────────────────────────────────────────
    # RQ1
    # ────────────────────────────────────────────────────────────────────
    lines.append("## RQ1 — Detection + Sensitivity Analysis")
    lines.append("")

    if rq1_metrics:
        r = rq1_metrics.get("results", {})
        lines.append("### RQ1.1 — Baseline detection metrics (test split)")
        lines.append("")
        lines.append(f"- Source: `xgboost_test_predictions.npz` (n={r.get('n_test_samples')} samples, "
                     f"{r.get('n_attacks_test')} attacks / {r.get('n_benign_test')} benign)")
        lines.append(f"- Threshold (F2-tuned): {fmt(r.get('threshold_used'))}")
        lines.append(f"- Sensitivity: **{fmt(r.get('sensitivity'))}**")
        lines.append(f"- Specificity: **{fmt(r.get('specificity'))}**")
        lines.append(f"- F2 score: **{fmt(r.get('f2_score'))}**")
        lines.append(f"- F1 score: {fmt(r.get('f1_score'))}")
        lines.append(f"- AUC: **{fmt(r.get('auc'))}**")
        lines.append(f"- PR-AUC: {fmt(r.get('pr_auc'))}")
        lines.append(f"- FNR_CRITICAL (proxy = FN/(FN+TP)): **{fmt(r.get('fnr_critical'))}**")
        cm = r.get("confusion_matrix", {})
        lines.append(f"- Confusion matrix: TP={cm.get('TP')} FN={cm.get('FN')} FP={cm.get('FP')} TN={cm.get('TN')}")
        assertions_ok = all((r.get("assertions") or {}).values())
        lines.append(f"- Hard assertions: {'PASSED' if assertions_ok else 'FAILED'}")
        lines.append(f"- Figures: `results/figures/roc_curves.pdf`, `pr_curves.pdf`, `confusion_matrix.pdf`")
        lines.append("")

    if rq1_tb:
        lines.append("### RQ1.2 — Track B per-class AUC")
        lines.append("")
        r = rq1_tb.get("results", {})
        for k in ("ehms", "medsec25"):
            v = r.get(k) or {}
            if isinstance(v, dict) and v.get("status") == "pending":
                lines.append(f"- {k.upper()}: pending — {v.get('reason')}")
            else:
                pairs = ", ".join(f"`{cls}`={fmt(auc, 3)}" for cls, auc in (v or {}).items())
                lines.append(f"- {k.upper()}: {pairs}")
        lines.append("")

    if rq1_sens:
        r = rq1_sens.get("results", {})
        pp = r.get("perturbation_results", {})
        bb = r.get("baselines", {})
        lines.append("### RQ1.3 — Composite-risk weight sensitivity")
        lines.append("")
        lines.append(f"- 30 perturbations (±10% per weight, renormalised to sum=1):")
        lines.append(f"  - mean agreement: **{fmt(pp.get('agreement_mean'))}**")
        lines.append(f"  - std / min / max: {fmt(pp.get('agreement_std'))} / {fmt(pp.get('agreement_min'))} / {fmt(pp.get('agreement_max'))}")
        lines.append(f"  - IQR p25–p75: [{fmt(pp.get('agreement_p25'))}, {fmt(pp.get('agreement_p75'))}]")
        lines.append("- Baselines (vs ARCHITECTURE.md weights 0.40/0.25/0.15/0.20):")
        for name in ("equal_weights", "c_detect_only", "multiplicative"):
            b = bb.get(name) or {}
            lines.append(f"  - `{name}`: agreement={fmt(b.get('agreement'))}, FNR_CRITICAL Δ={fmt(b.get('fnr_critical_delta'))}")
        lines.append(f"- N alerts evaluated: {r.get('n_alerts_evaluated')}")
        lines.append(f"- Figure: `results/figures/sensitivity_histogram.pdf`")
        lines.append("")

    if rq1_truth:
        rows = rq1_truth.get("rows", []) or []
        discreps = rq1_truth.get("discrepancies", []) or []
        lines.append("### RQ1.4 — Tier × Patchable × Maintenance truth table")
        lines.append("")
        lines.append(f"- 16 (tier × patchable × maintenance) combinations derived from `src.risk_scorer.score_alert()`")
        lines.append(f"- Discrepancies between code and documented expected behaviour: **{len(discreps)}**")
        if discreps:
            for d in discreps:
                lines.append(f"  - {d.get('key')}: code={d.get('code_result')} vs doc={d.get('doc_expected')}")
        lines.append(f"- Full table: `results/rq1_truth_table.md` (and `.yaml`)")
        lines.append("")

    if rq1_corr:
        r = rq1_corr.get("results", {})
        if r.get("status") == "pending":
            lines.append("### RQ1.5 — D_crit vs D_clinical_tier correlation")
            lines.append("")
            lines.append(f"- Status: pending — {r.get('reason')}")
        else:
            lines.append("### RQ1.5 — D_crit vs D_clinical_tier correlation")
            lines.append("")
            lines.append(f"- Pearson r = **{fmt(r.get('pearson_r'))}** (p = {fmt(r.get('p_value'))})")
            lines.append(f"- N devices: {r.get('device_count')}")
            lines.append(f"- Interpretation: {r.get('interpretation')}")
        lines.append("")

    # ────────────────────────────────────────────────────────────────────
    # RQ2
    # ────────────────────────────────────────────────────────────────────
    lines.append("## RQ2 — MVE Faithfulness + User Study")
    lines.append("")

    if rq2_shap:
        r = rq2_shap.get("results", {})
        if r.get("status") == "pending":
            lines.append(f"### RQ2.1 — SHAP stability")
            lines.append(f"- pending — {r.get('reason')}")
        else:
            agg = r.get("aggregate", {})
            lines.append("### RQ2.1 — SHAP stability")
            lines.append("")
            lines.append(f"- Method: TreeSHAP on signed Track A pipeline; 10 perturbations × U(0.99, 1.01) multiplicative noise; Jaccard top-3.")
            lines.append(f"- N alerts: {agg.get('n_alerts')}")
            lines.append(f"- Mean stability: **{fmt(agg.get('mean_stability'))}**")
            lines.append(f"- Median stability: {fmt(agg.get('median_stability'))}")
            lines.append(f"- Fraction stable (≥0.90): **{fmt(agg.get('pct_stable'))}**")
            byc = agg.get("by_fusion_class", {})
            for cls, info in byc.items():
                lines.append(f"  - `{cls}`: mean={fmt(info.get('mean'))}, n={info.get('n')}, pct_stable={fmt(info.get('pct_stable'))}")
            if r.get("computation_sampled"):
                lines.append(f"- Sampling: {r.get('sample_size')} stratified random sample (budget exceeded)")
            lines.append(f"- Figure: `results/figures/shap_stability_distribution.pdf`")
        lines.append("")

    if rq2_align:
        r = rq2_align.get("results", {})
        lines.append("### RQ2.2 — MVE-SHAP alignment (stratified)")
        lines.append("")
        lines.append("- Top-3 XGBoost SHAP features matched against Layer-1 narrative (full feature name, narrative phrase, or token of length ≥4 from configured feature_categories).")
        byc = r.get("by_fusion_class", {})
        for cls, info in byc.items():
            lines.append(f"- `{cls}` (n={info.get('n_alerts')}): all-3={fmt(info.get('all_3_present'))}, "
                         f"2+={fmt(info.get('two_plus_present'))}, "
                         f"any={fmt(info.get('any_present'))}, "
                         f"MITRE ref'd={fmt(info.get('mitre_referenced'))}, "
                         f"xgb_low_conf SHAP src={info.get('shap_source_xgb_low_conf')}")
            if info.get("interpretation"):
                lines.append(f"    - _{info['interpretation']}_")
        agg = r.get("aggregate_with_caveats", {})
        lines.append(f"- Aggregate (caveat: see by_fusion_class): all-3={fmt(agg.get('overall_all_3'))} over n={agg.get('n_alerts_total')}")
        lines.append(f"- _{agg.get('note')}_")
        lines.append("")

    if rq2_mitre:
        r = rq2_mitre.get("results", {})
        cov = r.get("config_coverage", {})
        l1 = r.get("layer1_grounding", {})
        lines.append("### RQ2.3 — MITRE ATT&CK coverage")
        lines.append("")
        lines.append(f"- Attack categories defined: {cov.get('total_attack_categories')}")
        lines.append(f"- Categories with ≥1 technique: {cov.get('mapped_categories')}")
        lines.append(f"- Orphan categories: {cov.get('orphan_categories')}")
        lines.append(f"- MITRE framework version: `{cov.get('mitre_framework_version')}`")
        lines.append(f"- Techniques by confidence: {cov.get('techniques_by_confidence')}")
        lines.append(f"- Layer-1 MITRE technique-ID grounding: {l1.get('n_alerts_with_mitre')}/{l1.get('n_alerts_total')} alerts ({fmt(l1.get('alerts_referencing_mitre'))})")
        lines.append(f"  - _{l1.get('note')}_")
        lines.append("")

    if rq2_us:
        r = rq2_us.get("results", {})
        if r.get("status") == "pending":
            lines.append(f"### RQ2.4 — User study faithfulness")
            lines.append(f"- pending — {r.get('reason')}")
        else:
            lines.append("### RQ2.4 — User study (LLM-persona) faithfulness analysis")
            lines.append("")
            lines.append(f"- Stat test: {r.get('stat_test')}")
            lines.append(f"- N survey files: {r.get('n_survey_files')}")
            by_role = r.get("by_role", {})
            for role, info in by_role.items():
                lines.append(f"- **{role}** (n_responses={info.get('n_responses')})")
                for metric, m in info.get("per_metric", {}).items():
                    if "p_value_raw" in m:
                        lines.append(f"  - `{metric}`: A median={fmt(m['A'].get('median'))} vs "
                                     f"B median={fmt(m['B'].get('median'))} → "
                                     f"U={fmt(m.get('statistic_U'))}, p={fmt(m.get('p_value_raw'))}, "
                                     f"p_holm={fmt(m.get('p_value_holm_bonferroni'))}, "
                                     f"Cliff δ={fmt(m.get('cliffs_delta'))}")
            lines.append(f"- _{r.get('note')}_")
        lines.append("")

    # ────────────────────────────────────────────────────────────────────
    # RQ3
    # ────────────────────────────────────────────────────────────────────
    lines.append("## RQ3 — Architectural Safety + HITL")
    lines.append("")

    if rq3_tests:
        r = rq3_tests.get("results", {})
        lines.append("### RQ3.1 — Test suite summary")
        lines.append("")
        lines.append(f"- Pytest result: **{r.get('overall_status')}** "
                     f"(passed={r.get('passed')}, failed={r.get('failed')}, skipped={r.get('skipped')})")
        lines.append(f"- Test files: {len(r.get('test_files', []))}")
        lines.append(f"- Pytest exit code: {r.get('pytest_exit_code')}")
        lines.append(f"- Raw log: `results/rq3_pytest_raw.log`")
        # Top files
        ff = sorted(r.get("test_files", []), key=lambda x: -x.get("test_count", 0))[:5]
        for f in ff:
            lines.append(f"  - `{f['file']}` — {f.get('test_count')} tests ({f.get('status')})")
        lines.append("")

    if rq3_grep:
        r = rq3_grep.get("results", {})
        lines.append("### RQ3.2 — No-auto-execution verification")
        lines.append("")
        lines.append(f"- Grep check (subprocess/os.system/iptables/netcat/etc.): **{r.get('grep_check', {}).get('status')}** (matches={r.get('grep_check', {}).get('n_matches')})")
        lines.append(f"- Import check (`import subprocess`): **{r.get('import_check', {}).get('status')}** (matches={r.get('import_check', {}).get('n_matches')})")
        lines.append(f"- Verdict: **{r.get('overall_verdict')}**")
        lines.append("")

    if rq3_audit:
        r = rq3_audit.get("results", {})
        lines.append("### RQ3.3 — Audit log hash chain")
        lines.append("")
        lines.append(f"- Total logs checked: {r.get('total_logs')}")
        lines.append(f"- Total entries scanned: {r.get('total_entries')}")
        lines.append(f"- All hash chains intact: **{r.get('all_chains_intact')}**")
        for lf in r.get("logs_checked", []):
            extra = ""
            if lf.get("n_chain_restarts") is not None:
                extra = f" (chain_restarts={lf['n_chain_restarts']})"
            lines.append(f"  - `{lf.get('file')}`: n_entries={lf.get('n_entries')}, status={lf.get('verification_status')}{extra}")
        lines.append("")

    if rq3_role:
        r = rq3_role.get("results", {})
        if r.get("status") == "pending":
            lines.append(f"### RQ3.4 — Cross-role consistency: pending — {r.get('reason')}")
        else:
            lines.append("### RQ3.4 — Cross-role consistency")
            lines.append("")
            lines.append(f"- Alerts checked: {r.get('alerts_checked')}, anchors_present={r.get('anchors_present')}")
            lines.append(f"- Invariant 9 (shared anchor): all_identical={r.get('invariant_9_shared_anchor', {}).get('all_identical')} "
                         f"(n_violations={r.get('invariant_9_shared_anchor', {}).get('n_violations')})")
            lines.append(f"- Invariant 6 (severity consistency): all_identical={r.get('invariant_6_severity', {}).get('all_identical')} "
                         f"(n_violations={r.get('invariant_6_severity', {}).get('n_violations')})")
            lines.append(f"- Invariant 6 (action authorization, nurse_manager most-restrictive): all_authorized={r.get('invariant_6_action_authorization', {}).get('all_authorized')} "
                         f"(n_violations={r.get('invariant_6_action_authorization', {}).get('n_violations')})")
            lines.append(f"- Overall: **{r.get('overall_status')}**")
            note = r.get("invariant_6_action_authorization", {}).get("note")
            if note:
                lines.append(f"  - _{note}_")
        lines.append("")

    if rq3_us:
        r = rq3_us.get("results", {})
        if r.get("status") == "pending":
            lines.append(f"### RQ3.5 — HITL user study: pending — {r.get('reason')}")
        else:
            lines.append("### RQ3.5 — HITL user study (LLM-persona simulation)")
            lines.append("")
            lines.append(f"- N survey files: {r.get('n_files_loaded')}")
            for role, info in r.get("per_role", {}).items():
                lines.append(f"- **{role}**: {info.get('n_responses')} responses, action distribution: {info.get('action_distribution')}")
                for cond, c in info.get("by_condition", {}).items():
                    lines.append(f"  - Condition {cond}: n={c.get('n')}, accuracy={fmt(c.get('accuracy'))}, mean_confidence={fmt(c.get('mean_confidence'))}, escalation_rate={fmt(c.get('escalation_rate'))}")
            chi = r.get("escalation_chi_square_A_vs_B", {})
            if chi:
                lines.append("- χ² escalation A vs B:")
                for role, c in chi.items():
                    if "p_value" in c:
                        lines.append(f"  - `{role}`: χ²={c.get('chi2')}, p={c.get('p_value')}")
        lines.append("")

    # ────────────────────────────────────────────────────────────────────
    # Pending + Failed
    # ────────────────────────────────────────────────────────────────────
    lines.append("## Pending Items")
    lines.append("")
    pending: list[str] = []
    failed: list[str] = []
    for name, payload in [
        ("RQ1.1", rq1_metrics), ("RQ1.2", rq1_tb), ("RQ1.3", rq1_sens),
        ("RQ1.4", rq1_truth), ("RQ1.5", rq1_corr),
        ("RQ2.1", rq2_shap), ("RQ2.2", rq2_align), ("RQ2.3", rq2_mitre), ("RQ2.4", rq2_us),
        ("RQ3.1", rq3_tests), ("RQ3.2", rq3_grep), ("RQ3.3", rq3_audit),
        ("RQ3.4", rq3_role), ("RQ3.5", rq3_us),
    ]:
        if payload is None:
            failed.append(f"- {name}: output file not generated")
        else:
            r = payload.get("results", {}) if isinstance(payload, dict) else {}
            if isinstance(r, dict) and r.get("status") == "pending":
                pending.append(f"- {name}: {r.get('reason')}")
    if not pending:
        lines.append("None.")
    else:
        lines.extend(pending)
    lines.append("")
    if failed:
        lines.append("## Failed Items")
        lines.append("")
        lines.extend(failed)
        lines.append("")

    # Caveats
    lines.append("## Caveats and known discrepancies")
    lines.append("")
    lines.append("- **fusion_class is `BENIGN` for all 20 alerts** in `evaluation_alerts.json` (the current snapshot). "
                 "RQ2.2 stratification only reports the BENIGN slice as a result. The fusion classifier did not write "
                 "KNOWN_ATTACK / CONFIRMED_ANOMALY / NOVEL_ANOMALY labels in this evaluation export — investigate before "
                 "citing per-class alignment numbers in the thesis.")
    lines.append("- **Test split metrics** are computed against `xgboost_test_predictions.npz` (the model's frozen test "
                 "predictions). The `compute_rq1_metrics.py` script in `module6_evaluation/` reads "
                 "`evaluation_alerts.json` (demo-sourced), which is a different population — see prior `[DISCREPANCY]` "
                 "flag in `docs/section313_data_flow_extraction.md`.")
    lines.append("- **RQ1.4 truth-table** flags 2 discrepancies between code and a hypothesised documented expectation "
                 "(MEDIUM × patchable × no_maint, MEDIUM × unpatchable × maint). These reflect the score level chosen "
                 "for the MEDIUM probe (0.45) vs the F2-tuned base threshold (0.425) plus `risk_adaptive_thresholds.yaml` "
                 "multipliers. Not a code bug — the discrepancy is between the assumed coarse policy and the actual "
                 "fine-grained policy. Authoritative behaviour is in `rq1_truth_table.md`.")
    lines.append("- **User study data is LLM-persona simulation**, not human participants. "
                 "Statistical tests still apply (Mann-Whitney U over per-alert correctness/confidence) but the "
                 "interpretation must reflect that LLM personas standing in for IT-generalist/biomed-engineer/nurse-manager "
                 "are an approximation; the response variances differ from human responses.")
    lines.append("- **DAE-driven novel-anomaly faithfulness** — RQ2.2 cannot evidence reduced SHAP faithfulness for "
                 "novel-anomaly alerts because the snapshot contains no NOVEL_ANOMALY rows; see `shap_source_xgb_low_conf` "
                 "counter (zero across all classes in the snapshot).")
    lines.append("- **Pearson r for ordinal D_crit/D_clinical_tier** is a coarse estimator; Spearman ρ may be more "
                 "appropriate for the device-tier ordinal scale. The reported r=0.612 with p=0.045 (n=11) suggests "
                 "partial overlap rather than complete redundancy — consistent with ARCHITECTURE.md limitation L3.")
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Random seed: 42 (numpy, sklearn, scipy bootstrap)")
    lines.append(f"- Code commit: `{git_commit()}`")
    lines.append(f"- Python: {sys.version.split()[0]}")
    lines.append("- Provenance metadata (input file SHA-256 hashes) embedded in every per-RQ JSON.")
    lines.append("- Re-run sequence:")
    lines.append("  ```bash")
    lines.append("  python -m analysis.compute_rq1")
    lines.append("  python -m analysis.compute_rq2")
    lines.append("  python -m analysis.compute_rq3")
    lines.append("  python -m analysis.build_thesis_results")
    lines.append("  ```")
    lines.append("")

    # Output organization
    lines.append("## Output organisation")
    lines.append("")
    lines.append("```")
    for name in [
        "rq1_metrics.json", "rq1_track_b_per_class.json", "rq1_sensitivity_analysis.json",
        "rq1_truth_table.md", "rq1_truth_table.yaml", "rq1_dcrit_dclinical_correlation.json",
        "rq2_shap_stability.json", "rq2_mve_shap_alignment.json", "rq2_mitre_coverage.json",
        "rq2_user_study.json",
        "rq3_test_summary.json", "rq3_no_auto_execution.json", "rq3_audit_integrity.json",
        "rq3_cross_role_consistency.json", "rq3_user_study.json",
        "computation_log.txt",
    ]:
        p = RESULTS_DIR / name
        size = p.stat().st_size if p.exists() else 0
        lines.append(f"results/{name} ({size} bytes)")
    lines.append("")
    lines.append("results/figures/")
    for fname in ["roc_curves.pdf", "pr_curves.pdf", "confusion_matrix.pdf",
                  "sensitivity_histogram.pdf", "shap_stability_distribution.pdf"]:
        p = RESULTS_DIR / "figures" / fname
        if p.exists():
            lines.append(f"  {fname} ({p.stat().st_size} bytes)")
    lines.append("```")
    lines.append("")
    lines.append(f"_Generated by `analysis/build_thesis_results.py` at {now_iso()}._")

    out_path = RESULTS_DIR / "THESIS_RESULTS.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
