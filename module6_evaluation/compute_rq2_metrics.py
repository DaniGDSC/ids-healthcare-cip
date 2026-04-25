#!/usr/bin/env python3
import json
from pathlib import Path


def main():
    json_path = Path("results/reports/evaluation_alerts.json")
    if not json_path.exists():
        print("ERROR: evaluation_alerts.json not found. Run module6_evaluation first.")
        return

    with open(json_path) as f:
        alerts = json.load(f)

    surfaced = [a for a in alerts if a.get("should_surface", False)]
    total_surfaced = len(surfaced)
    critical_surfaced = sum(1 for a in surfaced if a.get("risk_level") == "CRITICAL")
    critical_alert_rate = critical_surfaced / total_surfaced if total_surfaced > 0 else 0

    true_critical = [a for a in alerts if a.get("true_severity") == "CRITICAL"]
    total_true_critical = len(true_critical)
    fn_critical = sum(1 for a in true_critical if not a.get("should_surface", False))
    fnr_critical = fn_critical / total_true_critical if total_true_critical > 0 else 0

    tp = sum(1 for a in alerts if a.get("ground_truth") == "attack" and a.get("should_surface") is True)
    fn = sum(1 for a in alerts if a.get("ground_truth") == "attack" and a.get("should_surface") is False)
    fp = sum(1 for a in alerts if a.get("ground_truth") == "benign" and a.get("should_surface") is True)
    tn = sum(1 for a in alerts if a.get("ground_truth") == "benign" and a.get("should_surface") is False)

    metrics = {
        "critical_alert_rate": round(critical_alert_rate, 4),
        "fnr_critical": round(fnr_critical, 4),
        "total_surfaced_alerts": total_surfaced,
        "true_critical_count": total_true_critical,
        "false_negative_critical": fn_critical,
        "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else 0,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0,
    }

    out_path = Path("results/rq2_metrics.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== RQ2 Metrics ===")
    print(f"FNR_CRITICAL: {fnr_critical:.2%}")
    print(f"CRITICAL alert rate: {critical_alert_rate:.2%}")
    print(f"Sensitivity: {metrics['sensitivity']:.2%}")
    print(f"Specificity: {metrics['specificity']:.2%}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
