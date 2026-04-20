import re

with open("/home/un1/project/ids-healthcare-cip/module6_evaluation/module6_evaluation.py", "r") as f:
    text = f.read()

# Add imports
if "from src.risk_scorer" not in text:
    text = text.replace(
        "import pandas as pd",
        "import pandas as pd\n\nfrom src.risk_scorer import score_alert\nfrom src.mve_generator import generate_mve"
    )

# Fix curate_evaluation_alerts
# We'll replace the three alert generation blocks.
import_start = """
def curate_evaluation_alerts() -> list:
"""

new_loops = """
    for tier, cfg in tier_targets.items():
        tier_mask = levels == tier
        for cat in cfg["attack_cats"]:
            # M6-E2: vectorised string comparison via pre-cast array
            cat_mask = cats_str == cat
            combined = tier_mask & cat_mask & (y_true == 1)
            candidates = np.where(combined)[0]
            candidates = [c for c in candidates if c not in used_idx]

            if len(candidates) > 0:
                idx = int(candidates[np.argmax(R[candidates])])
                used_idx.add(idx)

    # Benign calibration: 4 benign at various risk levels
    for target_r in [0.20, 0.30, 0.45, 0.55]:
        # M6-E3: pass set directly to np.isin (no list() copy); reuse all_indices
        benign_mask = (y_true == 0) & (~np.isin(all_indices, used_idx))
        candidates = np.where(benign_mask)[0]
        if len(candidates) == 0:
            continue
        # Pick closest to target_r
        idx = int(candidates[np.argmin(np.abs(R[candidates] - target_r))])
        used_idx.add(idx)

    # Fill remaining to reach 20
    while len(used_idx) < 20:
        # M6-E3: reuse all_indices, pass set to np.isin
        remaining = np.where(~np.isin(all_indices, used_idx))[0]
        if len(remaining) == 0:
            break
        idx = int(remaining[np.argmax(R[remaining])])
        used_idx.add(idx)

    # Now generate the alerts for all selected indices
    for idx in used_idx:
        raw_row = df.iloc[idx]
        anomaly_score = float(R[idx])
        
        device_cls = _derive_device_class(idx, df)
        device_criticality = DEVICE_CONTEXT.get(device_cls, DEVICE_CONTEXT["other"]).get("device_criticality", "LOW")
        patchable = device_cls not in {"infusion_pump", "ventilator", "insulin_pump", "patient_monitor", "pacs_server"}
        
        device_context = {
            "criticality": device_criticality,
            "patchable": patchable,
            "similar_events_past_30d": 0,
            "is_maintenance_window": False,
            "is_known_vendor_ip": False,
            "device_type": device_cls
        }
        
        scored_alert = score_alert(anomaly_score, device_context, event_context=None)
        
        raw_alert_dict = raw_row.to_dict()
        raw_alert_dict["alert_name"] = "EVAL_TEST_" + str(attack_cats[idx])
        raw_alert_dict["protocol"] = "TCP" # mock
        
        mve_output = generate_mve(
            raw_alert=raw_alert_dict,
            device_context=device_context,
            baseline={"normal_destinations": [], "normal_protocols": ["HTTPS"]},
            user_context=None
        )
        
        alerts.append(_build_eval_alert(
            idx, R, levels, y_true, attack_cats,
            analyst_by_idx, clinician_by_idx, examples_by_idx,
            test_df=df, raw_row=raw_row,
            scored_alert=scored_alert, mve_output=mve_output
        ))

    logger.info("  Curated %d evaluation alerts", len(alerts))
"""

# ... Wait we can't do exact string replacements automatically here, I should use multi_replace.
