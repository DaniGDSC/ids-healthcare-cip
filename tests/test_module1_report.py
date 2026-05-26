"""report.py renderer tests — verifies C1 stale-hardcoded fixes hold.

These tests guard against regression to the manuscript-blocking bugs
that caused the rendered §4.1 to print:
  - "ffill bio, fill_zero net" when config was median/dropna
  - "Stratified 70/30" when config was 60/15/15/10
  - Train+Test only (missing val/demo rows) — 2-way era leftover
  - Hardcoded threshold 0.95
"""
from __future__ import annotations


from module1_preprocessing.phase1.report import render_preprocessing_report


def _report(*, bio="median", net="dropna", threshold=0.95, n_red=3):
    """Build a synthetic but realistic pipeline report dict."""
    return {
        "ingestion": {"raw_rows": 16318, "raw_columns": 44, "files_loaded": 1},
        "identifier_removal": {
            "n_dropped": 5,
            "columns_dropped": ["SrcAddr", "DstAddr", "SrcMac", "DstMac", "Packet_num"],
        },
        "cleaning": {
            "biometric_strategy": bio,
            "network_strategy": net,
            "biometric_cells_filled": 42,
            "rows_dropped": 100,
            "rows_remaining": 16218,
        },
        "redundancy": {
            "threshold": threshold,
            "columns_dropped": [f"col_{i}" for i in range(n_red)],
            "n_dropped": n_red,
            "n_refused_protected": 0,
        },
        "split": {
            "train_samples": 9730, "val_samples": 2432,
            "test_samples": 2432, "demo_samples": 1624,
            "train_ratio_global": 0.6000, "val_ratio_global": 0.1500,
            "test_ratio_global": 0.1500, "demo_ratio_global": 0.1000,
            "train_attack_rate": 0.205, "val_attack_rate": 0.205,
            "test_attack_rate": 0.205, "demo_attack_rate": 0.205,
            "stratified": True, "stratify_target": "Attack Category",
        },
        "track_a": {"smote_enabled": True, "smote_strategy": "auto", "smote_k_neighbors": 5},
        "output": {"n_features": 38, "feature_names": []},
        "random_state": 42,
        "elapsed_seconds": 12.3,
    }


# ── C1.1: missing-strategy strings must come from config, not hardcoded ──


def test_steps_table_renders_median_dropna_not_ffill_fill_zero():
    md = render_preprocessing_report(_report(bio="median", net="dropna"))
    assert "median bio, dropna net" in md
    # Anti-regression assertion: previous hardcoded string is gone
    assert "ffill bio, fill_zero net" not in md


def test_steps_table_renders_ffill_when_configured():
    md = render_preprocessing_report(_report(bio="ffill", net="dropna"))
    assert "ffill bio, dropna net" in md


def test_steps_table_renders_fill_zero_when_configured():
    md = render_preprocessing_report(_report(bio="median", net="fill_zero"))
    assert "median bio, fill_zero net" in md


# ── C1.2: split ratio in steps table must be 4-way, not 70/30 ──


def test_steps_table_renders_4way_split_not_70_30():
    md = render_preprocessing_report(_report())
    assert "Stratified 4-way" in md
    # Anti-regression
    assert "Stratified 70/30" not in md
    # Ratios from config
    assert "60%" in md and "15%" in md and "10%" in md


# ── C1.3: §4.1.4 split table must show all 4 splits ──


def test_split_table_renders_all_four_partitions():
    md = render_preprocessing_report(_report())
    # Header line + 4 partition rows
    assert "| Train |" in md
    assert "| Val |" in md
    assert "| Test |" in md
    assert "| Demo |" in md


def test_split_table_includes_attack_rate_and_purpose():
    md = render_preprocessing_report(_report())
    assert "Attack rate" in md
    assert "Purpose" in md
    assert "FROZEN" in md  # test + demo are frozen


# ── C1.4: threshold in feature-reduction table must come from config ──


def test_feature_reduction_threshold_from_config():
    md = render_preprocessing_report(_report(threshold=0.92))
    assert "|*r*| ≥ 0.92" in md


def test_feature_reduction_threshold_uses_default_when_missing():
    rep = _report(threshold=0.95)
    rep["redundancy"].pop("threshold")  # simulate older report
    md = render_preprocessing_report(rep)
    # Falls back to 0.95
    assert "|*r*| ≥ 0.95" in md


# ── 4.1.2 justification strings must come from strategy lookup ──


def test_4_1_2_justification_describes_median_when_median():
    md = render_preprocessing_report(_report(bio="median"))
    assert "patient-safe" in md
    assert "Forward-fill" not in md  # not ffill description


def test_4_1_2_justification_describes_ffill_when_ffill():
    md = render_preprocessing_report(_report(bio="ffill"))
    assert "session_column" in md
    assert "patient-safe" not in md  # not median description


def test_4_1_2_justification_describes_fill_zero_when_fill_zero():
    md = render_preprocessing_report(_report(net="fill_zero"))
    assert "capture-loss" in md or "explicitly accepted" in md


# ── Renderer warns on missing report keys ──


def test_missing_keys_logged_at_warning(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    render_preprocessing_report({})  # empty → all sections render as fallback
    assert any("missing keys" in r.message for r in caplog.records)
