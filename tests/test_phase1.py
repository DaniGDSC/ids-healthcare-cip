"""
Unit tests for Phase 1 preprocessing:
  - DataCleaner  (data_cleaner.py)
  - HIPAACompliance (hipaa_compliance.py)
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase1_preprocessing.data_cleaner import DataCleaner
from src.phase1_preprocessing.hipaa_compliance import HIPAACompliance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(**kwargs) -> pd.DataFrame:
    """Build a tiny DataFrame; keyword args are passed straight through."""
    return pd.DataFrame(kwargs)


def _numeric_df() -> pd.DataFrame:
    """A well-formed numeric DataFrame with no issues."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.integers(1, 100, size=(20, 4)),
        columns=["a", "b", "c", "d"],
    ).astype(float)


# ===========================================================================
# TestDataCleaner
# ===========================================================================

class TestDataCleaner:
    """Unit tests for DataCleaner."""

    # ------------------------------------------------------------------
    # Duplicate removal
    # ------------------------------------------------------------------

    def test_remove_duplicates(self):
        df = pd.DataFrame({"x": [1, 1, 2, 2, 3], "y": [10, 10, 20, 20, 30]})
        cleaner = DataCleaner(remove_duplicates=True, remove_constant_features=False)
        result = cleaner.clean(df)
        assert len(result) == 3
        assert cleaner.get_cleaning_report()["duplicates_removed"] == 2

    def test_no_remove_duplicates_flag(self):
        df = pd.DataFrame({"x": [1, 1, 2], "y": [5, 5, 6]})
        cleaner = DataCleaner(remove_duplicates=False, remove_constant_features=False)
        result = cleaner.clean(df)
        assert len(result) == 3  # duplicates kept

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------

    def test_drop_high_missing_columns(self):
        """Columns with > threshold fraction of NaN should be dropped."""
        df = pd.DataFrame(
            {
                "good": [1.0, 2.0, 3.0, 4.0],
                "mostly_nan": [np.nan, np.nan, np.nan, 1.0],  # 75 % missing
            }
        )
        cleaner = DataCleaner(drop_na_threshold=0.5, remove_constant_features=False)
        result = cleaner.clean(df)
        assert "mostly_nan" not in result.columns
        assert "good" in result.columns
        assert cleaner.get_cleaning_report()["columns_dropped_missing"] == 1

    def test_fill_missing_median(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        cleaner = DataCleaner(
            drop_na_threshold=0.9, fill_strategy="median", remove_constant_features=False
        )
        result = cleaner.clean(df)
        assert result["x"].isna().sum() == 0
        assert result["x"].iloc[1] == pytest.approx(np.nanmedian([1, 3]))
        assert cleaner.get_cleaning_report()["cells_filled"] == 1

    def test_fill_missing_mean(self):
        df = pd.DataFrame({"x": [2.0, np.nan, 4.0]})
        cleaner = DataCleaner(
            drop_na_threshold=0.9, fill_strategy="mean", remove_constant_features=False
        )
        result = cleaner.clean(df)
        assert result["x"].iloc[1] == pytest.approx(3.0)

    def test_fill_missing_zero(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        cleaner = DataCleaner(
            drop_na_threshold=0.9, fill_strategy="zero", remove_constant_features=False
        )
        result = cleaner.clean(df)
        assert result["x"].iloc[1] == 0.0

    def test_invalid_fill_strategy_raises(self):
        df = pd.DataFrame({"x": [1.0, np.nan]})
        cleaner = DataCleaner(fill_strategy="bogus", remove_constant_features=False)
        with pytest.raises(ValueError, match="Unsupported fill strategy"):
            cleaner.clean(df)

    # ------------------------------------------------------------------
    # Infinite value handling
    # ------------------------------------------------------------------

    def test_replace_infinite_values(self):
        df = pd.DataFrame({"x": [1.0, np.inf, -np.inf, 4.0]})
        cleaner = DataCleaner(
            replace_inf=True, inf_replacement_value=0.0, remove_constant_features=False
        )
        result = cleaner.clean(df)
        assert not np.isinf(result["x"]).any()
        report = cleaner.get_cleaning_report()
        assert report["inf_values_replaced"] == 2

    def test_no_replace_inf_flag(self):
        df = pd.DataFrame({"x": [1.0, np.inf, 4.0]})
        cleaner = DataCleaner(replace_inf=False, remove_constant_features=False)
        result = cleaner.clean(df)
        assert np.isinf(result["x"]).any()

    # ------------------------------------------------------------------
    # Constant feature removal
    # ------------------------------------------------------------------

    def test_remove_constant_features(self):
        """Near-zero variance column must be dropped."""
        df = pd.DataFrame(
            {
                "varying": [1.0, 2.0, 3.0, 100.0],
                "constant": [5.0, 5.0, 5.0, 5.0],
            }
        )
        cleaner = DataCleaner(remove_constant_features=True, constant_threshold=0.01)
        result = cleaner.clean(df)
        assert "constant" not in result.columns
        assert "varying" in result.columns
        assert cleaner.get_cleaning_report()["constant_columns_removed"] == 1

    # ------------------------------------------------------------------
    # Negative value handling
    # ------------------------------------------------------------------

    def test_handle_negative_values_all_numeric(self):
        """All negative values clipped to 0 when handle_negatives=True."""
        df = pd.DataFrame({"a": [-1.0, 2.0, -3.0], "b": [4.0, -5.0, 6.0]})
        cleaner = DataCleaner(
            handle_negatives=True, remove_constant_features=False
        )
        result = cleaner.clean(df)
        assert (result >= 0).all().all()
        report = cleaner.get_cleaning_report()
        assert report["negative_values_clipped"] == 3  # -1, -3, -5

    def test_handle_negative_values_explicit_cols(self):
        """Only the specified columns are clipped."""
        df = pd.DataFrame({"a": [-1.0, 2.0], "b": [-3.0, 4.0]})
        cleaner = DataCleaner(
            handle_negatives=True, negative_cols=["a"], remove_constant_features=False
        )
        result = cleaner.clean(df)
        # Column 'a' clipped
        assert result["a"].min() >= 0
        # Column 'b' left unchanged
        assert result["b"].iloc[0] == -3.0

    def test_handle_negatives_disabled_by_default(self):
        """Default DataCleaner must NOT touch negative values."""
        df = pd.DataFrame({"x": [-5.0, -10.0]})
        cleaner = DataCleaner(remove_constant_features=False, replace_inf=False)
        result = cleaner.clean(df)
        assert (result["x"] < 0).all()

    # ------------------------------------------------------------------
    # Cleaning report
    # ------------------------------------------------------------------

    def test_cleaning_report_keys(self):
        """get_cleaning_report() must return all expected keys."""
        expected_keys = {
            "duplicates_removed",
            "columns_dropped_missing",
            "rows_dropped_missing",
            "cells_filled",
            "inf_values_replaced",
            "negative_values_clipped",
            "constant_columns_removed",
        }
        cleaner = DataCleaner()
        cleaner.clean(_numeric_df())
        report = cleaner.get_cleaning_report()
        assert set(report.keys()) == expected_keys

    def test_cleaning_report_resets_each_call(self):
        """Report counters must reset between successive clean() calls."""
        df_with_dups = pd.DataFrame({"x": [1.0, 1.0, 2.0], "y": [1.0, 1.0, 2.0]})
        cleaner = DataCleaner(remove_constant_features=False)
        cleaner.clean(df_with_dups)
        cleaner.clean(_numeric_df())  # second call — clean data
        report = cleaner.get_cleaning_report()
        assert report["duplicates_removed"] == 0

    # ------------------------------------------------------------------
    # get_data_summary
    # ------------------------------------------------------------------

    def test_data_summary(self):
        df = _numeric_df()
        cleaner = DataCleaner()
        summary = cleaner.get_data_summary(df)
        assert summary["n_rows"] == 20
        assert summary["n_columns"] == 4
        assert summary["missing_values"] == 0
        assert summary["memory_usage_mb"] > 0


# ===========================================================================
# TestHIPAACompliance
# ===========================================================================

class TestHIPAACompliance:
    """Unit tests for HIPAACompliance."""

    # ------------------------------------------------------------------
    # IP anonymization
    # ------------------------------------------------------------------

    def test_anonymize_ip_addresses_removes_original(self):
        df = pd.DataFrame({"Src IP": ["192.168.1.1", "10.0.0.1"]})
        hipaa = HIPAACompliance(enabled=True, salt="test-salt")
        result = hipaa.anonymize_ip_addresses(df, ip_columns=["Src IP"])
        assert "192.168.1.1" not in result["Src IP"].values
        assert "10.0.0.1" not in result["Src IP"].values

    def test_anonymize_consistent_within_instance(self):
        """Same IP → same hash within a single HIPAACompliance instance."""
        df = pd.DataFrame({"Src IP": ["1.2.3.4", "1.2.3.4", "5.6.7.8"]})
        hipaa = HIPAACompliance(enabled=True, salt="fixed-salt")
        result = hipaa.anonymize_ip_addresses(df, ip_columns=["Src IP"])
        assert result["Src IP"].iloc[0] == result["Src IP"].iloc[1]
        assert result["Src IP"].iloc[0] != result["Src IP"].iloc[2]

    def test_salt_changes_hash(self):
        """Different salts must produce different hashes (re-ID protection)."""
        df = pd.DataFrame({"Src IP": ["192.168.1.1"]})
        h1 = HIPAACompliance(enabled=True, salt="salt-A")
        h2 = HIPAACompliance(enabled=True, salt="salt-B")
        r1 = h1.anonymize_ip_addresses(df.copy(), ["Src IP"])
        r2 = h2.anonymize_ip_addresses(df.copy(), ["Src IP"])
        assert r1["Src IP"].iloc[0] != r2["Src IP"].iloc[0]

    def test_anonymize_skips_missing(self):
        """NaN IP values must remain NaN after hashing."""
        df = pd.DataFrame({"Src IP": ["1.2.3.4", None]})
        hipaa = HIPAACompliance(enabled=True, salt="s")
        result = hipaa.anonymize_ip_addresses(df, ["Src IP"])
        assert pd.isna(result["Src IP"].iloc[1])

    def test_anonymize_disabled(self):
        """When enabled=False, data must be returned unchanged."""
        df = pd.DataFrame({"Src IP": ["1.2.3.4"]})
        hipaa = HIPAACompliance(enabled=False)
        result = hipaa.anonymize_ip_addresses(df, ["Src IP"])
        assert result["Src IP"].iloc[0] == "1.2.3.4"

    def test_anonymize_updates_deid_report(self):
        df = pd.DataFrame({"Src IP": ["1.2.3.4", "5.6.7.8"]})
        hipaa = HIPAACompliance(enabled=True, salt="s")
        hipaa.anonymize_ip_addresses(df, ["Src IP"])
        assert hipaa.get_deidentification_report()["ips_anonymized"] == 2

    # ------------------------------------------------------------------
    # PHI column removal
    # ------------------------------------------------------------------

    def test_remove_phi_columns(self):
        df = pd.DataFrame({"Src IP": ["x"], "Flow Duration": [100], "Timestamp": ["t"]})
        hipaa = HIPAACompliance(enabled=True)
        result = hipaa.remove_phi_columns(df, ["Src IP", "Timestamp"])
        assert "Src IP" not in result.columns
        assert "Timestamp" not in result.columns
        assert "Flow Duration" in result.columns

    def test_remove_phi_ignores_absent_columns(self):
        """remove_phi_columns must silently skip columns not in the DataFrame."""
        df = pd.DataFrame({"a": [1]})
        hipaa = HIPAACompliance(enabled=True)
        result = hipaa.remove_phi_columns(df, ["NonExistentCol"])
        # Original column still present, no error
        assert "a" in result.columns

    def test_remove_phi_updates_deid_report(self):
        df = pd.DataFrame({"Src IP": ["x"], "keep": [1]})
        hipaa = HIPAACompliance(enabled=True)
        hipaa.remove_phi_columns(df, ["Src IP"])
        assert hipaa.get_deidentification_report()["phi_columns_removed"] == 1

    # ------------------------------------------------------------------
    # Timestamp truncation
    # ------------------------------------------------------------------

    def test_truncate_timestamps_month(self):
        df = pd.DataFrame({"ts": ["2018-02-14 10:30:00", "2018-11-07 23:59:59"]})
        hipaa = HIPAACompliance(enabled=True)
        result = hipaa.truncate_timestamps(df, ["ts"], granularity="month")
        assert result["ts"].iloc[0] == "2018-02"
        assert result["ts"].iloc[1] == "2018-11"

    def test_truncate_timestamps_year(self):
        df = pd.DataFrame({"ts": ["2018-02-14 10:30:00"]})
        hipaa = HIPAACompliance(enabled=True)
        result = hipaa.truncate_timestamps(df, ["ts"], granularity="year")
        assert result["ts"].iloc[0] == "2018"

    def test_truncate_timestamps_day(self):
        df = pd.DataFrame({"ts": ["2018-02-14 10:30:00"]})
        hipaa = HIPAACompliance(enabled=True)
        result = hipaa.truncate_timestamps(df, ["ts"], granularity="day")
        assert result["ts"].iloc[0] == "2018-02-14"

    def test_truncate_timestamps_invalid_granularity(self):
        df = pd.DataFrame({"ts": ["2018-02-14"]})
        hipaa = HIPAACompliance(enabled=True)
        with pytest.raises(ValueError, match="Invalid granularity"):
            hipaa.truncate_timestamps(df, ["ts"], granularity="second")

    def test_truncate_timestamps_updates_deid_report(self):
        df = pd.DataFrame({"ts": ["2018-02-14", "2018-03-01"]})
        hipaa = HIPAACompliance(enabled=True)
        hipaa.truncate_timestamps(df, ["ts"], granularity="year")
        assert hipaa.get_deidentification_report()["timestamps_truncated"] == 2

    # ------------------------------------------------------------------
    # Pseudonymization
    # ------------------------------------------------------------------

    def test_pseudonymize_columns_replaces_values(self):
        df = pd.DataFrame({"user_id": ["alice", "bob", "alice"]})
        hipaa = HIPAACompliance(enabled=True, salt="ps")
        result = hipaa.pseudonymize_columns(df, ["user_id"])
        # Original values gone
        assert "alice" not in result["user_id"].values
        assert "bob" not in result["user_id"].values

    def test_pseudonymize_consistent(self):
        """Same raw value must always map to same pseudonym."""
        df = pd.DataFrame({"user_id": ["alice", "bob", "alice"]})
        hipaa = HIPAACompliance(enabled=True, salt="ps")
        result = hipaa.pseudonymize_columns(df, ["user_id"])
        assert result["user_id"].iloc[0] == result["user_id"].iloc[2]
        assert result["user_id"].iloc[0] != result["user_id"].iloc[1]

    def test_pseudonymize_prefix(self):
        df = pd.DataFrame({"col": ["x"]})
        hipaa = HIPAACompliance(enabled=True, salt="ps")
        result = hipaa.pseudonymize_columns(df, ["col"])
        assert result["col"].iloc[0].startswith("PSEUDO_")

    def test_pseudonymize_updates_deid_report(self):
        df = pd.DataFrame({"col": ["a", "b", "a"]})
        hipaa = HIPAACompliance(enabled=True, salt="ps")
        hipaa.pseudonymize_columns(df, ["col"])
        assert hipaa.get_deidentification_report()["columns_pseudonymized"] == 3

    # ------------------------------------------------------------------
    # De-identification report
    # ------------------------------------------------------------------

    def test_deidentification_report_keys(self):
        hipaa = HIPAACompliance(enabled=True)
        report = hipaa.get_deidentification_report()
        assert "ips_anonymized" in report
        assert "phi_columns_removed" in report
        assert "timestamps_truncated" in report
        assert "columns_pseudonymized" in report
        assert "access_log_entries" in report

    def test_deidentification_report_accumulates(self):
        """Stats accumulate across multiple method calls."""
        df_ip = pd.DataFrame({"Src IP": ["1.1.1.1", "2.2.2.2"]})
        df_phi = pd.DataFrame({"Timestamp": ["t"], "keep": [1]})
        hipaa = HIPAACompliance(enabled=True, salt="cumulative")
        hipaa.anonymize_ip_addresses(df_ip, ["Src IP"])
        hipaa.remove_phi_columns(df_phi, ["Timestamp"])
        report = hipaa.get_deidentification_report()
        assert report["ips_anonymized"] == 2
        assert report["phi_columns_removed"] == 1

    # ------------------------------------------------------------------
    # Access log & export
    # ------------------------------------------------------------------

    def test_access_log_records_entry(self):
        hipaa = HIPAACompliance(enabled=True)
        hipaa.log_data_access(
            user="test_user",
            action="read",
            data_description="test dataset",
            record_count=42,
        )
        log = hipaa.get_access_log()
        assert len(log) == 1
        assert log[0]["user"] == "test_user"
        assert log[0]["record_count"] == 42

    def test_export_access_log_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.json"
            hipaa = HIPAACompliance(enabled=True)
            hipaa.log_data_access(
                user="pipeline",
                action="process",
                data_description="network flows",
                record_count=1000,
            )
            hipaa.export_access_log(str(log_file))
            assert log_file.exists()
            data = json.loads(log_file.read_text())
            assert isinstance(data, list)
            assert data[0]["user"] == "pipeline"

    def test_export_access_log_empty_warns(self, caplog):
        import logging

        hipaa = HIPAACompliance(enabled=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            with caplog.at_level(logging.WARNING):
                hipaa.export_access_log(str(Path(tmpdir) / "empty.json"))
        assert "No access log entries" in caplog.text

    # ------------------------------------------------------------------
    # Data retention validation
    # ------------------------------------------------------------------

    def test_retention_compliant(self):
        hipaa = HIPAACompliance(enabled=True)
        assert hipaa.validate_data_retention(data_age_days=30) is True

    def test_retention_non_compliant(self):
        hipaa = HIPAACompliance(enabled=True)
        assert hipaa.validate_data_retention(data_age_days=9999) is False

    def test_retention_disabled_always_true(self):
        hipaa = HIPAACompliance(enabled=False)
        assert hipaa.validate_data_retention(data_age_days=99999) is True

    def test_retention_boundary(self):
        """Exactly 2555 days should be compliant (inclusive)."""
        hipaa = HIPAACompliance(enabled=True)
        assert hipaa.validate_data_retention(data_age_days=2555) is True
        assert hipaa.validate_data_retention(data_age_days=2556) is False


# ===========================================================================
# Integration: DataCleaner + HIPAACompliance
# ===========================================================================

class TestCleaner_HIPAA_Integration:
    """Verify DataCleaner and HIPAACompliance can be composed without data leaks."""

    def test_no_phi_after_pipeline(self):
        """IP columns must not survive a full clean → deid pipeline."""
        df = pd.DataFrame(
            {
                "Src IP": ["10.0.0.1", "10.0.0.2", "10.0.0.1"],
                "Flow Duration": [100.0, 200.0, 100.0],
                "Pkt Size": [64.0, 128.0, 64.0],
            }
        )

        # Step 1: Clean
        cleaner = DataCleaner(remove_constant_features=False)
        df_clean = cleaner.clean(df)

        # Step 2: HIPAA de-id
        hipaa = HIPAACompliance(enabled=True, salt="integration-salt")
        df_deid = hipaa.anonymize_ip_addresses(df_clean, ["Src IP"])
        df_deid = hipaa.remove_phi_columns(df_deid, ["Src IP"])

        assert "Src IP" not in df_deid.columns
        # Cleaning report is accessible
        assert cleaner.get_cleaning_report()["duplicates_removed"] >= 0

    def test_cleaning_report_is_copy(self):
        """Mutating the report dict must not affect the cleaner's internal state."""
        cleaner = DataCleaner(remove_constant_features=False)
        cleaner.clean(_numeric_df())
        report = cleaner.get_cleaning_report()
        report["duplicates_removed"] = 9999
        assert cleaner.get_cleaning_report()["duplicates_removed"] != 9999


# ===========================================================================
# TestPhase1Pipeline — new WUSTL-EHMS pipeline tests
# ===========================================================================

def _wustl_ehms_df(n: int = 100) -> pd.DataFrame:
    """Build a synthetic WUSTL-EHMS–like DataFrame for testing."""
    rng = np.random.default_rng(42)
    data: Dict[str, Any] = {}
    # Network features (subset)
    for col in ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
                 "sttl", "dttl", "sload", "dload", "sinpkt", "dinpkt"]:
        data[col] = rng.random(n) * 100
    # High-correlation pair: make "sload_dup" ≈ sload (r > 0.99)
    data["sload_dup"] = data["sload"] + rng.normal(0, 0.001, n)
    # Biometric features
    data["HR"] = rng.normal(75, 5, n)
    data["SpO2"] = rng.normal(97, 1, n)
    data["SBP"] = rng.normal(120, 10, n)
    data["DBP"] = rng.normal(80, 5, n)
    # Identifiers (should be dropped by HIPAA step)
    data["srcip"] = ["10.0.0.1"] * n
    data["dstip"] = ["192.168.1.1"] * n
    # Label
    data["label"] = rng.choice(["Normal", "Attack"], size=n, p=[0.87, 0.13])
    return pd.DataFrame(data)


from typing import Dict, Any as _Any  # noqa: E402  (already imported above, harmless)


class TestPhase1Pipeline:
    """Unit tests for the rewritten Phase1Pipeline (WUSTL-EHMS)."""

    @staticmethod
    def _make_config(**overrides) -> Dict[str, _Any]:
        """Minimal config dict for testing."""
        cfg: Dict[str, _Any] = {
            "data": {"input_dir": "/tmp", "output_dir": "/tmp", "label_column": "label"},
            "hipaa": {"enabled": True, "remove_columns": ["srcip", "dstip"]},
            "missing_values": {
                "biometric_strategy": "ffill",
                "biometric_columns": ["HR", "SpO2", "SBP", "DBP"],
                "network_strategy": "dropna",
                "network_missing_threshold": 0.05,
            },
            "correlation_removal": {"enabled": True, "threshold": 0.95},
            "splitting": {"train_ratio": 0.70, "test_ratio": 0.30, "stratify": True, "random_state": 42},
            "normalization": {"method": "robust"},
            "output": {},
            "logging": {"file": "/tmp/test_phase1.log"},
        }
        cfg.update(overrides)
        return cfg

    @staticmethod
    def _pipeline(cfg=None):
        from src.phase1_preprocessing.run_phase1 import Phase1Pipeline as P1
        cfg = cfg or TestPhase1Pipeline._make_config()
        return P1(cfg, logging.getLogger("test_phase1"))

    # ---- Step 2 ----
    def test_hipaa_drops_phi_columns(self):
        p = self._pipeline()
        df = _wustl_ehms_df()
        result = p._hipaa_sanitize(df)
        assert "srcip" not in result.columns
        assert "dstip" not in result.columns
        # Non-identifier columns survive
        assert "dur" in result.columns
        assert "HR" in result.columns

    # ---- Step 3 ----
    def test_missing_ffill_biometrics(self):
        p = self._pipeline()
        df = _wustl_ehms_df()
        # Inject NaNs in biometric columns
        df.loc[2, "HR"] = np.nan
        df.loc[3, "SpO2"] = np.nan
        result = p._handle_missing(df)
        assert result["HR"].isna().sum() == 0
        assert result["SpO2"].isna().sum() == 0

    def test_missing_dropna_network(self):
        p = self._pipeline()
        df = _wustl_ehms_df()
        # Inject NaN in a network feature (< 5% so column kept, row dropped)
        df.loc[0, "dur"] = np.nan
        rows_before = len(df)
        result = p._handle_missing(df)
        assert len(result) == rows_before - 1

    # ---- Step 4 ----
    def test_redundancy_elimination(self):
        p = self._pipeline()
        df = _wustl_ehms_df()
        # sload_dup should be eliminated (corr ≈ 1.0 with sload)
        result = p._drop_redundant(df)
        assert "sload_dup" not in result.columns
        assert "sload" in result.columns  # original kept

    # ---- Step 5 ----
    def test_stratified_split_ratio(self):
        p = self._pipeline()
        df = _wustl_ehms_df(n=200)
        # Remove identifiers first
        df = df.drop(columns=["srcip", "dstip"])
        X_train, X_test, y_train, y_test, feat = p._stratified_split(df)
        total = len(X_train) + len(X_test)
        train_pct = len(X_train) / total
        assert 0.65 <= train_pct <= 0.75  # ~70%

    # ---- Step 6 ----
    def test_robust_scaler_fit_on_train(self):
        p = self._pipeline()
        rng = np.random.default_rng(0)
        X_train = rng.random((100, 5))
        X_test = rng.random((30, 5))
        X_tr_s, X_te_s, normalizer = p._scale(X_train, X_test)
        # Scaled train should have IQR ≈ 1 per column (robust scaler property)
        for col_idx in range(X_tr_s.shape[1]):
            q1 = np.percentile(X_tr_s[:, col_idx], 25)
            q3 = np.percentile(X_tr_s[:, col_idx], 75)
            assert 0.8 <= (q3 - q1) <= 1.2  # IQR close to 1

    # ---- Full pipeline (e2e with tmpdir) ----
    def test_full_pipeline_run(self, tmp_path):
        """End-to-end: write a fake CSV → run pipeline → check outputs."""
        # Prepare fake CSV in tmp input dir
        input_dir = tmp_path / "raw"
        input_dir.mkdir()
        df = _wustl_ehms_df(n=200)
        df.to_csv(input_dir / "wustl.csv", index=False)

        output_dir = tmp_path / "processed"
        scaler_dir = tmp_path / "models" / "scalers"

        cfg = self._make_config()
        cfg["data"]["input_dir"] = str(input_dir)
        cfg["data"]["output_dir"] = str(output_dir)
        cfg["output"]["scaler_file"] = str(scaler_dir / "robust_scaler.pkl")

        from src.phase1_preprocessing.run_phase1 import Phase1Pipeline as P1
        p = P1(cfg, logging.getLogger("test_e2e"))
        # Override project_root-based scaler path for tmpdir
        report = p.run()

        assert (output_dir / "train_phase1.csv").exists()
        assert (output_dir / "test_phase1.csv").exists()
        assert (output_dir / "phase1_report.json").exists()
        assert report["elapsed_seconds"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
