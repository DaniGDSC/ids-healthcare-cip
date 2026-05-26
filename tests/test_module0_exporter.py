"""Exporter tests — Open/Closed compliance + atomic write."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from module0_analysis.exporter import (
    BaseExporter,
    CsvExporter,
    JsonExporter,
    MarkdownExporter,
    ParquetExporter,
    ReportExporter,
)
from module0_analysis import Phase0Config


def _config(out_dir: Path) -> Phase0Config:
    return Phase0Config(
        data_path=Path("data.csv"),
        output_dir=out_dir,
        label_column="Label",
        required_columns=["Label"],
        leakage_columns=[],
        network_feature_count=0,
        biometric_feature_count=0,
        correlation_threshold=0.95,
        missing_value_warn_pct=5.0,
        outlier_iqr_multiplier=1.5,
        top_variance_k=5,
        random_state=42,
        train_ratio=0.7,
        test_ratio=0.3,
        stats_report_file="stats.json",
        high_correlations_file="corr.csv",
        correlation_matrix_file="matrix.parquet",
        quality_report_file="report.md",
    )


def test_json_exporter_writes_indented(tmp_path):
    JsonExporter().export({"a": 1, "b": 2}, tmp_path / "x.json")
    text = (tmp_path / "x.json").read_text()
    assert json.loads(text) == {"a": 1, "b": 2}
    assert "\n" in text  # indented


def test_json_exporter_atomic_no_tmp_left(tmp_path):
    JsonExporter().export({"a": 1}, tmp_path / "x.json")
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == []


def test_csv_exporter_no_index(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    CsvExporter().export(df, tmp_path / "x.csv")
    text = (tmp_path / "x.csv").read_text()
    # pandas default index column would prefix lines with "0,"/"1," — absent.
    assert text.splitlines()[0] == "a,b"


def test_parquet_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    ParquetExporter().export(df, tmp_path / "x.parquet")
    df2 = pd.read_parquet(tmp_path / "x.parquet")
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df2.reset_index(drop=True))


def test_markdown_exporter_utf8(tmp_path):
    content = "## Title\n\nWith unicode: Ý ñ é\n"
    MarkdownExporter().export(content, tmp_path / "x.md")
    assert (tmp_path / "x.md").read_text(encoding="utf-8") == content


def test_base_exporter_is_abstract():
    with pytest.raises(TypeError):
        BaseExporter()  # type: ignore[abstract]


def test_report_exporter_uses_injected_exporters(tmp_path):
    """Dependency Inversion — orchestrator must accept substitutes."""
    cfg = _config(tmp_path / "out")
    calls = []

    class FakeJson(BaseExporter):
        def export(self, data, path):
            calls.append(("json", path.name))

    class FakeCsv(BaseExporter):
        def export(self, data, path):
            calls.append(("csv", path.name))

    rex = ReportExporter(cfg, json_exporter=FakeJson(), csv_exporter=FakeCsv())
    rex.export_stats_report({"d": 1}, {"m": 1}, {"c": 1})
    rex.export_high_correlations([("a", "b", 0.99)])
    assert ("json", "stats.json") in calls
    assert ("csv", "corr.csv") in calls


def test_report_exporter_creates_output_dir(tmp_path):
    out = tmp_path / "out" / "deeply" / "nested"
    cfg = _config(out)
    ReportExporter(cfg)
    assert out.exists() and out.is_dir()
