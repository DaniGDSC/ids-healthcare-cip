"""Phase 0 analysis driver — produces results/phase0_analysis/ artifacts.

Module 0 ships its analysis classes (DataLoader, StatisticsAnalyzer,
CorrelationAnalyzer, ReportExporter) but no top-level orchestrator.
This script wires them together to produce the artifacts Module 1
consumes: ``stats_report.json`` and ``high_correlations.csv``.

Prerequisite: run ``python -m module0_analysis.phase0.bootstrap_integrity``
first to baseline the dataset hash.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from module0_analysis.phase0 import (  # noqa: E402
    CorrelationAnalyzer,
    DataLoader,
    Phase0Config,
    ReportExporter,
    StatisticsAnalyzer,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = PROJECT_ROOT / "module0_analysis/phase0/config.yaml"
    cfg = Phase0Config.from_yaml(cfg_path)

    loader = DataLoader(cfg)
    df = loader.load()
    loader.validate(df)

    stats = StatisticsAnalyzer(df, cfg)
    descriptive = stats.descriptive_stats()
    missing = stats.missing_values()
    class_dist = stats.class_distribution()

    corr = CorrelationAnalyzer(df, cfg)
    matrix = corr.correlation_matrix()
    pairs = corr.high_correlation_pairs()

    exporter = ReportExporter(cfg)
    exporter.export_stats_report(descriptive, missing, class_dist)
    exporter.export_high_correlations(pairs)
    exporter.export_correlation_matrix(matrix)

    print(f"OK: wrote {cfg.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
