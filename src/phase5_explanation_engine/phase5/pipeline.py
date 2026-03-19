"""ExplanationPipeline — orchestrates all Phase 5 steps via DI.

Usage::

    python -m src.phase5_explanation_engine.phase5.pipeline
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

from .alert_filter import AlertFilter
from .artifact_reader import Phase4ArtifactReader
from .bar_chart_visualizer import BarChartVisualizer
from .config import Phase5Config
from .context_enricher import ContextEnricher
from .explanation_generator import ExplanationGenerator
from .exporter import ExplanationExporter
from .feature_importance import FeatureImportanceRanker
from .line_graph_visualizer import LineGraphVisualizer
from .report import render_explanation_report
from .shap_computer import SHAPComputer
from .waterfall_visualizer import WaterfallVisualizer

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
_TIMESTEPS: int = 20


def _get_git_commit() -> str:
    """Get current git commit hash for artifact versioning."""
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _detect_hardware() -> Dict[str, str]:
    """Detect GPU/CPU availability and return hardware info dict."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        info = {"device": f"GPU: {gpus[0].name}", "cuda": "available"}
    else:
        cpu_info = platform.processor() or platform.machine()
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        info = {
            "device": f"CPU: {cpu_info}",
            "cuda": "N/A (CPU execution)",
        }
    info["tensorflow"] = tf.__version__
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    return info


class ExplanationPipeline:
    """Orchestrate Phase 5: verify -> filter -> SHAP -> rank -> enrich -> viz -> export.

    All components are injected via the constructor (Dependency Inversion).

    Args:
        config: Validated Phase 5 configuration.
        artifact_reader: Phase 4 artifact reader.
        alert_filter: Non-NORMAL sample filter.
        shap_computer: SHAP value calculator.
        feature_ranker: Feature importance ranker.
        context_enricher: Sample context enricher.
        waterfall_viz: Waterfall chart visualizer.
        bar_chart_viz: Bar chart visualizer.
        line_graph_viz: Line graph (timeline) visualizer.
        exporter: Explanation artifact exporter.
        project_root: Absolute project root for path resolution.
    """

    def __init__(
        self,
        config: Phase5Config,
        artifact_reader: Phase4ArtifactReader,
        alert_filter: AlertFilter,
        shap_computer: SHAPComputer,
        feature_ranker: FeatureImportanceRanker,
        context_enricher: ContextEnricher,
        waterfall_viz: WaterfallVisualizer,
        bar_chart_viz: BarChartVisualizer,
        line_graph_viz: LineGraphVisualizer,
        exporter: ExplanationExporter,
        project_root: Path,
    ) -> None:
        self._config = config
        self._reader = artifact_reader
        self._filter = alert_filter
        self._shap = shap_computer
        self._ranker = feature_ranker
        self._enricher = context_enricher
        self._waterfall = waterfall_viz
        self._bar_chart = bar_chart_viz
        self._timeline = line_graph_viz
        self._exporter = exporter
        self._root = project_root

    def run(self) -> Dict[str, Any]:
        """Execute all pipeline steps and return summary dict.

        Returns:
            Summary dict with sample counts, top features, charts, duration.
        """
        t0 = time.time()
        cfg = self._config

        logger.info("═══════════════════════════════════════════════════")
        logger.info("  Phase 5 Explanation Engine (SOLID)")
        logger.info("═══════════════════════════════════════════════════")

        # 1. Reproducibility seeds
        np.random.seed(cfg.random_state)  # noqa: NPY002
        tf.random.set_seed(cfg.random_state)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        rng = np.random.default_rng(cfg.random_state)

        # 2. Hardware detection
        hw_info = _detect_hardware()
        git_commit = _get_git_commit()

        # 3. Verify Phase 2/3/4 artifacts (SHA-256)
        p2_metadata, p3_metadata, _ = self._reader.verify_all()

        # 4. Rebuild classification model
        model = self._reader.rebuild_model(p2_metadata, p3_metadata)

        # 5. Load risk report + baseline
        risk_report = self._reader.load_risk_report()
        baseline = self._reader.load_baseline()

        # 6. Filter non-NORMAL samples
        filtered, level_counts = self._filter.filter(risk_report["sample_assessments"], rng)

        # 7. Prepare SHAP background + explanation data
        train_path = self._root / cfg.phase1_train
        test_path = self._root / cfg.phase1_test
        sample_indices = [s["sample_index"] for s in filtered]

        background = self._shap.prepare_background(train_path, rng)
        X_explain, _, feature_names = self._shap.prepare_explanation_data(test_path, sample_indices)

        # 8. Compute SHAP values
        shap_values = self._shap.compute(model, background, X_explain)

        # 9. Rank feature importance
        importance_df, top_features = self._ranker.rank(shap_values, feature_names)

        # 10. Enrich samples with context + explanations
        enriched = self._enricher.enrich(filtered, shap_values, feature_names)

        # 11. Generate visualizations
        charts_dir = self._root / cfg.output_dir / cfg.charts_dir
        chart_files = self._generate_all_charts(
            enriched,
            shap_values,
            feature_names,
            importance_df,
            baseline["baseline_threshold"],
            charts_dir,
        )

        duration_s = time.time() - t0

        # 12. Export artifacts
        logger.info("── Exporting Phase 5 artifacts ──")
        artifact_hashes: Dict[str, str] = {}

        _, sha = self._exporter.export_shap_values(shap_values, feature_names, cfg.shap_values_file)
        artifact_hashes[cfg.shap_values_file] = sha

        _, sha = self._exporter.export_explanation_report(
            enriched,
            importance_df,
            level_counts,
            cfg.top_features,
            git_commit,
            cfg.explanation_report_file,
        )
        artifact_hashes[cfg.explanation_report_file] = sha

        self._exporter.export_metadata(
            enriched,
            level_counts,
            chart_files,
            hw_info,
            duration_s,
            git_commit,
            artifact_hashes,
            cfg.background_samples,
            cfg.metadata_file,
        )

        # 13. Generate markdown report
        report_md = render_explanation_report(
            enriched_samples=enriched,
            importance_df=importance_df,
            level_counts=level_counts,
            chart_files=chart_files,
            baseline_threshold=baseline["baseline_threshold"],
            hw_info=hw_info,
            duration_s=duration_s,
            git_commit=git_commit,
            config=cfg,
        )
        report_dir = self._root / "results" / "phase0_analysis"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "report_section_explanation.md"
        report_path.write_text(report_md)
        logger.info("  Report saved: %s", report_path.name)

        logger.info("═══════════════════════════════════════════════════")
        logger.info("  Phase 5 complete — %.2fs", duration_s)
        logger.info(
            "  Explained: %d samples, %d charts",
            len(enriched),
            len(chart_files),
        )
        logger.info("═══════════════════════════════════════════════════")

        return {
            "samples_explained": len(enriched),
            "level_counts": level_counts,
            "top_features": top_features,
            "charts_generated": chart_files,
            "duration_s": round(duration_s, 2),
        }

    def _generate_all_charts(
        self,
        enriched_samples: List[Dict[str, Any]],
        shap_values: np.ndarray,
        feature_names: List[str],
        importance_df: Any,
        baseline_threshold: float,
        charts_dir: Path,
    ) -> List[str]:
        """Generate all visualization charts using injected visualizers."""
        logger.info("── Generating visualizations ──")
        charts_dir.mkdir(parents=True, exist_ok=True)
        generated: List[str] = []
        cfg = self._config

        # 1. Feature importance bar chart
        bar_path = charts_dir / "feature_importance.png"
        self._bar_chart.plot({"importance_df": importance_df}, bar_path)
        generated.append("feature_importance.png")
        logger.info("  Saved: feature_importance.png")

        # 2. Waterfall charts for CRITICAL + HIGH samples
        crit_high = [
            (i, s)
            for i, s in enumerate(enriched_samples)
            if s["risk_level"] in ("CRITICAL", "HIGH")
        ]
        for idx, (shap_idx, sample) in enumerate(crit_high[: cfg.max_waterfall_charts]):
            if shap_idx >= len(shap_values):
                break
            fname = f"waterfall_{sample['sample_index']}.png"
            self._waterfall.plot(
                {
                    "sample": sample,
                    "shap_vals": shap_values[shap_idx],
                    "feature_names": feature_names,
                    "baseline_threshold": baseline_threshold,
                },
                charts_dir / fname,
            )
            generated.append(fname)
        logger.info(
            "  Saved: %d waterfall charts",
            min(len(crit_high), cfg.max_waterfall_charts),
        )

        # 3. Timeline charts for first N incidents
        timeline_samples = [s for s in enriched_samples if s["risk_level"] in ("CRITICAL", "HIGH")]
        for idx, sample in enumerate(timeline_samples[: cfg.max_timeline_charts]):
            center = sample["sample_index"]
            window_scores = []
            for s in enriched_samples:
                if abs(s["sample_index"] - center) < _TIMESTEPS:
                    window_scores.append((s["sample_index"], s["anomaly_score"]))
            window_scores.sort()

            if len(window_scores) < 3:
                scores = [sample["anomaly_score"]] * _TIMESTEPS
            else:
                scores = [s[1] for s in window_scores[:_TIMESTEPS]]

            fname = f"timeline_{sample['sample_index']}.png"
            self._timeline.plot(
                {
                    "anomaly_scores": scores,
                    "baseline_threshold": baseline_threshold,
                    "incident_id": sample["sample_index"],
                },
                charts_dir / fname,
            )
            generated.append(fname)
        logger.info(
            "  Saved: %d timeline charts",
            min(len(timeline_samples), cfg.max_timeline_charts),
        )

        return generated


def main() -> None:
    """Entry point: construct components from config, run pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    config_path = PROJECT_ROOT / "config" / "phase5_config.yaml"
    config = Phase5Config.from_yaml(config_path)

    reader = Phase4ArtifactReader(
        project_root=PROJECT_ROOT,
        phase4_dir=config.phase4_dir,
        phase4_metadata=config.phase4_metadata,
        phase3_dir=config.phase3_dir,
        phase3_metadata=config.phase3_metadata,
        phase2_dir=config.phase2_dir,
        phase2_metadata=config.phase2_metadata,
        label_column=config.label_column,
    )
    alert_filter = AlertFilter(max_samples=config.max_explain_samples)
    shap_computer = SHAPComputer(
        n_background=config.background_samples,
        label_column=config.label_column,
    )
    feature_ranker = FeatureImportanceRanker(top_k=config.top_features)
    explanation_gen = ExplanationGenerator(templates=config.explanation_templates)
    context_enricher = ContextEnricher(explanation_generator=explanation_gen)
    waterfall_viz = WaterfallVisualizer(top_k=config.top_features)
    bar_chart_viz = BarChartVisualizer(
        top_k=config.top_features,
        biometric_columns=frozenset(config.biometric_columns),
    )
    line_graph_viz = LineGraphVisualizer()
    exporter = ExplanationExporter(output_dir=PROJECT_ROOT / config.output_dir)

    pipeline = ExplanationPipeline(
        config=config,
        artifact_reader=reader,
        alert_filter=alert_filter,
        shap_computer=shap_computer,
        feature_ranker=feature_ranker,
        context_enricher=context_enricher,
        waterfall_viz=waterfall_viz,
        bar_chart_viz=bar_chart_viz,
        line_graph_viz=line_graph_viz,
        exporter=exporter,
        project_root=PROJECT_ROOT,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
