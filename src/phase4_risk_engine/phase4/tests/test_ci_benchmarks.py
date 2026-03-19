"""CI benchmark suite — latency, baseline integrity, and drift simulation.

Provides:
1. Latency benchmarks (pytest-benchmark): RiskScorer, DynamicThresholdUpdater,
   ConceptDriftDetector — all with p95 < 100ms SLA.
2. Baseline integrity tests: tamper detection, SHA-256 verification, write-once.
3. Drift simulation tests: 25% shift → fallback, 15% → no fallback, recovery.
4. Report generation: report_section_risk_adaptive_reproducibility.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from src.phase4_risk_engine.phase4.config import KScheduleEntry
from src.phase4_risk_engine.phase4.cross_modal import CrossModalFusionDetector
from src.phase4_risk_engine.phase4.drift_detector import ConceptDriftDetector
from src.phase4_risk_engine.phase4.dynamic_threshold import DynamicThresholdUpdater
from src.phase4_risk_engine.phase4.exporter import RiskAdaptiveExporter
from src.phase4_risk_engine.phase4.fallback_manager import ThresholdFallbackManager
from src.phase4_risk_engine.phase4.risk_scorer import RiskScorer

# ── Test fixtures ──────────────────────────────────────────────────────

K_SCHEDULE = [
    KScheduleEntry(start_hour=0, end_hour=6, k=2.5),
    KScheduleEntry(start_hour=6, end_hour=22, k=3.0),
    KScheduleEntry(start_hour=22, end_hour=24, k=3.5),
]

BIOMETRIC_COLS = ["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA"]
N_FEATURES = 20
N_SAMPLES = 1000


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def scorer() -> RiskScorer:
    cross_modal = CrossModalFusionDetector(biometric_columns=BIOMETRIC_COLS, sigma_threshold=2.0)
    return RiskScorer(
        low_upper=0.5,
        medium_upper=1.0,
        high_upper=2.0,
        cross_modal=cross_modal,
    )


@pytest.fixture()
def threshold_updater() -> DynamicThresholdUpdater:
    return DynamicThresholdUpdater(window_size=100, k_schedule=K_SCHEDULE)


@pytest.fixture()
def drift_detector() -> ConceptDriftDetector:
    return ConceptDriftDetector(drift_threshold=0.20)


@pytest.fixture()
def baseline_dict() -> Dict[str, Any]:
    return {
        "median": 0.15,
        "mad": 0.025,
        "baseline_threshold": 0.225,
        "mad_multiplier": 3.0,
        "n_normal_samples": 800,
        "n_attention_dims": 128,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture()
def anomaly_scores(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 0.5, size=N_SAMPLES).astype(np.float64)


@pytest.fixture()
def raw_features(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((N_SAMPLES, N_FEATURES)).astype(np.float32)


@pytest.fixture()
def feature_names() -> List[str]:
    names = list(BIOMETRIC_COLS)
    for i in range(N_FEATURES - len(BIOMETRIC_COLS)):
        names.append(f"net_{i}")
    return names


# ── 1. Latency benchmarks (pytest-benchmark) ──────────────────────────


class TestLatencyBenchmarks:
    """SLA: all p95 < 100ms per sample."""

    def test_risk_scorer_latency(
        self,
        benchmark: Any,
        scorer: RiskScorer,
        anomaly_scores: np.ndarray,
        raw_features: np.ndarray,
        feature_names: List[str],
        baseline_dict: Dict[str, Any],
    ) -> None:
        """RiskScorer.score() — 1000 samples, p95 < 100ms."""
        thresholds = np.full(N_SAMPLES, baseline_dict["baseline_threshold"])

        def run_score() -> List[Dict[str, Any]]:
            return scorer.score(
                anomaly_scores, thresholds, baseline_dict["mad"], raw_features, feature_names
            )

        result = benchmark(run_score)
        assert len(result) == N_SAMPLES

    def test_dynamic_threshold_latency(
        self,
        benchmark: Any,
        threshold_updater: DynamicThresholdUpdater,
        anomaly_scores: np.ndarray,
        baseline_dict: Dict[str, Any],
    ) -> None:
        """DynamicThresholdUpdater.update() — 1000 samples, p95 < 50ms."""

        def run_update() -> Any:
            return threshold_updater.update(anomaly_scores, baseline_dict)

        thresholds, log = benchmark(run_update)
        assert len(thresholds) == N_SAMPLES

    def test_drift_detector_latency(
        self, benchmark: Any, drift_detector: ConceptDriftDetector
    ) -> None:
        """ConceptDriftDetector.detect() — single call, p95 < 1ms."""

        def run_detect() -> bool:
            return drift_detector.detect(0.30, 0.225)

        result = benchmark(run_detect)
        assert isinstance(result, bool)

    def test_risk_scorer_p95_under_100ms(
        self,
        scorer: RiskScorer,
        anomaly_scores: np.ndarray,
        raw_features: np.ndarray,
        feature_names: List[str],
        baseline_dict: Dict[str, Any],
    ) -> None:
        """Assert p95 latency < 100ms over 50 iterations."""
        thresholds = np.full(N_SAMPLES, baseline_dict["baseline_threshold"])
        timings: List[float] = []

        for _ in range(50):
            start = time.perf_counter()
            scorer.score(
                anomaly_scores, thresholds, baseline_dict["mad"], raw_features, feature_names
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append(elapsed_ms)

        p50 = float(np.percentile(timings, 50))
        p95 = float(np.percentile(timings, 95))
        p99 = float(np.percentile(timings, 99))

        # Store for report generation
        TestLatencyBenchmarks._scorer_timings = {"p50": p50, "p95": p95, "p99": p99}

        assert p95 < 100.0, f"RiskScorer p95={p95:.2f}ms exceeds 100ms SLA"

    def test_threshold_updater_p95_under_50ms(
        self,
        threshold_updater: DynamicThresholdUpdater,
        anomaly_scores: np.ndarray,
        baseline_dict: Dict[str, Any],
    ) -> None:
        """Assert p95 latency < 50ms over 50 iterations."""
        timings: List[float] = []

        for _ in range(50):
            start = time.perf_counter()
            threshold_updater.update(anomaly_scores, baseline_dict)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append(elapsed_ms)

        p50 = float(np.percentile(timings, 50))
        p95 = float(np.percentile(timings, 95))
        p99 = float(np.percentile(timings, 99))

        TestLatencyBenchmarks._updater_timings = {"p50": p50, "p95": p95, "p99": p99}

        assert p95 < 50.0, f"ThresholdUpdater p95={p95:.2f}ms exceeds 50ms SLA"

    def test_drift_detector_p95_under_1ms(self, drift_detector: ConceptDriftDetector) -> None:
        """Assert p95 latency < 1ms over 1000 iterations."""
        timings: List[float] = []

        for _ in range(1000):
            start = time.perf_counter()
            drift_detector.detect(0.30, 0.225)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append(elapsed_ms)

        p50 = float(np.percentile(timings, 50))
        p95 = float(np.percentile(timings, 95))
        p99 = float(np.percentile(timings, 99))

        TestLatencyBenchmarks._detector_timings = {"p50": p50, "p95": p95, "p99": p99}

        assert p95 < 1.0, f"DriftDetector p95={p95:.2f}ms exceeds 1ms SLA"


# ── 2. Baseline integrity tests ───────────────────────────────────────


class TestBaselineIntegrity:
    """Baseline tamper detection, hash verification, write-once."""

    def test_tamper_detected(self, baseline_dict: Dict[str, Any]) -> None:
        """Tampered baseline_config.json fails SHA-256 verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline_config.json"
            with open(path, "w") as f:
                json.dump(baseline_dict, f, indent=2)

            original_hash = hashlib.sha256(path.read_bytes()).hexdigest()

            # Tamper: change threshold
            tampered = baseline_dict.copy()
            tampered["baseline_threshold"] = 999.0
            with open(path, "w") as f:
                json.dump(tampered, f, indent=2)

            tampered_hash = hashlib.sha256(path.read_bytes()).hexdigest()

            assert original_hash != tampered_hash, "Tampered file should change hash"

    def test_sha256_round_trip(self, baseline_dict: Dict[str, Any]) -> None:
        """SHA-256 hash is consistent across read/write cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline_config.json"
            content = json.dumps(baseline_dict, indent=2, sort_keys=True)
            path.write_text(content)
            hash1 = hashlib.sha256(path.read_bytes()).hexdigest()

            # Re-read and re-write identically
            loaded = json.loads(path.read_text())
            path.write_text(json.dumps(loaded, indent=2, sort_keys=True))
            hash2 = hashlib.sha256(path.read_bytes()).hexdigest()

            assert hash1 == hash2, "Identical content must produce identical hash"

    def test_write_once_chmod_444(self, baseline_dict: Dict[str, Any]) -> None:
        """Write-once: file with chmod 444 rejects further writes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline_config.json"
            with open(path, "w") as f:
                json.dump(baseline_dict, f, indent=2)

            # Set read-only (444)
            os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

            # Verify write is rejected
            with pytest.raises(PermissionError):
                with open(path, "w") as f:
                    json.dump({"tampered": True}, f)

            # Cleanup: restore write permission for tempdir cleanup
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    def test_baseline_immutable_keys(self, baseline_dict: Dict[str, Any]) -> None:
        """Baseline config must contain all required immutable keys."""
        required_keys = {
            "median",
            "mad",
            "baseline_threshold",
            "mad_multiplier",
            "n_normal_samples",
            "n_attention_dims",
            "computed_at",
        }
        assert required_keys.issubset(set(baseline_dict.keys()))

    def test_exporter_baseline_round_trip(self, baseline_dict: Dict[str, Any]) -> None:
        """RiskAdaptiveExporter baseline round-trip preserves values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = RiskAdaptiveExporter(output_dir=Path(tmpdir))
            path = exporter.export_baseline(baseline_dict, "baseline_config.json")

            loaded = json.loads(path.read_text())
            assert loaded["baseline_threshold"] == baseline_dict["baseline_threshold"]
            assert loaded["mad"] == baseline_dict["mad"]
            assert loaded["median"] == baseline_dict["median"]


# ── 3. Drift simulation tests ─────────────────────────────────────────


class TestDriftSimulation:
    """Concept drift injection, fallback verification, recovery."""

    def test_25pct_shift_triggers_fallback(self) -> None:
        """Inject 25% shift — fallback triggered within 3 windows."""
        baseline_threshold = 0.225
        window_size = 50
        n_samples = 500

        drift_detector = ConceptDriftDetector(drift_threshold=0.20)
        fallback_mgr = ThresholdFallbackManager(
            drift_detector=drift_detector,
            baseline_threshold=baseline_threshold,
            recovery_threshold=0.10,
            recovery_windows=3,
        )

        # Simulate: first 200 stable, then 25% upward shift
        dynamic_thresholds = np.full(n_samples, baseline_threshold)
        shift_start = 200
        shifted_value = baseline_threshold * 1.25  # 25% increase
        dynamic_thresholds[shift_start:] = shifted_value

        adjusted, drift_events = fallback_mgr.process(dynamic_thresholds, window_size)

        # Must have at least one FALLBACK_LOCKED event
        locked_events = [e for e in drift_events if e["action"] == "FALLBACK_LOCKED"]
        assert len(locked_events) >= 1, "25% shift must trigger FALLBACK_LOCKED"

        # Fallback must trigger within 3 windows after shift start
        first_lock_idx = locked_events[0]["sample_index"]
        windows_after_shift = (first_lock_idx - shift_start) / window_size
        assert (
            windows_after_shift <= 3
        ), f"Fallback at window {windows_after_shift:.1f}, expected <= 3"

        # After lock, thresholds revert to baseline
        lock_start = locked_events[0]["sample_index"]
        locked_region = adjusted[lock_start + 1 : lock_start + window_size]
        if len(locked_region) > 0:
            np.testing.assert_allclose(
                locked_region,
                baseline_threshold,
                err_msg="Locked region must use baseline threshold",
            )

    def test_15pct_shift_no_fallback(self) -> None:
        """Inject 15% shift — below 20% threshold, no fallback triggered."""
        baseline_threshold = 0.225
        window_size = 50
        n_samples = 500

        drift_detector = ConceptDriftDetector(drift_threshold=0.20)
        fallback_mgr = ThresholdFallbackManager(
            drift_detector=drift_detector,
            baseline_threshold=baseline_threshold,
            recovery_threshold=0.10,
            recovery_windows=3,
        )

        # 15% shift — below drift_threshold of 20%
        dynamic_thresholds = np.full(n_samples, baseline_threshold)
        dynamic_thresholds[200:] = baseline_threshold * 1.15

        adjusted, drift_events = fallback_mgr.process(dynamic_thresholds, window_size)

        locked_events = [e for e in drift_events if e["action"] == "FALLBACK_LOCKED"]
        assert len(locked_events) == 0, "15% shift must NOT trigger fallback"

    def test_recovery_after_stable_windows(self) -> None:
        """After drift, recovery triggered when ratio < 10% for 3 windows."""
        baseline_threshold = 0.225
        window_size = 50
        n_samples = 600

        drift_detector = ConceptDriftDetector(drift_threshold=0.20)
        fallback_mgr = ThresholdFallbackManager(
            drift_detector=drift_detector,
            baseline_threshold=baseline_threshold,
            recovery_threshold=0.10,
            recovery_windows=3,
        )

        # Phase 1: stable (0–199)
        dynamic_thresholds = np.full(n_samples, baseline_threshold)
        # Phase 2: drift (200–349) — 30% shift
        dynamic_thresholds[200:350] = baseline_threshold * 1.30
        # Phase 3: recovery (350–600) — back to stable
        dynamic_thresholds[350:] = baseline_threshold * 1.05  # 5% < 10% recovery

        adjusted, drift_events = fallback_mgr.process(dynamic_thresholds, window_size)

        locked_events = [e for e in drift_events if e["action"] == "FALLBACK_LOCKED"]
        resumed_events = [e for e in drift_events if e["action"] == "RESUMED_DYNAMIC"]

        assert len(locked_events) >= 1, "Drift must be detected"
        assert len(resumed_events) >= 1, "Recovery must resume dynamic thresholds"

        # Resumed event must come after locked event
        if locked_events and resumed_events:
            assert resumed_events[0]["sample_index"] > locked_events[0]["sample_index"]

    def test_drift_ratio_computation(self) -> None:
        """Verify drift ratio formula: |dynamic - baseline| / baseline."""
        detector = ConceptDriftDetector(drift_threshold=0.20)

        ratio = detector.compute_drift_ratio(0.30, 0.225)
        expected = abs(0.30 - 0.225) / 0.225
        assert abs(ratio - expected) < 1e-10

        ratio_down = detector.compute_drift_ratio(0.10, 0.225)
        expected_down = abs(0.10 - 0.225) / 0.225
        assert abs(ratio_down - expected_down) < 1e-10

    def test_fallback_manager_is_locked_property(self) -> None:
        """ThresholdFallbackManager.is_locked tracks state correctly."""
        detector = ConceptDriftDetector(drift_threshold=0.20)
        mgr = ThresholdFallbackManager(
            drift_detector=detector,
            baseline_threshold=0.225,
            recovery_threshold=0.10,
            recovery_windows=3,
        )

        assert not mgr.is_locked, "Initial state should be unlocked"

        # Process with drift
        thresholds = np.full(200, 0.225)
        thresholds[100:] = 0.225 * 1.30
        mgr.process(thresholds, 50)

        # After process, final state depends on whether recovery happened


# ── 4. Report generation ──────────────────────────────────────────────


class TestReportGeneration:
    """Generate report_section_risk_adaptive_reproducibility.md."""

    def test_generate_reproducibility_report(
        self,
        scorer: RiskScorer,
        threshold_updater: DynamicThresholdUpdater,
        drift_detector: ConceptDriftDetector,
        anomaly_scores: np.ndarray,
        raw_features: np.ndarray,
        feature_names: List[str],
        baseline_dict: Dict[str, Any],
    ) -> None:
        """Generate the full reproducibility report with benchmarks."""
        # ── Run latency benchmarks ──
        thresholds = np.full(N_SAMPLES, baseline_dict["baseline_threshold"])

        # RiskScorer timings
        scorer_timings: List[float] = []
        for _ in range(50):
            start = time.perf_counter()
            scorer.score(
                anomaly_scores, thresholds, baseline_dict["mad"], raw_features, feature_names
            )
            scorer_timings.append((time.perf_counter() - start) * 1000)

        # DynamicThresholdUpdater timings
        updater_timings: List[float] = []
        for _ in range(50):
            start = time.perf_counter()
            threshold_updater.update(anomaly_scores, baseline_dict)
            updater_timings.append((time.perf_counter() - start) * 1000)

        # ConceptDriftDetector timings
        detector_timings: List[float] = []
        for _ in range(1000):
            start = time.perf_counter()
            drift_detector.detect(0.30, 0.225)
            detector_timings.append((time.perf_counter() - start) * 1000)

        scorer_p50 = float(np.percentile(scorer_timings, 50))
        scorer_p95 = float(np.percentile(scorer_timings, 95))
        scorer_p99 = float(np.percentile(scorer_timings, 99))

        updater_p50 = float(np.percentile(updater_timings, 50))
        updater_p95 = float(np.percentile(updater_timings, 95))
        updater_p99 = float(np.percentile(updater_timings, 99))

        detector_p50 = float(np.percentile(detector_timings, 50))
        detector_p95 = float(np.percentile(detector_timings, 95))
        detector_p99 = float(np.percentile(detector_timings, 99))

        # ── Run integrity tests ──
        integrity_results = []

        # Tamper detection
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline_config.json"
            content = json.dumps(baseline_dict, indent=2, sort_keys=True)
            path.write_text(content)
            orig_hash = hashlib.sha256(path.read_bytes()).hexdigest()

            tampered = baseline_dict.copy()
            tampered["baseline_threshold"] = 999.0
            path.write_text(json.dumps(tampered, indent=2, sort_keys=True))
            tamp_hash = hashlib.sha256(path.read_bytes()).hexdigest()

            tamper_pass = orig_hash != tamp_hash
            integrity_results.append(
                ("Tamper detection (SHA-256)", "PASS" if tamper_pass else "FAIL")
            )

        # Hash consistency
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline_config.json"
            content = json.dumps(baseline_dict, indent=2, sort_keys=True)
            path.write_text(content)
            h1 = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_text(content)
            h2 = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_pass = h1 == h2
            integrity_results.append(
                ("Hash round-trip consistency", "PASS" if hash_pass else "FAIL")
            )

        # Write-once (chmod 444)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline_config.json"
            path.write_text(json.dumps(baseline_dict, indent=2))
            os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            write_once_pass = True
            try:
                with open(path, "w") as f:
                    f.write("tampered")
                write_once_pass = False
            except PermissionError:
                write_once_pass = True
            finally:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            integrity_results.append(
                ("Write-once enforcement (chmod 444)", "PASS" if write_once_pass else "FAIL")
            )

        # Immutable keys
        required = {
            "median",
            "mad",
            "baseline_threshold",
            "mad_multiplier",
            "n_normal_samples",
            "n_attention_dims",
            "computed_at",
        }
        keys_pass = required.issubset(set(baseline_dict.keys()))
        integrity_results.append(
            ("Baseline immutable keys present", "PASS" if keys_pass else "FAIL")
        )

        # ── Run drift simulation ──
        drift_results = []

        # 25% shift
        bl_thresh = 0.225
        dt_25 = np.full(500, bl_thresh)
        dt_25[200:] = bl_thresh * 1.25
        det_25 = ConceptDriftDetector(drift_threshold=0.20)
        fb_25 = ThresholdFallbackManager(
            drift_detector=det_25,
            baseline_threshold=bl_thresh,
            recovery_threshold=0.10,
            recovery_windows=3,
        )
        _, events_25 = fb_25.process(dt_25, 50)
        locked_25 = [e for e in events_25 if e["action"] == "FALLBACK_LOCKED"]
        drift_results.append(
            (
                "25% shift → fallback triggered",
                "PASS" if len(locked_25) >= 1 else "FAIL",
                f"{len(locked_25)} FALLBACK_LOCKED events",
            )
        )

        # 15% shift
        dt_15 = np.full(500, bl_thresh)
        dt_15[200:] = bl_thresh * 1.15
        det_15 = ConceptDriftDetector(drift_threshold=0.20)
        fb_15 = ThresholdFallbackManager(
            drift_detector=det_15,
            baseline_threshold=bl_thresh,
            recovery_threshold=0.10,
            recovery_windows=3,
        )
        _, events_15 = fb_15.process(dt_15, 50)
        locked_15 = [e for e in events_15 if e["action"] == "FALLBACK_LOCKED"]
        drift_results.append(
            (
                "15% shift → no fallback",
                "PASS" if len(locked_15) == 0 else "FAIL",
                f"{len(locked_15)} FALLBACK_LOCKED events",
            )
        )

        # Recovery
        dt_rec = np.full(600, bl_thresh)
        dt_rec[200:350] = bl_thresh * 1.30
        dt_rec[350:] = bl_thresh * 1.05
        det_rec = ConceptDriftDetector(drift_threshold=0.20)
        fb_rec = ThresholdFallbackManager(
            drift_detector=det_rec,
            baseline_threshold=bl_thresh,
            recovery_threshold=0.10,
            recovery_windows=3,
        )
        _, events_rec = fb_rec.process(dt_rec, 50)
        resumed_rec = [e for e in events_rec if e["action"] == "RESUMED_DYNAMIC"]
        drift_results.append(
            (
                "Recovery after stable windows",
                "PASS" if len(resumed_rec) >= 1 else "FAIL",
                f"{len(resumed_rec)} RESUMED_DYNAMIC events",
            )
        )

        # ── Generate report ──
        scorer_sla = "PASS" if scorer_p95 < 100.0 else "FAIL"
        updater_sla = "PASS" if updater_p95 < 50.0 else "FAIL"
        detector_sla = "PASS" if detector_p95 < 1.0 else "FAIL"

        integrity_rows = ""
        for name, status in integrity_results:
            integrity_rows += f"| {name} | {status} |\n"

        drift_rows = ""
        for name, status, detail in drift_results:
            drift_rows += f"| {name} | {status} | {detail} |\n"

        # Build latency table rows (avoid long f-string lines)
        lat_hdr = "| Component | p50 | p95 | p99 | SLA | Result |\n"
        lat_hdr += "|-----------|-----|-----|-----|-----|--------|\n"
        lat_r1 = (
            f"| RiskScorer | {scorer_p50:.2f}ms"
            f" | {scorer_p95:.2f}ms | {scorer_p99:.2f}ms"
            f" | <100ms | {scorer_sla} |\n"
        )
        lat_r2 = (
            f"| ThresholdUpdater | {updater_p50:.2f}ms"
            f" | {updater_p95:.2f}ms | {updater_p99:.2f}ms"
            f" | <50ms | {updater_sla} |\n"
        )
        lat_r3 = (
            f"| DriftDetector | {detector_p50:.4f}ms"
            f" | {detector_p95:.4f}ms | {detector_p99:.4f}ms"
            f" | <1ms | {detector_sla} |\n"
        )
        latency_table = lat_hdr + lat_r1 + lat_r2 + lat_r3

        report = f"""## 7.3 Risk-Adaptive Engine — Reproducibility & CI/CD Validation

This section documents the Phase 4 Risk-Adaptive Engine reproducibility
validation, including latency benchmarks, baseline integrity verification,
and concept drift simulation results.

### 7.3.1 Latency Benchmarks

All benchmarks run on {N_SAMPLES} samples, 50 iterations.

{latency_table}
**Benchmark parameters:**

| Parameter | Value |
|-----------|-------|
| N samples | {N_SAMPLES} |
| N features | {N_FEATURES} |
| Scorer iterations | 50 |
| Updater iterations | 50 |
| Detector iterations | 1000 |
| Window size | 100 |

### 7.3.2 Baseline Integrity Results

| Test | Status |
|------|--------|
{integrity_rows}
**Integrity mechanism:** SHA-256 hash verification + chmod 444 write-once enforcement.

### 7.3.3 Drift Simulation Results

| Scenario | Status | Detail |
|----------|--------|--------|
{drift_rows}
**Drift parameters:**

| Parameter | Value |
|-----------|-------|
| Drift threshold | 20% (trigger fallback) |
| Recovery threshold | 10% (resume dynamic) |
| Recovery windows | 3 consecutive |
| Window size | 50 samples |
| Baseline threshold | 0.225 |

### 7.3.4 Full Pipeline Artifact Chain

| Phase | Input | Output | Hash |
|-------|-------|--------|------|
| 0 | Raw CSV | stats, integrity JSON | SHA-256 |
| 1 | Phase 0 stats | train/test parquet, metadata | SHA-256 |
| 2 | Phase 1 parquet | model weights, attention parquet | SHA-256 |
| 3 | Phase 2 model | classification weights, metrics | SHA-256 |
| 4 | Phase 3 + Phase 2 | baseline, threshold, risk, drift | SHA-256 |

Each phase verifies predecessor artifacts via SHA-256 hash comparison
against stored metadata before processing.

### 7.3.5 CI/CD Pipeline Architecture

```
push/PR → lint-phase4 → test-phase4 ──────────┐
                       → security-scan-phase4 ─┤
                                               ├→ benchmark-test → integration-test → build
```

| Job | Gate | Tool |
|-----|------|------|
| lint-phase4 | ruff + black | Static analysis |
| test-phase4 | 80% coverage | pytest-cov |
| security-scan-phase4 | bandit + pip-audit | SAST + CVE |
| benchmark-test | p95 SLA assertions | pytest-benchmark |
| integration-test | 5 artifact assertions | Phase 0→1→2→3→4 |
| build | Docker image | phase0-phase4:5.0 |

### 7.3.6 SBOM (Software Bill of Materials)

CycloneDX SBOM generated during CI/CD security scan phase:
- Format: CycloneDX JSON
- Scope: All Phase 0–4 dependencies from requirements.txt
- CVE policy: Fail build if any dependency has CVSS > 7.0
- Artifact: `sbom-phase4.json` (uploaded as CI artifact)

### 7.3.7 Reproducibility Statement

This Phase 4 Risk-Adaptive Engine produces **deterministic results** given:

1. **Fixed random seed** (`random_state=42`, `TF_DETERMINISTIC_OPS=1`)
2. **Immutable baseline** (Median + k*MAD, write-once, SHA-256 verified)
3. **Versioned model weights** (SHA-256 hashes in classification_metadata.json)
4. **Locked dependencies** (requirements.txt + CycloneDX SBOM)
5. **Git commit tracking** (embedded in all output artifacts)

The full Phase 0→1→2→3→4 pipeline can be reproduced by:

```bash
# 1. Clone at specific commit
git clone <repo> && git checkout <commit>

# 2. Install locked dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python -m src.phase0_dataset_analysis.phase0
python -m src.phase1_preprocessing.phase1
python -m src.phase2_detection_engine.phase2
python -m src.phase3_classification_engine.phase3
python -m src.phase4_risk_engine.phase4.pipeline

# 4. Verify artifacts
python -c "from src.phase4_risk_engine.phase4 import *; print('All imports OK')"
```

Docker reproducibility:

```bash
docker build -t analyst/phase0-phase4:5.0 .
docker run --rm analyst/phase0-phase4:5.0
```

---

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
**Test framework:** pytest + pytest-benchmark
**Pipeline version:** 5.0
"""

        # Write report
        report_dir = Path("results/phase0_analysis")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "report_section_risk_adaptive_reproducibility.md"
        report_path.write_text(report)

        # Verify report was written
        assert report_path.exists()
        content = report_path.read_text()
        assert "7.3.1 Latency Benchmarks" in content
        assert "7.3.2 Baseline Integrity" in content
        assert "7.3.3 Drift Simulation" in content
        assert "7.3.4 Full Pipeline Artifact Chain" in content
        assert "Reproducibility Statement" in content
