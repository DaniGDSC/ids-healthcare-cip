"""Streaming inference service — per-window model prediction + risk scoring.

Wraps the Phase 2.5 finetuned model and Phase 4 risk components into
a single callable service. In production, this runs as a long-lived
process consuming from the WindowBuffer.

Components loaded at startup (once):
  - CNN-BiLSTM-Attention model (Phase 2 backbone + Phase 2.5 head)
  - RiskScorer, CIARiskModifier, ClinicalImpactAssessor, AlertFatigueManager
  - AttentionAnomalyDetector (from Phase 2)

Per-window processing (~190ms):
  window (1, 20, 24) → model.predict() → risk score → attention anomaly
  → CIA → clinical → fatigue check → alert (if emitted)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf

from dashboard.streaming.feature_aligner import MODEL_FEATURES, N_FEATURES
from dashboard.streaming.window_buffer import WindowBuffer

logger = logging.getLogger(__name__)


class InferenceService:
    """Streaming inference with full risk scoring pipeline.

    Loads the model and all Phase 4 components at construction time.
    Call process_window() for each new window from the buffer.

    Args:
        project_root: Absolute path to project root.
        device_id: Default device identifier for CIA scoring.
    """

    def __init__(
        self,
        project_root: str | Path,
        device_id: str | None = None,
        database: Any = None,
    ) -> None:
        from config.production_loader import cfg

        self._root = Path(project_root)
        self._device_id = device_id or cfg("inference.default_device_id", "generic_iomt_sensor")
        self._db = database
        self._model: Optional[tf.keras.Model] = None
        self._scorer = None
        self._cia_modifier = None
        self._clinical_assessor = None
        self._fatigue_mgr = None
        self._attn_detector = None
        self._explainer = None
        self._calibrator = None
        self._baseline: Dict[str, Any] = {}
        self._inference_count = 0
        self._calibration_scores: list = []
        self._calibration_labels: list = []
        self._lock = threading.Lock()

        # Circuit breaker
        self._consecutive_failures = 0
        self._max_failures = int(cfg("inference.max_consecutive_failures", 3))
        self._circuit_open = False
        self._last_error: Optional[str] = None

    def load(self) -> None:
        """Load model and all scoring components. Call once at startup."""
        t0 = time.perf_counter()

        # Load Phase 2 metadata for architecture
        p2_meta_path = self._root / "data" / "phase2" / "detection_metadata.json"
        with open(p2_meta_path) as f:
            p2_meta = json.load(f)
        hp = p2_meta["hyperparameters"]

        # Load Phase 2.5 finetuned results for threshold
        ft_results_path = self._root / "data" / "phase2_5" / "finetuned_results.json"
        with open(ft_results_path) as f:
            ft_results = json.load(f)

        # Build model architecture
        from src.phase2_detection_engine.phase2.assembler import DetectionModelAssembler
        from src.phase2_detection_engine.phase2.attention_builder import AttentionBuilder
        from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
        from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder

        builders = [
            CNNBuilder(
                filters_1=hp["cnn_filters_1"], filters_2=hp["cnn_filters_2"],
                kernel_size=hp["cnn_kernel_size"], activation="relu", pool_size=2,
            ),
            BiLSTMBuilder(
                units_1=hp["bilstm_units_1"], units_2=hp["bilstm_units_2"],
                dropout_rate=hp["dropout_rate"],
            ),
            AttentionBuilder(units=hp["attention_units"]),
        ]
        det = DetectionModelAssembler(
            timesteps=hp["timesteps"], n_features=N_FEATURES, builders=builders,
        ).assemble()

        # Add classification head (matching Phase 2.5 architecture)
        x = tf.keras.layers.Dense(32, activation="relu", name="dense_head")(det.output)
        x = tf.keras.layers.Dropout(hp["dropout_rate"], name="drop_head")(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
        self._model = tf.keras.Model(det.input, x, name="inference_model")

        # Load finetuned weights
        weights_path = self._root / "data" / "phase2_5" / "finetuned_model.weights.h5"
        self._model.load_weights(str(weights_path))

        # Load Phase 4 baseline
        baseline_path = self._root / "data" / "phase4" / "baseline_config.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                self._baseline = json.load(f)
        else:
            self._baseline = {"median": 0.18, "mad": 0.025, "baseline_threshold": 0.255}

        # Initialize scoring components
        from src.phase4_risk_engine.phase4.risk_scorer import RiskScorer
        from src.phase4_risk_engine.phase4.cross_modal import CrossModalFusionDetector
        from src.phase4_risk_engine.phase4.cia_risk_modifier import CIARiskModifier
        from src.phase4_risk_engine.phase4.cia_threat_mapper import CIAThreatMapper
        from src.phase4_risk_engine.phase4.device_registry import DeviceRegistry
        from src.phase4_risk_engine.phase4.clinical_impact import ClinicalImpactAssessor
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager
        from src.phase2_detection_engine.phase2.attention_anomaly import AttentionAnomalyDetector

        biometric_cols = ["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"]

        self._scorer = RiskScorer(
            cross_modal=CrossModalFusionDetector(biometric_columns=biometric_cols),
        )
        self._cia_modifier = CIARiskModifier(
            threat_mapper=CIAThreatMapper(),
            device_registry=DeviceRegistry(),
        )
        self._clinical_assessor = ClinicalImpactAssessor(biometric_columns=biometric_cols)
        self._fatigue_mgr = AlertFatigueManager()
        self._attn_threshold = (
            self._baseline["median"]
            + self._baseline.get("mad_multiplier", 3.0) * self._baseline["mad"]
        )

        from src.production.score_calibrator import ScoreCalibrator
        self._calibrator = ScoreCalibrator()

        # Build multi-output model: score + backbone in single forward pass
        dense_head = self._model.get_layer("dense_head")
        backbone_output = dense_head.input  # attention context vector
        self._fast_model = tf.keras.Model(
            self._model.input,
            [self._model.output, backbone_output],
            name="fast_dual_output",
        )

        # Compile a single @tf.function that returns score + backbone + gradients
        # in ONE traced graph execution — eliminates Python overhead and enables
        # XLA fusion. The graph is compiled once on first call, then reused.
        @tf.function(reduce_retracing=True)
        def _infer_with_grads(x: tf.Tensor):
            with tf.GradientTape() as tape:
                tape.watch(x)
                pred, backbone = self._fast_model(x, training=False)
            grads = tape.gradient(pred, x)
            return pred, backbone, grads

        self._infer_with_grads = _infer_with_grads

        # Warm up the compiled function (trigger tracing once at load)
        _dummy = tf.zeros((1, hp["timesteps"], N_FEATURES), dtype=tf.float32)
        self._infer_with_grads(_dummy)

        elapsed = time.perf_counter() - t0
        logger.info(
            "InferenceService loaded: %d params, %.2fs (compiled inference)",
            self._model.count_params(), elapsed,
        )

    def process_window(
        self,
        window: np.ndarray,
        raw_features: Optional[np.ndarray] = None,
        attack_category: str = "unknown",
    ) -> Dict[str, Any]:
        """Process a single window with tiered inference.

        Tier 1 (every flow, ~15ms): single forward pass → score + attention
        Tier 2 (HIGH+ only, +25ms): GradientTape → explanation

        Args:
            window: Input array of shape (1, 20, 24).
            raw_features: Unscaled features for cross-modal check, shape (24,).
            attack_category: Attack type if known (for CIA scoring).

        Returns:
            Dict with risk_level, clinical_severity, alert_emit, explanation, etc.
        """
        t0 = time.perf_counter()

        # ── Circuit breaker check ──
        if self._circuit_open:
            # Auto-reset attempt every N flows
            self._consecutive_failures += 1
            reset_after = int(
                __import__("config.production_loader", fromlist=["cfg"]).cfg(
                    "inference.circuit_breaker_reset_after", 10
                )
            )
            if self._consecutive_failures % reset_after == 0:
                logger.info("Circuit breaker: attempting recovery")
                self._circuit_open = False
            else:
                return self._fallback_result(t0)

        # ── Input validation ──
        if window.ndim != 3 or window.shape[1:] != (20, N_FEATURES):
            raise ValueError(
                f"Expected window shape (batch, 20, {N_FEATURES}), got {window.shape}"
            )
        if not np.all(np.isfinite(window)):
            logger.warning("Window contains NaN/Inf — replacing with 0")
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            return self._process_window_inner(window, raw_features, attack_category, t0)
        except Exception as exc:
            with self._lock:
                self._consecutive_failures += 1
                self._last_error = str(exc)
            logger.error(
                "Inference failed (%d/%d): %s",
                self._consecutive_failures, self._max_failures, exc,
            )
            if self._consecutive_failures >= self._max_failures:
                self._circuit_open = True
                logger.critical("Circuit breaker OPEN — inference disabled after %d failures", self._max_failures)
            return self._fallback_result(t0)

    def _fallback_result(self, t0: float) -> Dict[str, Any]:
        """Return visible degradation signal when inference fails.

        Returns ADVISORY (not NORMAL) so clinical staff knows the system
        is degraded. Hiding failures as NORMAL violates IEC 62443 and
        FDA pre-market cybersecurity guidance.
        """
        return {
            "sample_index": self._inference_count,
            "anomaly_score": 0.0,
            "threshold": 0.0,
            "distance": 0.0,
            "risk_level": "LOW",
            "attention_flag": False,
            "clinical_severity": 2,
            "clinical_severity_name": "ADVISORY",
            "response_time_minutes": 480,
            "device_action": "none",
            "patient_safety_flag": False,
            "clinical_rationale": "SYSTEM DEGRADED: Inference engine unavailable. "
                                  "Manual monitoring recommended until system recovers.",
            "scenario": "normal_monitoring",
            "cia_scores": {"C": 0, "I": 0, "A": 0},
            "cia_max_dimension": "I",
            "attack_category": "unknown",
            "explanation": {"level": "none", "top_features": [], "timestep_importance": []},
            "percentile": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "inference_failed": True,
        }

    def _process_window_inner(
        self,
        window: np.ndarray,
        raw_features: Optional[np.ndarray],
        attack_category: str,
        t0: float,
    ) -> Dict[str, Any]:
        """Inner processing — separated for circuit breaker wrapping."""
        # ── Single compiled call: score + backbone + gradients ──
        window_tensor = tf.constant(window, dtype=tf.float32)
        pred_out, backbone_out, grads = self._infer_with_grads(window_tensor)
        score = float(pred_out.numpy().ravel()[0])
        if not 0.0 <= score <= 1.0:
            logger.warning("Score %.6f outside [0,1] — clipping", score)
            score = float(np.clip(score, 0.0, 1.0))
        attn_magnitude = float(np.linalg.norm(backbone_out.numpy()))
        attn_flag = bool(attn_magnitude > self._attn_threshold)

        # 3. Calibrated risk scoring
        # Raw sigmoid scores fall in a narrow range (0.88-0.98) where benign
        # and attack overlap. The calibrator maps raw scores to percentile
        # ranks within the observed benign distribution, producing a
        # meaningful risk spread: NORMAL(75%) / LOW(10%) / MEDIUM(8%) /
        # HIGH(4%) / CRITICAL(3%).
        # Look up device-specific detection sensitivity
        device_profile = self._cia_modifier._registry.lookup(self._device_id)
        device_pct = device_profile.detection_percentile

        calibrated = (
            self._calibrator.calibrate(score, device_percentile=device_pct)
            if self._calibrator.fitted else None
        )

        threshold = self._baseline.get("baseline_threshold", 0.255)
        mad = self._baseline["mad"]
        distance = score - threshold

        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        # Use calibrated risk level if calibrator is fitted
        if calibrated is not None:
            risk_result = RiskLevel(calibrated["risk_level"])
        else:
            risk_result = self._scorer.classify_single(
                distance=distance,
                mad=mad,
                feature_values=raw_features,
                feature_names=MODEL_FEATURES,
            )

        # Attention escalation
        if attn_flag and risk_result in (RiskLevel.NORMAL, RiskLevel.LOW):
            risk_result = RiskLevel.MEDIUM

        # 4. CIA modification
        cia_assessment = self._cia_modifier.modify(
            base_risk=risk_result,
            attack_category=attack_category,
            device_id=self._device_id,
            attention_flag=attn_flag,
        )

        # 5. Clinical impact
        clinical = self._clinical_assessor.assess(
            risk_level=cia_assessment.adjusted_risk_level,
            device_type=self._device_id,
            feature_values=raw_features,
            feature_names=MODEL_FEATURES,
            attention_flag=attn_flag,
        )

        # ── Explanation from already-computed gradients (zero extra cost) ──
        explanation = {"level": "none", "top_features": [], "timestep_importance": []}
        final_risk = cia_assessment.adjusted_risk_level
        if final_risk in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
            if grads is not None:
                grads_np = grads.numpy().squeeze()
                feat_imp = np.mean(np.abs(grads_np), axis=0)
                top_idx = np.argsort(feat_imp)[::-1][:5]
                explanation = {
                    "level": "attention_and_gradient",
                    "top_features": [
                        {"feature": MODEL_FEATURES[j], "importance": round(float(feat_imp[j]), 6)}
                        for j in top_idx
                    ],
                    "timestep_importance": np.mean(np.abs(grads_np), axis=1).tolist(),
                }

        # 6. Build result
        result: Dict[str, Any] = {
            "sample_index": self._inference_count,
            "anomaly_score": round(score, 6),
            "threshold": round(threshold, 6),
            "distance": round(distance, 6),
            "risk_level": final_risk.value,
            "attention_flag": attn_flag,
            "clinical_severity": clinical.clinical_severity.value,
            "clinical_severity_name": clinical.clinical_severity.name,
            "response_time_minutes": clinical.protocol.response_time_minutes,
            "device_action": clinical.protocol.device_action,
            "patient_safety_flag": clinical.patient_safety_flag,
            "clinical_rationale": clinical.rationale,
            "scenario": cia_assessment.scenario.value,
            "cia_scores": cia_assessment.cia_scores,
            "cia_max_dimension": cia_assessment.cia_max_dimension,
            "attack_category": attack_category,
            "explanation": explanation,
            "percentile": calibrated["percentile"] if calibrated else None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

        self._fatigue_mgr.process([result], device_id=self._device_id)
        with self._lock:
            self._inference_count += 1
            self._consecutive_failures = 0
            self._circuit_open = False

        # Persist to database
        if self._db is not None:
            try:
                result["device_id"] = self._device_id
                self._db.insert_prediction(result)
            except Exception as exc:
                logger.warning("DB insert_prediction failed: %s", exc)

        return result

    def calibrate_from_scores(self, score: float, gt_label: int) -> None:
        """Collect scores during calibration phase for auto-fitting.

        Call this for each flow during the first N flows. Once enough
        benign samples are collected, the calibrator auto-fits.

        Args:
            score: Raw sigmoid score.
            gt_label: Ground truth label (0=benign, 1=attack).
        """
        from config.production_loader import cfg
        cal_threshold = cfg("streaming.calibration_threshold", 200)

        with self._lock:
            self._calibration_scores.append(score)
            self._calibration_labels.append(gt_label)

            if not self._calibrator.fitted and len(self._calibration_scores) >= cal_threshold:
                self._calibrator.fit_from_buffer(
                    self._calibration_scores, self._calibration_labels,
                )
                if self._calibrator.fitted:
                    cal_cfg = self._calibrator.get_config()
                    logger.info(
                        "Calibrator auto-fitted: mode=%s, youden_j=%.3f",
                        cal_cfg.get("mode", "unknown"),
                        cal_cfg.get("youden_j", 0),
                    )
                    if self._db is not None:
                        try:
                            self._db.insert_calibration(cal_cfg)
                        except Exception as exc:
                            logger.warning("DB insert_calibration failed: %s", exc)

    def run_streaming(
        self,
        buffer: WindowBuffer,
        on_alert: Optional[Callable[[Dict[str, Any]], None]] = None,
        poll_interval: float = 0.05,
    ) -> None:
        """Run continuous streaming inference loop.

        Polls the WindowBuffer for new windows and processes each one.
        Blocks until interrupted.

        Args:
            buffer: WindowBuffer to consume windows from.
            on_alert: Callback for emitted alerts (e.g., SIEM publish).
            poll_interval: Seconds between buffer polls.
        """
        logger.info("Streaming inference started (poll=%.3fs)", poll_interval)
        last_flow_count = 0

        try:
            while True:
                current = buffer.flow_count
                if current <= last_flow_count:
                    time.sleep(poll_interval)
                    continue

                window = buffer.get_window()
                if window is None:
                    time.sleep(poll_interval)
                    continue

                result = self.process_window(window)
                buffer.record_prediction(result)

                if result.get("alert_emit", True) and on_alert:
                    on_alert(result)

                last_flow_count = current

        except KeyboardInterrupt:
            logger.info("Streaming inference stopped after %d inferences",
                        self._inference_count)

    @property
    def inference_count(self) -> int:
        return self._inference_count

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_loaded": self._model is not None,
            "model_params": self._model.count_params() if self._model else 0,
            "inference_count": self._inference_count,
            "device_id": self._device_id,
            "baseline_threshold": self._baseline.get("baseline_threshold", 0),
            "fatigue_summary": self._fatigue_mgr.get_summary() if self._fatigue_mgr else {},
            "circuit_open": self._circuit_open,
            "consecutive_failures": self._consecutive_failures,
            "last_error": self._last_error,
        }
