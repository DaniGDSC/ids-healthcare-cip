"""Streaming inference service — per-window model prediction + risk scoring.

Wraps the Phase 2.5 finetuned model and Phase 4 risk components into
a single callable service. In production, this runs as a long-lived
process consuming from the WindowBuffer.

Components loaded at startup (once):
  - CNN-BiLSTM-Attention model (Phase 2 backbone + Phase 2.5 head)
  - RiskScorer, CIARiskModifier, ClinicalImpactAssessor, AlertFatigueManager
  - AttentionAnomalyDetector, ConditionalExplainer

Per-window processing (~100ms):
  window (1, 20, 24) → model.predict() → risk score → CIA → clinical
  → fatigue check → conditional explanation → alert (if emitted)
"""

from __future__ import annotations

import json
import logging
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
        device_id: str = "generic_iomt_sensor",
    ) -> None:
        self._root = Path(project_root)
        self._device_id = device_id
        self._model: Optional[tf.keras.Model] = None
        self._scorer = None
        self._cia_modifier = None
        self._clinical_assessor = None
        self._fatigue_mgr = None
        self._attn_detector = None
        self._explainer = None
        self._baseline: Dict[str, Any] = {}
        self._inference_count = 0

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
        self._attn_detector = AttentionAnomalyDetector(
            baseline_median=self._baseline["median"],
            baseline_mad=self._baseline["mad"],
            mad_multiplier=self._baseline.get("mad_multiplier", 3.0),
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "InferenceService loaded: %d params, %.2fs",
            self._model.count_params(), elapsed,
        )

    def process_window(
        self,
        window: np.ndarray,
        raw_features: Optional[np.ndarray] = None,
        attack_category: str = "unknown",
    ) -> Dict[str, Any]:
        """Process a single window and return scored result.

        Args:
            window: Input array of shape (1, 20, 24).
            raw_features: Unscaled features for cross-modal check, shape (24,).
            attack_category: Attack type if known (for CIA scoring).

        Returns:
            Dict with risk_level, clinical_severity, alert_emit, explanation, etc.
        """
        t0 = time.perf_counter()

        # 1. Model inference
        score = float(self._model.predict(window, verbose=0).ravel()[0])

        # 2. Attention anomaly detection
        attn_magnitudes = self._attn_detector.compute_scores(self._model, window)
        attn_flag = bool(self._attn_detector.classify(attn_magnitudes)[0])

        # 3. Risk scoring
        threshold = self._baseline.get("baseline_threshold", 0.255)
        mad = self._baseline["mad"]
        distance = score - threshold

        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

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

        # 6. Alert fatigue
        result: Dict[str, Any] = {
            "sample_index": self._inference_count,
            "anomaly_score": round(score, 6),
            "threshold": round(threshold, 6),
            "distance": round(distance, 6),
            "risk_level": cia_assessment.adjusted_risk_level.value,
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
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

        self._fatigue_mgr.process([result], device_id=self._device_id)
        self._inference_count += 1

        return result

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
        }
