"""FastAPI inference endpoint for production deployment.

Provides:
  POST /predict          — single-window inference
  POST /ingest           — ingest a raw flow + biometrics
  GET  /health           — service health check
  GET  /status           — detailed pipeline status
  GET  /baseline         — current baseline configuration

Usage:
    uvicorn src.production.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — FastAPI may not be installed in all environments
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None  # type: ignore[misc,assignment]

from dashboard.streaming.feature_aligner import MODEL_FEATURES, N_FEATURES
from dashboard.streaming.window_buffer import WindowBuffer

# ── Module-level singletons (initialized on startup) ──────────────

_service = None
_buffer = None
_fuser = None
_bridge = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_app() -> Any:
    """Create FastAPI app with lazy initialization."""
    if FastAPI is None:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="RA-X-IoMT Inference API",
        description="Real-time IoMT intrusion detection inference endpoint",
        version="1.0.0",
    )

    # ── Request/Response models ────────────────────────────────────

    class FlowRequest(BaseModel):
        """Single flow for ingestion."""
        network_features: Dict[str, float] = Field(
            ..., description="16 network feature values",
        )
        biometrics: Dict[str, float] = Field(
            default_factory=dict,
            description="8 biometric feature values (optional, uses latest if omitted)",
        )
        device_id: str = Field(
            default="unknown",
            description="Device identifier for biometric lookup",
        )

    class WindowRequest(BaseModel):
        """Pre-formed window for direct inference."""
        window: List[List[float]] = Field(
            ..., description="Window of shape (20, 24) — 20 timesteps, 24 features",
        )

    class PredictionResponse(BaseModel):
        """Inference result."""
        risk_level: str
        clinical_severity: int
        clinical_severity_name: str
        anomaly_score: float
        attention_flag: bool
        patient_safety_flag: bool
        device_action: str
        response_time_minutes: int
        alert_emit: bool
        latency_ms: float

    class HealthResponse(BaseModel):
        """Service health check."""
        status: str
        model_loaded: bool
        inference_count: int
        buffer_state: str
        uptime_seconds: float

    # ── Startup ────────────────────────────────────────────────────

    _start_time = time.time()

    @app.on_event("startup")
    async def startup() -> None:
        """Load model and initialize all services."""
        global _service, _buffer, _fuser, _bridge

        from src.production.inference_service import InferenceService
        from src.production.feature_fuser import FeatureFuser
        from src.production.biometric_bridge import BiometricBridge

        _buffer = WindowBuffer(window_size=20, calibration_threshold=20)

        scaler_path = PROJECT_ROOT / "models" / "scalers" / "robust_scaler.pkl"
        _fuser = FeatureFuser(scaler_path, _buffer)
        _bridge = BiometricBridge(publish_fn=lambda t, p: None)

        _service = InferenceService(PROJECT_ROOT)
        _service.load()

        logger.info("API startup complete")

    # ── Endpoints ──────────────────────────────────────────────────

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(req: WindowRequest) -> Dict[str, Any]:
        """Run inference on a pre-formed window (20 timesteps, 24 features)."""
        if _service is None:
            raise HTTPException(503, "Service not initialized")

        window = np.array(req.window, dtype=np.float32)
        if window.shape != (20, N_FEATURES):
            raise HTTPException(
                400,
                f"Window must be (20, {N_FEATURES}), got {window.shape}",
            )

        window = window.reshape(1, 20, N_FEATURES)
        result = _service.process_window(window)
        return result

    @app.post("/ingest", response_model=PredictionResponse)
    async def ingest(req: FlowRequest) -> Dict[str, Any]:
        """Ingest a single flow, fuse with biometrics, and predict.

        If the buffer doesn't have a full window yet, returns a
        CALIBRATING response.
        """
        if _service is None or _fuser is None or _bridge is None:
            raise HTTPException(503, "Service not initialized")

        # Update biometrics if provided
        if req.biometrics:
            _bridge.update(req.device_id, req.biometrics)

        # Get latest biometrics for this device
        latest_bio = _bridge.get_latest(req.device_id)

        # Fuse and feed buffer
        _fuser.fuse(req.network_features, latest_bio)

        # Check if we have a full window
        window = _buffer.get_window()
        if window is None:
            return {
                "risk_level": "CALIBRATING",
                "clinical_severity": 0,
                "clinical_severity_name": "CALIBRATING",
                "anomaly_score": 0.0,
                "attention_flag": False,
                "patient_safety_flag": False,
                "device_action": "none",
                "response_time_minutes": 0,
                "alert_emit": False,
                "latency_ms": 0.0,
            }

        result = _service.process_window(window)
        _buffer.record_prediction(result)
        return result

    @app.get("/health", response_model=HealthResponse)
    async def health() -> Dict[str, Any]:
        """Service health check."""
        return {
            "status": "healthy" if _service else "starting",
            "model_loaded": _service is not None and _service._model is not None,
            "inference_count": _service.inference_count if _service else 0,
            "buffer_state": _buffer.state.value if _buffer else "NONE",
            "uptime_seconds": round(time.time() - _start_time, 1),
        }

    @app.get("/status")
    async def status() -> Dict[str, Any]:
        """Detailed pipeline status."""
        result: Dict[str, Any] = {"api": "running"}
        if _service:
            result["inference"] = _service.get_status()
        if _buffer:
            result["buffer"] = _buffer.get_status()
        if _fuser:
            result["fuser"] = _fuser.get_status()
        if _bridge:
            result["bridge"] = {
                "active_devices": _bridge.active_devices,
                "update_count": _bridge.update_count,
            }
        return result

    @app.get("/baseline")
    async def baseline() -> Dict[str, Any]:
        """Current baseline configuration."""
        if _service is None:
            raise HTTPException(503, "Service not initialized")
        return _service._baseline

    return app


# Create app instance (imported by uvicorn)
app = _get_app() if FastAPI is not None else None
