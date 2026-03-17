"""Phase 2 detection engine package — SOLID-architected pipeline.

Public API
----------
Phase2Config             — pydantic-validated configuration
Phase1ArtifactReader     — reads Phase 1 outputs (DI)
DataReshaper             — sliding window reshape
BaseLayerBuilder         — abstract builder interface
CNNBuilder               — Conv1D feature extractor
BiLSTMBuilder            — bidirectional LSTM encoder
AttentionBuilder         — Bahdanau attention mechanism
BahdanauAttention        — custom Keras attention layer
DetectionModelAssembler  — chains builders into a Model
DetectionExporter        — weights, parquet, JSON export
DetectionPipeline        — orchestrates all steps
render_detection_report  — generates §5.1 Markdown
"""

from .artifact_reader import Phase1ArtifactReader
from .assembler import DetectionModelAssembler
from .attention_builder import AttentionBuilder, BahdanauAttention
from .base import BaseLayerBuilder
from .bilstm_builder import BiLSTMBuilder
from .cnn_builder import CNNBuilder
from .config import Phase2Config
from .exporter import DetectionExporter
from .pipeline import DetectionPipeline
from .report import render_detection_report
from .reshaper import DataReshaper

__all__ = [
    "BaseLayerBuilder",
    "Phase2Config",
    "Phase1ArtifactReader",
    "DataReshaper",
    "CNNBuilder",
    "BiLSTMBuilder",
    "AttentionBuilder",
    "BahdanauAttention",
    "DetectionModelAssembler",
    "DetectionExporter",
    "DetectionPipeline",
    "render_detection_report",
]
