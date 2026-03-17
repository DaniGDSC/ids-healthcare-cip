"""Phase 2 Detection Engine — CNN→BiLSTM→Attention feature extraction.

Re-exports from the ``phase2`` subpackage.
"""

from .phase2 import (
    AttentionBuilder,
    BahdanauAttention,
    BaseLayerBuilder,
    BiLSTMBuilder,
    CNNBuilder,
    DataReshaper,
    DetectionExporter,
    DetectionModelAssembler,
    DetectionPipeline,
    Phase1ArtifactReader,
    Phase2Config,
    render_detection_report,
)
