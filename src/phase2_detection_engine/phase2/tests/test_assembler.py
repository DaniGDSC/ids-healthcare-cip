"""Unit tests for DetectionModelAssembler (shapes, no head, config)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf

from src.phase2_detection_engine.phase2.assembler import DetectionModelAssembler
from src.phase2_detection_engine.phase2.attention_builder import AttentionBuilder
from src.phase2_detection_engine.phase2.base import BaseLayerBuilder
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder


class TestDetectionModelAssembler:
    """Test model assembly with builder chain."""

    def _default_builders(self):
        return [
            CNNBuilder(filters_1=32, filters_2=64, kernel_size=3, pool_size=2),
            BiLSTMBuilder(units_1=32, units_2=16, dropout_rate=0.1),
            AttentionBuilder(units=32),
        ]

    def test_output_shape(self) -> None:
        assembler = DetectionModelAssembler(
            timesteps=20,
            n_features=5,
            builders=self._default_builders(),
        )
        model = assembler.assemble()
        # BiLSTM units_2=16, bidirectional=32 → attention output = 32
        assert model.output_shape == (None, 32)

    def test_input_shape(self) -> None:
        assembler = DetectionModelAssembler(
            timesteps=20,
            n_features=5,
            builders=self._default_builders(),
        )
        model = assembler.assemble()
        assert model.input_shape == (None, 20, 5)

    def test_no_classification_head(self) -> None:
        """Model output must NOT be a Dense/softmax classification layer."""
        assembler = DetectionModelAssembler(
            timesteps=20,
            n_features=5,
            builders=self._default_builders(),
        )
        model = assembler.assemble()
        last_layer = model.layers[-1]
        # The output dim should be bilstm_units_2 * 2 = 32
        # NOT a small classification dim like 2 or num_classes
        assert model.output_shape[-1] == 32
        # Last layer must not be Dense (no classification head)
        assert not isinstance(last_layer, tf.keras.layers.Dense)

    def test_param_count_positive(self) -> None:
        assembler = DetectionModelAssembler(
            timesteps=20,
            n_features=5,
            builders=self._default_builders(),
        )
        model = assembler.assemble()
        assert model.count_params() > 0

    def test_forward_pass(self) -> None:
        assembler = DetectionModelAssembler(
            timesteps=20,
            n_features=5,
            builders=self._default_builders(),
        )
        model = assembler.assemble()
        rng = np.random.RandomState(42)
        X = rng.randn(4, 20, 5).astype(np.float32)
        out = model.predict(X, verbose=0)
        assert out.shape == (4, 32)

    def test_no_builders_raises(self) -> None:
        assembler = DetectionModelAssembler(
            timesteps=20,
            n_features=5,
            builders=[],
        )
        with pytest.raises(ValueError, match="No builders"):
            assembler.assemble()

    def test_get_config(self) -> None:
        builders = self._default_builders()
        assembler = DetectionModelAssembler(
            timesteps=20,
            n_features=5,
            builders=builders,
            model_name="test_model",
        )
        cfg = assembler.get_config()
        assert cfg["timesteps"] == 20
        assert cfg["n_features"] == 5
        assert cfg["model_name"] == "test_model"
        assert len(cfg["builders"]) == 3
        assert cfg["builders"][0]["type"] == "CNNBuilder"

    def test_custom_builder(self) -> None:
        """A custom BaseLayerBuilder can be plugged in (Open/Closed)."""

        class PassthroughBuilder(BaseLayerBuilder):
            def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
                # Use a Keras layer to maintain graph connectivity
                return tf.keras.layers.Activation("linear")(input_tensor)

            def get_config(self) -> Dict[str, Any]:
                return {"type": "passthrough"}

        assembler = DetectionModelAssembler(
            timesteps=10,
            n_features=3,
            builders=[PassthroughBuilder()],
        )
        model = assembler.assemble()
        assert model.output_shape == (None, 10, 3)
