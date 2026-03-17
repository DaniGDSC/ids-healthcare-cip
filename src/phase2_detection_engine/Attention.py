"""Attention stage — third stage of the CNN→BiLSTM→Attention pipeline.

Data flow
---------
BiLSTM output : (batch, 7, 128)
    ↓
Dense(128, activation='tanh')    → score per timestep  (batch, 7, 128)
    ↓
Dense(1,   activation='softmax') → attention weights   (batch, 7, 1)
    ↓
Multiply(input, weights)         → weighted sequence   (batch, 7, 128)
    ↓
GlobalAveragePooling1D()         → context vector      (batch, 128)
    ↓
Output → Classification layer
"""

from __future__ import annotations

import tensorflow as tf

keras  = tf.keras
layers = tf.keras.layers

from src.phase2_detection_engine.CNN_1D import build_cnn_extractor   # noqa: E402
from src.phase2_detection_engine.BiLTSM import build_bilstm_encoder  # noqa: E402
from src.phase2_detection_engine.load_dataset import load_both       # noqa: E402


class BahdanauAttention(layers.Layer):
    """Additive (Bahdanau-style) self-attention over a sequence.

    Steps
    -----
    1. Dense(units, tanh)  — project each timestep to a score vector
    2. Dense(1, softmax)   — collapse to a scalar weight per timestep
    3. Multiply            — scale each timestep by its weight
    4. GlobalAveragePool   — sum weighted timesteps → context vector
    """

    def __init__(self, units: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.score_dense  = layers.Dense(units, activation="tanh", name="score")
        self.weight_dense = layers.Dense(1,     use_bias=False,    name="weights")
        self.gap          = layers.GlobalAveragePooling1D(name="context")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # x : (batch, timesteps, features)
        scores  = self.score_dense(x)                        # (batch, 7, 128)
        weights = self.weight_dense(scores)                  # (batch, 7,   1)
        weights = tf.nn.softmax(weights, axis=1)             # normalize over 7 timesteps
        weighted = x * weights                               # (batch, 7, 128)  — broadcast
        context  = self.gap(weighted)           # (batch, 128)
        return context

    def get_config(self) -> dict:
        return {**super().get_config(), "units": self.units}


def build_attention_block(
    input_shape: tuple = (7, 128),
    *,
    units: int = 128,
) -> keras.Model:
    """Return the Attention block as a standalone Keras model.

    Parameters
    ----------
    input_shape : (timesteps, features) — matches BiLSTM output (7, 128)
    units       : hidden size of the score Dense layer

    Output shape : (batch, 128) — context vector for Classification
    """
    inp     = keras.Input(shape=input_shape, name="attn_input")
    context = BahdanauAttention(units=units, name="attention")(inp)
    return keras.Model(inp, context, name="attention_block")


if __name__ == "__main__":
    train_data, _ = load_both()

    # CNN
    X_train = tf.expand_dims(train_data.X, axis=-1)      # (19980, 29, 1)
    cnn     = build_cnn_extractor(n_features=29)
    cnn_out = cnn(X_train, training=False)                # (19980,  7, 128)

    # BiLSTM
    bilstm     = build_bilstm_encoder(input_shape=(7, 128))
    bilstm_out = bilstm(cnn_out, training=False)          # (19980,  7, 128)

    # Attention
    attn    = build_attention_block(input_shape=(7, 128))
    attn.summary()

    context = attn(bilstm_out[:4], training=False)
    print(f"\nBiLSTM output   : {bilstm_out.shape}")
    print(f"Attention output : {context.shape}   ← Classification input")
