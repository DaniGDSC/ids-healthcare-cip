"""BiLSTM stage — second stage of the CNN→BiLSTM→Attention pipeline.

Data flow
---------
CNN output  : (batch, 7, 128)
    ↓
Bidirectional(LSTM(128, return_sequences=True))
    ↓  (batch, 7, 256)   ← forward 128 + backward 128
Dropout(0.3)
    ↓
Bidirectional(LSTM(64, return_sequences=True))
    ↓  (batch, 7, 128)   ← forward 64 + backward 64
Dropout(0.3)
    ↓  (batch, 7, 128)   → input for Attention
"""

from __future__ import annotations

import tensorflow as tf

keras  = tf.keras
layers = tf.keras.layers

from src.phase2_detection_engine.CNN_1D import build_cnn_extractor          # noqa: E402
from src.phase2_detection_engine.load_dataset import load_both              # noqa: E402


def build_bilstm_encoder(
    input_shape: tuple = (7, 128),
    *,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Return the BiLSTM encoder (no classification head).

    Parameters
    ----------
    input_shape  : (timesteps, features) — matches CNN output (7, 128)
    dropout_rate : applied after each BiLSTM block

    Output shape : (batch, 7, 128) — sequence retained for Attention input
    """
    inp = keras.Input(shape=input_shape, name="bilstm_input")

    # Block 1 — BiLSTM(128) → (batch, 7, 256)
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True), name="bilstm1"
    )(inp)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    # Block 2 — BiLSTM(64) → (batch, 7, 128)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True), name="bilstm2"
    )(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    return keras.Model(inp, x, name="bilstm_encoder")


if __name__ == "__main__":
    train_data, _ = load_both()

    # CNN stage
    X_train = tf.expand_dims(train_data.X, axis=-1)   # (19980, 29, 1)
    cnn = build_cnn_extractor(n_features=X_train.shape[1])
    cnn_out = cnn(X_train, training=False)             # (19980, 7, 128)

    # BiLSTM stage
    bilstm = build_bilstm_encoder(input_shape=cnn_out.shape[1:])
    bilstm.summary()

    sample_out = bilstm(cnn_out[:4], training=False)
    print(f"\nCNN    output : {cnn_out.shape}")
    print(f"BiLSTM output : {sample_out.shape}   ← Attention input")
