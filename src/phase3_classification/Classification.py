"""Classification stage — final stage of the CNN→BiLSTM→Attention pipeline.

Data flow
---------
Attention output : (batch, 128)
    ↓
Dense(64, activation='relu')
    ↓  (batch, 64)
Dropout(0.3)
    ↓
Dense(1, activation='sigmoid')
    ↓  (batch, 1)
Output : probability ∈ [0, 1]
    ≥ 0.5 → Attack (1)
    < 0.5 → Normal (0)
"""

from __future__ import annotations

import tensorflow as tf

keras  = tf.keras
layers = tf.keras.layers

from src.phase2_detection_engine.CNN_1D     import build_cnn_extractor   # noqa: E402
from src.phase2_detection_engine.BiLTSM     import build_bilstm_encoder  # noqa: E402
from src.phase2_detection_engine.Attention  import build_attention_block # noqa: E402
from src.phase2_detection_engine.load_dataset import load_both           # noqa: E402


def build_classifier(
    input_dim: int = 128,
    *,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Return the classification head as a standalone Keras model.

    Parameters
    ----------
    input_dim    : size of the context vector from Attention (128)
    dropout_rate : dropout before final sigmoid

    Output shape : (batch, 1) — attack probability
    """
    inp = keras.Input(shape=(input_dim,), name="clf_input")

    x   = layers.Dense(64, activation="relu", name="dense")(inp)
    x   = layers.Dropout(dropout_rate, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    return keras.Model(inp, out, name="classifier")


def build_full_model(n_features: int = 29) -> keras.Model:
    """Assemble the end-to-end CNN → BiLSTM → Attention → Classifier model.

    Parameters
    ----------
    n_features : number of input features from Phase 1 (29)

    Input  shape : (batch, 29)
    Output shape : (batch, 1)  — attack probability
    """
    inp = keras.Input(shape=(n_features,), name="input")

    # (batch, 29) → (batch, 29, 1)
    x = layers.Reshape((n_features, 1), name="reshape")(inp)

    # CNN extractor  → (batch, 7, 128)
    cnn = build_cnn_extractor(n_features)
    x = cnn(x)

    # BiLSTM encoder → (batch, 7, 128)
    bilstm = build_bilstm_encoder(input_shape=(7, 128))
    x = bilstm(x)

    # Attention      → (batch, 128)
    attn = build_attention_block(input_shape=(7, 128))
    x = attn(x)

    # Classifier     → (batch, 1)
    clf = build_classifier(input_dim=128)
    out = clf(x)

    model = keras.Model(inp, out, name="cnn_bilstm_attention_ids")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc_roc", curve="ROC"),
            keras.metrics.AUC(name="auc_pr",  curve="PR"),
        ],
    )
    return model


if __name__ == "__main__":
    train_data, test_data = load_both()

    model = build_full_model(n_features=29)
    model.summary()

    # Single forward pass to verify end-to-end shapes
    sample_prob = model(train_data.X[:4], training=False)
    print(f"\nInput  : {train_data.X.shape}")
    print(f"Output : {sample_prob.shape}   values: {sample_prob.numpy().flatten()}")
    print(f"\nThreshold 0.5 → {['Attack' if p >= 0.5 else 'Normal' for p in sample_prob.numpy().flatten()]}")
