"""CNN-1D feature extractor — first stage of the CNN→BiLSTM pipeline.

Data flow
---------
load_dataset.load()
    ↓  X: (batch, 29)  →  reshape  →  (batch, 29, 1)
Conv1D (filters=64, kernel_size=3, activation='relu')
    ↓  (batch, 29, 64)
MaxPooling1D (pool_size=2)
    ↓  (batch, 14, 64)
Conv1D (filters=128, kernel_size=3, activation='relu')
    ↓  (batch, 14, 128)
MaxPooling1D (pool_size=2)
    ↓  (batch, 7, 128)   ← sequence output → input for BiLSTM
"""

from __future__ import annotations

import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers

from src.phase2_detection_engine.load_dataset import load_both  # noqa: E402


def build_cnn_extractor(n_features: int = 29) -> keras.Model:
    """Return the CNN-1D feature extractor (no classification head).

    Output shape: (batch, 7, 128) — a compressed sequence for BiLSTM input.
    """
    inp = keras.Input(shape=(n_features, 1), name="cnn_input")

    x = layers.Conv1D(filters=64, kernel_size=3, activation="relu",
                      padding="same", name="conv1")(inp)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = layers.Conv1D(filters=128, kernel_size=3, activation="relu",
                      padding="same", name="conv2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    return keras.Model(inp, x, name="cnn_extractor")


def extract(model: keras.Model,
            split: str = "train") -> tuple[tf.Tensor, tf.Tensor]:
    """Run the CNN extractor on a dataset split.

    Parameters
    ----------
    model : the cnn_extractor model from build_cnn_extractor()
    split : "train" or "test"

    Returns
    -------
    cnn_out : tf.Tensor  shape (n_samples, 7, 128) — BiLSTM input
    y        : tf.Tensor  shape (n_samples,)        — labels
    """
    from src.phase2_detection_engine.load_dataset import load
    data = load(split)

    # (n_samples, 29) → (n_samples, 29, 1)
    X = tf.expand_dims(data.X, axis=-1)

    cnn_out = model(X, training=False)
    return cnn_out, data.y


if __name__ == "__main__":
    train_data, test_data = load_both()

    # reshape: (n, 29) → (n, 29, 1)
    X_train = tf.expand_dims(train_data.X, axis=-1)   # (19980, 29, 1)
    X_test  = tf.expand_dims(test_data.X,  axis=-1)   # (4896,  29, 1)
    y_train = train_data.y
    y_test  = test_data.y

    cnn = build_cnn_extractor(n_features=X_train.shape[1])
    cnn.summary()

    # Verify output shape before handing off to BiLSTM
    sample_out = cnn(X_train[:4], training=False)
    print(f"\nInput  shape : {X_train.shape}")
    print(f"CNN output   : {sample_out.shape}   ← BiLSTM input")
