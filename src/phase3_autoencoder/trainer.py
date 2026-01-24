"""Autoencoder training utilities."""

import logging
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class AutoencoderTrainer:
    """Train autoencoder models."""

    def __init__(self, optimizer: str = 'adam', learning_rate: float = 1e-3,
                 batch_size: int = 256, epochs: int = 100,
                 early_stopping: dict = None, reduce_lr: dict = None,
                 validation_split: float = 0.0, shuffle: bool = True):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping or {}
        self.reduce_lr = reduce_lr or {}
        self.validation_split = validation_split
        self.shuffle = shuffle

    def _build_callbacks(self, tensorboard_dir=None):
        callbacks = []

        if self.early_stopping.get('enabled', False):
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor=self.early_stopping.get('monitor', 'val_loss'),
                patience=self.early_stopping.get('patience', 10),
                restore_best_weights=self.early_stopping.get('restore_best_weights', True)
            ))

        if self.reduce_lr.get('enabled', False):
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor=self.reduce_lr.get('monitor', 'val_loss'),
                factor=self.reduce_lr.get('factor', 0.5),
                patience=self.reduce_lr.get('patience', 5),
                min_lr=self.reduce_lr.get('min_lr', 1e-6)
            ))

        if tensorboard_dir:
            callbacks.append(keras.callbacks.TensorBoard(log_dir=tensorboard_dir))

        return callbacks

    def compile_model(self, model: keras.Model):
        if self.optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        return model

    def train(self, model: keras.Model, X_train: np.ndarray, X_val: np.ndarray = None,
              tensorboard_dir=None):
        callbacks = self._build_callbacks(tensorboard_dir)
        history = model.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            validation_split=self.validation_split if X_val is None else 0.0,
            callbacks=callbacks,
            verbose=2
        )
        return history