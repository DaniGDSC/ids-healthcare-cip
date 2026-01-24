"""Autoencoder model architecture."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import List, Tuple


def build_encoder(input_dim: int, encoder_layers: List[int], latent_dim: int,
                  activation: str = 'relu', dropout_rate: float = 0.0,
                  use_batch_norm: bool = True, l2_reg: float = 0.0):
    """Build encoder model."""
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in encoder_layers:
        x = layers.Dense(units, activation=activation,
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
    latent = layers.Dense(latent_dim, name="latent_vector")(x)
    return keras.Model(inputs, latent, name="encoder")


def build_decoder(latent_dim: int, decoder_layers: List[int], output_dim: int,
                  activation: str = 'relu', output_activation: str = 'sigmoid',
                  l2_reg: float = 0.0):
    """Build decoder model."""
    inputs = keras.Input(shape=(latent_dim,))
    x = inputs
    for units in decoder_layers:
        x = layers.Dense(units, activation=activation,
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
    outputs = layers.Dense(output_dim, activation=output_activation)(x)
    return keras.Model(inputs, outputs, name="decoder")


def build_autoencoder(input_dim: int, encoder_layers: List[int], decoder_layers: List[int],
                      latent_dim: int, activation: str = 'relu', output_activation: str = 'sigmoid',
                      dropout_rate: float = 0.0, use_batch_norm: bool = True, l2_reg: float = 0.0):
    """Build full autoencoder model."""
    encoder = build_encoder(input_dim, encoder_layers, latent_dim,
                            activation, dropout_rate, use_batch_norm, l2_reg)
    decoder = build_decoder(latent_dim, decoder_layers, input_dim,
                            activation, output_activation, l2_reg)

    inputs = keras.Input(shape=(input_dim,))
    latent = encoder(inputs)
    reconstructed = decoder(latent)

    autoencoder = keras.Model(inputs, reconstructed, name="autoencoder")
    return autoencoder, encoder, decoder