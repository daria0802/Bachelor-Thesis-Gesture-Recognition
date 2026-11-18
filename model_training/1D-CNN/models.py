# File:        models.py
# Author:      Daria Kinash
# Description: 1D CNN architecture for gesture and activity recognition.

from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers, models


@dataclass
class CNN1DConfig:
    """Parameters that control the CNN architecture."""
    filters: tuple = (16, 32)       # number of filters in each conv layer
    kernel_sizes: tuple = (3, 3)    # kernel size for each conv layer
    pool_sizes: tuple = (2, 2)      # max-pool size for each layer
    dropout: float = 0.0            # dropout rate applied after each block (0 = disabled)
    dense_units: int = 16           # units in the fully connected layer before output
    use_batchnorm: bool = False     # add BatchNormalization after each conv
    l2: float = 0.0                 # L2 regularization coefficient for Conv1D and Dense


def compile_model(model, learning_rate=1e-3):
    """Compile the model with Adam optimizer and sparse categorical cross-entropy loss."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(input_shape, num_classes, learning_rate=1e-3, cnn_cfg=None):
    """Build and compile a 1D CNN model."""
    cfg = cnn_cfg or CNN1DConfig()
    reg = tf.keras.regularizers.l2(cfg.l2) if cfg.l2 and cfg.l2 > 0 else None

    n_layers = len(cfg.filters)

    # if only one value is given, repeat it for every layer
    kernels = list(cfg.kernel_sizes)
    if len(kernels) == 1:
        kernels = kernels * n_layers

    pools = list(cfg.pool_sizes)
    if len(pools) == 1:
        pools = pools * n_layers

    model = models.Sequential(name="CNN1D")
    model.add(layers.Input(shape=input_shape, name="input"))

    for i in range(n_layers):
        if cfg.use_batchnorm:
            model.add(
                layers.Conv1D(
                    filters=int(cfg.filters[i]),
                    kernel_size=int(kernels[i]),
                    padding="same",
                    activation=None,
                    kernel_regularizer=reg,
                    name=f"conv_{i + 1}",
                )
            )
            model.add(layers.BatchNormalization(name=f"bn_{i + 1}"))
            model.add(layers.Activation("relu", name=f"relu_{i + 1}"))
        else:
            model.add(
                layers.Conv1D(
                    filters=int(cfg.filters[i]),
                    kernel_size=int(kernels[i]),
                    padding="same",
                    activation="relu",
                    kernel_regularizer=reg,
                    name=f"conv_{i + 1}",
                )
            )
        if pools[i] > 1:
            model.add(layers.MaxPool1D(pool_size=int(pools[i]), name=f"pool_{i + 1}"))
        if cfg.dropout > 0:
            model.add(layers.Dropout(cfg.dropout, name=f"dropout_{i + 1}"))

    model.add(layers.GlobalAveragePooling1D(name="gap"))
    if cfg.dense_units > 0:
        model.add(
            layers.Dense(
                cfg.dense_units,
                activation="relu",
                kernel_regularizer=reg,
                name="dense",
            )
        )
        if cfg.dropout > 0:
            model.add(layers.Dropout(cfg.dropout, name="dropout_dense"))
    model.add(layers.Dense(num_classes, activation="softmax", name="out"))
    return compile_model(model, learning_rate=learning_rate)
