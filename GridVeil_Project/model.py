"""Hybrid CNN-LSTM-Attention model for GridVeil forecasting."""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Attention,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


OUTPUT_UNITS = 3


def build_hybrid_model(input_shape):
    """Build and compile the hybrid CNN-LSTM-Attention forecasting model.

    Args:
        input_shape: Tuple containing (timesteps, number_of_features).

    Returns:
        A compiled, untrained TensorFlow Keras model.
    """
    inputs = Input(shape=input_shape, name="time_series_input")

    # Convolution extracts short-term patterns from adjacent time steps.
    convolution = Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        name="local_pattern_extractor",
    )(inputs)
    pooled = MaxPooling1D(pool_size=2, name="temporal_pooling")(convolution)

    # The LSTM processes the pooled sequence and preserves each time step.
    recurrent = LSTM(
        units=64,
        return_sequences=True,
        name="long_term_sequence_encoder",
    )(pooled)

    # Self-attention weights the relative importance of encoded time steps.
    attended = Attention(name="temporal_attention")([recurrent, recurrent])
    aggregated = GlobalAveragePooling1D(name="attention_pooling")(attended)

    hidden = Dense(32, activation="relu", name="forecasting_hidden_layer")(
        aggregated
    )
    outputs = Dense(
        OUTPUT_UNITS,
        activation="linear",
        name="zone_forecasts",
    )(hidden)

    model = Model(inputs=inputs, outputs=outputs, name="GridVeil_CNN_LSTM_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    return model


def get_callbacks(checkpoint_path):
    """Create callbacks that retain the best validation-loss model."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
    ]
