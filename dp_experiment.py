"""Synthetic-data privacy utility experiment.

This script implements bounded-label Laplace perturbation on the synthetic
Libya-pattern series described in the report. It is a reproducible privacy
experiment, not a formal end-to-end DP-SGD guarantee for the neural network.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
except ImportError:
    tf = None


DATA_PATH = Path(__file__).parent / "data" / "synthetic" / "Synthetic_Libya_Electricity.csv"
RESULTS_PATH = Path(__file__).with_name("dp_results.json")
SEQUENCE_LENGTH = 24
TRAIN_FRACTION = 0.80


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--sensitivity-mw", type=float, default=42.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_windows(dataframe):
    """Build hourly 24-step windows and one-step-ahead load targets."""
    numeric_features = dataframe[["Load", "Temperature", "Hour", "Month"]].to_numpy(float)
    targets = dataframe["Load"].to_numpy(float)
    features = np.stack(
        [numeric_features[index : index + SEQUENCE_LENGTH]
         for index in range(len(dataframe) - SEQUENCE_LENGTH)]
    )
    labels = np.array(
        [targets[index + SEQUENCE_LENGTH] for index in range(len(dataframe) - SEQUENCE_LENGTH)]
    )
    return features, labels


def build_model(input_shape, seed):
    if tf is None:
        return None
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=input_shape)
    encoded = tf.keras.layers.Conv1D(32, 3, activation="relu")(inputs)
    encoded = tf.keras.layers.MaxPooling1D(2)(encoded)
    encoded = tf.keras.layers.LSTM(32, return_sequences=True)(encoded)
    encoded = tf.keras.layers.Attention()([encoded, encoded])
    encoded = tf.keras.layers.GlobalAveragePooling1D()(encoded)
    outputs = tf.keras.layers.Dense(1)(encoded)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_and_score(
    train_features,
    train_labels,
    evaluation_features,
    evaluation_labels,
    target_scaler,
    epochs,
    seed,
):
    model = build_model(train_features.shape[1:], seed)
    if model is None:
        estimator = RandomForestRegressor(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        )
        estimator.fit(train_features.reshape(len(train_features), -1), train_labels)
        predictions = estimator.predict(
            evaluation_features.reshape(len(evaluation_features), -1)
        ).reshape(-1, 1)
    else:
        model.fit(train_features, train_labels, epochs=epochs, batch_size=64, verbose=0)
        predictions = model.predict(
            evaluation_features, batch_size=64, verbose=0
        ).reshape(-1, 1)
    actual_mw = target_scaler.inverse_transform(evaluation_labels.reshape(-1, 1))
    predicted_mw = target_scaler.inverse_transform(predictions)
    return mean_absolute_percentage_error(actual_mw, predicted_mw) * 100


def main():
    args = parse_args()
    if args.epsilon <= 0 or args.sensitivity_mw <= 0:
        raise ValueError("epsilon and sensitivity-mw must be positive")

    dataframe = pd.read_csv(DATA_PATH, parse_dates=["Datetime"]).sort_values("Datetime")
    split_row = int(len(dataframe) * TRAIN_FRACTION)
    train_frame = dataframe.iloc[:split_row].copy()
    test_frame = dataframe.iloc[split_row:].copy()
    train_features, train_labels = make_windows(train_frame)
    test_features, test_labels = make_windows(test_frame)

    feature_scaler = StandardScaler().fit(train_features.reshape(-1, train_features.shape[-1]))
    target_scaler = StandardScaler().fit(train_labels.reshape(-1, 1))
    train_features = feature_scaler.transform(
        train_features.reshape(-1, train_features.shape[-1])
    ).reshape(train_features.shape)
    test_features = feature_scaler.transform(
        test_features.reshape(-1, test_features.shape[-1])
    ).reshape(test_features.shape)
    scaled_train_labels = target_scaler.transform(train_labels.reshape(-1, 1)).reshape(-1)
    scaled_test_labels = target_scaler.transform(test_labels.reshape(-1, 1)).reshape(-1)

    rng = np.random.default_rng(args.seed)
    noise_scale_mw = args.sensitivity_mw / args.epsilon
    noisy_labels_mw = train_labels + rng.laplace(
        0.0, noise_scale_mw, size=train_labels.shape
    )
    noisy_labels = target_scaler.transform(noisy_labels_mw.reshape(-1, 1)).reshape(-1)
    clean_mape = train_and_score(
        train_features, scaled_train_labels, test_features, scaled_test_labels,
        target_scaler, args.epochs, args.seed,
    )
    private_mape = train_and_score(
        train_features, noisy_labels, test_features, scaled_test_labels,
        target_scaler, args.epochs, args.seed,
    )
    results = {
        "dataset": str(DATA_PATH),
        "split": "80/20 chronological",
        "lookback_hours": SEQUENCE_LENGTH,
        "epsilon": args.epsilon,
        "sensitivity_mw": args.sensitivity_mw,
        "noise_scale_mw": noise_scale_mw,
        "clean_mape_percent": clean_mape,
        "private_mape_percent": private_mape,
        "engine": "tensorflow_cnn_lstm_attention" if tf is not None else "sklearn_random_forest_fallback",
        "formal_end_to_end_dp": False,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
