"""Train and evaluate the GridVeil hybrid forecasting model."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from data_preprocessing import preprocess_data
from model import build_hybrid_model, get_callbacks


EPOCHS = 50
BATCH_SIZE = 64
CHECKPOINT_PATH = "best_gridveil_model.keras"
LOSS_CURVE_PATH = "training_loss_curve.png"


def final_timestep_targets(target_windows):
    """Select one three-zone target vector for each input sequence."""
    return target_windows[:, -1, :]


def main():
    """Run preprocessing, training, evaluation, and loss plotting."""
    processed_data = preprocess_data()
    sequences = processed_data["sequences"]

    train_features, train_target_windows = sequences["train"]
    validation_features, validation_target_windows = sequences["validation"]
    test_features, test_target_windows = sequences["test"]

    # The model forecasts the three zones at the final timestep in each window.
    train_targets = final_timestep_targets(train_target_windows)
    validation_targets = final_timestep_targets(validation_target_windows)
    test_targets = final_timestep_targets(test_target_windows)

    input_shape = train_features.shape[1:]
    model = build_hybrid_model(input_shape)
    callbacks = get_callbacks(CHECKPOINT_PATH)

    history = model.fit(
        train_features,
        train_targets,
        validation_data=(validation_features, validation_targets),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    test_predictions = model.predict(test_features, batch_size=BATCH_SIZE, verbose=1)

    # Metrics are calculated on the scaled targets to measure model learning.
    mean_absolute_error_value = mean_absolute_error(
        test_targets, test_predictions
    )
    root_mean_squared_error = np.sqrt(
        mean_squared_error(test_targets, test_predictions)
    )
    mean_absolute_percentage_error_value = mean_absolute_percentage_error(
        test_targets, test_predictions
    )
    r_squared = r2_score(test_targets, test_predictions)

    print(f"Test MAE: {mean_absolute_error_value:.6f}")
    print(f"Test RMSE: {root_mean_squared_error:.6f}")
    print(f"Test MAPE: {mean_absolute_percentage_error_value:.6f}")
    print(f"Test R-squared (R2): {r_squared:.6f}")

    # Flag test predictions whose errors exceed the 95th percentile of training errors.
    train_predictions = model.predict(train_features, batch_size=BATCH_SIZE, verbose=1)
    train_absolute_errors = np.abs(train_targets - train_predictions)
    anomaly_threshold = np.percentile(train_absolute_errors, 95)

    test_absolute_errors = np.abs(test_targets - test_predictions)
    anomalies = np.any(test_absolute_errors > anomaly_threshold, axis=1)

    print(f"Anomaly Threshold: {anomaly_threshold:.6f}")
    print(f"Anomalies detected in test set: {np.sum(anomalies)}")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("GridVeil Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=150)
    plt.close()
    print(f"Training loss curve saved to: {LOSS_CURVE_PATH}")


if __name__ == "__main__":
    main()
