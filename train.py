"""Train and evaluate the GridVeil hybrid forecasting model."""

import json
from pathlib import Path

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
CHECKPOINT_PATH = "results/best_gridveil_model.keras"
LOSS_CURVE_PATH = "results/training_loss_curve.png"
DASHBOARD_DATA_PATH = "dashboard/dashboard_data.json"


def final_timestep_targets(target_windows):
    """Select one three-zone target vector for each input sequence."""
    return target_windows[:, -1, :]


def main():
    """Run preprocessing, training, evaluation, and loss plotting."""
    Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(LOSS_CURVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    processed_data = preprocess_data()
    sequences = processed_data["sequences"]
    target_scaler = processed_data["target_scaler"]

    # Timestamps for the test split's target rows (one per sliding window,
    # taken at the window's final timestep), used only for dashboard labels.
    SEQUENCE_LENGTH = 24
    test_slice = processed_data["split_slices"]["test"]
    test_timestamps = processed_data["data"].index[test_slice][SEQUENCE_LENGTH - 1:]

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

    test_predictions_scaled = model.predict(test_features, batch_size=BATCH_SIZE, verbose=1)

    # --- FIX (issue 1): invert the StandardScaler BEFORE computing any metric.
    # test_targets / test_predictions_scaled are z-scores (mean=0, std=1) at this
    # point; MAE/RMSE/MAPE/R2 and the anomaly threshold must be computed on the
    # real MW load, not on the scaled representation.
    test_targets_mw = target_scaler.inverse_transform(test_targets)
    test_predictions_mw = target_scaler.inverse_transform(test_predictions_scaled)

    mean_absolute_error_value = mean_absolute_error(test_targets_mw, test_predictions_mw)
    root_mean_squared_error = np.sqrt(
        mean_squared_error(test_targets_mw, test_predictions_mw)
    )
    mean_absolute_percentage_error_value = mean_absolute_percentage_error(
        test_targets_mw, test_predictions_mw
    )
    r_squared = r2_score(test_targets_mw, test_predictions_mw)

    print(f"Test MAE (MW): {mean_absolute_error_value:.4f}")
    print(f"Test RMSE (MW): {root_mean_squared_error:.4f}")
    print(f"Test MAPE: {mean_absolute_percentage_error_value * 100:.4f}%")
    print(f"Test R-squared (R2): {r_squared:.6f}")

    # --- FIX (issue 2): anomaly threshold and detection also moved to real MW
    # units, computed on TRAIN errors only (no leakage) and applied to TEST
    # errors only. This is completely unrelated to the 70/15/15 split sizes —
    # the old dashboard numbers (44,554 / 7,834) were actually the train+val
    # cutoff row and the test-set window count, not anomaly counts at all.
    train_predictions_scaled = model.predict(train_features, batch_size=BATCH_SIZE, verbose=1)
    train_targets_mw = target_scaler.inverse_transform(train_targets)
    train_predictions_mw = target_scaler.inverse_transform(train_predictions_scaled)

    train_absolute_errors_mw = np.abs(train_targets_mw - train_predictions_mw)
    anomaly_threshold_mw = np.percentile(train_absolute_errors_mw, 95)

    test_absolute_errors_mw = np.abs(test_targets_mw - test_predictions_mw)
    is_unusual = np.any(test_absolute_errors_mw > anomaly_threshold_mw, axis=1)
    # A second, stricter band for "High deviation" (99th percentile of train errors).
    high_deviation_threshold_mw = np.percentile(train_absolute_errors_mw, 99)
    is_high_deviation = np.any(test_absolute_errors_mw > high_deviation_threshold_mw, axis=1)

    status = np.where(is_high_deviation, "High deviation",
                       np.where(is_unusual, "Unusual", "Normal"))
    status_counts = {
        "Normal": int(np.sum(status == "Normal")),
        "Unusual": int(np.sum(status == "Unusual")),
        "High deviation": int(np.sum(status == "High deviation")),
    }

    print(f"Anomaly threshold (MW, 95th pct of train errors): {anomaly_threshold_mw:.4f}")
    print(f"High-deviation threshold (MW, 99th pct of train errors): {high_deviation_threshold_mw:.4f}")
    print(f"Test set size (rows, NOT the anomaly count): {len(test_targets_mw)}")
    print(f"Status counts on the test set: {status_counts}")

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

    # --- FIX (issues 3 & 4): write ONE JSON file the dashboard reads from, so the
    # HTML never has hand-typed numbers that can drift from what was actually
    # computed. Uses total load (sum of the 3 zones) for the display table/chart.
    export_dashboard_data(
        test_targets_mw=test_targets_mw,
        test_predictions_mw=test_predictions_mw,
        test_absolute_errors_mw=test_absolute_errors_mw,
        status=status,
        status_counts=status_counts,
        mape=mean_absolute_percentage_error_value * 100,
        anomaly_threshold_mw=anomaly_threshold_mw,
        test_timestamps=test_timestamps,
    )


def export_dashboard_data(test_targets_mw, test_predictions_mw, test_absolute_errors_mw,
                           status, status_counts, mape, anomaly_threshold_mw,
                           test_timestamps, last_n_hours=168):
    """Write the exact numbers the dashboard displays, computed from this run."""
    total_actual = test_targets_mw.sum(axis=1)
    total_pred = test_predictions_mw.sum(axis=1)
    total_abs_err = test_absolute_errors_mw.sum(axis=1)
    pct_error = np.divide(
        total_abs_err, np.abs(total_actual),
        out=np.zeros_like(total_abs_err), where=total_actual != 0,
    ) * 100

    tail = slice(-last_n_hours, None)
    anomaly_rows = [
        {
            "time": test_timestamps[i].strftime("%b %d, %H:%M"),
            "expected": round(float(total_actual[i]), 1),
            "actual": round(float(total_pred[i]), 1),
            "pct_error": round(float(pct_error[i]), 1),
            "status": str(status[i]),
        }
        for i in range(len(status))
        if status[i] != "Normal"
    ][-10:]

    dashboard_data = {
        "forecast_actual": [round(float(v), 1) for v in total_actual[tail]],
        "forecast_pred": [round(float(v), 1) for v in total_pred[tail]],
        "anomaly_table": anomaly_rows,
        "status_counts": status_counts,
        "test_set_size": int(len(status)),
        "anomaly_threshold_mw": round(float(anomaly_threshold_mw), 2),
        "next_hour_pred": round(float(total_pred[-1]), 1),
        "mape": round(float(mape), 2),
    }

    output_path = (Path(__file__).parent / DASHBOARD_DATA_PATH).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dashboard_data, indent=2))
    print(f"Dashboard data written to: {output_path}")


if __name__ == "__main__":
    main()
