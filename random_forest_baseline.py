"""Reproducible Random Forest baseline for GridVeil forecasting."""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from data_preprocessing import preprocess_data


RANDOM_STATE = 42
N_ESTIMATORS = 50
RESULTS_PATH = Path(__file__).with_name("rf_results.json")


def flatten_windows(windows):
    """Convert each temporal feature window into one tabular feature vector."""
    return windows.reshape(windows.shape[0], -1)


def main():
    """Train and evaluate the baseline using the project's chronological split."""
    processed_data = preprocess_data()
    train_features, train_target_windows = processed_data["sequences"]["train"]
    test_features, test_target_windows = processed_data["sequences"]["test"]
    target_scaler = processed_data["target_scaler"]

    train_targets = train_target_windows[:, -1, :]
    test_targets = test_target_windows[:, -1, :]
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(flatten_windows(train_features), train_targets)
    predictions = model.predict(flatten_windows(test_features))

    actual = target_scaler.inverse_transform(test_targets)
    predicted = target_scaler.inverse_transform(predictions)
    results = {
        "model": "RandomForestRegressor",
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "split": "70/15/15 chronological",
        "lookback_steps": 24,
        "mae_mw": float(mean_absolute_error(actual, predicted)),
        "rmse_mw": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_percent": float(mean_absolute_percentage_error(actual, predicted) * 100),
        "r2": float(r2_score(actual, predicted)),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"RF MAE (MW): {results['mae_mw']:.6f}")
    print(f"RF RMSE (MW): {results['rmse_mw']:.6f}")
    print(f"RF MAPE: {results['mape_percent']:.6f}%")
    print(f"RF R-squared: {results['r2']:.6f}")
    print(f"RF results written to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
