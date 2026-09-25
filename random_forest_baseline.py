"""Reproducible Random Forest baseline for GridVeil forecasting."""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from data_preprocessing import preprocess_data

RANDOM_STATE = 42
N_ESTIMATORS = 100
RESULTS_PATH = Path(__file__).with_name("rf_results.json")
DATA_PATH = Path(__file__).parent / "powerconsumption.csv"


def main():
    """Train and evaluate the baseline on the same 10-minute windows as the hybrid model."""
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
    model.fit(train_features.reshape(len(train_features), -1), train_targets)
    predicted = model.predict(test_features.reshape(len(test_features), -1))
    actual = target_scaler.inverse_transform(test_targets)
    predicted = target_scaler.inverse_transform(predicted)
    results = {
        "model": "RandomForestRegressor",
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "dataset": str(DATA_PATH),
        "split": "70/15/15 chronological",
        "lookback_steps": 24,
        "target": "three zone loads at final timestep",
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
