"""Reproducible Random Forest baseline for GridVeil forecasting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

RANDOM_STATE = 42
N_ESTIMATORS = 100
RESULTS_PATH = Path(__file__).with_name("rf_results.json")
DATA_PATH = Path(__file__).parent / "data" / "tetouan" / "Tetouan_Hourly_Features.csv"
TARGET_COLUMN = "Next_Hour_Load"
FEATURE_COLUMNS = [
    "Temperature", "Humidity", "WindSpeed", "GeneralDiffuseFlows",
    "DiffuseFlows", "PowerConsumption_Zone1", "PowerConsumption_Zone2",
    "PowerConsumption_Zone3", "Total_Load", "Hour", "Day_of_Week",
    "Month", "Previous_Load",
]


def main():
    """Train and evaluate the report's hourly baseline split."""
    dataframe = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    train_end = int(len(dataframe) * 0.70)
    test_start = train_end + int(len(dataframe) * 0.15)
    train_features = dataframe.iloc[:train_end][FEATURE_COLUMNS]
    train_targets = dataframe.iloc[:train_end][TARGET_COLUMN]
    test_features = dataframe.iloc[test_start:][FEATURE_COLUMNS]
    test_targets = dataframe.iloc[test_start:][TARGET_COLUMN]
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train_features, train_targets)
    predicted = model.predict(test_features)
    actual = test_targets.to_numpy()
    results = {
        "model": "RandomForestRegressor",
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "dataset": str(DATA_PATH),
        "split": "70/15/15 chronological",
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
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
