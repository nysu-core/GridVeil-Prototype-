"""Data loading, feature engineering, scaling, and sequence preparation for GridVeil."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).with_name("powerconsumption.csv")
TARGET_COLUMNS = [
    "PowerConsumption_Zone1",
    "PowerConsumption_Zone2",
    "PowerConsumption_Zone3",
]
SEQUENCE_LENGTH = 24
EXPECTED_INTERVAL = pd.Timedelta(minutes=10)


def load_data(file_path=DATA_PATH):
    """Load the raw data, parse timestamps, and sort it chronologically."""
    dataframe = pd.read_csv(file_path)
    dataframe["Datetime"] = pd.to_datetime(dataframe["Datetime"])
    dataframe = dataframe.sort_values("Datetime").reset_index(drop=True)
    return dataframe


def verify_time_steps(dataframe):
    """Print and return monotonicity and 10-minute interval checks."""
    timestamps = dataframe["Datetime"]
    differences = timestamps.diff().dropna()
    is_strictly_monotonic = timestamps.is_monotonic_increasing and not timestamps.duplicated().any()
    is_equidistant = differences.eq(EXPECTED_INTERVAL).all()
    missing_values = dataframe.isnull().sum().sum()

    print(f"Time steps strictly monotonic: {is_strictly_monotonic}")
    print(f"Time steps equidistant at 10 minutes: {is_equidistant}")
    print(f"Total missing values: {missing_values}")
    return is_strictly_monotonic, is_equidistant


def create_features(dataframe):
    """Create calendar, behavioral, and cumulative time features."""
    dataframe = dataframe.copy()
    datetime_values = dataframe["Datetime"]
    iso_calendar = datetime_values.dt.isocalendar()

    dataframe["hour"] = datetime_values.dt.hour
    dataframe["minute"] = datetime_values.dt.minute
    dataframe["dayofweek"] = datetime_values.dt.dayofweek
    dataframe["dayofmonth"] = datetime_values.dt.day
    dataframe["dayofyear"] = datetime_values.dt.dayofyear
    dataframe["weekofyear"] = iso_calendar.week.astype(int)
    dataframe["month"] = datetime_values.dt.month
    dataframe["quarter"] = datetime_values.dt.quarter
    dataframe["season"] = datetime_values.dt.month.map(
        {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
         6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
    )

    dataframe["is_weekend"] = (dataframe["dayofweek"] >= 5).astype(int)
    dataframe["is_working_day"] = (dataframe["dayofweek"] < 5).astype(int)
    dataframe["is_business_hours"] = datetime_values.dt.hour.between(9, 17).astype(int)
    dataframe["is_peak_hour"] = datetime_values.dt.hour.isin([8, 12, 18]).astype(int)
    dataframe["is_month_start"] = datetime_values.dt.is_month_start.astype(int)
    dataframe["is_month_end"] = datetime_values.dt.is_month_end.astype(int)
    dataframe["is_quarter_start"] = datetime_values.dt.is_quarter_start.astype(int)
    dataframe["is_quarter_end"] = datetime_values.dt.is_quarter_end.astype(int)

    dataframe["minute_of_day"] = datetime_values.dt.hour * 60 + datetime_values.dt.minute
    dataframe["minute_of_week"] = dataframe["dayofweek"] * 1440 + dataframe["minute_of_day"]

    return dataframe.set_index("Datetime")


def create_sliding_windows(features, targets, sequence_length=SEQUENCE_LENGTH):
    """Convert feature and target arrays into (samples, timesteps, features) tensors."""
    if len(features) != len(targets):
        raise ValueError("Features and targets must contain the same number of rows.")
    if len(features) < sequence_length:
        raise ValueError("The data split must contain at least one complete sequence.")

    feature_windows = np.stack(
        [features[start : start + sequence_length] for start in range(len(features) - sequence_length + 1)]
    )
    target_windows = np.stack(
        [targets[start : start + sequence_length] for start in range(len(targets) - sequence_length + 1)]
    )
    return feature_windows, target_windows


def preprocess_data(file_path=DATA_PATH, sequence_length=SEQUENCE_LENGTH):
    """Run the complete leakage-safe preprocessing pipeline."""
    dataframe = load_data(file_path)
    verify_time_steps(dataframe)
    engineered_data = create_features(dataframe)

    feature_columns = [column for column in engineered_data.columns if column not in TARGET_COLUMNS]
    feature_data = engineered_data[feature_columns].copy()
    target_data = engineered_data[TARGET_COLUMNS].copy()

    train_end = int(len(engineered_data) * 0.70)
    validation_end = train_end + int(len(engineered_data) * 0.15)
    split_slices = {
        "train": slice(0, train_end),
        "validation": slice(train_end, validation_end),
        "test": slice(validation_end, len(engineered_data)),
    }

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(feature_data.iloc[split_slices["train"]])
    target_scaler.fit(target_data.iloc[split_slices["train"]])

    sequences = {}
    for split_name, split_slice in split_slices.items():
        scaled_features = feature_scaler.transform(feature_data.iloc[split_slice])
        scaled_targets = target_scaler.transform(target_data.iloc[split_slice])
        sequences[split_name] = create_sliding_windows(
            scaled_features, scaled_targets, sequence_length
        )

    return {
        "data": engineered_data,
        "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "sequences": sequences,
        # FIX: exposed so downstream code (train.py) can recover real
        # timestamps for the test split instead of guessing offsets, and so
        # nothing ever confuses a split boundary with an anomaly count again.
        "split_slices": split_slices,
    }


if __name__ == "__main__":
    processed = preprocess_data()
    for split_name, (features, targets) in processed["sequences"].items():
        print(
            f"{split_name.capitalize()} tensors: "
            f"features={features.shape}, targets={targets.shape}"
        )