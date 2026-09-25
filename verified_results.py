"""Load and validate the supplied verified external experiment results."""

import json
from pathlib import Path


RESULTS_PATH = Path(__file__).parent / "results" / "verified_report_results.json"


def load_verified_results(path=RESULTS_PATH):
    """Return the verified result set and fail if a required metric is missing."""
    results = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "forecasting": ["mape_percent", "training_epochs"],
        "random_forest": ["mape_percent"],
        "anomaly_detection": ["threshold", "anomalies"],
        "differential_privacy": ["epsilon", "clean_mape_percent", "private_mape_percent"],
    }
    for section, fields in required.items():
        missing = [field for field in fields if field not in results.get(section, {})]
        if missing:
            raise ValueError(f"Missing verified result fields in {section}: {missing}")
    return results


def main():
    results = load_verified_results()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
