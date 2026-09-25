"""Run this once after every train.py run to refresh the dashboard.

Merges the freshly computed forecast/anomaly numbers (dashboard_data.json,
written by train.py) into the existing `const DATA = {...};` object embedded
in dashboard/gridveil.html — without touching forecast_labels or the OVERVIEW
block, which are unrelated to the model run.

Usage (from GridVeil_Project/):
    python sync_dashboard_data.py
"""

import json
import re
from pathlib import Path

DASHBOARD_HTML = Path(__file__).parent / "dashboard/gridveil.html"
DASHBOARD_JSON = Path(__file__).parent / "dashboard/dashboard_data.json"


def main():
    html_path = DASHBOARD_HTML.resolve()
    json_path = DASHBOARD_JSON.resolve()

    html = html_path.read_text(encoding="utf-8")
    new_fields = json.loads(json_path.read_text())

    match = re.search(r"const DATA = (\{.*?\});", html, re.DOTALL)
    if not match:
        raise SystemExit("Could not find 'const DATA = {...};' in the dashboard HTML.")

    current_data = json.loads(match.group(1))
    # Keep forecast_labels (static hour labels) as-is; overwrite everything
    # else with the values just computed by train.py.
    current_data.update(new_fields)

    new_data_str = "const DATA = " + json.dumps(current_data) + ";"
    updated_html = html[:match.start()] + new_data_str + html[match.end():]
    html_path.write_text(updated_html, encoding="utf-8")
    print(f"Synced {json_path.name} into {html_path}")


if __name__ == "__main__":
    main()
