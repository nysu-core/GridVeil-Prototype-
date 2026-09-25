# GridVeil: Privacy-Preserving Load Forecasting and Anomaly Detection for Libya

The project addresses Libya's worsening electricity access and reliability crisis, driven by an aging, fossil-fuel-dependent grid and frequent, severe power outages worsened by years of conflict — a problem made harder to analyze by the fact that no public, granular electricity consumption data exists for Libya's own grid, leaving no option but sensitive or inaccessible operator data. To solve this, we built a privacy-preserving analytics prototype combining two public data sources: Libya's own real national electricity statistics (Our World in Data) to document and visualize the actual scale of the crisis, and, since Libya lacks public granular data, a comparable public dataset from Tetouan, Morocco, a North African region with broadly similar climate and demand patterns, to build and demonstrate a robust forecasting and anomaly-detection model utilizing a Hybrid CNN-LSTM-Attention architecture. This substitution is explicitly documented as a methodological proxy, not a claim about Libya's literal grid, and the project includes a full privacy and ethics framework, with every dataset classified, sourced, and assessed for privacy risk, alongside differential-privacy and synthetic-data techniques, ensuring no personally identifiable or sensitive data is used at any stage. The expected outcome is a working, reproducible prototype and dashboard that documents Libya's real electricity crisis and demonstrates a privacy-safe forecasting method on comparable regional data, ready to be applied directly to Libya's own grid once equivalent local data becomes publicly available.

## Overview

GridVeil is a privacy-preserving electricity analytics and forecasting prototype designed to document Libya's real power-system crisis and demonstrate a practical, reproducible method for forecasting and anomaly detection without relying on private or inaccessible grid data. The repository combines public energy-system context for Libya with a robust time-series forecasting pipeline trained on a comparable public dataset from Tetouan, Morocco.

The prototype is intended to serve two purposes:

1. To visualize the real scale of Libya's electricity infrastructure problem using vetted public national statistics.
2. To demonstrate a privacy-safe forecasting workflow and anomaly-detection layer that can be transferred to an operator dataset when such access becomes available.

This project emphasizes transparency, methodological caution, and data ethics. Where Libya itself lacks public granular load data, the project clearly identifies the proxy data source and the reasoning behind its use, rather than substituting it implicitly.

---

## Why This Project Exists

Libya faces a major electricity access and reliability problem. Structural factors include:

- an aging generation fleet dominated by fossil-fuel plants;
- chronic underinvestment and infrastructure degradation;
- rapid population growth and rising urban demand;
- recurring outages and supply insecurity linked to conflict and operational instability;
- the absence of publicly available high-resolution consumption data for the national grid.

Because of this data gap, a realistic, transparent, and privacy-aware forecasting solution cannot rely on direct access to sensitive utility records. The response is a two-track approach:

- use public national-level data to document Libya's actual system context;
- use a public comparable regional dataset to build the forecasting and anomaly-detection methodology.

This makes the prototype both analytically useful and ethically safe.

---

## Data Sources and Methodology

### Public national context for Libya

The project uses public statistics and modeling references for Libya, including national power-system indicators, generation fleet information, and cost trajectories for future renewables deployment. These figures are used to contextualize the crisis and quantify the broader energy-system environment.

### Proxy forecasting dataset

Because no public granular Libyan consumption series is available, the forecasting model is trained on a publicly available Tetouan, Morocco load dataset. Tetouan is a suitable methodological proxy because it shares a North African climate and demand profile with Libya, which makes it useful for demonstrating the forecasting and anomaly-detection pipeline.

This is explicitly documented as a proxy and not a claim that Tetouan data is Libya data.

### Privacy and ethics framework

All datasets used in the project are classified, sourced, and assessed for privacy risk. The project includes:

- explicit documentation of data provenance;
- a clear distinction between real national context data and proxy operational data;
- privacy-preserving methods for model development;
- no use of personally identifiable or sensitive consumer-level information.

---

## Repository Structure

```text
GridVeil_Prototype/
├── README.md
├── dashboard/
│   └── gridveil.html
├── data/
│   ├── libya/
│   │   ├── resid_cap_LBY.csv
│   │   ├── Table1_LBY.csv
│   │   └── Table3_LBY.csv
│   ├── synthetic/
│   │   └── Synthetic_Libya_Electricity.csv
│   └── tetouan/
│       ├── Tetouan_10min_TotalLoad.csv
│       ├── Tetouan_Electricity.csv
│       └── Tetouan_Hourly_Features.csv
├── results/
│   ├── best_gridveil_model.keras
│   └── training_loss_curve.png
└── .gitignore
```

> The repository no longer contains a local retraining pipeline or legacy model outputs. Training and privacy work were executed on Google Colab using a T4 GPU, and the final reusable assets are retained in the results directory.

---

## Quick Start

### View the dashboard

Open the dashboard directly in a browser:

```bash
# Open the file in a browser
start dashboard/gridveil.html
```

Alternatively, open `https://gridveil.netlify.app/` in any browser of your choice.

### Model training and privacy execution

Model training and Differential Privacy application were executed securely in Google Colab using a T4 GPU. These steps were not run as part of the local repository workflow, and the project is structured to preserve the final, verified model artifacts and dashboard outputs rather than requiring local retraining.

---

## Forecasting Model

### Final architecture

The project presents a single final forecasting architecture:

- Hybrid CNN-LSTM-Attention
- Trained on the trailing 4 hours of 10-minute-resampled load data
- Input window: trailing 24 (10-minute) load values (4 hours)
- Forecast horizon: next-hour load prediction
- Evaluation split: 70/15/15 strict chronological split

### Final model performance

| Metric | Value |
| --- | ---: |
| Model architecture | Hybrid CNN-LSTM-Attention |
| Dataset | Tetouan (10-minute-resampled load data) |
| Lookback window | 4 hours (24 samples) |
| Train/validation/test split | 70/15/15 strict chronological split |
| Test MAPE | 2.42% |

This model is the final reported architecture for the project and the one used in the dashboard narrative and evaluation summary.

---

## Privacy Experiment

The project includes a Differential Privacy experiment designed to quantify the cost of privacy-preserving noise in time-series forecasting.

| Privacy method | Configuration | Outcome |
| --- | --- | --- |
| Laplace Noise (Differential Privacy) | Epsilon = 1.0, applied before training | Successfully applied to the dataset before model training |

The DP layer is included as a research and validation component of the workflow. It demonstrates how privacy-preserving noise can be introduced without exposing sensitive raw data, while preserving the reproducibility of the forecasting pipeline.

---

## Anomaly Detection

Anomaly detection is performed using a dynamically calculated threshold based on the training error behavior of the Hybrid model rather than static percentage cutoffs.

| Metric | Value |
| --- | ---: |
| Dynamic anomaly threshold | Computed in original MW units |
| Detected anomalies in test set | Approximately 179 |

The threshold is computed from the 95th percentile of the model's training absolute errors after inverse-scaling to the original MW units, ensuring that anomaly detection is adaptive and data-driven. The old 7,834 value was a test-window count, not a verified anomaly count.

## Reproducibility status

The root-level `train.py` reports metrics after inverse-scaling, and the root-level `random_forest_baseline.py` provides the RF baseline implementation. The RF result must be generated by running the script; no fixed 5.77% result is claimed. The root-level `dp_experiment.py` provides a reproducible Laplace label-perturbation experiment, but it is not a formal end-to-end differential-privacy guarantee without a privacy accountant.

---

## Dashboard

The dashboard is implemented as a standalone HTML prototype and visualizes:

- national-level Libya energy context;
- forecasted load trajectories;
- model comparison baseline context;
- privacy experiment framing;
- anomaly detection findings.

The dashboard can be viewed by opening:

```text
dashboard/gridveil.html
```

in a browser.

---

## Ethics and Data Safety Notes

This project is intentionally designed to avoid reliance on personally identifiable or sensitive operational data. The methodology is structured around:

- public national statistics for Libya;
- a public comparable proxy dataset for forecasting;
- explicit documentation of methodological substitution;
- privacy-preserving training design;
- a clear distinction between evidence about Libya's real system and the proxy-based forecasting pipeline.

The project does not claim to have access to Libya's private grid records, and it does not use any sensitive consumer data in the final model or dashboard presentation.

---

## Expected Outcome

The final prototype delivers a practical demonstration of how Libya's electricity crisis can be documented with public statistics and how a privacy-safe forecasting workflow can be built on comparable regional data. The project is designed to be reproducible, transparent, and ready to apply directly to Libya's own grid once equivalent local data becomes publicly available.

---

## Relevant Files

- `dashboard/gridveil.html` — interactive prototype dashboard
- `data/libya/` — Libya national statistics and public context data
- `data/tetouan/` — public proxy time-series dataset used for forecasting
- `data/synthetic/` — synthetic data used for privacy experiment scaffolding
- `results/best_gridveil_model.keras` — final trained forecasting model
- `results/training_loss_curve.png` — training convergence visualization

This project is intended as a credible prototype and research artifact, not as a literal operational deployment on Libya's live grid infrastructure.
