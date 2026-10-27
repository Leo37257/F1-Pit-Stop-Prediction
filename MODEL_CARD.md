# Model card: F1 pit-next-lap classifier

| | |
|---|---|
| **Model** | XGBoost (selected by FLAML AutoML) inside a scikit-learn pipeline: one-hot `Compound` and `TyreLife_bucket`, 29 numeric lap features passed through |
| **Output** | Probability that the driver pits on the next lap; a pit call when the probability is at least **0.36** |
| **Artifact** | `models/champion/` (MLflow sklearn flavour); the threshold is stored in `CHAMPION.json` and read by the API, CLI and monitoring |
| **Maintainer** | Leo Liu |

## Intended use

Analytics and demonstration: given a driver's lap-by-lap history in a stint, estimate the chance of a pit stop on the next
lap. It is **not** a race-strategy tool. It was trained on synthetic data and has never been checked against real team
decisions.

## Data

[Kaggle Playground Series S6E5, "Predicting F1 Pit Stops"](https://kaggle.com/competitions/playground-series-s6e5),
CC BY 4.0. The data is **synthetic**: lap numbers skip, tyre age and lap counts do not always agree, and the 2023 season
has a pit-next-lap rate of 1.0% against 27–30% in the other seasons.

| Split | Seasons | Laps | Pit-next-lap rate |
|---|---|---:|---:|
| Train | 2022–2024 | 346,246 | 26.7% / 1.0% / 29.5% |
| Test | 2025 | 92,894 | 28.4% |

## Performance (2025, never used for any choice)

| | Value |
|---|---:|
| macro-F1 at threshold 0.36 | **0.801** (0.785 at 0.5) |
| ROC-AUC | 0.894 |
| Brier score | 0.117 |
| Predicted vs actual pit rate | 32.1% vs 28.4% |

The threshold was chosen on out-of-fold predictions for 2022 and 2024 (5-fold GroupKFold by season and race,
[`evaluation/threshold_tuning.py`](evaluation/threshold_tuning.py)).

## Where it is weaker

| Slice | Laps | macro-F1 |
|---|---:|---:|
| Compound: MEDIUM / INTERMEDIATE | 47,783 / 3,366 | 0.798 / 0.799 |
| Compound: HARD | 34,034 | 0.748 |
| Compound: SOFT | 7,709 | **0.724** |
| Race: Miami / Australia / Austria | 1,287 / 2,533 / 4,323 | **0.606** / 0.653 / 0.694 |

Full table: [`monitoring_reports/slice_performance.csv`](monitoring_reports/slice_performance.csv).

**What did not fix it** ([`evaluation/feature_experiments.py`](evaluation/feature_experiments.py), same protocol, selection
on out-of-fold scores only):

| Added features | Out-of-fold macro-F1 | 2025 macro-F1 |
|---|---:|---:|
| none (champion) | 0.7856 | 0.8010 |
| race as a category | 0.7709 | not run |
| driver's race history (pits so far, compounds used, laps into stint) | 0.7862 | 0.8005 |
| both | 0.7722 | not run |

Race identity makes it worse (the model memorises track noise); history features are neutral. The champion was kept.

## Monitoring

- Model-free drift checks (PSI, KS) with thresholds: PSI 0.10 warn / 0.25 alert; macro-F1 drop 0.03 warn / 0.07 alert.
- Natural drift is measured against the 2024 season. Against 2024, 2 of 31 features alert and macro-F1 holds.
- The Airflow DAG runs `monitor_drift` after each retrain and fails the run only if macro-F1 drops past the alert line.
- Injected stress tests show `TyreLife` is the most sensitive input: +20 laps costs 0.207 macro-F1.

## Limitations

- Synthetic training data; real-world performance is unknown.
- Lag and rolling features need the driver's full stint history in each request; single laps get missing lags.
- Slightly under-confident on average (mean probability 0.258 vs observed 0.284).
- The 2023 season is unrepresentative; it is kept in training as the original team pipeline did, but excluded from threshold tuning and from the drift baseline.
- The public demo runs on a free Render instance that sleeps when idle.
