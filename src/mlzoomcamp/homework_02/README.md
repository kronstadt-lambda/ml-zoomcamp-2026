# Homework 02 — Module 2: Machine Learning for Regression

> Official assignment: [ML Zoomcamp 2026 Homework 2](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/02-regression/homework.md)

## Overview

This homework builds and evaluates linear regression models for predicting car fuel efficiency. It covers exploratory analysis, missing-value handling, regularization, split-seed stability, and final test evaluation.

## Results

| Question | Topic | Documented result | Evidence status |
|---|---|---|---|
| Q1 | Missing values | `horsepower` | Verified execution |
| Q2 | Horsepower median | `254.0` | Verified execution |
| Q3 | Imputation comparison | Mean imputation has the lower rounded validation RMSE (`2.202` vs. `2.205`) | Verified execution |
| Q4 | Ridge regularization | `r=100` has the lowest rounded validation RMSE (`2.2052`) | Verified execution |
| Q5 | Split-seed stability | Standard deviation of validation RMSE: `0.029` | Verified execution |
| Q6 | Final test evaluation | Test RMSE: `2.236` | Verified execution |

## Implementation approach

The implementation normalizes column labels, derives `vehicle_age` from `model_year` using 2026 as the reference year, and models the numerical predictors `engine_displacement`, `horsepower`, `vehicle_weight`, and `vehicle_age`. The numerical pipeline imputes missing horsepower values and standardizes the specified predictors before model fitting.

The dataset is split into 60% training, 20% validation, and 20% test data with the requested random seeds. Q3 compares zero and training-mean imputation with unregularized linear regression. Q4 evaluates Ridge candidates on validation data, Q5 measures variation across seeds, and Q6 retrains Ridge on combined training and validation data before evaluating the held-out test set.

## Reproducibility

From the repository root:

```bash
uv sync
uv run python -m mlzoomcamp.homework_02.main
```

The script displays exploratory plots and prints the question results. For a non-interactive environment, run:

```bash
MPLBACKEND=Agg uv run python -m mlzoomcamp.homework_02.main
```

## Data and dependencies

Place `car_fuel_efficiency_2026.csv` in `data/`. The dataset is available from the [official 2026 course release](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv). The implementation uses NumPy, pandas, Matplotlib, seaborn, and scikit-learn, as declared in `pyproject.toml`.

## Notes and limitations

`vehicle_age` is an affine transformation of `model_year`; it is used in the implementation to express the temporal feature as vehicle age. The reported values document this implementation's verified execution and do not indicate homework-platform submission or acceptance.
