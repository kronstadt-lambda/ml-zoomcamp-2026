# Homework 06 — Module 6: Decision Trees and Ensemble Learning

> Official assignment: [ML Zoomcamp 2026 Homework 06](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/06-trees/homework.md)

## Overview

This homework builds car fuel-efficiency regression models with a decision
tree, random forests, and XGBoost. It compares validation RMSE while varying
the number of trees, maximum tree depth, and XGBoost learning rate.

## Results

| Question | Topic | Documented result | Evidence status |
|---|---|---|---|
| Q1 | Decision-stump root feature | `model_year` | Verified execution |
| Q2 | 10-tree random-forest validation RMSE | `1.833` | Verified execution |
| Q3 | Best `n_estimators` by validation RMSE | `150` (`1.769`) | Verified execution |
| Q4 | Best `max_depth` by mean validation RMSE | `10` (`1.754`) | Verified execution |
| Q5 | Most important specified feature | `vehicle_weight` (`0.203313`) | Verified execution |
| Q6 | Best XGBoost `eta` | `0.1` (`1.724810`) | Verified execution |

The results were produced locally from the pinned course dataset in the locked
environment. They document repository execution only; platform submission and
acceptance are not asserted here.

## Implementation approach

- Missing values are filled with `0`, then data is split with the assignment's
  exact 60/20/20 train/validation/test procedure and `random_state=1`.
- Features are encoded with `DictVectorizer(sparse=True)`.
- Q1 trains a depth-one `DecisionTreeRegressor` and maps its vectorized root
  split back to the original feature column.
- Q2–Q5 evaluate `RandomForestRegressor` models on the validation split;
  one-hot feature importances are aggregated back to original columns for Q5.
- Q6 builds training and validation XGBoost `DMatrix` objects, uses a
  watchlist, and compares `eta` values `0.3` and `0.1` across 100 rounds.

## Reproducibility

From the repository root:

```bash
uv sync --locked
uv run python -m mlzoomcamp.homework_06.main
```

## Data and dependencies

- Dataset: `data/car_fuel_efficiency_2026.csv`
- Dataset source: [pinned 2026 course release](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv)
- Direct dependencies: pandas, SciPy, scikit-learn, and XGBoost
- Environment management and locked versions: `uv`, `pyproject.toml`, and
  `uv.lock`

## Notes and limitations

The reported metrics depend on the pinned dataset, exact split, package
versions, and stated random seed. Homework 06 has no dedicated automated
pytest tests; reproducibility is verified by running the module above.
