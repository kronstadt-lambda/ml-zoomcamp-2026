# Homework 01 — Module 1: Introduction to Machine Learning

> Official assignment: [ML Zoomcamp 2026 cohort homework](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework)

## Overview

This homework practices basic Python data analysis with Pandas and NumPy using the 2026 Car Fuel Efficiency dataset. The implementation checks package information, inspects records and missing values, filters observations, imputes a column with its mode, and performs the matrix operations specified in the assignment.

## Results

| Question | Topic | Documented result | Evidence status |
|---|---|---|---|
| Q1 | Pandas version | `3.0.6` | Verified execution environment |
| Q2 | Dataset record count | `10000` | Verified execution |
| Q3 | Fuel types | `3` | Verified execution |
| Q4 | Columns with missing values | `2` | Verified execution |
| Q5 | Maximum Asia fuel efficiency | `41.2` | Verified execution |
| Q6 | Horsepower median after mode imputation | Changed: `254.0` to `252.0` (decreased); mode: `252.0` | Verified execution |
| Q7 | Sum of linear-regression weights | `0.36919696904925486` (matching the `0.369` option) | Verified execution |

The results above were produced locally from the bundled dataset and the current locked environment. They document execution evidence only; platform submission and acceptance are not asserted here.

## Implementation approach

`main.py` loads `data/car_fuel_efficiency_2026.csv` through the shared `DATA_DIR` path. It uses Pandas for the exploratory questions and missing-value imputation, then NumPy to compute `X.T @ X`, its inverse, and the resulting weight vector for Q7.

## Reproducibility

From the repository root:

```bash
uv sync
uv run python -m mlzoomcamp.homework_01.main
```

The module command was verified locally. The Q1 expression is evaluated in the source without being printed by the script; its documented value was verified separately in the same `uv` environment.

## Data and dependencies

- Dataset: `data/car_fuel_efficiency_2026.csv`
- Dataset source: [pinned 2026 course release](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv)
- Direct dependencies used by this homework: Pandas and NumPy
- Environment management and locked versions: `uv`, `pyproject.toml`, and `uv.lock`

## Notes and limitations

- `pytest` currently discovers no tests for this repository.
- Ruff reports two non-functional style findings in `main.py`: import ordering and the standalone Q1 version expression. They were intentionally left unchanged because the academic implementation belongs to the student.
