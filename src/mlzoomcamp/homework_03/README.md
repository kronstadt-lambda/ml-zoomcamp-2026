# Homework 03 — Machine Learning for Classification

> Official assignment: [ML Zoomcamp 2026 Homework 03](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/03-classification/homework.md)

## Overview

This homework builds a binary lead-conversion classifier for the `converted`
target. It covers missing-value handling, exploratory analysis, categorical
mutual information, one-hot encoding, logistic regression, feature
elimination, and regularization tuning.

## Results

| Question | Topic | Documented result | Evidence status |
|---|---|---|---|
| Q1 | Mode of `industry` | `technology` | Verified execution |
| Q2 | Strongest listed numerical correlation | `interaction_count` and `lead_score` | Verified execution |
| Q3 | Largest categorical mutual-information score, using training data only | `lead_source` | Verified execution |
| Q4 | Validation accuracy for logistic regression | `0.65` | Verified execution |
| Q5 | Smallest candidate feature-elimination difference | `number_of_courses_viewed` (`0.002`) | Verified execution |
| Q6 | Best regularization value | `C = 0.001` (validation accuracy `0.645`) | Verified execution |

## Implementation approach

- Numerical missing values are filled with `0.0`; categorical missing values
  are filled with `"NA"`.
- The target is separated before the reproducible 60/20/20 train/validation/test
  split (`random_state=42`).
- Mutual information is calculated on the training split only and rounded to
  two decimals for Q3.
- Categorical predictors are one-hot encoded with unknown categories ignored.
- Q4 uses `LogisticRegression(solver="liblinear", C=1.0, max_iter=1000,
  random_state=42)`.
- Q5 evaluates each raw feature exclusion against the unrounded Q4 baseline.
- Q6 evaluates `C` values `0.000001`, `0.00001`, `0.0001`, and `0.001`,
  selecting the smallest value in the event of a rounded tie.

## Reproducibility

From the repository root:

```bash
uv sync
uv run python -m mlzoomcamp.homework_03.main
```

For non-interactive environments that cannot display the exploratory plots:

```bash
MPLBACKEND=Agg uv run python -m mlzoomcamp.homework_03.main
```

## Data and dependencies

The required dataset is `data/course_lead_scoring_2026.csv`. It is the pinned
2026 lead-scoring release referenced by the official assignment. The project
uses Python 3.11 with pandas, scikit-learn, matplotlib, and seaborn, managed by
`uv`.

## Notes and limitations

The script displays exploratory plots during execution. The README documents
repository execution results only; it does not represent submission or
acceptance on the course platform.
