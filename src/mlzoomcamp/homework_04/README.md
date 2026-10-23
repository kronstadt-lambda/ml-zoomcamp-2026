# Homework 04 — Module 4: Evaluation Metrics for Classification

> Official assignment: [ML Zoomcamp 2026 Homework 04](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/04-evaluation/homework.md)

## Overview

This homework evaluates a binary lead-conversion classifier using ROC AUC,
precision, recall, F1, five-fold cross-validation, and regularization tuning.
It uses the 2026 lead-scoring dataset with `converted` as the target.

## Results

| Question | Topic | Documented result | Evidence status |
|---|---|---|---|
| Q1 | Highest numerical ROC AUC | `lead_score` (`0.788`) | Verified execution |
| Q2 | Validation ROC AUC | `0.732` | Verified execution |
| Q3 | Precision/recall intersection threshold | `0.63` | Verified execution |
| Q4 | Maximum F1 threshold | `0.41` | Verified execution |
| Q5 | Five-fold ROC AUC standard deviation | `0.007` | Verified execution |
| Q6 | Best regularization value by mean five-fold ROC AUC | `C = 0.001` (mean `0.749`, std `0.010`) | Verified execution |

The results were produced locally using the bundled pinned dataset and the
locked environment. They document repository execution only; platform
submission and acceptance are not asserted here.

## Implementation approach

- Missing numerical feature values are replaced with `0.0`; missing categorical
  values are replaced with `"NA"`.
- The dataset uses the assignment's exact 60/20/20 train/validation/test split
  with `random_state=1`.
- Q1 calculates ROC AUC separately for each specified numerical feature and
  reverses score direction only when its raw AUC is below `0.5`.
- Q2 one-hot encodes record dictionaries with `DictVectorizer` and trains
  logistic regression with `solver="liblinear"`, `C=1.0`, and `max_iter=1000`.
- Q3 and Q4 evaluate thresholds from `0.00` through `1.00` in steps of `0.01`.
- Q5 and Q6 use five-fold `KFold(shuffle=True, random_state=1)` validation;
  the vectorizer is fitted independently within every training fold.

## Reproducibility

From the repository root:

```bash
uv sync
uv run python -m mlzoomcamp.homework_04.main
```

The script displays the precision/recall plot. In a non-interactive
environment, use:

```bash
MPLBACKEND=Agg uv run python -m mlzoomcamp.homework_04.main
```

## Data and dependencies

- Dataset: `data/course_lead_scoring_2026.csv`
- Dataset source: [pinned 2026 course release](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv)
- Direct dependencies: NumPy, pandas, Matplotlib, and scikit-learn
- Environment management and locked versions: `uv`, `pyproject.toml`, and
  `uv.lock`

## Notes and limitations

The evaluation results are tied to the pinned dataset, exact split, and stated
cross-validation configuration. The plot is an execution aid and is not saved
as a repository artifact.
