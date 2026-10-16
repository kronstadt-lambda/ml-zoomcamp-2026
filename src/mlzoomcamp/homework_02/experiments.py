from typing import Literal, Sequence, Mapping

import numpy as np
import pandas as pd

from mlzoomcamp.homework_02.modeling import (
    evaluate_rmse,
    train_linear_regression,
    train_ridge_regression,
)
from mlzoomcamp.homework_02.preprocessing import (
    build_numeric_preprocessor,
    prepare_train_eval,
    train_val_test_split,
)


# -------------------------------------------------------------------------------------
# Q3 Which option gives better RMSE? (imputation met)
# -------------------------------------------------------------------------------------
def evaluate_linear_regression_strategy(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    impute_columns: Sequence[str] = (),
    impute_strategy: Literal[
        "mean",
        "median",
        "most_frequent",
        "constant",
    ] = "mean",
    fill_value: float | None = None,
    scale_columns: Sequence[str] = (),
) -> float:
    """Train and evaluate one preprocessing/modeling strategy."""

    preprocessor = build_numeric_preprocessor(
        impute_columns=impute_columns,
        impute_strategy=impute_strategy,
        fill_value=fill_value,
        scale_columns=scale_columns,
    )

    (
        X_train_prepared,
        y_train,
        X_eval_prepared,
        y_eval,
    ) = prepare_train_eval(
        df_train,
        df_eval,
        features=features,
        target=target,
        preprocessor=preprocessor,
    )

    model = train_linear_regression(
        X_train_prepared,
        y_train,
    )

    return evaluate_rmse(model, X_eval_prepared, y_eval)


def compare_zero_and_mean_imputation(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    impute_columns: Sequence[str],
    scale_columns: Sequence[str] = (),
) -> dict[str, float]:
    """Compare zero and mean imputation on a validation dataset."""

    return {
        "zero": evaluate_linear_regression_strategy(
            df_train,
            df_val,
            features=features,
            target=target,
            impute_columns=impute_columns,
            impute_strategy="constant",
            fill_value=0,
            scale_columns=scale_columns,
        ),
        "mean": evaluate_linear_regression_strategy(
            df_train,
            df_val,
            features=features,
            target=target,
            impute_columns=impute_columns,
            impute_strategy="mean",
            scale_columns=scale_columns,
        ),
    }


# -------------------------------------------------------------------------------------
# Q4 There's one column with missing values. What is it? (r diffs)
# -------------------------------------------------------------------------------------
def tune_ridge_regularization(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    impute_columns: Sequence[str],
    scale_columns: Sequence[str] = (),
    alphas: Sequence[float] = (
        0,
        0.01,
        0.1,
        1,
        5,
        10,
        100,
    ),
) -> dict[float, float]:
    """Evaluate Ridge candidates on the validation dataset."""

    preprocessor = build_numeric_preprocessor(
        impute_columns=impute_columns,
        impute_strategy="constant",
        fill_value=0,
        scale_columns=scale_columns,
    )

    (
        X_train_prepared,
        y_train,
        X_val_prepared,
        y_val,
    ) = prepare_train_eval(
        df_train,
        df_val,
        features=features,
        target=target,
        preprocessor=preprocessor,
    )

    scores = {}

    for alpha in alphas:
        model = train_ridge_regression(
            X_train_prepared,
            y_train,
            alpha=alpha,
        )

        scores[alpha] = evaluate_rmse(
            model,
            X_val_prepared,
            y_val,
        )

    return scores


def select_best_alpha(
    scores: Mapping[float, float],
) -> float:
    """Select the alpha with the lowest rounded RMSE."""

    if not scores:
        raise ValueError("scores cannot be empty")

    return min(
        scores,
        key=lambda alpha: (
            round(scores[alpha], 4),
            alpha,
        ),
    )


# -------------------------------------------------------------------------------------
# Q5 What's the value of std? (seeds diffs)
# -------------------------------------------------------------------------------------
def evaluate_seed_scores(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    seeds: Sequence[int],
    impute_columns: Sequence[str],
    scale_columns: Sequence[str] = (),
) -> dict[int, float]:
    """Evaluate validation RMSE for multiple split seeds."""

    scores = {}

    for seed in seeds:
        df_train, df_val, _ = train_val_test_split(
            df,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=seed,
        )

        scores[seed] = evaluate_linear_regression_strategy(
            df_train,
            df_val,
            features=features,
            target=target,
            impute_columns=impute_columns,
            impute_strategy="constant",
            fill_value=0,
            scale_columns=scale_columns,
        )

    return scores


def calculate_score_std(
    scores: Mapping[int, float],
) -> float:
    """Calculate the standard deviation of validation scores."""

    return float(np.std(list(scores.values())))

# -------------------------------------------------------------------------------------
# Q6 What's the RMSE on the test dataset? (test dataset)
# -------------------------------------------------------------------------------------
def evaluate_ridge_on_test(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    impute_columns: Sequence[str],
    scale_columns: Sequence[str] = (),
    seed: int = 9,
    alpha: float = 0.001,
) -> float:
    """Train Ridge on train+validation and evaluate it on test."""

    df_train, df_val, df_test = train_val_test_split(
        df,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=seed,
    )

    df_train_val = pd.concat(
        [df_train, df_val],
        ignore_index=True,
    )

    preprocessor = build_numeric_preprocessor(
        impute_columns=impute_columns,
        impute_strategy="constant",
        fill_value=0,
        scale_columns=scale_columns,
    )

    (
        X_train_val,
        y_train_val,
        X_test,
        y_test,
    ) = prepare_train_eval(
        df_train_val,
        df_test,
        features=features,
        target=target,
        preprocessor=preprocessor,
    )

    model = train_ridge_regression(
        X_train_val,
        y_train_val,
        alpha=alpha,
    )

    return evaluate_rmse(
        model,
        X_test,
        y_test,
    )
