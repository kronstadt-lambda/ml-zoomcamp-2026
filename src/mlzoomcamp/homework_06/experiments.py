"""Random-forest experiment helpers for Homework 06."""
from collections.abc import Sequence

import pandas as pd
import xgboost as xgb
from scipy.sparse import spmatrix

from mlzoomcamp.homework_06.evaluation import calculate_rmse
from mlzoomcamp.homework_06.modeling import (
    train_random_forest_regressor,
)


def evaluate_random_forest_estimators(
    X_train: spmatrix,
    y_train: pd.Series,
    X_val: spmatrix,
    y_val: pd.Series,
    *,
    n_estimators_values: Sequence[int],
    random_state: int,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Evaluate validation RMSE for each requested estimator count."""

    results = []

    for n_estimators in n_estimators_values:
        model = train_random_forest_regressor(
            X_train,
            y_train,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        y_val_pred = model.predict(X_val)
        rmse = calculate_rmse(y_val, y_val_pred)

        results.append(
            {
                "n_estimators": n_estimators,
                "rmse": rmse,
                "rmse_rounded": round(rmse, 3),
            }
        )

    return pd.DataFrame(results)

def evaluate_random_forest_depth_grid(
    X_train: spmatrix,
    y_train: pd.Series,
    X_val: spmatrix,
    y_val: pd.Series,
    *,
    max_depth_values: Sequence[int],
    n_estimators_values: Sequence[int],
    random_state: int,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Evaluate validation RMSE for every depth and estimator combination."""

    results = []

    for max_depth in max_depth_values:
        for n_estimators in n_estimators_values:
            model = train_random_forest_regressor(
                X_train,
                y_train,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=n_jobs,
            )

            y_val_pred = model.predict(X_val)
            rmse = calculate_rmse(y_val, y_val_pred)

            results.append(
                {
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                    "rmse": rmse,
                }
            )

    return pd.DataFrame(results)


def mean_rmse_by_max_depth(
    grid_results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the mean validation RMSE for each maximum depth."""

    required_columns = {"max_depth", "rmse"}

    if not required_columns.issubset(grid_results.columns):
        raise ValueError(
            "grid_results must contain max_depth and rmse columns"
        )

    return (
        grid_results.groupby("max_depth", as_index=False)["rmse"]
        .mean()
        .rename(columns={"rmse": "mean_rmse"})
        .sort_values("max_depth", ignore_index=True)
    )

def evaluate_xgboost_eta(
    X_train: spmatrix,
    y_train: pd.Series,
    X_val: spmatrix,
    y_val: pd.Series,
    *,
    feature_names: Sequence[str],
    eta_values: Sequence[float],
    num_boost_round: int,
) -> pd.DataFrame:
    """Evaluate validation RMSE for each requested XGBoost eta value."""

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=list(feature_names),
    )
    dval = xgb.DMatrix(
        X_val,
        label=y_val,
        feature_names=list(feature_names),
    )

    watchlist = [
        (dtrain, "train"),
        (dval, "validation"),
    ]

    results = []

    for eta in eta_values:
        xgb_params = {
            "eta": eta,
            "max_depth": 6,
            "min_child_weight": 1,
            "objective": "reg:squarederror",
            "nthread": 8,
            "seed": 1,
            "verbosity": 1,
        }

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            verbose_eval=False,
        )

        y_val_pred = model.predict(dval)
        rmse = calculate_rmse(y_val, y_val_pred)

        results.append(
            {
                "eta": eta,
                "rmse": rmse,
            }
        )

    return pd.DataFrame(results)
