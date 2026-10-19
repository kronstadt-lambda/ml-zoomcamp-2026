"""Regression metrics with clear true-target and prediction inputs."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Calculate mean absolute error."""

    return float(mean_absolute_error(y_true, y_pred))


def mse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Calculate mean squared error."""

    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Calculate root mean squared error."""

    return float(root_mean_squared_error(y_true, y_pred))


def regression_summary(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> pd.Series:
    """Return MAE, MSE, and RMSE in an interpretable metric summary."""

    return pd.Series(
        {
            "mae": mae(y_true, y_pred),
            "mse": mse(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
        },
        dtype=float,
    )
