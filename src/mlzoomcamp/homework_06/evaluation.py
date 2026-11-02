import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def calculate_rmse(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Return root mean squared error for regression predictions."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    return float(root_mean_squared_error(y_true, y_pred))
