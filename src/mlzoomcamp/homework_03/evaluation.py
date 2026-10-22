"""Classification-evaluation helpers used by Homework 03."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def binary_accuracy(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Calculate binary classification accuracy from hard predictions."""

    return float(accuracy_score(y_true, y_pred))
