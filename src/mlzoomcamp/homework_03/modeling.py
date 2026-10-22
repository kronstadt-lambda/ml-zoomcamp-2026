"""Classification-model helpers used by Homework 03."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def build_logistic_regression(
    *,
    solver: str = "lbfgs",
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int | None = None,
    class_weight: dict[int, float] | str | None = None,
) -> LogisticRegression:
    """Fit and return a binary logistic-regression estimator."""

    if C <= 0:
        raise ValueError("C must be greater than zero")
    if max_iter < 1:
        raise ValueError("max_iter must be greater than zero")

    model = LogisticRegression(
        solver=solver,
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
    )

    return model


def predict_classes(
    model: Any,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Return hard class predictions from a fitted classifier."""

    return np.asarray(model.predict(X))
