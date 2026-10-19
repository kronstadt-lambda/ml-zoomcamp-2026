"""Binary-classification fitting and probability helpers."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def sigmoid(scores: float | np.ndarray) -> float | np.ndarray:
    """Apply the logistic sigmoid used in the didactic classification material."""

    values = np.asarray(scores, dtype=float)
    probabilities = np.empty_like(values, dtype=float)
    positive = values >= 0
    probabilities[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    probabilities[~positive] = exp_values / (1 + exp_values)

    return float(probabilities) if probabilities.ndim == 0 else probabilities


def fit_logistic_regression(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
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

    return model.fit(X, y)


def predict_positive_probabilities(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    *,
    positive_class: int | str = 1,
) -> np.ndarray:
    """Return probabilities for an explicit positive class from a fitted model."""

    classes = np.asarray(model.classes_)
    matching_indices = np.flatnonzero(classes == positive_class)
    if len(matching_indices) != 1:
        raise ValueError(f"positive_class '{positive_class}' is not present in model.classes_")

    probabilities = model.predict_proba(X)

    return np.asarray(probabilities[:, matching_indices[0]], dtype=float)


def predict_classes(
    model: Any,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Return hard class predictions from a fitted classifier."""

    return np.asarray(model.predict(X))
