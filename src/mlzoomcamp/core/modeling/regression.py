"""Regression fitting and prediction helpers.

Scikit-learn estimators and the didactic normal equation are intentionally
exposed through distinct functions.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


def fit_linear_regression(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    fit_intercept: bool = True,
    positive: bool = False,
    n_jobs: int | None = None,
) -> LinearRegression:
    """Fit and return scikit-learn's LinearRegression estimator."""

    model = LinearRegression(
        fit_intercept=fit_intercept,
        positive=positive,
        n_jobs=n_jobs,
    )

    return model.fit(X, y)


def fit_ridge_regression(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    alpha: float,
    fit_intercept: bool = True,
    solver: str = "auto",
    random_state: int | None = None,
) -> Ridge:
    """Fit and return a Ridge estimator using explicit regularization."""

    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    model = Ridge(
        alpha=alpha,
        fit_intercept=fit_intercept,
        solver=solver,
        random_state=random_state,
    )

    return model.fit(X, y)


def predict_regression(
    model: Any,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Return numeric predictions from a fitted regression estimator."""

    return np.asarray(model.predict(X), dtype=float)


def fit_normal_equation(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    regularization: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Fit the didactic normal-equation regression implementation.

    ``regularization`` adds a value to every diagonal element of the Gram
    matrix, matching the regularized mechanism studied in the course.
    """

    if regularization < 0:
        raise ValueError("regularization must be non-negative")

    feature_matrix = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float).reshape(-1)
    if feature_matrix.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if len(feature_matrix) != len(target):
        raise ValueError("X and y must contain the same number of rows")

    design_matrix = np.column_stack([np.ones(feature_matrix.shape[0]), feature_matrix])
    gram_matrix = design_matrix.T @ design_matrix
    if regularization:
        gram_matrix = gram_matrix + regularization * np.eye(gram_matrix.shape[0])

    weights = np.linalg.solve(gram_matrix, design_matrix.T @ target)

    return float(weights[0]), weights[1:]


def predict_from_weights(
    X: pd.DataFrame | np.ndarray,
    intercept: float,
    weights: np.ndarray,
) -> np.ndarray:
    """Predict with intercept and weights returned by ``fit_normal_equation``."""

    feature_matrix = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if feature_matrix.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if feature_matrix.shape[1] != len(weights):
        raise ValueError("weights must match the number of feature columns")

    return intercept + feature_matrix @ weights
