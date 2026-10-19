"""Plots for prepared data, predictions, and evaluation results."""

from collections.abc import Sequence
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from mlzoomcamp.core.features.analysis import correlation_matrix


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    *,
    bins: int = 40,
    color: str = "black",
    alpha: float = 1.0,
    xlabel: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot a histogram for a numerical DataFrame column."""

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x=column, bins=bins, color=color, alpha=alpha, ax=ax)
    ax.set_ylabel("Frequency")
    ax.set_xlabel(xlabel or column)
    ax.set_title(title or f"Distribution of {column}")

    return ax


def plot_missing_values(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    color: str = "steelblue",
    title: str = "Missing values by column",
    ax: Axes | None = None,
) -> Axes:
    """Plot missing-value counts for selected columns."""

    selected_columns = list(columns) if columns is not None else list(df.columns)
    missing = df.loc[:, selected_columns].isna().sum().sort_values(ascending=False)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=missing.to_numpy(), y=missing.index, color=color, errorbar=None, ax=ax)
    ax.set_xlabel("Number of missing values")
    ax.set_ylabel("Column")
    ax.set_title(title)

    return ax


def plot_numeric_relationships(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot scatterplots of numerical features against a target."""

    selected_features = list(features)
    if not selected_features:
        raise ValueError("features must contain at least one column")
    if ncols < 1:
        raise ValueError("ncols must be greater than zero")

    nrows = ceil(len(selected_features) / ncols)
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)
    figure, axes_grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = list(axes_grid.flat)

    for axis, feature in zip(axes, selected_features):
        sns.scatterplot(data=df, x=feature, y=target, alpha=0.6, ax=axis)
        axis.set_title(f"{target} vs {feature}")
    for axis in axes[len(selected_features):]:
        figure.delaxes(axis)
    figure.tight_layout()

    return figure, axes[: len(selected_features)]


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    annot: bool = True,
    cmap: str = "coolwarm",
    title: str = "Correlation matrix",
    ax: Axes | None = None,
) -> Axes:
    """Plot a heatmap from a prepared numerical correlation matrix."""

    correlations = correlation_matrix(df, columns)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlations, annot=annot, cmap=cmap, vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)

    return ax


def plot_feature_distributions(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    ncols: int = 2,
    bins: int = 30,
    kde: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot histograms for selected numerical feature columns."""

    selected_columns = (
        list(df.select_dtypes(include="number").columns)
        if columns is None
        else list(columns)
    )
    if not selected_columns:
        raise ValueError("columns must contain at least one numerical column")
    if ncols < 1:
        raise ValueError("ncols must be greater than zero")

    nrows = ceil(len(selected_columns) / ncols)
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)
    figure, axes_grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = list(axes_grid.flat)

    for axis, column in zip(axes, selected_columns):
        sns.histplot(data=df, x=column, bins=bins, kde=kde, color="steelblue", ax=axis)
        axis.set_title(f"Distribution of {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Frequency")
    for axis in axes[len(selected_columns):]:
        figure.delaxes(axis)
    figure.tight_layout()

    return figure, axes[: len(selected_columns)]


def plot_actual_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot regression predictions against their true values."""

    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=true_values, y=predicted_values, alpha=0.6, ax=ax)
    lower = min(true_values.min(), predicted_values.min())
    upper = max(true_values.max(), predicted_values.max())
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black")
    ax.set_xlabel("Actual value")
    ax.set_ylabel("Predicted value")
    ax.set_title("Actual vs predicted")

    return ax


def plot_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot regression residuals against predicted values."""

    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=predicted_values, y=true_values - predicted_values, alpha=0.6, ax=ax)
    ax.axhline(0, linestyle="--", color="black")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs predicted")

    return ax


def plot_confusion_matrix(
    matrix: np.ndarray,
    *,
    display_labels: Sequence[str] = ("Negative", "Positive"),
    ax: Axes | None = None,
) -> Axes:
    """Plot a precomputed binary confusion matrix without fitting a model."""

    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix),
        display_labels=display_labels,
    )
    display.plot(ax=ax, colorbar=False)

    return display.ax_


def plot_roc_curve(
    y_true: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot an ROC curve from true binary labels and positive probabilities."""

    display = RocCurveDisplay.from_predictions(y_true, y_probability, ax=ax)

    return display.ax_
