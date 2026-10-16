"""Reusable plotting functions for exploratory data analysis."""

from math import ceil
from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd


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
    """Plot the distribution of a numerical column."""

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(
        data=df,
        x=column,
        bins=bins,
        color=color,
        alpha=alpha,
        ax=ax,
    )

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
    """Plot the number of missing values for each selected column."""

    selected_columns = list(columns) if columns is not None else list(df.columns)
    missing = df[selected_columns].isna().sum()
    missing = missing.sort_values(ascending=False)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        x=missing.to_numpy(),
        y=missing.index,
        color=color,
        errorbar=None,
        ax=ax,
    )

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
    """Plot scatterplots of several features against a target column."""

    features = list(features)

    if not features:
        raise ValueError("features must contain at least one column")

    if ncols < 1:
        raise ValueError("ncols must be greater than zero")

    nrows = ceil(len(features) / ncols)

    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)

    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )

    axes = list(axes_grid.flat)

    for ax, feature in zip(axes, features):
        sns.scatterplot(
            data=df,
            x=feature,
            y=target,
            alpha=0.6,
            ax=ax,
        )
        ax.set_title(f"{target} vs {feature}")

    for ax in axes[len(features):]:
        fig.delaxes(ax)

    fig.tight_layout()

    return fig, axes[:len(features)]


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    annot: bool = True,
    cmap: str = "coolwarm",
    title: str = "Correlation matrix",
    ax: Axes | None = None,
) -> Axes:
    """Plot a correlation heatmap for selected numerical columns."""

    selected_columns = list(columns) if columns is not None else list(df.columns)
    correlation = df[selected_columns].corr()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=annot,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        ax=ax,
    )

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
    """Plot the distributions of selected numerical features."""

    if columns is None:
        columns = df.select_dtypes(include="number").columns

    columns = list(columns)

    if not columns:
        raise ValueError("columns must contain at least one numerical column")

    if ncols < 1:
        raise ValueError("ncols must be greater than zero")

    nrows = ceil(len(columns) / ncols)

    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)

    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )

    axes = list(axes_grid.flat)

    for ax, column in zip(axes, columns):
        sns.histplot(
            data=df,
            x=column,
            bins=bins,
            kde=kde,
            color="steelblue",
            ax=ax,
        )
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

    for ax in axes[len(columns):]:
        fig.delaxes(ax)

    fig.tight_layout()

    return fig, axes[:len(columns)]
