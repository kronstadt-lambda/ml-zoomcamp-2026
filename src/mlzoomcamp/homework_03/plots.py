"""Plots used by Homework 03 classification exploration."""

from collections.abc import Sequence
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a Pearson correlation matrix for selected numerical columns."""

    selected_columns = (
        list(df.select_dtypes(include="number").columns)
        if columns is None
        else list(columns)
    )

    return df.loc[:, selected_columns].corr()

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

    correlations = _correlation_matrix(df, columns)
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



# -------------------------------------------------------------------------------------
# Classification EDA
# -------------------------------------------------------------------------------------

def _classification_data(
    df: pd.DataFrame,
    target: str,
    positive_label: object,
) -> pd.DataFrame:
    """Validate a binary target and exclude records without a known target."""

    if target not in df.columns:
        raise KeyError(f"Target column not found: {target}")

    prepared = df.loc[df[target].notna()].copy()
    classes = prepared[target].unique()
    if len(classes) != 2:
        raise ValueError(f"{target!r} must contain exactly two observed classes")
    if positive_label not in classes:
        raise ValueError(f"Positive class {positive_label!r} not found in {target!r}")
    return prepared


def _classification_grid(
    features: Sequence[str],
    *,
    ncols: int,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, list[Axes]]:
    """Create a grid with one axis per feature."""

    if not features:
        raise ValueError("features must contain at least one column")
    if ncols < 1:
        raise ValueError("ncols must be greater than zero")

    nrows = ceil(len(features) / ncols)
    size = figsize or (6 * ncols, 4 * nrows)
    figure, axes_grid = plt.subplots(nrows, ncols, figsize=size, squeeze=False)
    axes = list(axes_grid.flat)
    for axis in axes[len(features):]:
        figure.delaxes(axis)
    return figure, axes[:len(features)]


def plot_target_balance(
    df: pd.DataFrame,
    target: str,
    *,
    positive_label: object = 1,
    ax: Axes | None = None,
) -> Axes:
    """Plot the class counts and percentages of a binary target."""

    prepared = _classification_data(df, target, positive_label)
    counts = prepared[target].value_counts().sort_index()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar([str(c) for c in counts.index], counts.to_numpy())
    total = counts.sum()
    for bar, count in zip(bars, counts.to_numpy()):
        ax.annotate(
            f"n={count} ({count / total:.1%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
        )
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_xlabel(target)
    ax.set_ylabel("Number of observations")
    ax.set_title(f"Class balance: {target}")
    return ax


def plot_continuous_by_target(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    positive_label: object = 1,
    kind: str = "box",
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Compare each continuous feature's distribution across the two classes.

    Missing feature values are omitted from the corresponding plot.
    """

    if kind not in {"box", "violin"}:
        raise ValueError("kind must be 'box' or 'violin'")
    selected = list(features)
    prepared = _classification_data(df, target, positive_label)
    missing = set(selected) - set(prepared.columns)
    if missing:
        raise KeyError(f"Features not found: {sorted(missing)}")
    for column in selected:
        if not pd.api.types.is_numeric_dtype(prepared[column]):
            raise TypeError(f"Continuous feature {column!r} must be numeric")

    figure, axes = _classification_grid(selected, ncols=ncols, figsize=figsize)
    for axis, feature in zip(axes, selected):
        if kind == "box":
            sns.boxplot(data=prepared, x=target, y=feature, ax=axis)
        else:
            sns.violinplot(data=prepared, x=target, y=feature, inner="quart", ax=axis)
        axis.set_title(f"{feature} by {target}")
    figure.tight_layout()
    return figure, axes


def plot_categorical_target_rates(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    positive_label: object = 1,
    min_count: int = 1,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot P(target=positive_label | category) with counts for each category.

    Missing feature values are represented by '<missing>' instead of dropped.
    Rates are descriptive, not a model or a causal effect.
    """

    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    selected = list(features)
    prepared = _classification_data(df, target, positive_label)
    missing = set(selected) - set(prepared.columns)
    if missing:
        raise KeyError(f"Features not found: {sorted(missing)}")

    figure, axes = _classification_grid(selected, ncols=ncols, figsize=figsize)
    for axis, feature in zip(axes, selected):
        categories = prepared[feature].astype("string").fillna("<missing>")
        grouped = pd.DataFrame({
            "category": categories,
            "positive": prepared[target].eq(positive_label).astype(float),
        }).groupby("category", dropna=False)["positive"].agg(["mean", "count"])
        grouped = grouped.loc[grouped["count"] >= min_count]
        grouped = grouped.sort_values("mean", ascending=False)
        if grouped.empty:
            axis.text(0.5, 0.5, "No categories meet min_count", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue

        bars = axis.bar(grouped.index.astype(str), grouped["mean"].to_numpy())
        for bar, count in zip(bars, grouped["count"].to_numpy()):
            axis.annotate(
                f"n={count}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
            )
        axis.set_ylim(0, 1.10)
        axis.set_ylabel(f"P({target}={positive_label})")
        axis.set_xlabel(feature)
        axis.set_title(f"Conversion rate by {feature}")
        axis.tick_params(axis="x", labelrotation=35)
    figure.tight_layout()
    return figure, axes


def plot_discrete_target_rates(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    positive_label: object = 1,
    max_unique: int = 30,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot class-positive rates for each observed numerical count value.

    Counts with missing values are omitted. A high cardinality raises an error;
    use grouping/binning chosen by the experiment instead of unreadable plots.
    """

    selected = list(features)
    prepared = _classification_data(df, target, positive_label)
    missing = set(selected) - set(prepared.columns)
    if missing:
        raise KeyError(f"Features not found: {sorted(missing)}")
    if max_unique < 1:
        raise ValueError("max_unique must be positive")

    figure, axes = _classification_grid(selected, ncols=ncols, figsize=figsize)
    for axis, feature in zip(axes, selected):
        if not pd.api.types.is_numeric_dtype(prepared[feature]):
            raise TypeError(f"Discrete feature {feature!r} must be numeric")
        observed = prepared.loc[prepared[feature].notna(), [feature, target]].copy()
        if observed[feature].nunique() > max_unique:
            plt.close(figure)
            raise ValueError(f"{feature!r} has too many unique values; bin explicitly")
        observed["positive"] = observed[target].eq(positive_label).astype(float)
        grouped = observed.groupby(feature)["positive"].agg(["mean", "count"]).sort_index()
        if grouped.empty:
            axis.text(0.5, 0.5, "No nonmissing observations", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue
        axis.plot(grouped.index.to_numpy(), grouped["mean"].to_numpy(), marker="o")
        for x, rate, count in zip(grouped.index, grouped["mean"], grouped["count"]):
            axis.annotate(f"{count}", (x, rate), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        axis.set_ylim(0, 1.10)
        axis.set_xlabel(feature)
        axis.set_ylabel(f"P({target}={positive_label})")
        axis.set_title(f"Positive rate by {feature} (labels: n)")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure, axes


def plot_missingness_by_target(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    positive_label: object = 1,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Compare positive-class rates for observed vs missing feature values."""

    selected = list(features)
    prepared = _classification_data(df, target, positive_label)
    missing = set(selected) - set(prepared.columns)
    if missing:
        raise KeyError(f"Features not found: {sorted(missing)}")

    figure, axes = _classification_grid(selected, ncols=ncols, figsize=figsize)
    for axis, feature in zip(axes, selected):
        indicator = prepared[feature].isna()
        grouped = pd.DataFrame({
            "missing": indicator,
            "positive": prepared[target].eq(positive_label).astype(float),
        }).groupby("missing")["positive"].agg(["mean", "count"])
        grouped = grouped.reindex([False, True]).dropna(subset=["mean"])
        bars = axis.bar(
            ["Observed" if not k else "Missing" for k in grouped.index],
            grouped["mean"].to_numpy(),
        )
        for bar, count in zip(bars, grouped["count"].to_numpy()):
            axis.annotate(f"n={count}",
                          (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        axis.set_ylim(0, 1.1)
        axis.set_ylabel(f"P({target}={positive_label})")
        axis.set_title(f"Missingness of {feature} vs {target}")
    figure.tight_layout()
    return figure, axes
