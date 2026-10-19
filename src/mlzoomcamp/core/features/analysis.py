"""Statistical feature-analysis functions that do not select features."""

from collections.abc import Sequence
from itertools import combinations

import pandas as pd
from sklearn.metrics import mutual_info_score

from mlzoomcamp.core.data import require_columns


def binary_target_group_summary(
    df: pd.DataFrame,
    *,
    feature: str,
    target: str,
) -> pd.DataFrame:
    """Summarize a categorical feature against a binary target.

    The returned table contains group target rate, count, difference from the
    global target rate, and relative risk. It does not rank or select features.
    """

    require_columns(df, [feature, target])
    target_values = df[target].dropna()
    if not target_values.isin([0, 1, False, True]).all():
        raise ValueError("target must contain binary values")

    global_rate = target_values.mean()
    summary = df.groupby(feature, dropna=False)[target].agg(["mean", "count"])
    summary = summary.rename(columns={"mean": "target_rate"})
    summary["difference_from_global"] = summary["target_rate"] - global_rate
    summary["risk_ratio"] = (
        summary["target_rate"] / global_rate if global_rate else float("nan")
    )

    return summary


def categorical_mutual_information(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
) -> pd.Series:
    """Return mutual-information scores for categorical features and a target."""

    selected_features = list(dict.fromkeys(features))
    require_columns(df, [*selected_features, target])

    scores = {
        feature: mutual_info_score(df[feature], df[target])
        for feature in selected_features
    }

    return pd.Series(scores, dtype=float).sort_values(ascending=False)


def numeric_target_correlations(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    absolute: bool = False,
) -> pd.Series:
    """Return Pearson correlations between numeric features and a target."""

    selected_features = list(dict.fromkeys(features))
    require_columns(df, [*selected_features, target])
    correlations = df.loc[:, selected_features].corrwith(df[target])

    return correlations.abs() if absolute else correlations


def correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a Pearson correlation matrix for selected numerical columns."""

    if columns is None:
        selected_columns = list(df.select_dtypes(include="number").columns)
    else:
        selected_columns = list(columns)
        require_columns(df, selected_columns)

    return df.loc[:, selected_columns].corr()


def find_correlated_feature_pairs(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    min_absolute_correlation: float = 0.9,
) -> pd.DataFrame:
    """Return highly correlated feature pairs for redundancy inspection.

    This diagnostic deliberately does not remove any feature.
    """

    if not 0 <= min_absolute_correlation <= 1:
        raise ValueError("min_absolute_correlation must be between zero and one")

    correlations = correlation_matrix(df, columns)
    pairs = []

    for left, right in combinations(correlations.columns, 2):
        correlation = correlations.loc[left, right]
        if abs(correlation) >= min_absolute_correlation:
            pairs.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "correlation": correlation,
                    "absolute_correlation": abs(correlation),
                }
            )

    return pd.DataFrame(
        pairs,
        columns=["feature_1", "feature_2", "correlation", "absolute_correlation"],
    ).sort_values("absolute_correlation", ascending=False, ignore_index=True)
