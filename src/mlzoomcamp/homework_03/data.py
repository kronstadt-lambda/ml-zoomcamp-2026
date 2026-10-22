"""Data validation and target-splitting helpers used by Homework 03."""

from collections.abc import Sequence

import pandas as pd


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a descriptive error when one or more required columns are absent."""

    missing_columns = sorted(set(columns) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return names of string-like columns suitable for categorical inspection."""

    return list(df.select_dtypes(include=["object", "string", "str"]).columns)


def missing_value_counts(df: pd.DataFrame) -> pd.Series:
    """Return missing-value counts indexed by column name."""

    return df.isna().sum()


def split_features_target(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictor columns from a named target column."""

    require_columns(df, [target])

    return df.drop(columns=target).copy(), df[target].copy()
