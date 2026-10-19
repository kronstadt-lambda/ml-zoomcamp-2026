"""Deterministic feature-construction functions."""

import numpy as np
import pandas as pd

from mlzoomcamp.core.data import require_columns


def years_to_age(
    df: pd.DataFrame,
    year_column: str,
    reference_year: int,
    output_column: str,
    *,
    drop_year_column: bool = False,
) -> pd.DataFrame:
    """Return a copy with an age feature derived from a year-based column."""

    require_columns(df, [year_column])
    transformed = df.copy()
    transformed[output_column] = reference_year - transformed[year_column]

    if drop_year_column:
        transformed = transformed.drop(columns=year_column)

    return transformed


def add_log1p_feature(
    df: pd.DataFrame,
    source_column: str,
    output_column: str,
    *,
    drop_source_column: bool = False,
) -> pd.DataFrame:
    """Return a copy with ``log1p`` applied to a numeric source feature."""

    require_columns(df, [source_column])
    if (df[source_column].dropna() < -1).any():
        raise ValueError("log1p requires source values greater than or equal to -1")

    transformed = df.copy()
    transformed[output_column] = np.log1p(transformed[source_column])

    if drop_source_column:
        transformed = transformed.drop(columns=source_column)

    return transformed
