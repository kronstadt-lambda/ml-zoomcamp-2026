"""Pure exploratory-data-analysis summaries without printing or plotting."""

from collections.abc import Sequence

import pandas as pd

from mlzoomcamp.core.data import categorical_columns, require_columns


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize type, non-null values, missing values, and cardinality per column."""

    total_rows = len(df)
    missing = df.isna().sum()

    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null_count": df.notna().sum(),
            "missing_count": missing,
            "missing_ratio": missing / total_rows if total_rows else 0.0,
            "unique_count": df.nunique(dropna=True),
        }
    )


def numeric_summary(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return descriptive statistics for selected numerical columns."""

    if columns is None:
        selected_columns = list(df.select_dtypes(include="number").columns)
    else:
        selected_columns = list(columns)
        require_columns(df, selected_columns)

    return df.loc[:, selected_columns].describe().T


def categorical_value_counts(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    normalize: bool = False,
    dropna: bool = False,
) -> dict[str, pd.Series]:
    """Return value counts for each requested categorical column."""

    selected_columns = categorical_columns(df) if columns is None else list(columns)
    require_columns(df, selected_columns)

    return {
        column: df[column].value_counts(normalize=normalize, dropna=dropna)
        for column in selected_columns
    }
