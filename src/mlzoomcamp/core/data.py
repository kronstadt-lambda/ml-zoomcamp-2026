"""Dataset loading, validation, normalization, and splitting utilities."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Read a CSV dataset using pandas without imposing dataset-specific options."""

    return pd.read_csv(path, **kwargs)


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a descriptive error when one or more required columns are absent."""

    missing_columns = sorted(set(columns) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped, lowercase, underscore-separated column names."""

    normalized = df.copy()
    normalized.columns = (
        normalized.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    return normalized


def normalize_text_values(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a copy with normalized values in selected string columns.

    When ``columns`` is omitted, only pandas string-like columns are changed.
    Missing values remain missing.
    """

    normalized = df.copy()

    if columns is None:
        selected_columns = list(
            normalized.select_dtypes(include=["object", "string", "str"]).columns
        )
    else:
        selected_columns = list(dict.fromkeys(columns))
        require_columns(normalized, selected_columns)

    for column in selected_columns:
        if not pd.api.types.is_string_dtype(normalized[column]):
            raise TypeError(f"Column '{column}' must contain string-like values")

        normalized[column] = (
            normalized[column]
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )

    return normalized


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return names of string-like columns suitable for categorical inspection."""

    return list(df.select_dtypes(include=["object", "string", "str"]).columns)


def missing_value_counts(df: pd.DataFrame) -> pd.Series:
    """Return missing-value counts indexed by column name."""

    return df.isna().sum()


def select_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a validated, deduplicated, copied subset of columns."""

    selected_columns = list(dict.fromkeys(columns))
    require_columns(df, selected_columns)

    return df.loc[:, selected_columns].copy()


def split_features_target(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictor columns from a named target column."""

    require_columns(df, [target])

    return df.drop(columns=target).copy(), df[target].copy()


def split_train_val_test(
    df: pd.DataFrame,
    *,
    validation_size: float,
    test_size: float,
    random_state: int,
    stratify: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into reproducible train, validation, and test sets.

    ``stratify`` is optional and must align with ``df``. The function preserves
    original row indices so experiments may decide whether resetting them is useful.
    """

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between zero and one")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than one")
    if stratify is not None and not stratify.index.equals(df.index):
        raise ValueError("stratify must have the same index as df")

    full_train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_stratify = stratify.loc[full_train.index] if stratify is not None else None
    validation_share = validation_size / (1 - test_size)
    train, validation = train_test_split(
        full_train,
        test_size=validation_share,
        random_state=random_state,
        stratify=train_stratify,
    )

    return train.copy(), validation.copy(), test.copy()
