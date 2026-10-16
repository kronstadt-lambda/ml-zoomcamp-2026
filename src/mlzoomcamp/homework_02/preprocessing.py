from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and string values."""

    normalized = df.copy()

    normalized.columns = (
        normalized.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    string_columns = normalized.select_dtypes(
        include=["object", "string", "str"]
    ).columns

    for column in string_columns:
        normalized[column] = (
            normalized[column]
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )

    return normalized


def train_val_test_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train, validation and test sets."""

    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    n = len(df)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    rng = np.random.RandomState(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)

    return (
        df.iloc[indices[:n_train]].copy(),
        df.iloc[indices[n_train:n_train + n_val]].copy(),
        df.iloc[indices[n_train + n_val:]].copy(),
    )


def select_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return a copy containing only the requested columns."""

    columns = list(dict.fromkeys(columns))
    missing_columns = set(columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Columns not found: {sorted(missing_columns)}"
        )

    return df.loc[:, columns].copy()


def split_features_target(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictor columns from the target column."""

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    return df.drop(columns=target), df[target].copy()


def build_numeric_preprocessor(
    *,
    impute_columns: Sequence[str] = (),
    impute_strategy: Literal[
        "mean",
        "median",
        "most_frequent",
        "constant",
    ] = "mean",
    fill_value: float | None = None,
    scale_columns: Sequence[str] = (),
) -> ColumnTransformer:
    """Build a configurable preprocessor for numerical columns."""

    impute_columns = list(dict.fromkeys(impute_columns))
    scale_columns = list(dict.fromkeys(scale_columns))

    if impute_strategy == "constant" and fill_value is None:
        raise ValueError(
            "fill_value is required for constant imputation"
        )

    imputed_and_scaled = [
        column
        for column in scale_columns
        if column in impute_columns
    ]

    impute_only = [
        column
        for column in impute_columns
        if column not in scale_columns
    ]

    scale_only = [
        column
        for column in scale_columns
        if column not in impute_columns
    ]

    transformers = []

    if imputed_and_scaled:
        transformers.append(
            (
                "impute_and_scale",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy=impute_strategy,
                                fill_value=fill_value,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                imputed_and_scaled,
            )
        )

    if impute_only:
        transformers.append(
            (
                "impute",
                SimpleImputer(
                    strategy=impute_strategy,
                    fill_value=fill_value,
                ),
                impute_only,
            )
        )

    if scale_only:
        transformers.append(
            (
                "scale",
                StandardScaler(),
                scale_only,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    preprocessor.set_output(transform="pandas")

    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fit a preprocessor using training features only."""

    return preprocessor.fit(X_train)


def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Transform a feature dataset with an already-fitted preprocessor."""

    return preprocessor.transform(X)


def prepare_train_eval(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str,
    preprocessor: ColumnTransformer,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Prepare an existing train/evaluation split."""

    selected_columns = [*features, target]

    train_selected = select_columns(
        df_train,
        selected_columns,
    )
    eval_selected = select_columns(
        df_eval,
        selected_columns,
    )

    X_train, y_train = split_features_target(
        train_selected,
        target,
    )
    X_eval, y_eval = split_features_target(
        eval_selected,
        target,
    )

    fitted_preprocessor = fit_preprocessor(
        preprocessor,
        X_train,
    )

    X_train_prepared = transform_features(
        fitted_preprocessor,
        X_train,
    )
    X_eval_prepared = transform_features(
        fitted_preprocessor,
        X_eval,
    )

    return (
        X_train_prepared,
        y_train,
        X_eval_prepared,
        y_eval,
    )
