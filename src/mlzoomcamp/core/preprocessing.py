"""Explicit builders for scikit-learn preprocessing components."""

from collections.abc import Sequence
from typing import Literal

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ImputeStrategy = Literal["mean", "median", "most_frequent", "constant"]


def build_numeric_preprocessor(
    *,
    impute_columns: Sequence[str] = (),
    impute_strategy: ImputeStrategy = "mean",
    fill_value: float | None = None,
    scale_columns: Sequence[str] = (),
) -> ColumnTransformer:
    """Build a numeric imputation/scaling transformer with explicit columns."""

    impute_columns = list(dict.fromkeys(impute_columns))
    scale_columns = list(dict.fromkeys(scale_columns))

    if impute_strategy == "constant" and fill_value is None:
        raise ValueError("fill_value is required for constant imputation")

    imputed_and_scaled = [
        column for column in scale_columns if column in impute_columns
    ]
    impute_only = [
        column for column in impute_columns if column not in scale_columns
    ]
    scale_only = [
        column for column in scale_columns if column not in impute_columns
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
                SimpleImputer(strategy=impute_strategy, fill_value=fill_value),
                impute_only,
            )
        )
    if scale_only:
        transformers.append(("scale", StandardScaler(), scale_only))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")

    return preprocessor


def build_categorical_preprocessor(
    columns: Sequence[str],
    *,
    fill_value: str = "missing",
    handle_unknown: Literal["error", "ignore"] = "ignore",
    sparse_output: bool = False,
) -> ColumnTransformer:
    """Build an imputation plus one-hot-encoding transformer for categories."""

    selected_columns = list(dict.fromkeys(columns))
    if not selected_columns:
        raise ValueError("columns must contain at least one categorical feature")

    transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=fill_value)),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown=handle_unknown,
                    sparse_output=sparse_output,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[("categorical", transformer, selected_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_dict_vectorizer(*, sparse: bool = False) -> DictVectorizer:
    """Create a DictVectorizer; fit it on train records before transforming others."""

    return DictVectorizer(sparse=sparse)
