"""Scikit-learn preprocessing builders used by Homework 03."""

from typing import Literal

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

ImputeStrategy = Literal["mean", "median", "most_frequent", "constant"]


def build_simple_imputer(
    *,
    strategy: ImputeStrategy = "mean",
    fill_value: str | float | None = None,
) -> SimpleImputer:
    if strategy == "constant" and fill_value is None:
        raise ValueError("fill_value is required for constant imputation")

    return SimpleImputer(
        strategy=strategy,
        fill_value=fill_value,
    )


def build_one_hot_encoder(
    *,
    handle_unknown: Literal["error", "ignore"] = "ignore",
    sparse_output: bool = False,
) -> OneHotEncoder:
    return OneHotEncoder(
        handle_unknown=handle_unknown,
        sparse_output=sparse_output,
    )
