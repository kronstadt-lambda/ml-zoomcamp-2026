import pandas as pd
import pytest

from mlzoomcamp.core.data import (
    normalize_column_names,
    normalize_text_values,
    split_features_target,
    split_train_val_test,
)
from mlzoomcamp.core.eda import column_overview
from mlzoomcamp.core.features import (
    add_log1p_feature,
    binary_target_group_summary,
    categorical_mutual_information,
    find_correlated_feature_pairs,
    years_to_age,
)


def test_normalization_keeps_input_unchanged() -> None:
    source = pd.DataFrame({" Customer Type ": [" Premium User ", None]})

    normalized = normalize_text_values(normalize_column_names(source))

    assert list(normalized.columns) == ["customer_type"]
    assert normalized.loc[0, "customer_type"] == "premium_user"
    assert source.columns.tolist() == [" Customer Type "]
    assert pd.isna(normalized.loc[1, "customer_type"])


def test_split_features_target_and_train_val_test_are_reproducible() -> None:
    frame = pd.DataFrame({"feature": range(20), "target": [0, 1] * 10})

    first_split = split_train_val_test(
        frame,
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
        stratify=frame["target"],
    )
    second_split = split_train_val_test(
        frame,
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
        stratify=frame["target"],
    )
    X, y = split_features_target(frame, "target")

    assert [len(partition) for partition in first_split] == [12, 4, 4]
    assert all(left.equals(right) for left, right in zip(first_split, second_split))
    assert X.columns.tolist() == ["feature"]
    assert y.name == "target"


def test_eda_and_feature_analysis_return_interpretable_results() -> None:
    frame = pd.DataFrame(
        {
            "category": ["a", "a", "b", "b"],
            "target": [0, 0, 1, 1],
            "x": [1.0, 2.0, 3.0, 4.0],
            "x_duplicate": [1.0, 2.0, 3.0, 4.0],
        }
    )

    overview = column_overview(frame)
    group_summary = binary_target_group_summary(
        frame,
        feature="category",
        target="target",
    )
    scores = categorical_mutual_information(
        frame,
        features=["category"],
        target="target",
    )
    pairs = find_correlated_feature_pairs(
        frame,
        columns=["x", "x_duplicate"],
        min_absolute_correlation=0.99,
    )

    assert overview.loc["category", "unique_count"] == 2
    assert group_summary.loc["b", "risk_ratio"] == 2.0
    assert scores.index.tolist() == ["category"]
    assert pairs.loc[0, "feature_1"] == "x"


def test_feature_engineering_returns_copies() -> None:
    frame = pd.DataFrame({"year": [2020], "value": [9.0]})

    aged = years_to_age(frame, "year", 2026, "age")
    transformed = add_log1p_feature(frame, "value", "log_value")

    assert aged.loc[0, "age"] == 6
    assert transformed.loc[0, "log_value"] == pytest.approx(2.302585)
    assert "age" not in frame
    assert "log_value" not in frame
