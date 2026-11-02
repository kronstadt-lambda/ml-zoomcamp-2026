"""Model-training and tree-inspection helpers for Homework 06."""

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor, export_text


@dataclass(frozen=True)
class RootSplitFeature:
    """The root split represented as vectorized and original features."""

    vectorized_feature: str
    original_feature: str


def train_decision_tree_regressor(
    X_train: spmatrix,
    y_train: pd.Series,
    *,
    max_depth: int,
) -> DecisionTreeRegressor:
    """Fit a decision-tree regressor with the requested maximum depth."""

    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)

    return model


def get_original_feature_name(feature_name: str) -> str:
    """Map a one-hot feature name to its original dataframe column."""

    return feature_name.split("=", maxsplit=1)[0]


def get_root_split_feature(
    model: DecisionTreeRegressor,
    vectorizer: DictVectorizer,
) -> RootSplitFeature:
    """Return the vectorized and original feature used at the tree root."""

    root_feature_index = int(model.tree_.feature[0])

    if root_feature_index < 0:
        raise ValueError("The fitted tree has no split at its root")

    feature_names = vectorizer.get_feature_names_out()
    vectorized_feature = str(feature_names[root_feature_index])

    return RootSplitFeature(
        vectorized_feature=vectorized_feature,
        original_feature=get_original_feature_name(vectorized_feature),
    )


def export_tree_rules(
    model: DecisionTreeRegressor,
    vectorizer: DictVectorizer,
) -> str:
    """Return a readable representation of the fitted decision tree."""

    feature_names = list(vectorizer.get_feature_names_out())

    return export_text(
        model,
        feature_names=feature_names,
    )

def train_random_forest_regressor(
    X_train: spmatrix,
    y_train: pd.Series,
    *,
    n_estimators: int,
    random_state: int,
    n_jobs: int | None = None,
    max_depth: int | None = None,
) -> RandomForestRegressor:
    """Fit a random-forest regressor with the requested parameters."""

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)

    return model

def feature_importance_by_original_feature(
    model: RandomForestRegressor,
    vectorizer: DictVectorizer,
) -> pd.DataFrame:
    """Return feature importances aggregated by original dataframe column."""

    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            "Vectorizer features and model importances must have the same length"
        )

    importance_table = pd.DataFrame(
        {
            "vectorized_feature": feature_names,
            "importance": importances,
        }
    )

    importance_table["original_feature"] = (
        importance_table["vectorized_feature"]
        .map(get_original_feature_name)
    )

    return (
        importance_table.groupby(
            "original_feature",
            as_index=False,
        )["importance"]
        .sum()
        .sort_values("importance", ascending=False, ignore_index=True)
    )
