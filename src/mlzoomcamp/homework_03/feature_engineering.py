"""Feature-analysis helpers used by Homework 03."""

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import mutual_info_score

from mlzoomcamp.homework_03.data import require_columns


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
