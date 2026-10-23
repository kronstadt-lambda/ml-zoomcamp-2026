"""Evaluation helpers for Homework 04."""

from collections.abc import Sequence

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score



def univariate_roc_auc_scores(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
) -> pd.DataFrame:
    """Calculate orientation-corrected ROC-AUC for numeric feature scores.

    A feature with raw AUC below 0.5 is negated so that larger scores
    consistently indicate a higher likelihood of the positive class.
    """

    selected_features = list(dict.fromkeys(features))

    missing_columns = sorted(set(selected_features) - set(X.columns))
    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")

    if not X.index.equals(y.index):
        raise ValueError("X and y must have matching indexes")

    if y.isna().any() or y.nunique() != 2:
        raise ValueError("y must be a non-missing binary target")

    if X[selected_features].isna().any().any():
        raise ValueError("Features must not contain missing values")

    results = []

    for feature in selected_features:
        scores = X[feature]
        raw_auc = float(roc_auc_score(y, scores))

        direction = -1 if raw_auc < 0.5 else 1
        oriented_scores = direction * scores
        oriented_auc = float(roc_auc_score(y, oriented_scores))

        results.append(
            {
                "feature": feature,
                "raw_auc": raw_auc,
                "direction": direction,
                "auc": oriented_auc,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("auc", ascending=False, ignore_index=True)
    )

def precision_recall_by_thresholds(
    y_true: pd.Series,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Calculate precision and recall for each probability threshold."""

    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    results = []

    for threshold in thresholds:
        predictions = scores >= threshold

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(results)
