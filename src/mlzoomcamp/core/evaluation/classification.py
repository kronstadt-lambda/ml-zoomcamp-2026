"""Binary-classification thresholding and metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_binary_probabilities(
    probabilities: pd.Series | np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Convert estimated positive-class probabilities into binary decisions."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")

    return np.asarray(probabilities, dtype=float) >= threshold


def binary_accuracy(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Calculate binary classification accuracy from hard predictions."""

    return float(accuracy_score(y_true, y_pred))


def binary_precision(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    zero_division: float = 0,
) -> float:
    """Calculate binary precision from hard predictions."""

    return float(precision_score(y_true, y_pred, zero_division=zero_division))


def binary_recall(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    zero_division: float = 0,
) -> float:
    """Calculate binary recall from hard predictions."""

    return float(recall_score(y_true, y_pred, zero_division=zero_division))


def binary_f1(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    zero_division: float = 0,
) -> float:
    """Calculate binary F1 from hard predictions."""

    return float(f1_score(y_true, y_pred, zero_division=zero_division))


def binary_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return the two-by-two confusion matrix for binary hard predictions."""

    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def binary_roc_auc(
    y_true: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray,
) -> float:
    """Calculate ROC-AUC from true labels and estimated positive probabilities."""

    return float(roc_auc_score(y_true, y_probability))


def classification_summary(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray | None = None,
) -> pd.Series:
    """Return hard-prediction metrics and optional ROC-AUC in one summary."""

    metrics = {
        "accuracy": binary_accuracy(y_true, y_pred),
        "precision": binary_precision(y_true, y_pred),
        "recall": binary_recall(y_true, y_pred),
        "f1": binary_f1(y_true, y_pred),
    }
    if y_probability is not None:
        metrics["roc_auc"] = binary_roc_auc(y_true, y_probability)

    return pd.Series(metrics, dtype=float)
