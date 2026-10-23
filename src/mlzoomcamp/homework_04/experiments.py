"""Cross-validation experiments for Homework 04."""
import numpy as np
import pandas as pd
from collections.abc import Sequence
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def logistic_auc_cv_scores(
    df_full_train: pd.DataFrame,
    *,
    target: str,
    C: float = 1.0,
) -> pd.Series:
    """Return one validation ROC-AUC score for each of five folds."""

    if target not in df_full_train.columns:
        raise ValueError(f"Target column not found: {target}")

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=1,
    )

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train_fold = df_full_train.iloc[train_idx]
        df_val_fold = df_full_train.iloc[val_idx]

        y_train_fold = df_train_fold[target]
        y_val_fold = df_val_fold[target]

        X_train_fold = df_train_fold.drop(columns=[target])
        X_val_fold = df_val_fold.drop(columns=[target])

        dv = DictVectorizer(sparse=False)

        X_train_encoded = dv.fit_transform(
            X_train_fold.to_dict(orient="records")
        )
        X_val_encoded = dv.transform(
            X_val_fold.to_dict(orient="records")
        )

        model = LogisticRegression(
            solver="liblinear",
            C=C,
            max_iter=1000,
        )

        model.fit(X_train_encoded, y_train_fold)

        y_val_pred = model.predict_proba(X_val_encoded)[:, 1]

        scores.append(roc_auc_score(y_val_fold, y_val_pred))

    return pd.Series(
        scores,
        index=range(1, 6),
        name="auc",
    )

def tune_logistic_c_values(
    df_full_train: pd.DataFrame,
    *,
    target: str,
    c_values: Sequence[float],
) -> pd.DataFrame:
    """Evaluate mean and standard deviation of ROC-AUC for each C."""

    results = []

    for C in c_values:
        scores = logistic_auc_cv_scores(
            df_full_train,
            target=target,
            C=C,
        )

        mean_auc = float(np.mean(scores))
        std_auc = float(np.std(scores))

        results.append(
            {
                "C": C,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_auc_rounded": round(mean_auc, 3),
                "std_auc_rounded": round(std_auc, 3),
            }
        )

    return pd.DataFrame(results)
