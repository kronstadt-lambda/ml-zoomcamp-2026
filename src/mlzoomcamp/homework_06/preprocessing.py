"""Dataset preparation helpers for Homework 06."""

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


@dataclass
class PreparedRegressionData:
    """Prepared splits, sparse matrices, labels, and fitted vectorizer."""

    df: pd.DataFrame
    X_train: csr_matrix
    X_val: csr_matrix
    X_test: csr_matrix
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    vectorizer: DictVectorizer


def split_features_target(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target from one dataset split."""

    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    return df.drop(columns=[target]).copy(), df[target].copy()


def prepare_regression_data(
    df: pd.DataFrame,
    *,
    target: str,
) -> PreparedRegressionData:
    """Prepare the Homework 06 dataset before Question 1."""

    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    df_prepared = df.copy().fillna(0)

    df_full_train, df_test = train_test_split(
        df_prepared,
        test_size=0.2,
        random_state=1,
    )

    df_train, df_val = train_test_split(
        df_full_train,
        test_size=0.25,
        random_state=1,
    )

    X_train_df, y_train = split_features_target(df_train, target)
    X_val_df, y_val = split_features_target(df_val, target)
    X_test_df, y_test = split_features_target(df_test, target)

    vectorizer = DictVectorizer(sparse=True)

    X_train = vectorizer.fit_transform(
        X_train_df.to_dict(orient="records")
    )
    X_val = vectorizer.transform(
        X_val_df.to_dict(orient="records")
    )
    X_test = vectorizer.transform(
        X_test_df.to_dict(orient="records")
    )

    return PreparedRegressionData(
        df=df_prepared,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        vectorizer=vectorizer,
    )
