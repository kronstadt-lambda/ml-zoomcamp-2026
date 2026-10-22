from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer

from mlzoomcamp.homework_03.data import require_columns
from mlzoomcamp.homework_03.evaluation import binary_accuracy
from mlzoomcamp.homework_03.modeling import (
    build_logistic_regression,
    predict_classes,
)
from mlzoomcamp.homework_03.preprocessing import build_one_hot_encoder

def _prepare_logistic_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    *,
    features: Sequence[str],
    categorical_features: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_features = list(dict.fromkeys(features))
    selected_categorical = [
        feature
        for feature in categorical_features
        if feature in selected_features
    ]

    require_columns(df_train, selected_features)
    require_columns(df_val, selected_features)

    transformers = []

    if selected_categorical:
        transformers.append(
            (
                "categorical",
                build_one_hot_encoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                selected_categorical,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")

    X_train = preprocessor.fit_transform(df_train[selected_features])
    X_val = preprocessor.transform(df_val[selected_features])

    return X_train, X_val

def evaluate_logistic_validation_accuracy(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    df_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    features: Sequence[str],
    categorical_features: Sequence[str],
    C: float = 1.0,
) -> float:
    """Fit the Q4 logistic model and return unrounded validation accuracy."""


    X_train, X_val = _prepare_logistic_features(
        df_train,
        df_val,
        features=features,
        categorical_features=categorical_features,
    )

    model = build_logistic_regression(
        solver="liblinear",
        C=C,
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_val_pred = predict_classes(model, X_val)

    return binary_accuracy(y_val, y_val_pred)


def evaluate_leave_one_out_features(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    df_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    features: Sequence[str],
    categorical_features: Sequence[str],
) -> tuple[float, pd.DataFrame]:
    """Compare validation accuracy after excluding each raw feature once."""

    selected_features = list(dict.fromkeys(features))

    baseline_accuracy = evaluate_logistic_validation_accuracy(
        df_train,
        y_train,
        df_val,
        y_val,
        features=selected_features,
        categorical_features=categorical_features,
    )

    results = []

    for excluded_feature in selected_features:
        remaining_features = [
            feature
            for feature in selected_features
            if feature != excluded_feature
        ]

        accuracy_without_feature = evaluate_logistic_validation_accuracy(
            df_train,
            y_train,
            df_val,
            y_val,
            features=remaining_features,
            categorical_features=categorical_features,
        )

        results.append(
            {
                "feature": excluded_feature,
                "accuracy_without_feature": accuracy_without_feature,
                "difference": baseline_accuracy - accuracy_without_feature,
            }
        )

    return baseline_accuracy, pd.DataFrame(results).set_index("feature")

def evaluate_logistic_c_values(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    df_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    features: Sequence[str],
    categorical_features: Sequence[str],
    c_values: Sequence[float],
) -> pd.DataFrame:
    """Evaluate logistic regression for several C values."""

    X_train, X_val = _prepare_logistic_features(
        df_train,
        df_val,
        features=features,
        categorical_features=categorical_features,
    )

    results = []

    for C in c_values:
        model = build_logistic_regression(
            solver="liblinear",
            C=C,
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_val_pred = predict_classes(model, X_val)
        accuracy = binary_accuracy(y_val, y_val_pred)

        results.append(
            {
                "C": C,
                "accuracy": accuracy,
                "accuracy_rounded": round(accuracy, 3),
            }
        )

    return pd.DataFrame(results)
