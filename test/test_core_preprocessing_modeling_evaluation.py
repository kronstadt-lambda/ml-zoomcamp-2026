import numpy as np
import pandas as pd
import pytest

from mlzoomcamp.core.evaluation import (
    binary_accuracy,
    binary_confusion_matrix,
    binary_roc_auc,
    mae,
    mse,
    rmse,
    threshold_binary_probabilities,
)
from mlzoomcamp.core.modeling import (
    fit_linear_regression,
    fit_logistic_regression,
    fit_normal_equation,
    fit_ridge_regression,
    predict_from_weights,
    predict_positive_probabilities,
    predict_regression,
    sigmoid,
)
from mlzoomcamp.core.preprocessing import (
    build_categorical_preprocessor,
    build_dict_vectorizer,
    build_numeric_preprocessor,
)


def test_preprocessors_fit_only_on_training_data() -> None:
    train = pd.DataFrame({"number": [1.0, np.nan], "category": ["a", "b"]})
    validation = pd.DataFrame({"number": [3.0], "category": ["new"]})

    numeric = build_numeric_preprocessor(
        impute_columns=["number"],
        impute_strategy="mean",
        scale_columns=["number"],
    )
    categorical = build_categorical_preprocessor(["category"])

    numeric_train = numeric.fit_transform(train[["number"]])
    numeric_validation = numeric.transform(validation[["number"]])
    categorical.fit(train[["category"]])
    categorical_validation = categorical.transform(validation[["category"]])

    assert numeric_train.shape == (2, 1)
    assert numeric_validation.shape == (1, 1)
    assert categorical_validation.shape == (1, 2)


def test_dict_vectorizer_and_regression_models() -> None:
    records = [{"category": "a", "value": 1}, {"category": "b", "value": 2}]
    vectorizer = build_dict_vectorizer(sparse=False)
    matrix = vectorizer.fit_transform(records)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    linear = fit_linear_regression(X, y)
    ridge = fit_ridge_regression(X, y, alpha=0.1)
    intercept, weights = fit_normal_equation(X, y)

    assert matrix.shape == (2, 3)
    assert predict_regression(linear, X)[0] == pytest.approx(1.0)
    assert predict_regression(ridge, X).shape == y.shape
    assert predict_from_weights(X, intercept, weights) == pytest.approx(y)


def test_logistic_model_and_evaluation_functions() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = fit_logistic_regression(X, y, solver="liblinear", random_state=42)
    probabilities = predict_positive_probabilities(model, X)
    decisions = threshold_binary_probabilities(probabilities, threshold=0.5)

    assert sigmoid(np.array([-1000.0, 0.0, 1000.0])) == pytest.approx([0.0, 0.5, 1.0])
    assert binary_accuracy(y, decisions) >= 0.75
    assert binary_confusion_matrix(y, decisions).shape == (2, 2)
    assert binary_roc_auc(y, probabilities) == 1.0
    assert mae([1, 2], [1, 4]) == 1.0
    assert mse([1, 2], [1, 4]) == 2.0
    assert rmse([1, 2], [1, 4]) == pytest.approx(np.sqrt(2))
