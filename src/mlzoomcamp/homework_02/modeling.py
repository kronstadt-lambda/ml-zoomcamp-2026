import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    """Fit an unregularized linear regression model."""

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def train_ridge_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float,
) -> Ridge:
    """Fit a Ridge regression model."""

    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    return model


def predict(
    model: LinearRegression,
    X: pd.DataFrame,
) -> pd.Series:
    """Generate predictions with a fitted linear regression model."""

    return model.predict(X)


def evaluate_rmse(
    model: LinearRegression,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """Calculate RMSE for a fitted regression model."""

    predictions = predict(model, X)

    return float(root_mean_squared_error(y, predictions))
