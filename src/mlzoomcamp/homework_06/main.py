# %%
"""Homework 06 dataset preparation before Question 1."""

from math import isclose

import pandas as pd
from scipy.sparse import issparse

from mlzoomcamp.config import DATA_DIR
from mlzoomcamp.homework_06.evaluation import calculate_rmse
from mlzoomcamp.homework_06.experiments import (
    evaluate_random_forest_depth_grid,
    evaluate_random_forest_estimators,
    evaluate_xgboost_eta,
    mean_rmse_by_max_depth,
)
from mlzoomcamp.homework_06.modeling import (
    export_tree_rules,
    feature_importance_by_original_feature,
    get_root_split_feature,
    train_decision_tree_regressor,
    train_random_forest_regressor,
)
from mlzoomcamp.homework_06.preprocessing import (
    prepare_regression_data,
)

DATASET_PATH = DATA_DIR / "car_fuel_efficiency_2026.csv"
TARGET = "fuel_efficiency_mpg"

# %%
# Load, prepare, and verify the Homework 06 dataset.
df = pd.read_csv(DATASET_PATH)

prepared = prepare_regression_data(
        df,
        target=TARGET,
        )

assert prepared.df.isna().sum().sum() == 0

assert len(prepared.y_train) == 6000
assert len(prepared.y_val) == 2000
assert len(prepared.y_test) == 2000

assert issparse(prepared.X_train)
assert issparse(prepared.X_val)
assert issparse(prepared.X_test)

feature_count = len(prepared.vectorizer.get_feature_names_out())

assert prepared.X_train.shape[1] == feature_count
assert prepared.X_val.shape[1] == feature_count
assert prepared.X_test.shape[1] == feature_count

print("Dataset preparation completed.")
print(f"Train shape: {prepared.X_train.shape}")
print(f"Validation shape: {prepared.X_val.shape}")
print(f"Test shape: {prepared.X_test.shape}")
print(f"Vectorized features: {feature_count}")

# %%
# -------------------------------------------------------------------------------------
# Q1 Which original feature is used for splitting the data?
# -------------------------------------------------------------------------------------
decision_stump = train_decision_tree_regressor(
    prepared.X_train,
    prepared.y_train,
    max_depth=1,
)

root_split = get_root_split_feature(
    decision_stump,
    prepared.vectorizer,
)

assert decision_stump.get_depth() == 1

print("Vectorized root feature:", root_split.vectorized_feature)
print("Original root feature:", root_split.original_feature)
print(export_tree_rules(decision_stump, prepared.vectorizer))
# %%
# -------------------------------------------------------------------------------------
# Q2 What's the RMSE of this model on the validation data?
# -------------------------------------------------------------------------------------
random_forest = train_random_forest_regressor(
    prepared.X_train,
    prepared.y_train,
    n_estimators=10,
    random_state=1,
    n_jobs=-1,
)

y_val_pred = random_forest.predict(prepared.X_val)

validation_rmse = calculate_rmse(
    prepared.y_val,
    y_val_pred,
)

print(f"Validation RMSE: {validation_rmse:.3f}")
# %%
# -------------------------------------------------------------------------------------
# Q3 Which value gives the lowest validation RMSE?
# -------------------------------------------------------------------------------------
estimator_results = evaluate_random_forest_estimators(
    prepared.X_train,
    prepared.y_train,
    prepared.X_val,
    prepared.y_val,
    n_estimators_values=[10, 50, 100, 150],
    random_state=1,
    n_jobs=-1,
)

print(estimator_results.to_string(index=False))

best_result = estimator_results.loc[
    estimator_results["rmse_rounded"].idxmin()
]

print(f"Best n_estimators: {int(best_result['n_estimators'])}")
print(f"Best validation RMSE: {best_result['rmse_rounded']:.3f}")
# %%
# -------------------------------------------------------------------------------------
# Q4 What's the best max_depth, using the mean RMSE?
# -------------------------------------------------------------------------------------
depth_grid_results = evaluate_random_forest_depth_grid(
    prepared.X_train,
    prepared.y_train,
    prepared.X_val,
    prepared.y_val,
    max_depth_values=[10, 15, 20, 25],
    n_estimators_values=[10, 50, 100, 150],
    random_state=1,
    n_jobs=-1,
)

print("RMSE by max_depth and n_estimators:")
print(
    depth_grid_results.pivot(
        index="max_depth",
        columns="n_estimators",
        values="rmse",
    ).round(3)
)

depth_summary = mean_rmse_by_max_depth(depth_grid_results)

print("\nMean RMSE by max_depth:")
print(
    depth_summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)

best_depth_result = depth_summary.loc[
    depth_summary["mean_rmse"].idxmin()
]

print(f"\nBest max_depth: {int(best_depth_result['max_depth'])}")
print(f"Best mean RMSE: {best_depth_result['mean_rmse']:.3f}")
# %%
# -------------------------------------------------------------------------------------
# Q5 What's the most important feature (among these 4)?
# -------------------------------------------------------------------------------------
feature_importance_forest = train_random_forest_regressor(
    prepared.X_train,
    prepared.y_train,
    n_estimators=10,
    max_depth=20,
    random_state=1,
    n_jobs=-1,
)

feature_importances = feature_importance_by_original_feature(
    feature_importance_forest,
    prepared.vectorizer,
)

candidate_features = [
    "vehicle_weight",
    "horsepower",
    "acceleration",
    "engine_displacement",
]

candidate_importances = (
    feature_importances.loc[
        feature_importances["original_feature"].isin(candidate_features)
    ]
    .sort_values("importance", ascending=False, ignore_index=True)
)

if len(candidate_importances) != len(candidate_features):
    raise ValueError("One or more Q5 candidate features are missing")

print("Feature importances for Q5 candidates:")
print(
    candidate_importances.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

most_important_feature = candidate_importances.iloc[0]

print(
    "Most important candidate:",
    most_important_feature["original_feature"],
)
# %%
# -------------------------------------------------------------------------------------
# Q6 Which eta leads to the best RMSE score on the validation dataset?
# -------------------------------------------------------------------------------------
xgboost_results = evaluate_xgboost_eta(
    prepared.X_train,
    prepared.y_train,
    prepared.X_val,
    prepared.y_val,
    feature_names=prepared.vectorizer.get_feature_names_out(),
    eta_values=[0.3, 0.1],
    num_boost_round=100,
)

print("XGBoost eta validation RMSE:")
print(
    xgboost_results.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

eta_03_rmse = xgboost_results.loc[
    xgboost_results["eta"].eq(0.3),
    "rmse",
].iloc[0]

eta_01_rmse = xgboost_results.loc[
    xgboost_results["eta"].eq(0.1),
    "rmse",
].iloc[0]

if isclose(eta_03_rmse, eta_01_rmse, abs_tol=1e-12):
    print("Best eta: Both give equal value")
else:
    best_eta = xgboost_results.loc[
        xgboost_results["rmse"].idxmin(),
        "eta",
    ]
    print(f"Best eta: {best_eta}")
# %%
