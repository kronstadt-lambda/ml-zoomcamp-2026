# %%
"""
Script en python para poder resolver el Homework01
"""
# %%
import pandas as pd
from matplotlib import pyplot as plt
from mlzoomcamp.core.config import DATA_DIR
from mlzoomcamp.homework_02.plots import (
    plot_distribution,
    plot_missing_values,
    plot_numeric_relationships,
    plot_correlation_matrix,
    plot_feature_distributions,
)
from mlzoomcamp.homework_02.preprocessing import (
    normalize_columns,
    train_val_test_split,
)
from mlzoomcamp.homework_02.feature_engineering import (
    years_to_age
)
from mlzoomcamp.homework_02.experiments import (
    compare_zero_and_mean_imputation,
    select_best_alpha,
    tune_ridge_regularization,
    evaluate_seed_scores,
    calculate_score_std,
    evaluate_ridge_on_test,
)
# %%
# EDA
DATASET_PATH = DATA_DIR / "car_fuel_efficiency_2026.csv"
df = pd.read_csv(DATASET_PATH)

# %%
print('num of records: ', df.shape[0])
# %%
print(df.dtypes)
# %%
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print("--------")
# %%
df = normalize_columns(df)

df_model = years_to_age(
    df,
    year_column="model_year",
    reference_year=2026,
    output_column="vehicle_age",
    drop_year_column=True,
)

model_features = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "vehicle_age",
]
target = 'fuel_efficiency_mpg'

# %%
# plots
_ = plot_distribution(df_model, target, color='red')
plt.show()

# %%
_ = plot_numeric_relationships(df_model, model_features, target, ncols=2)
plt.show()
# %%
_ = plot_feature_distributions(df_model, model_features)
plt.show()
# %%
_ = plot_correlation_matrix(df_model, model_features)
plt.show()

# %%
# -------------------------------------------------------------------------------------
# Q1 There's one column with missing values. What is it?
# -------------------------------------------------------------------------------------

print(df_model.isnull().sum())

_ = plot_missing_values(df_model, model_features)
plt.show()
print('the column with missing values is horsepower')
# %%
# -------------------------------------------------------------------------------------
# Q2
# -------------------------------------------------------------------------------------
print('The median of horsepower is: ', df_model['horsepower'].median())

# %%
# Prepare and split the dataset
df_train, df_val, df_test = train_val_test_split(
    df_model,
    val_ratio=0.2,
    test_ratio=0.2,
    random_state=42,
)
# %%
# -------------------------------------------------------------------------------------
# Q3
# -------------------------------------------------------------------------------------
scores = compare_zero_and_mean_imputation(
    df_train,
    df_val,
    features=model_features,
    target=target,
    impute_columns=["horsepower"],
    scale_columns=[
        "engine_displacement",
        "horsepower",
        "vehicle_weight",
    ],
)

rounded_scores = {
    option: round(score, 3)
    for option, score in scores.items()
}

print(rounded_scores)
print("Best RMSE is for: ", min(rounded_scores, key=rounded_scores.get))
# %%
# -------------------------------------------------------------------------------------
# Q4
# -------------------------------------------------------------------------------------
ridge_scores = tune_ridge_regularization(
    df_train,
    df_val,
    features=model_features,
    target=target,
    impute_columns=["horsepower"],
    scale_columns=[
        "engine_displacement",
        "horsepower",
        "vehicle_weight",
    ],
    alphas=[0, 0.01, 0.1, 1, 5, 10, 100],
)

rounded_ridge_scores = {
    alpha: round(score, 4)
    for alpha, score in ridge_scores.items()
}

best_alpha = select_best_alpha(ridge_scores)

print(rounded_ridge_scores)
print("Best alpha:", best_alpha)
print(
    "Best RMSE:",
    rounded_ridge_scores[best_alpha],
)
# %%
# -------------------------------------------------------------------------------------
# Q5
# -------------------------------------------------------------------------------------
seed_scores = evaluate_seed_scores(
    df_model,
    features=model_features,
    target=target,
    seeds=list(range(10)),
    impute_columns=["horsepower"],
    scale_columns=[
        "engine_displacement",
        "horsepower",
        "vehicle_weight",
    ],
)

score_std = calculate_score_std(seed_scores)

print("RMSE by seed:")
print(
    {
        seed: round(score, 3)
        for seed, score in seed_scores.items()
    }
)

print("Standard deviation:", round(score_std, 3))
# %%
# -------------------------------------------------------------------------------------
# Q6
# -------------------------------------------------------------------------------------
test_rmse = evaluate_ridge_on_test(
    df_model,
    features=model_features,
    target=target,
    impute_columns=["horsepower"],
    scale_columns=[
        "engine_displacement",
        "horsepower",
        "vehicle_weight",
    ],
    seed=9,
    alpha=0.001,
)

print("Test RMSE:", round(test_rmse, 3))
# %%
