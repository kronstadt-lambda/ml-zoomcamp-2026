
# %%
from pathlib import Path

import pandas as pd

from mlzoomcamp.homework_03.feature_engineering import (
    categorical_mutual_information,
)
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from mlzoomcamp.homework_03.data import (
    missing_value_counts,
    split_features_target,
)
from mlzoomcamp.homework_03.eda import (
    column_overview,
    numeric_summary,
    categorical_value_counts,
)
from mlzoomcamp.homework_03.preprocessing import build_simple_imputer
from mlzoomcamp.homework_03.plots import (
    plot_target_balance,
    plot_feature_distributions,
    plot_continuous_by_target,
    plot_categorical_target_rates,
    plot_discrete_target_rates,
    plot_missing_values,
    plot_missingness_by_target,
    plot_correlation_matrix,
)
from mlzoomcamp.homework_03.experiments import (
    evaluate_leave_one_out_features,
    evaluate_logistic_c_values,
    evaluate_logistic_validation_accuracy,
)

pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)
# %%
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DATASET_PATH = DATA_DIR / "course_lead_scoring_2026.csv"
df = pd.read_csv(DATASET_PATH)

# %%
# EDA
print('num of records: ', df.shape[0])
print(column_overview(df))
# %%
for col in df.columns:
    print(col)
    print(df[col].unique()[:9])
    print(df[col].nunique())
    print("--------")
# %%
target = 'converted'
feature_groups = {
    "continuous": ["annual_income", "lead_score"],
    "discrete": [
        "number_of_courses_viewed",
        "interaction_count",
    ],
    "categorical": [
        "lead_source",
        "industry",
        "employment_status",
        "location",
    ],
    "ordinal": [],
    "identifiers": [],
}

# %%
numerical = feature_groups['continuous'] + feature_groups['discrete']
numeric_summary(df, numerical)

# %%
categorical_value_counts(df, feature_groups['categorical'])

# %%
# target
plot_target_balance(df, target)
plt.show()
# %%
# Numerical feature distributions
plot_feature_distributions(df, numerical, kde=False)
plt.show()

# %%
# Continuous features by target
plot_continuous_by_target(
    df,
    feature_groups["continuous"],
    target,
)
plt.show()
# %%
# Discrete features and conversion rate
plot_discrete_target_rates(
    df,
    feature_groups["discrete"],
    target,
)
plt.show()
# %%
# Categorical features and conversion rate
plot_categorical_target_rates(
    df,
    feature_groups["categorical"],
    target,
    min_count=10,
)
plt.show()
# %%
# Numerical correlations
plot_correlation_matrix(
    df,
    numerical + [target],
)
plt.show()
# %%
# Missing Values
plot_missing_values(df)
plt.show()
# %%
# Missingness and target relationships
plot_missingness_by_target(
    df,
    ["annual_income", "lead_score", "lead_source"],
    target,
)
plt.show()
# %%
# Data preparation
categorical = feature_groups["categorical"]
features = numerical + categorical

missing_before = missing_value_counts(df[features])
print("Missing values before preparation:")
print(missing_before[missing_before > 0])

if missing_before.gt(0).any():
    numerical_imputer = build_simple_imputer(
        strategy="constant",
        fill_value=0.0,
    )
    categorical_imputer = build_simple_imputer(
        strategy="constant",
        fill_value="NA",
    )

    df.loc[:, numerical] = numerical_imputer.fit_transform(df[numerical])
    df.loc[:, categorical] = categorical_imputer.fit_transform(df[categorical])

missing_after = missing_value_counts(df[features])

print("\nMissing values after preparation:")
print(missing_after[missing_after > 0])

assert missing_after.eq(0).all()
# %%
# -------------------------------------------------------------------------------------
# Q1 What is the most frequent observation (mode) for the column 'industry'?
# -------------------------------------------------------------------------------------
print(df['industry'].value_counts())
print("More frecuent value observation in industry column is: ", df["industry"].mode().iloc[0])

# %%
# -------------------------------------------------------------------------------------
# Q2 What are the two features that have the biggest correlation?
# -------------------------------------------------------------------------------------
correlation_table = df[numerical].corr().round(3)
print(correlation_table.to_string())
print("Features with biggest correlation are: interaction_count and lead_score")
# %%
# Split the data
df_feat, y = split_features_target(df, target)

df_full_train, df_test = train_test_split(
    df_feat,
    test_size=0.2,
    random_state=42,
)

df_train, df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=42,
)

# Recover labels using the original row indices
y_train = y.loc[df_train.index].copy()
y_val = y.loc[df_val.index].copy()
y_test = y.loc[df_test.index].copy()

assert target not in df_train.columns
assert target not in df_val.columns
assert target not in df_test.columns

print(len(df_train), len(df_val), len(df_test))
# %%
# -------------------------------------------------------------------------------------
# Q3 Which of these variables has the biggest mutual information score?
# -------------------------------------------------------------------------------------

mi_scores = categorical_mutual_information(
    df_train.assign(**{target: y_train}),
    features=feature_groups["categorical"],
    target=target,
)

mi_scores = mi_scores.map(lambda score: round(score, 2))

print(mi_scores)
print(
    "Feature with the biggest mutual information score:",
    mi_scores.idxmax(),
)

# %%
# -------------------------------------------------------------------------------------
# Q4 What accuracy did you get?
# -------------------------------------------------------------------------------------
model_features = numerical + categorical

validation_accuracy = evaluate_logistic_validation_accuracy(
    df_train,
    y_train,
    df_val,
    y_val,
    features=model_features,
    categorical_features=categorical,
)

print("Validation accuracy:", round(validation_accuracy, 2))
# %%
# -------------------------------------------------------------------------------------
# Q5 Which of following feature has the smallest difference?
# -------------------------------------------------------------------------------------
baseline_accuracy, elimination_scores = evaluate_leave_one_out_features(
    df_train,
    y_train,
    df_val,
    y_val,
    features=model_features,
    categorical_features=categorical,
)

print("Baseline accuracy:", baseline_accuracy)
print(elimination_scores)

candidates = [
    "lead_source",
    "number_of_courses_viewed",
    "interaction_count",
]

candidate_differences = elimination_scores.loc[candidates, "difference"]

print(candidate_differences)
print("Feature with the smallest difference:", candidate_differences.idxmin())
# %%
# -------------------------------------------------------------------------------------
# Q6 Which of these C leads to the best accuracy on the validation set?
# -------------------------------------------------------------------------------------
c_values = [
    0.000001,
    0.00001,
    0.0001,
    0.001,
]

c_scores = evaluate_logistic_c_values(
    df_train,
    y_train,
    df_val,
    y_val,
    features=model_features,
    categorical_features=categorical,
    c_values=c_values,
)

print(c_scores)

best_result = c_scores.sort_values(
    ["accuracy_rounded", "C"],
    ascending=[False, True],
).iloc[0]

print("Best C:", best_result["C"])
print("Best validation accuracy:", best_result["accuracy_rounded"])
# %%
