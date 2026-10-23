# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlzoomcamp.config import DATA_DIR
from mlzoomcamp.homework_04.evaluation import (
    precision_recall_by_thresholds,
    univariate_roc_auc_scores,
)
from mlzoomcamp.homework_04.experiments import (
    logistic_auc_cv_scores,
    tune_logistic_c_values,
)

# Carga de datos
DATASET_PATH = DATA_DIR / "course_lead_scoring_2026.csv"

df = pd.read_csv(DATASET_PATH)

target = "converted"

numerical = [
    "annual_income",
    "lead_score",
    "number_of_courses_viewed",
    "interaction_count",
]

categorical = [
    "lead_source",
    "industry",
    "employment_status",
    "location",
]

features = numerical + categorical


# Imputación de valores faltantes
missing_before = df[features].isna().sum()
print("Missing values before preparation:")
print(missing_before[missing_before > 0])

numerical_imputer = SimpleImputer(
    strategy="constant",
    fill_value=0.0,
)

categorical_imputer = SimpleImputer(
    strategy="constant",
    fill_value="NA",
)

df.loc[:, numerical] = numerical_imputer.fit_transform(df[numerical])
df.loc[:, categorical] = categorical_imputer.fit_transform(df[categorical])

missing_after = df[features].isna().sum()

print("\nMissing values after preparation:")
print(missing_after[missing_after > 0])

assert missing_after.eq(0).all()


# División exacta exigida por HW04
df_full_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=1,
)

df_train, df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=1,
)


# Separar target de las matrices de características
y_train = df_train[target]
y_val = df_val[target]
y_test = df_test[target]

X_train = df_train.drop(columns=[target])
X_val = df_val.drop(columns=[target])
X_test = df_test.drop(columns=[target])

assert target not in X_train.columns
assert target not in X_val.columns
assert target not in X_test.columns
# %%
# -------------------------------------------------------------------------------------
# Q1 Which numerical variable (among the following 4) has the highest AUC?
# -------------------------------------------------------------------------------------
auc_scores = univariate_roc_auc_scores(
    X=X_train,
    y=y_train,
    features=numerical,
)

print(auc_scores.round(3))
print("The numerical variable with more highest AUC is: ", auc_scores.loc[auc_scores["auc"].idxmax(), "feature"])

# %%
# -------------------------------------------------------------------------------------
# Q2 What's the AUC of this model on the validation dataset? (round to 3 digits)
# -------------------------------------------------------------------------------------

train_dicts = X_train.to_dict(orient="records")
val_dicts = X_val.to_dict(orient="records")

dv = DictVectorizer(sparse=False)

X_train_encoded = dv.fit_transform(train_dicts)
X_val_encoded = dv.transform(val_dicts)

model = LogisticRegression(
    solver="liblinear",
    C=1.0,
    max_iter=1000,
)

model.fit(X_train_encoded, y_train)

y_val_pred = model.predict_proba(X_val_encoded)[:, 1]

validation_auc = roc_auc_score(y_val, y_val_pred)

print("The AUC for this val dataset is: ", round(validation_auc, 3))
# %%
# -------------------------------------------------------------------------------------
# Q3 At which threshold precision and recall curves intersect?
# -------------------------------------------------------------------------------------
thresholds = np.linspace(0.0, 1.0, 101)

threshold_scores = precision_recall_by_thresholds(
    y_true=y_val,
    scores=y_val_pred,
    thresholds=thresholds,
)

plt.plot(
    threshold_scores["threshold"],
    threshold_scores["precision"],
    label="Precision",
)
plt.plot(
    threshold_scores["threshold"],
    threshold_scores["recall"],
    label="Recall",
)

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.show()

valid_thresholds = threshold_scores.loc[
    ~(
        threshold_scores["precision"].eq(0)
        & threshold_scores["recall"].eq(0)
    )
].copy()

valid_thresholds["difference"] = (
    valid_thresholds["precision"] - valid_thresholds["recall"]
).abs()

intersection = valid_thresholds.loc[
    valid_thresholds["difference"].idxmin()
]

print(
    f"Precision and recall intersect at threshold: "
    f"{intersection['threshold']:.2f}"
)
# %%
# -------------------------------------------------------------------------------------
# Q4 At which threshold F1 is maximal?
# -------------------------------------------------------------------------------------
threshold_scores["f1"] = (
    2
    * threshold_scores["precision"]
    * threshold_scores["recall"]
    / (
        threshold_scores["precision"]
        + threshold_scores["recall"]
    )
).fillna(0.0)

best_f1 = threshold_scores.loc[threshold_scores["f1"].idxmax()]

print(f"F1 is maximal at threshold: {best_f1['threshold']:.2f}")
# %%
# -------------------------------------------------------------------------------------
# Q5 How large is standard deviation of the scores across different folds?
# -------------------------------------------------------------------------------------
cv_scores = logistic_auc_cv_scores(
    df_full_train,
    target=target,
    C=1.0,
)

cv_std = np.std(cv_scores)

print(cv_scores)
print(f"Standard deviation of AUC scores: {cv_std:.3f}")
# %%
# -------------------------------------------------------------------------------------
# Q6 Which C leads to the best mean ROC AUC?
# -------------------------------------------------------------------------------------
c_values = [0.000001, 0.001, 1]

tuning_results = tune_logistic_c_values(
    df_full_train,
    target=target,
    c_values=c_values,
)

print(tuning_results.to_string(index=False))

best_result = (
    tuning_results
    .sort_values(
        by=[
            "mean_auc_rounded",
            "std_auc_rounded",
            "C",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )
    .iloc[0]
)

print(f"Best C: {best_result['C']}")
# %%
