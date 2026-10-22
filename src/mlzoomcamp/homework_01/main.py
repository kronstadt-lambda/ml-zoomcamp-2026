# %%
"""
Script en python para poder resolver el Homework01
"""
# %%
import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DATASET_PATH = DATA_DIR / "car_fuel_efficiency_2026.csv"

# %%
# -------------------------------------------------------------------------------------
# Q1 Pandas version
# -------------------------------------------------------------------------------------
pd.__version__

# %%
# -------------------------------------------------------------------------------------
# Q2 How many records are in the dataset?
# -------------------------------------------------------------------------------------
ds = pd.read_csv(DATASET_PATH)
print("Records count: ", ds.shape[0])

# %%
# -------------------------------------------------------------------------------------
# Q3 How many fuel types are presented in the dataset?
# -------------------------------------------------------------------------------------
print("Number of Fuel types: ", len( ds.fuel_type.unique() ))


# %%
# -------------------------------------------------------------------------------------
# Q4 How many columns in the dataset have missing values?
# -------------------------------------------------------------------------------------
missing_values_per_column = ds.isnull().sum()
columns_with_missing_values = missing_values_per_column[missing_values_per_column > 0]
print("Number of columns with missing values:", len(columns_with_missing_values))

# %%
# -------------------------------------------------------------------------------------
# Q5 What's the maximum fuel efficiency of cars from Asia?
# -------------------------------------------------------------------------------------
print("El maximo valor de fuel efficiency para origen Asia es",
      ds.loc[ ds['origin'].eq('Asia'), 'fuel_efficiency_mpg'].max())

# %%
# -------------------------------------------------------------------------------------
# Q6 Median value of horsepower
# -------------------------------------------------------------------------------------
horsepower = ds['horsepower']
median_before = horsepower.median()
most_freq = horsepower.mode().iat[0]
ds['horsepower'] = horsepower.fillna(most_freq)

median_after = ds['horsepower'].median()

print("Did the median value of Horsepower change?: ", "No" if median_before == median_after else "Yes")

# %%
# -------------------------------------------------------------------------------------
# Q7 Sum of weights
# -------------------------------------------------------------------------------------
carsFromAsia = ds.loc[ds['origin'].eq('Asia'), ['vehicle_weight', 'model_year']]
sevenValues = carsFromAsia.head(7)
X = sevenValues.to_numpy()
XTX = X.T.dot(X)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
XTX_inv = np.linalg.inv(XTX)
w = XTX_inv.dot(X.T).dot(y)
print("La suma de los valores de `w` es: ", w.sum())
# %%
