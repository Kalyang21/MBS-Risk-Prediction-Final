# -----------------------------
# data preprocessing (my attempt)
# -----------------------------

import pandas as pd
import numpy as np

print("loading data...")

df = pd.read_csv("Cleaned_LoanExport_Final.csv", low_memory=False)

print("data loaded")
print("shape:", df.shape)

# -----------------------------
# checking duplicates
# -----------------------------

print("\nchecking duplicates...")

dup = df.duplicated().sum()
print("duplicates found:", dup)

if dup > 0:
    print("removing duplicates...")
    df = df.drop_duplicates()
    print("done removing")

# -----------------------------
# dropping unnecessary stuff
# -----------------------------

print("\nremoving some columns i think not needed")

cols_to_drop = ['LoanID', 'CustomerID']

for col in cols_to_drop:
    if col in df.columns:
        df = df.drop(col, axis=1)

print("columns left:", len(df.columns))

# -----------------------------
# missing values
# -----------------------------

print("\nchecking missing values...")

missing_before = df.isnull().sum().sum()
print("missing before:", missing_before)

# filling numeric columns with median (seems safe)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

missing_after = df.isnull().sum().sum()
print("missing after:", missing_after)

# -----------------------------
# encoding target
# -----------------------------

print("\nencoding target column...")

target = "loanriskcategory"

if df[target].dtype == 'object':
    df[target] = df[target].astype('category').cat.codes

print("target encoded")

# -----------------------------
# selecting features
# -----------------------------

print("\nselecting features...")

drop_cols = [target, 'firstpaymentdate', 'maturitydate', 'computedmaturity']

X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# only numeric for now (keeping it simple)
X = X.select_dtypes(include=['int64', 'float64'])

y = df[target]

print("X shape:", X.shape)

# -----------------------------
# outliers (just trying something basic)
# -----------------------------

print("\nhandling extreme values...")

for col in X.columns:
    low = X[col].quantile(0.01)
    high = X[col].quantile(0.99)
    X[col] = np.clip(X[col], low, high)

print("clipped extreme values")

# -----------------------------
# final dataset
# -----------------------------

final_df = pd.concat([X, y], axis=1)

print("\nfinal shape:", final_df.shape)

# -----------------------------
# saving
# -----------------------------

final_df.to_csv("processed_data.csv", index=False)

print("\ndone preprocessing, saved file as processed_data.csv")