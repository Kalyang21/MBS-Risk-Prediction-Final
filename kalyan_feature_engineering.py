# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 2. Load Preprocessed Data
df = pd.read_csv('Cleaned_LoanExport_Final.csv', low_memory=False)
print(" Data loaded successfully!")

# 3. Drop Irrelevant Features
columns_to_drop = ['LoanID', 'CustomerID']  # adjust names if needed
df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
print(" Dropped irrelevant columns.")

# 4. Safe New Feature Creation

# Loan Age in Months (only if OriginationDate exists)
if 'OriginationDate' in df.columns:
    df['OriginationDate'] = pd.to_datetime(df['OriginationDate'], errors='coerce')
    df['LoanAgeMonths'] = (pd.Timestamp.now() - df['OriginationDate']).dt.days // 30
    df['OriginationYear'] = df['OriginationDate'].dt.year
    df['OriginationMonth'] = df['OriginationDate'].dt.month
    print(" Created LoanAgeMonths, OriginationYear, OriginationMonth features.")
else:
    print("OriginationDate column not found. Skipping LoanAgeMonths, OriginationYear, OriginationMonth creation.")

# Create DTI_Ratio (safe check Loan Amount / Borrower Income)
loan_amount_col = None
borrower_income_col = None

# Search for correct loan amount column
for col in df.columns:
    if 'loanamount' in col.lower().replace(' ', ''):
        loan_amount_col = col
    if 'borrowerincome' in col.lower().replace(' ', ''):
        borrower_income_col = col

if loan_amount_col and borrower_income_col:
    df['DTI_Ratio'] = df[loan_amount_col] / df[borrower_income_col]
    print(f" Created DTI_Ratio using '{loan_amount_col}' and '{borrower_income_col}' columns.")
else:
    print("Loan Amount or Borrower Income column not found. Skipping DTI_Ratio creation.")

# Create Credit Score Band
def credit_score_band(score):
    if pd.isnull(score):
        return np.nan
    if score >= 750:
        return 'Excellent'
    elif score >= 700:
        return 'Good'
    elif score >= 650:
        return 'Fair'
    else:
        return 'Poor'

# Find Credit Score Column
credit_score_col = None
for col in df.columns:
    if 'creditscore' in col.lower().replace(' ', ''):
        credit_score_col = col

if credit_score_col:
    # Converting Credit Score to numeric first
    df[credit_score_col] = pd.to_numeric(df[credit_score_col], errors='coerce')

    # Now apply credit_score_band safely
    df['CreditScoreBand'] = df[credit_score_col].apply(credit_score_band)
    print(f" Created CreditScoreBand from '{credit_score_col}' column.")
else:
    print("Credit Score column not found. Skipping CreditScoreBand creation.")

# 5. One-Hot Encode Categorical Columns (if present)
categorical_cols = ['LoanPurpose', 'PropertyType', 'CreditScoreBand']
for col in categorical_cols:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

print(" Categorical Encoding Completed.")

# 6. Handle Missing Numerical Values
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

print(" Handled missing numerical values.")

# 7. Standard Scaling
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print(" Standardized numerical features.")

# 8. Save Final Engineered Dataset
df.to_csv('engineered_data.csv', index=False)
print("\n Feature Engineering Completed Successfully!")
print(" Final dataset saved as 'engineered_data.csv'.")

