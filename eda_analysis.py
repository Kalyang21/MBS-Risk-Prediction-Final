import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading data
print("loading data...")
data = pd.read_csv("Cleaned_LoanExport_Final.csv", low_memory=False)

print("data loaded")
print(data.head())

# just checking columns
print("columns:", data.columns)

# -------------------------
# Plot 1: Credit Score
# -------------------------
print("plotting credit score...")

plt.figure(figsize=(10,6))   # making it bigger
sns.histplot(data["creditscore"], bins=30)

plt.title("Credit Score Distribution")

plt.show()   # manually save from popup

# -------------------------
# Plot 2: Loan vs DTI
# -------------------------
print("plotting loan vs dti...")

plt.figure(figsize=(10,6))
plt.scatter(data["origupb"], data["dti"])

plt.xlabel("Loan Amount")
plt.ylabel("DTI")
plt.title("Loan Amount vs DTI")

plt.show()

# -------------------------
# Plot 3: Heatmap
# -------------------------
print("creating heatmap...")

num_data = data.select_dtypes(include=["int64", "float64"])

print("numeric data shape:", num_data.shape)

plt.figure(figsize=(12,8))   # slightly bigger
sns.heatmap(num_data.corr(), cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()

print("done")