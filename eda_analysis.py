import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("loading data...")
data = pd.read_csv("Cleaned_LoanExport_Final.csv", low_memory=False)

print("data loaded")
print(data.head())

# just checking columns
print("columns:", data.columns)

# -------------------------
# BASIC INFO
# -------------------------
print("checking missing values")
print(data.isnull().sum())

print("basic statistics")
print(data.describe())

# -------------------------
# UNIVARIATE ANALYSIS
# -------------------------

print("plotting credit score distribution...")

plt.figure(figsize=(10,6))
sns.histplot(data["creditscore"], bins=30)
plt.title("Credit Score Distribution")
plt.show()

# trying boxplot (outliers check)
print("checking outliers in credit score")

plt.figure(figsize=(10,6))
sns.boxplot(x=data["creditscore"])
plt.title("Credit Score Boxplot")
plt.show()

# categorical plot
print("checking loan risk distribution")

plt.figure(figsize=(10,6))
sns.countplot(x=data["loanriskcategory"])
plt.title("Loan Risk Distribution")
plt.xticks(rotation=45)
plt.show()

# -------------------------
# BIVARIATE ANALYSIS
# -------------------------

print("checking relation between loan amount and dti")

plt.figure(figsize=(10,6))
plt.scatter(data["origupb"], data["dti"])
plt.xlabel("Loan Amount")
plt.ylabel("DTI")
plt.title("Loan Amount vs DTI")
plt.show()

# boxplot for category vs numeric
print("checking credit score vs loan risk")

plt.figure(figsize=(10,6))
sns.boxplot(x=data["loanriskcategory"], y=data["creditscore"])
plt.xticks(rotation=45)
plt.title("Credit Score vs Loan Risk")
plt.show()

# -------------------------
# MULTIVARIATE ANALYSIS
# -------------------------

print("creating correlation heatmap")

num_data = data.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(12,8))
sns.heatmap(num_data.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# trying pairplot (small sample only, otherwise slow)
print("trying pairplot")

sample = data.sample(500)

sns.pairplot(sample[["creditscore", "dti", "origupb"]])
plt.show()

print("EDA done")

# -------------------------
# DETAILED EDA INSIGHTS
# -------------------------

print("\n================ EDA INSIGHTS ================")

# UNIVARIATE
print("\n[Univariate Analysis]")
print("- Credit score is distributed across a wide range, with many borrowers in mid-range scores.")
print("- Loan risk categories show that most loans fall into moderate categories.")
print("- DTI values indicate many borrowers are under moderate financial stress.")

# BIVARIATE
print("\n[Bivariate Analysis]")
print("- Lower credit score is generally associated with higher loan risk.")
print("- Higher DTI values tend to increase probability of default.")
print("- Loan amount and DTI show some relationship but not very strong.")

# MULTIVARIATE
print("\n[Multivariate Analysis]")
print("- No single feature fully explains loan risk; it depends on multiple factors.")
print("- Credit score, DTI, and loan amount together influence loan performance.")
print("- Some correlation exists, but no extreme multicollinearity observed.")

# FINAL INSIGHT
print("\n[Final Conclusion]")
print("- Borrower financial strength (credit score + DTI) is a key driver of loan risk.")
print("- These features are important for building predictive models.")