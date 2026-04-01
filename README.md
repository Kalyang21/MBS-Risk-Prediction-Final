# Mortgage-Backed Securities Analysis and Risk Prediction

## Overview

This project focuses on analyzing and predicting loan risk in Mortgage-Backed Securities (MBS) using machine learning techniques.

The goal is to understand borrower behavior and identify factors that influence loan performance, including default probability, financial risk, and loan characteristics such as credit score and debt-to-income ratio (DTI).

The analysis is based on Freddie Mac’s Single-Family Loan Performance dataset, which contains large-scale real-world mortgage data with detailed information on borrower profiles, credit performance, and loan status over time.

Using this data, I applied data analysis and machine learning methods to explore patterns and build predictive models for loan risk.

---

## Data Source

Freddie Mac Single-Family Loan Performance Data (Last 5 Years)

The dataset includes large-scale mortgage records with:

* Borrower credit information
* Loan characteristics
* Monthly loan performance status
* Delinquency and risk indicators

This dataset provides a strong foundation for analyzing financial risk in mortgage-backed securities.

---

## Approach

### 1. Data Preprocessing

* Loaded and explored the dataset
* Handled missing values using median imputation
* Encoded the target variable
* Removed unnecessary columns
* Selected relevant numerical features

### 2. Exploratory Data Analysis (EDA)

* Analyzed distributions of key variables such as credit score and loan risk
* Studied relationships between credit score, DTI, and loan risk
* Visualized patterns using histograms, boxplots, scatter plots, and heatmaps
* Derived insights through univariate, bivariate, and multivariate analysis

### 3. Feature Engineering

* Filtered and prepared relevant features
* Applied standardization using scaling
* Used sampling to optimize model training time

### 4. Model Building

The following models were trained:

* Support Vector Classifier (SVC)
* Logistic Regression
* Random Forest

### 5. Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

Random Forest achieved the best overall performance among the models.

---

## Key Insights

* Credit score and DTI are strong indicators of loan risk
* Lower credit scores are generally associated with higher default probability
* Higher DTI reflects increased financial stress and higher risk
* Loan performance is influenced by a combination of factors rather than a single variable

---

## Model Comparison (Summary)

| Model               | Observation                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| Logistic Regression | Good baseline performance, assumes linear relationships                 |
| SVC                 | Performs reasonably well but less effective for complex patterns        |
| Random Forest       | Best performance due to its ability to capture non-linear relationships |

---

## Scope and Next Steps

This project was designed as a structured implementation of machine learning for financial risk prediction, with a focus on clarity and interpretability.

To keep the workflow efficient, a sampled dataset was used during model training.

Further improvements that can enhance this work include:

* Training models on the full dataset for better generalization
* Performing hyperparameter tuning
* Exploring advanced models such as XGBoost or LightGBM
* Extending feature engineering with additional domain-specific features

---

## Project Structure

eda_analysis.py
kalyan_feature_engineering.py
kalyan_SVC_model.ipynb
EDA plot images (PNG files)
README.md

---

## How to Run

1. Clone the repository

2. Place the dataset file (`Cleaned_LoanExport_Final.csv`) in the project directory

3. Run the EDA script:

   python eda_analysis.py

4. Open the notebook for model training and evaluation:

   kalyan_SVC_model.ipynb

---

## Conclusion

This project demonstrates how machine learning techniques can be applied to financial risk prediction using real-world mortgage data.

The results highlight the importance of borrower financial indicators such as credit score and DTI, and show that ensemble methods like Random Forest are effective in capturing complex patterns in structured financial datasets.

---


