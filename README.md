# Mortgage-Backed Securities Analysis and Prediction

## 📌 Overview
This project applies Machine Learning and Data Analytics techniques to analyze and predict **prepayment risk, default probability, and loss severity** in Mortgage-Backed Securities (MBS) using **Freddie Mac's loan performance data**.

## 📊 Data Source
- Freddie Mac Single-Family Loan Performance Data (Last 5 Years)
- Covers **53.8 million mortgages** with credit performance, loss details, and monthly loan status.

## 🚀 Project Pipeline
1. **Data Preprocessing** – Cleaning, feature engineering, and encoding.
2. **Exploratory Data Analysis** – Trend analysis and visualization.
3. **Machine Learning Models**:
   - Prepayment Risk Prediction (XGBoost, Random Forest)
   - Default Probability Classification (Logistic Regression, CatBoost)
   - Loss Severity Estimation (LSTM, LightGBM)
4. **Model Evaluation & Optimization** – Using AUC-ROC, RMSE, and Hyperparameter Tuning.
5. **Deployment** – Interactive risk assessment dashboard.

## 📜 Setup Instructions
Clone the repository and install dependencies:
```bash
git clone https://github.com/Technocolabs100/Mortgage-Backed-Securities-Analysis-and-Prediction.git
cd Mortgage-Backed-Securities-Analysis-Prediction
pip install -r requirements.txt
