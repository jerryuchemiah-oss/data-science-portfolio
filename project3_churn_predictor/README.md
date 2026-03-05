# 📉 Customer Churn Predictor — IBM Telco Dataset

**Author:** Uche Jeremiah Nzubechukwu  
**Stack:** Python · Pandas · Scikit-learn · Matplotlib  
**Data:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 real customers  

## Overview
ML classification pipeline predicting telecom customer churn using the industry-standard IBM Telco dataset. Covers full EDA, feature engineering, 3-model comparison, and actionable business insights.

## Dataset
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it in this folder.  
**7,043 customers · 21 features · ~26% churn rate**

## Features
- Full EDA: churn by contract type, internet service, payment method, tenure, monthly charges
- Feature engineering: services count, new customer flag, charges per month ratio
- 3 models: Logistic Regression, Random Forest, Gradient Boosting (class-weight balanced)
- 5-fold Stratified CV for reliable evaluation
- ROC curves, confusion matrix, feature importance charts
- Business insights section with real churn rates by segment

## Quick Start
```bash
pip install numpy pandas scikit-learn matplotlib
# Download dataset from Kaggle first
python churn_predictor.py
```

## Results
| Model | Accuracy | AUC-ROC | CV AUC |
|-------|----------|---------|--------|
| Logistic Regression | ~80% | ~0.84 | ~0.83 |
| Random Forest | ~81% | ~0.86 | ~0.85 |
| Gradient Boosting | ~82% | ~0.87 | ~0.86 |

## Key Findings (from real data)
- Month-to-month contracts churn at **~43%** vs 11% for 2-year contracts
- Fiber optic customers churn at **~42%** — highest of any internet service
- Electronic check payment users churn at **~45%**
- Customers in their **first 12 months** are highest risk

## Upgrade Path
- SHAP values for explainability
- FastAPI deployment for live scoring
- Streamlit dashboard for business users
